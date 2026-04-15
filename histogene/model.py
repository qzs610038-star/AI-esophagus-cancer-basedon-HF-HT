"""
HisToGene 模型适配版 - 用于 PFMval 项目
基于原始 HisToGene (Pang et al., 2021) 的 ViT-MLP 架构
适配为 8 通路 ssGSEA 评分预测
"""

import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    """多头自注意力"""
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """单个 Transformer 块"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class HisToGeneModel(nn.Module):
    """
    HisToGene 适配版模型

    输入：
        - images: (B, 3, img_size, img_size) 的图像张量
        - pos_x: (B,) 的 x 坐标索引 (long)
        - pos_y: (B,) 的 y 坐标索引 (long)

    输出：
        - predictions: (B, n_targets) 的预测值
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 dim=1024, depth=8, heads=16, mlp_dim=2048,
                 n_pos=128, n_targets=8, dropout=0.3):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: 将图像分成小块并线性映射
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 位置编码（分离 X/Y 嵌入，保持 HisToGene 原始设计）
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Patch 位置嵌入（ViT 内部的位置编码）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer 编码器
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout)
              for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)

        # 回归头：输出 n_targets 个通路评分
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_targets)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, images, pos_x, pos_y):
        B = images.shape[0]
        p = self.patch_size

        # 1. 将图像分成 patches 并展平
        patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                            p1=p, p2=p)

        # 2. Patch embedding
        x = self.patch_embed(patches)  # (B, num_patches, dim)

        # 3. 添加空间位置编码（来自坐标）
        x_pos = self.x_embed(pos_x).unsqueeze(1)  # (B, 1, dim)
        y_pos = self.y_embed(pos_y).unsqueeze(1)  # (B, 1, dim)

        # 4. CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, dim)

        # 5. 添加 ViT 位置嵌入
        x = x + self.pos_embedding

        # 6. 将空间坐标信息加到 CLS token 上
        x = x.clone()
        x[:, 0:1, :] = x[:, 0:1, :] + x_pos + y_pos

        x = self.dropout(x)

        # 7. Transformer 编码
        x = self.transformer(x)
        x = self.norm(x)

        # 8. 取 CLS token 的输出进行回归
        cls_output = x[:, 0]  # (B, dim)

        # 9. 回归头
        output = self.head(cls_output)  # (B, n_targets)
        return output
