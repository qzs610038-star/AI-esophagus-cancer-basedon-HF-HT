"""
EGN-v1 模型定义 - PFMval 项目
基于 ViT-Large + GCN + Exemplar Graph Neural Network 的通路评分预测模型

与 EGNv2 的关键差异：
- 特征提取器: ViT-Large (dim=1024, depth=8, heads=16) vs ResNet-50
- GNN: GCN (GCNConv) vs GraphSAGE (SAGEConv)
- 图构建: KNN 图 (特征相似性) vs 空间半径图 (坐标距离)
- hidden_dim: 1024 (匹配 ViT 输出) vs 512
"""

import torch
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 尝试导入 timm，如果不可用则使用自实现 ViT
_USE_TIMM = False
try:
    import timm
    _USE_TIMM = True
except ImportError:
    print("[ViT] timm 库不可用，使用自实现 ViT-Large")


# ═══════════════════════════════════════════════════════════════
#  自实现 ViT 组件
# ═══════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """将图像分割为 patches 并嵌入"""

    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_dim=1024, num_heads=16, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLPBlock(nn.Module):
    """Transformer MLP 块"""

    def __init__(self, embed_dim=1024, mlp_dim=4096, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, embed_dim=1024, num_heads=16, mlp_dim=4096, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CustomViTLarge(nn.Module):
    """
    自实现 ViT-Large
    patch_size=32, embed_dim=1024, depth=8, num_heads=16, mlp_dim=4096
    """

    def __init__(self, img_size=224, patch_size=32, embed_dim=1024,
                 depth=8, num_heads=16, mlp_dim=4096, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # 取 CLS token 作为全局特征
        cls_out = x[:, 0]  # (B, embed_dim)
        return cls_out


# ═══════════════════════════════════════════════════════════════
#  ViT 特征提取器
# ═══════════════════════════════════════════════════════════════

class ViTFeatureExtractor(nn.Module):
    """
    基于 ViT-Large 的特征提取器
    优先使用 timm 库加载预训练权重，不可用时使用自实现
    输入: (B, 3, 224, 224) → 输出: (B, 1024)
    """

    def __init__(self, freeze_layers=4, img_size=224, patch_size=32,
                 embed_dim=1024, depth=8, num_heads=16, mlp_dim=4096,
                 dropout=0.0):
        """
        Args:
            freeze_layers: 冻结前 N 个 Transformer 块
            img_size: 输入图像尺寸
            patch_size: patch 尺寸
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            mlp_dim: MLP 隐藏维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.freeze_layers = freeze_layers

        if _USE_TIMM:
            try:
                # 加载标准 ViT-Large 预训练权重，然后截断到前 depth 层
                self.vit = timm.create_model(
                    'vit_large_patch32_224',
                    pretrained=True,
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                # 截断 blocks 到指定深度以匹配自定义配置
                if hasattr(self.vit, 'blocks') and depth < len(self.vit.blocks):
                    self.vit.blocks = self.vit.blocks[:depth]
                    print(f"[ViTFeatureExtractor] 使用 timm 预训练 ViT-Large (patch32, 截断到 {depth} 层)")
                else:
                    print(f"[ViTFeatureExtractor] 使用 timm 预训练 ViT-Large (patch32, {depth} 层)")
                self._use_timm = True
            except Exception as e:
                print(f"[ViTFeatureExtractor] timm 加载失败: {e}，使用自实现")
                self.vit = CustomViTLarge(
                    img_size=img_size, patch_size=patch_size,
                    embed_dim=embed_dim, depth=depth,
                    num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
                )
                self._use_timm = False
        else:
            self.vit = CustomViTLarge(
                img_size=img_size, patch_size=patch_size,
                embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
            )
            self._use_timm = False

        # 冻结指定层
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n):
        """冻结前 n 个 Transformer 块"""
        if self._use_timm:
            # timm ViT 的 blocks
            blocks = self.vit.blocks if hasattr(self.vit, 'blocks') else []
            for i, block in enumerate(blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = False
            # 冻结 patch_embed 和 pos_embed
            if hasattr(self.vit, 'patch_embed'):
                for param in self.vit.patch_embed.parameters():
                    param.requires_grad = False
            if hasattr(self.vit, 'pos_embed'):
                self.vit.pos_embed.requires_grad = False
            if hasattr(self.vit, 'cls_token'):
                self.vit.cls_token.requires_grad = False
        else:
            # 自实现 ViT
            for i, block in enumerate(self.vit.blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = False
            # 冻结 patch_embed 和位置编码
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            self.vit.pos_embed.requires_grad = False
            self.vit.cls_token.requires_grad = False

        frozen_count = sum(1 for p in self.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.parameters())
        print(f"[ViTFeatureExtractor] 冻结 {frozen_count}/{total_count} 个参数组 (前 {n}/{len(self.vit.blocks) if hasattr(self.vit, 'blocks') else 0} 层)")

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) 输入图像
        Returns:
            (B, 1024) 特征向量
        """
        if self._use_timm:
            # timm ViT 的 forward_features 返回 (B, num_tokens, embed_dim)
            feats = self.vit.forward_features(x)
            if feats.ndim == 3:
                feats = feats[:, 0]  # 取 CLS token
        else:
            feats = self.vit(x)  # (B, 1024)
        return feats


# ═══════════════════════════════════════════════════════════════
#  Exemplar 库
# ═══════════════════════════════════════════════════════════════

class ExemplarLibrary:
    """代表库管理 - 存储训练集特征和目标，支持 KNN 检索"""

    def __init__(self, features, targets):
        """
        Args:
            features: Tensor(K, D) 代表特征
            targets:  Tensor(K, 30) 代表目标值
        """
        self.features = features.cpu()
        self.targets = targets.cpu()
        # 预拟合 NearestNeighbors
        self._nn = NearestNeighbors(
            n_neighbors=min(10, len(features)),
            algorithm='auto', metric='euclidean'
        )
        self._nn.fit(self.features.numpy())

    def retrieve(self, query_features, k=10):
        """
        查询最近 k 个代表

        Args:
            query_features: Tensor(N, D) 查询特征
            k: 最近邻数量
        Returns:
            indices: (N, k) 最近邻索引
            distances: (N, k) 最近邻距离
        """
        if k > len(self.features):
            k = len(self.features)
        query_np = query_features.detach().cpu().numpy()
        distances, indices = self._nn.kneighbors(query_np, n_neighbors=k)
        return indices, distances

    def get_features(self, indices):
        """获取指定索引的代表特征"""
        return self.features[indices]

    def get_targets(self, indices):
        """获取指定索引的代表目标值"""
        return self.targets[indices]

    def save(self, path):
        """保存代表库到文件"""
        torch.save({
            'features': self.features,
            'targets': self.targets,
        }, path)
        print(f"[ExemplarLibrary] 已保存到 {path}")

    @classmethod
    def load(cls, path):
        """从文件加载代表库"""
        data = torch.load(path, weights_only=False)
        lib = cls(data['features'], data['targets'])
        print(f"[ExemplarLibrary] 已从 {path} 加载 {len(lib.features)} 个代表")
        return lib


# ═══════════════════════════════════════════════════════════════
#  EGN-v1 完整模型
# ═══════════════════════════════════════════════════════════════

class EGNv1Model(nn.Module):
    """
    完整 EGN-v1 模型（图级训练版本）

    架构：ViT 特征提取 → GCN 图卷积(2层) → Exemplar 融合 → MLP 回归器

    输入:
        node_features: (N, in_dim) ViT 提取的特征
        edge_index: (2, E) 图边
        exemplar_agg_features: (N, hidden_dim) 聚合后的 exemplar 特征（可选）

    输出:
        (N, n_targets) 每个节点的 30 通路预测
    """

    def __init__(self, in_dim=1024, hidden_dim=1024, n_targets=30,
                 graph_layers=2, dropout=0.5, k_exemplars=10):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_targets = n_targets
        self.k_exemplars = k_exemplars

        # 1. 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. GCN 图卷积层（使用 GCNConv，不是 SAGEConv）
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for _ in range(graph_layers):
            self.graph_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.graph_norms.append(nn.LayerNorm(hidden_dim))

        self.graph_dropout = nn.Dropout(dropout)

        # 3. Exemplar 融合层
        self.exemplar_fuse = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, node_features, edge_index, exemplar_agg_features=None):
        """
        Args:
            node_features: (N, in_dim) 节点特征
            edge_index: (2, E) 边索引
            exemplar_agg_features: (N, hidden_dim) 聚合后的 exemplar 特征（可选）
        Returns:
            (N, n_targets) 预测值
        """
        # 1. 特征投影
        h = self.feature_proj(node_features)  # (N, hidden_dim)

        # 2. GCN 图卷积
        for conv, norm in zip(self.graph_convs, self.graph_norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = h + h_res  # 残差连接
            h = nn.functional.gelu(h)
            h = self.graph_dropout(h)

        # 3. Exemplar 融合（如果提供了 exemplar 特征）
        if exemplar_agg_features is not None:
            h = self.exemplar_fuse(torch.cat([h, exemplar_agg_features], dim=-1))
        # 否则直接使用图卷积输出

        # 4. 回归
        out = self.regressor(h)  # (N, n_targets)
        return out
