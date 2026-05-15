"""
HisToGene UNI2-h Token序列模型
使用轻量级 Transformer 编码器处理 UNI2-h 完整 token 序列，
保留原始 HisToGene 的坐标编码和回归头结构。

架构概述：
    UNI2-h tokens [B, num_tokens, 1536]
    → LightweightTokenEncoder → [B, 1536]
    → Linear+LN 投影 (dim)
    → +坐标嵌入
    → MLP 回归头
    → n_targets
"""

import torch
import torch.nn as nn


class LightweightTokenEncoder(nn.Module):
    """将 [B, num_tokens, 1536] 编码为 [B, embed_dim]
    
    使用投影到较小维度 + Transformer Encoder + 全局平均池化 + 输出投影的轻量设计。
    参数量约 3.2M，控制在合理范围。
    """

    def __init__(self, embed_dim=1536, hidden_dim=512, n_heads=8, n_layers=1, dropout=0.3):
        super().__init__()
        # 输入投影：1536 → hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 输出投影：hidden_dim → embed_dim
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        Args:
            tokens: [B, num_tokens, 1536]
        Returns:
            encoded: [B, embed_dim]
        """
        x = self.input_proj(tokens)      # [B, num_tokens, hidden_dim]
        x = self.encoder(x)              # [B, num_tokens, hidden_dim]
        x = x.mean(dim=1)               # [B, hidden_dim] 全局平均池化
        x = self.output_proj(x)          # [B, embed_dim]
        x = self.norm(x)
        return x


class HisToGeneUNITokens(nn.Module):
    """
    HisToGene Token序列模型：UNI2-h token序列 + 轻量编码器 + 坐标编码 + MLP 回归头

    输入：
        - tokens: (B, num_tokens, feature_dim) UNI2-h token序列
        - pos_x: (B,) X 坐标索引 (long)
        - pos_y: (B,) Y 坐标索引 (long)

    输出：
        - predictions: (B, n_targets) ssGSEA 通路评分预测
    """

    def __init__(self, feature_dim=1536, dim=1024, n_pos=128,
                 n_targets=30, mlp_dim=2048, dropout=0.3,
                 encoder_hidden_dim=512, n_encoder_layers=1, n_encoder_heads=8):
        super().__init__()

        # Token编码器：将token序列编码为单个特征向量
        self.token_encoder = LightweightTokenEncoder(
            embed_dim=feature_dim,
            hidden_dim=encoder_hidden_dim,
            n_heads=n_encoder_heads,
            n_layers=n_encoder_layers,
            dropout=dropout
        )

        # 以下与 HisToGeneUNI 完全一致
        # 投影层：将特征映射到模型维度
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.LayerNorm(dim)
        )

        # 坐标编码
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        # 回归头
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_targets)
        )

    def forward(self, tokens, pos_x, pos_y):
        """
        Args:
            tokens: (B, num_tokens, feature_dim) — UNI2-h token序列
            pos_x: (B,) — X 坐标索引 (long)
            pos_y: (B,) — Y 坐标索引 (long)
        Returns:
            predictions: (B, n_targets) — ssGSEA 通路评分预测
        """
        encoded = self.token_encoder(tokens)  # (B, feature_dim)
        x = self.proj(encoded)                # (B, dim)
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)  # 加入坐标编码
        x = self.head(x)                      # (B, n_targets)
        return x

    def count_parameters(self):
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = HisToGeneUNITokens()
    total_params = model.count_parameters()
    print(f"HisToGeneUNITokens 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print()

    # 分模块统计
    encoder_params = sum(p.numel() for p in model.token_encoder.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in model.proj.parameters() if p.requires_grad)
    embed_params = sum(p.numel() for p in model.x_embed.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in model.y_embed.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    print(f"  Token Encoder: {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    print(f"  投影层: {proj_params:,} ({proj_params/1e6:.2f}M)")
    print(f"  坐标嵌入: {embed_params:,} ({embed_params/1e6:.2f}M)")
    print(f"  回归头: {head_params:,} ({head_params/1e6:.2f}M)")
    print()

    # 测试前向传播
    batch_size = 4
    num_tokens = 257  # UNI2-h typical token count
    dummy_tokens = torch.randn(batch_size, num_tokens, 1536)
    dummy_pos_x = torch.randint(0, 128, (batch_size,))
    dummy_pos_y = torch.randint(0, 128, (batch_size,))

    with torch.no_grad():
        output = model(dummy_tokens, dummy_pos_x, dummy_pos_y)
    print(f"  输入: tokens {dummy_tokens.shape}, pos_x {dummy_pos_x.shape}, pos_y {dummy_pos_y.shape}")
    print(f"  输出: {output.shape}")
    print("  前向传播测试通过!")
