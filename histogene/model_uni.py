"""
HisToGene UNI2-h 特征模型（方案A）
用 UNI2-h 预提取特征替代 ViT 图像编码器，
保留原始 HisToGene 的坐标编码和回归头结构。

架构概述：
    UNI2-h feature (1536) → Linear+LN 投影 (dim) → +坐标嵌入 → MLP 回归头 → n_targets
"""

import torch
import torch.nn as nn


class HisToGeneUNI(nn.Module):
    """
    HisToGene 方案A：UNI2-h 预提取特征 + 坐标编码 + MLP 回归头

    输入：
        - features: (B, feature_dim) UNI2-h 预提取特征
        - pos_x: (B,) X 坐标索引 (long)
        - pos_y: (B,) Y 坐标索引 (long)

    输出：
        - predictions: (B, n_targets) ssGSEA 通路评分预测
    """

    def __init__(self, feature_dim=1536, dim=1024, n_pos=128,
                 n_targets=30, mlp_dim=2048, dropout=0.3):
        super().__init__()

        # 投影层：将 UNI2-h 特征映射到模型维度
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.LayerNorm(dim)
        )

        # 坐标编码（与原始 HisToGene 完全一致）
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        # 回归头（与原始 HisToGene 的回归头结构一致）
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_targets)
        )

    def forward(self, features, pos_x, pos_y):
        """
        Args:
            features: (B, feature_dim) — UNI2-h 预提取特征
            pos_x: (B,) — X 坐标索引 (long)
            pos_y: (B,) — Y 坐标索引 (long)
        Returns:
            predictions: (B, n_targets) — ssGSEA 通路评分预测
        """
        x = self.proj(features)                          # (B, dim)
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)  # 加入坐标编码
        x = self.head(x)                                 # (B, n_targets)
        return x

    def count_parameters(self):
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
