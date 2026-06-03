"""
model_online_cls.py — 在线 CLS 模式模型
=================================

将 UNI2-H backbone（可选 LoRA）+ CLS token → 投影 → 坐标编码 → MLP 回归头。
架构与 histogene/model_uni.py 的 HisToGeneUNI 完全一致，
唯一区别：输入是原始图像 [B,3,224,224] 而非预提取特征 [B,1536]。

用法:
    from uni2h.uni2h_utils import load_uni2h_backbone
    backbone, transform, _ = load_uni2h_backbone(device=device)
    model = OnlineCLSModel(backbone).to(device)
    preds = model(images, pos_x, pos_y)  # images: [B,3,224,224], preds: [B,30]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OnlineCLSModel(nn.Module):
    """在线 CLS 模型：图像 → UNI2-H backbone → CLS → MLP 回归。

    下游架构与 HisToGeneUNI 完全相同：
        feature[1536] → Linear+LN(1024) → +coords → LN+Linear(2048)+GELU+Dropout+Linear(30)
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 1536,
        dim: int = 1024,
        n_pos: int = 128,
        n_targets: int = 30,
        mlp_dim: int = 2048,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: UNI2-H ViT 模型（可含 LoRA 注入）
            feature_dim: backbone 输出特征维度（UNI2-H=1536）
            dim: 内部模型维度
            n_pos: 坐标编码表大小
            n_targets: 输出通路数
            mlp_dim: MLP 隐藏层维度
            dropout: dropout 概率
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # ── 投影层（与 HisToGeneUNI.proj 完全一致）──
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.LayerNorm(dim),
        )

        # ── 坐标编码（与 HisToGeneUNI 一致）──
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        # ── 回归头（与 HisToGeneUNI.head 一致）──
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_targets),
        )

        self.n_targets = n_targets

    def forward(
        self,
        images: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224] 原始 H&E patch（已预处理）
            pos_x: [B] X 坐标索引 (long, [0, n_pos-1])
            pos_y: [B] Y 坐标索引 (long, [0, n_pos-1])

        Returns:
            predictions: [B, n_targets] ssGSEA 通路评分预测
        """
        # 通过 backbone 提取 CLS token
        features = self.backbone(images)   # [B, feature_dim]

        # 下游（与 HisToGeneUNI.forward 完全一致）
        x = self.proj(features)            # [B, dim]
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)
        x = self.head(x)                   # [B, n_targets]
        return x

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """仅返回 backbone CLS 特征，用于分析/调试。"""
        return self.backbone(images)

    def count_parameters(self) -> int:
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_param_summary(self) -> None:
        """按模块打印参数统计。"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        embed_params = (sum(p.numel() for p in self.x_embed.parameters() if p.requires_grad) +
                        sum(p.numel() for p in self.y_embed.parameters() if p.requires_grad))
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total = backbone_params + proj_params + embed_params + head_params

        print(f"OnlineCLSModel 可训练参数: {total:,} ({total/1e6:.2f}M)")
        print(f"  Backbone (含LoRA): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  投影层:             {proj_params:,}")
        print(f"  坐标嵌入:           {embed_params:,}")
        print(f"  回归头:             {head_params:,}")


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("OnlineCLSModel 自检")
    print("=" * 60)

    # 用 nn.Sequential 模拟 backbone（轻量自检，不加载真实 UNI2-H）
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return x.mean(dim=[2, 3])[:, :1536]  # 从图像伪造 CLS

    dummy = nn.Sequential(
        nn.Conv2d(3, 1536, 14, 14),  # 模拟 ViT patch embedding
        DummyBackbone(),
    )

    model = OnlineCLSModel(dummy, feature_dim=1536)
    total = model.count_parameters()
    print(f"总可训练参数: {total:,} ({total/1e6:.2f}M)")

    # 测试前向
    images = torch.randn(4, 3, 224, 224)
    pos_x = torch.randint(0, 128, (4,))
    pos_y = torch.randint(0, 128, (4,))

    with torch.no_grad():
        out = model(images, pos_x, pos_y)
    assert out.shape == (4, 30), f"Expected (4,30), got {out.shape}"

    model.print_param_summary()
    print("  前向传播测试通过 ✓")
    print("=" * 60)
