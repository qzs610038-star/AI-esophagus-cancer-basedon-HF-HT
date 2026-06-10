"""
model_online_cls.py — 在线 CLS 模式模型
=================================

将 UNI2-H backbone（可选 LoRA）+ CLS token → 投影 → 坐标编码 → MLP 回归头。
架构与 histogene/model_uni.py 的 HisToGeneUNI 完全一致，
唯一区别：输入是原始图像 [B,3,224,224] 而非预提取特征 [B,1536]。

v2 (2026-06-10): 新增 freq_branch 参数，支持 CLS + 频域/均值 patch 旁路融合。

用法:
    from uni2h.uni2h_utils import load_uni2h_backbone
    backbone, transform, _ = load_uni2h_backbone(device=device)
    model = OnlineCLSModel(backbone).to(device)
    preds = model(images, pos_x, pos_y)  # images: [B,3,224,224], preds: [B,30]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_online_tokens import select_uni_tokens


class OnlineCLSModel(nn.Module):
    """在线 CLS 模型：图像 → UNI2-H backbone → CLS → MLP 回归。

    下游架构与 HisToGeneUNI 完全相同：
        feature[1536] → Linear+LN(1024) → +coords → LN+Linear(2048)+GELU+Dropout+Linear(30)

    v2: 支持 freq_branch 旁路，将 patch tokens 的频域/均值特征与 CLS 融合。
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
        # ── 频域旁路参数 ──
        freq_branch: str = "none",
        token_select_mode: str = "cls_patch64",
        freq_hidden_dim: int = 512,
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
            freq_branch: patch 旁路模式
                - "none": 原始 CLS-only 路径（不改动）
                - "mean": CLS + mean(patch tokens) 旁路（非频域对照）
                - "gfnet": CLS + GFNetBlock(patch tokens) 旁路（频域实验组）
            token_select_mode: token 选择模式（同 OnlineTokenModel）
            freq_hidden_dim: GFNet 旁路的内部维度
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.freq_branch = freq_branch
        self.token_select_mode = token_select_mode

        if freq_branch == "none":
            # ── 原始 CLS-only 路径（与 v1 完全一致）──
            self.proj = nn.Sequential(
                nn.Linear(feature_dim, dim),
                nn.LayerNorm(dim),
            )
            self.x_embed = nn.Embedding(n_pos, dim)
            self.y_embed = nn.Embedding(n_pos, dim)
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, n_targets),
            )
        else:
            # ── CLS + patch 旁路路径 ──
            # CLS 分支投影
            self.branch_cls_proj = nn.Sequential(
                nn.Linear(feature_dim, dim),
                nn.LayerNorm(dim),
            )
            # 坐标编码（仅 CLS 分支使用，与原始一致）
            self.x_embed = nn.Embedding(n_pos, dim)
            self.y_embed = nn.Embedding(n_pos, dim)

            # Patch 旁路
            if freq_branch == "gfnet":
                from model_gfnet import GFNetBlock
                self.freq_patch_proj = nn.Linear(feature_dim, freq_hidden_dim)
                self.freq_blocks = nn.ModuleList([
                    GFNetBlock(seq_len=64, dim=freq_hidden_dim, dropout=dropout)
                ])
                self.freq_patch_out = nn.Sequential(
                    nn.Linear(freq_hidden_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                )
            elif freq_branch == "mean":
                # mean 模式不需要额外参数 — 直接 mean pool
                pass
            else:
                raise ValueError(f"Unsupported freq_branch: {freq_branch}")

            # 融合后 MLP head（dim + feature_dim = CLS 投影 + patch 特征）
            self.head = nn.Sequential(
                nn.LayerNorm(dim + feature_dim),
                nn.Linear(dim + feature_dim, mlp_dim),
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
        if self.freq_branch == "none":
            # ── 原始 CLS-only 路径 ──
            features = self.backbone(images)       # [B, feature_dim]
            x = self.proj(features)                # [B, dim]
            x = x + self.x_embed(pos_x) + self.y_embed(pos_y)
            return self.head(x)

        # ── CLS + patch 旁路路径 ──
        # 获取完整 token 序列（需要 forward_features，不是 __call__）
        all_tokens = self.backbone.forward_features(images)  # [B, 265, feature_dim]
        selected = select_uni_tokens(
            all_tokens, num_tokens=65, mode=self.token_select_mode,
        )  # [B, 65, feature_dim]

        # CLS 分支
        cls_feat = self.branch_cls_proj(selected[:, 0, :])   # [B, dim]
        cls_feat = cls_feat + self.x_embed(pos_x) + self.y_embed(pos_y)

        # Patch 旁路
        patch_tokens = selected[:, 1:, :]                     # [B, 64, feature_dim]
        if self.freq_branch == "gfnet":
            z = self.freq_patch_proj(patch_tokens)            # [B, 64, freq_hidden_dim]
            for block in self.freq_blocks:
                z = block(z)
            patch_feat = self.freq_patch_out(z.mean(dim=1))   # [B, feature_dim]
        elif self.freq_branch == "mean":
            patch_feat = patch_tokens.mean(dim=1)             # [B, feature_dim]
        else:
            raise RuntimeError(f"Unknown freq_branch: {self.freq_branch}")

        fused = torch.cat([cls_feat, patch_feat], dim=-1)     # [B, dim + feature_dim]
        return self.head(fused)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """仅返回 backbone CLS 特征，用于分析/调试。"""
        return self.backbone(images)

    def count_parameters(self) -> int:
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_param_summary(self) -> None:
        """按模块打印参数统计。"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        parts = {}
        if self.freq_branch == "none":
            parts["投影层"] = sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        else:
            parts["CLS投影"] = sum(p.numel() for p in self.branch_cls_proj.parameters() if p.requires_grad)
            if self.freq_branch == "gfnet":
                gfnet_params = 0
                for attr in ["freq_patch_proj", "freq_blocks", "freq_patch_out"]:
                    if hasattr(self, attr):
                        m = getattr(self, attr)
                        if isinstance(m, nn.ModuleList):
                            gfnet_params += sum(p.numel() for p in m.parameters() if p.requires_grad)
                        elif isinstance(m, nn.Module):
                            gfnet_params += sum(p.numel() for p in m.parameters() if p.requires_grad)
                parts["GFNet旁路"] = gfnet_params
            elif self.freq_branch == "mean":
                parts["Mean旁路"] = 0

        embed_params = (sum(p.numel() for p in self.x_embed.parameters() if p.requires_grad) +
                        sum(p.numel() for p in self.y_embed.parameters() if p.requires_grad))
        parts["坐标嵌入"] = embed_params
        parts["回归头"] = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        total = backbone_params + sum(parts.values())
        print(f"OnlineCLSModel 可训练参数: {total:,} ({total/1e6:.2f}M)")
        print(f"  Backbone (含LoRA): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        for name, count in parts.items():
            print(f"  {name}: {'':6s}{count:,}")


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("OnlineCLSModel 自检")
    print("=" * 60)

    # 模拟 UNI2-H: __call__() 返回 CLS, forward_features() 返回完整 token 序列
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 1536, 14, 14)

        def forward(self, x):
            # 模拟 CLS token
            return self.patch_embed(x).mean(dim=[2, 3])[:, :1536]

        def forward_features(self, x):
            # 模拟完整 token 序列 [B, 265, 1536]
            b = x.size(0)
            base = torch.randn(b, 265, 1536, device=x.device)
            pos_signal = torch.arange(265, device=x.device).float().view(1, 265, 1) * 0.01
            return base + pos_signal

    dummy_backbone = DummyBackbone()

    images = torch.randn(4, 3, 224, 224)
    pos_x = torch.randint(0, 128, (4,))
    pos_y = torch.randint(0, 128, (4,))

    for fb in ("none", "mean", "gfnet"):
        print(f"\n--- freq_branch={fb} ---")
        model = OnlineCLSModel(
            dummy_backbone, feature_dim=1536,
            freq_branch=fb, token_select_mode="cls_patch64",
        )
        total = model.count_parameters()
        print(f"  总可训练参数: {total:,} ({total/1e6:.2f}M)")

        with torch.no_grad():
            out = model(images, pos_x, pos_y)
        assert out.shape == (4, 30), f"[{fb}] Expected (4,30), got {out.shape}"
        print(f"  [OK] Forward test passed -- output {tuple(out.shape)}")
        model.print_param_summary()

    print("\n" + "=" * 60)
    print("所有自检通过")
    print("=" * 60)
