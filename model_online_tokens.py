"""
model_online_tokens.py — 在线 Token 序列模式模型
===========================================

将 UNI2-H backbone（可选 LoRA）+ 完整 token 序列 → LightweightTokenEncoder →
投影 → 坐标编码 → MLP 回归头。
下游架构与 model_uni_tokens.py 的 HisToGeneUNITokens 完全一致，
唯一区别：输入是原始图像 [B,3,224,224] 而非预提取 token [B,N,1536]。

用法:
    from uni2h.uni2h_utils import load_uni2h_backbone
    backbone, transform, _ = load_uni2h_backbone(device=device)
    model = OnlineTokenModel(backbone, num_tokens=65).to(device)
    preds = model(images, pos_x, pos_y)  # images: [B,3,224,224], preds: [B,30]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用现有 LightweightTokenEncoder
from model_uni_tokens import LightweightTokenEncoder


# ═══════════════════════════════════════════════════════════════
# Token 选择工具
# ═══════════════════════════════════════════════════════════════

# UNI2-H 的 token 排列: [CLS, reg_0..reg_7, patch_0..patch_255] = 265 tokens
# 来源: uni2h_utils.py 中 reg_tokens=8, no_embed_class=True
_UNI2H_REG_TOKENS = 8


def select_uni_tokens(
    all_tokens: torch.Tensor,
    num_tokens: int = 65,
    mode: str = "legacy_firstN",
    reg_tokens: int = _UNI2H_REG_TOKENS,
) -> torch.Tensor:
    """从 UNI2-H 完整 token 序列中选择指定模式的子集。

    Args:
        all_tokens: [B, 265, D] — backbone.forward_features() 的完整输出
        num_tokens: 需要的 token 数量
        mode: 选择模式
            - "legacy_firstN": 历史行为，直接取前 N 个 token。
              包含 CLS + register tokens（如果 N 够大），用于复现已有结果。
            - "cls_patch64": CLS + 前 64 个 patch token（跳过 register tokens）。
              仅适用于 num_tokens=65，用于后续频域实验的干净空间序列。
            - "cls_pool8x8": CLS + 全部 256 patch token 经 2x2 average pooling
              压缩为 8x8=64 个 patch token。仅适用于 num_tokens=65。
        reg_tokens: register token 数量（默认 8，与 UNI2-H 配置一致）

    Returns:
        [B, num_tokens, D]
    """
    if mode == "legacy_firstN":
        return all_tokens[:, :num_tokens, :]

    if mode == "cls_patch64":
        if num_tokens != 65:
            raise ValueError(
                f"cls_patch64 is only defined for num_tokens=65, got {num_tokens}"
            )
        cls_token = all_tokens[:, 0:1, :]                        # [B, 1, D]
        patch_start = 1 + reg_tokens                               # 跳过 CLS + reg
        patch_tokens = all_tokens[:, patch_start:patch_start + 64, :]  # [B, 64, D]
        return torch.cat([cls_token, patch_tokens], dim=1)        # [B, 65, D]

    if mode == "cls_pool8x8":
        if num_tokens != 65:
            raise ValueError(
                f"cls_pool8x8 is only defined for num_tokens=65, got {num_tokens}"
            )
        cls_token = all_tokens[:, 0:1, :]                         # [B, 1, D]
        patch_start = 1 + reg_tokens                              # 跳过 CLS + reg
        patch_tokens = all_tokens[:, patch_start:patch_start + 256, :]  # [B, 256, D]
        if patch_tokens.size(1) != 256:
            raise ValueError(
                f"cls_pool8x8 requires 256 patch tokens, got {patch_tokens.size(1)}"
            )
        b, _, d = patch_tokens.shape
        patch_grid = patch_tokens.transpose(1, 2).reshape(b, d, 16, 16)
        pooled = F.avg_pool2d(patch_grid, kernel_size=2, stride=2)      # [B, D, 8, 8]
        pooled_flat = pooled.reshape(b, d, 64).transpose(1, 2)          # [B, 64, D]
        return torch.cat([cls_token, pooled_flat], dim=1)              # [B, 65, D]

    raise ValueError(f"Unsupported token_select_mode: {mode}")


class OnlineTokenModel(nn.Module):
    """在线 Token 模型：图像 → UNI2-H → token 序列 → TokenEncoder → MLP 回归。

    下游架构与 HisToGeneUNITokens 完全相同。
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
        # TokenEncoder 参数
        encoder_type: str = "transformer",
        encoder_hidden_dim: int = 512,
        n_encoder_layers: int = 1,
        n_encoder_heads: int = 8,
        token_drop_rate: float = 0.0,
        num_tokens: int = 65,
        token_select_mode: str = "legacy_firstN",
    ):
        """
        Args:
            backbone: UNI2-H ViT 模型（可含 LoRA 注入）
            feature_dim: 每个 token 的维度（UNI2-H=1536）
            dim: 内部模型维度
            n_pos: 坐标编码表大小
            n_targets: 输出通路数
            mlp_dim: MLP 隐藏层维度
            dropout: dropout 概率
            encoder_type: TokenEncoder 类型 ("transformer" | "gfnet")
            encoder_hidden_dim: TokenEncoder 隐藏维度
            n_encoder_layers: TokenEncoder Transformer/GFNet 层数
            n_encoder_heads: TokenEncoder 注意力头数 (仅 transformer)
            token_drop_rate: 训练时随机丢弃 token 的概率 (当前未使用，保留参数兼容)
            num_tokens: 保留的 token 数量（lite=65, full=265）
            token_select_mode: token 选择模式
                - "legacy_firstN": 历史行为，取前 N 个 token（含 register）
                - "cls_patch64": CLS + 前 64 个 patch token，跳过 register（仅 65-token）
                - "cls_pool8x8": CLS + 2x2 pooled 全空间 patch tokens（仅 65-token）
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.encoder_type = encoder_type
        self.token_select_mode = token_select_mode

        # ── Token 编码器 ──
        if encoder_type == "gfnet":
            from model_gfnet import GFNetTokenEncoder
            self.token_encoder = GFNetTokenEncoder(
                embed_dim=feature_dim,
                hidden_dim=encoder_hidden_dim,
                seq_len=num_tokens,
                n_layers=n_encoder_layers,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.token_encoder = LightweightTokenEncoder(
                embed_dim=feature_dim,
                hidden_dim=encoder_hidden_dim,
                n_heads=n_encoder_heads,
                n_layers=n_encoder_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # ── 以下与 HisToGeneUNITokens 完全一致 ──
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

        self.n_targets = n_targets

    def forward(
        self,
        images: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224] 原始 H&E patch
            pos_x: [B] X 坐标索引
            pos_y: [B] Y 坐标索引

        Returns:
            predictions: [B, n_targets] ssGSEA 通路评分预测
        """
        # 通过 backbone 提取完整 token 序列
        all_tokens = self.backbone.forward_features(images)  # [B, 265, feature_dim]

        # 按模式选择 token 子集
        tokens = select_uni_tokens(
            all_tokens,
            num_tokens=self.num_tokens,
            mode=self.token_select_mode,
        )  # [B, num_tokens, feature_dim]

        # TokenEncoder
        encoded = self.token_encoder(tokens)                   # [B, feature_dim]

        # 下游（与 HisToGeneUNITokens 一致）
        x = self.proj(encoded)                                # [B, dim]
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)
        x = self.head(x)                                      # [B, n_targets]
        return x

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """仅返回 backbone token 特征，用于分析/调试。"""
        all_tokens = self.backbone.forward_features(images)
        return select_uni_tokens(
            all_tokens,
            num_tokens=self.num_tokens,
            mode=self.token_select_mode,
        )

    def count_parameters(self) -> int:
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_param_summary(self) -> None:
        """按模块打印参数统计。"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.token_encoder.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        embed_params = (sum(p.numel() for p in self.x_embed.parameters() if p.requires_grad) +
                        sum(p.numel() for p in self.y_embed.parameters() if p.requires_grad))
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total = backbone_params + encoder_params + proj_params + embed_params + head_params

        print(f"OnlineTokenModel 可训练参数: {total:,} ({total/1e6:.2f}M)")
        print(f"  Backbone (含LoRA): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  TokenEncoder:      {encoder_params:,} ({encoder_params/1e6:.2f}M)")
        print(f"  投影层:             {proj_params:,}")
        print(f"  坐标嵌入:           {embed_params:,}")
        print(f"  回归头:             {head_params:,}")


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("OnlineTokenModel 自检")
    print("=" * 60)

    # 用轻量替代模拟 backbone：返回 [CLS, reg×8, patch×256] = 265 tokens
    class DummyViT(nn.Module):
        """模拟 UNI2-H: 返回 [B, 265, 1536] token 序列"""
        def __init__(self):
            super().__init__()

        def forward_features(self, x):
            b = x.size(0)
            # 生成可区分的 token: 每个位置有独特的微小偏移
            base = torch.randn(b, 265, 1536, device=x.device)
            # 在 dim=1 上叠加位置编码，使不同位置可区分
            pos_signal = torch.arange(265, device=x.device).float().view(1, 265, 1) * 0.01
            return base + pos_signal

    dummy_backbone = DummyViT()

    for mode in ("legacy_firstN", "cls_patch64", "cls_pool8x8"):
        print(f"\n--- token_select_mode={mode} ---")
        model = OnlineTokenModel(dummy_backbone, num_tokens=65, token_select_mode=mode)
        total = model.count_parameters()
        print(f"  总可训练参数: {total:,} ({total/1e6:.2f}M)")

        # 测试 select_uni_tokens 直接调用
        all_tokens = dummy_backbone.forward_features(torch.randn(2, 3, 224, 224))
        selected = select_uni_tokens(all_tokens, num_tokens=65, mode=mode)
        print(f"  select_uni_tokens output shape: {tuple(selected.shape)}")

        # 验证 token 身份
        if mode == "legacy_firstN":
            # 前 65: [CLS, reg0..reg7, patch0..patch55]
            assert torch.allclose(selected[:, 0, :], all_tokens[:, 0, :], atol=1e-5), \
                "legacy_firstN: position 0 should be CLS"
            assert torch.allclose(selected[:, 9, :], all_tokens[:, 9, :], atol=1e-5), \
                "legacy_firstN: position 9 should be patch_0"
            print("  [OK] token identity verified: [CLS, reg0..reg7, patch0..patch55]")
        elif mode == "cls_patch64":
            # 应包含: CLS at 0, patches 8-71 from original (skip reg0..reg7)
            assert torch.allclose(selected[:, 0, :], all_tokens[:, 0, :], atol=1e-5), \
                "cls_patch64: position 0 should be CLS"
            assert torch.allclose(selected[:, 1, :], all_tokens[:, 9, :], atol=1e-5), \
                "cls_patch64: position 1 should be patch_0 (original index 9)"
            assert torch.allclose(selected[:, 64, :], all_tokens[:, 72, :], atol=1e-5), \
                "cls_patch64: position 64 should be patch_63 (original index 72)"
            print("  [OK] token identity verified: [CLS, patch0..patch63] (register skipped)")
        elif mode == "cls_pool8x8":
            assert torch.allclose(selected[:, 0, :], all_tokens[:, 0, :], atol=1e-5), \
                "cls_pool8x8: position 0 should be CLS"
            expected = all_tokens[:, 9:9 + 256, :].reshape(2, 16, 16, 1536)[:, 0:2, 0:2, :].mean(dim=(1, 2))
            assert torch.allclose(selected[:, 1, :], expected, atol=1e-5), \
                "cls_pool8x8: position 1 should be mean of patch grid [0:2, 0:2]"
            print("  [OK] token identity verified: [CLS, pooled 8x8 full-grid patches]")

        # 测试完整前向
        images = torch.randn(4, 3, 224, 224)
        pos_x = torch.randint(0, 128, (4,))
        pos_y = torch.randint(0, 128, (4,))

        with torch.no_grad():
            out = model(images, pos_x, pos_y)
        assert out.shape == (4, 30), f"Expected (4,30), got {out.shape}"
        print(f"  [OK] Forward test passed -- output {tuple(out.shape)}")

    print("\n" + "=" * 60)
    print("所有自检通过")
    print("=" * 60)
