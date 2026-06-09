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

# 复用现有 LightweightTokenEncoder
from model_uni_tokens import LightweightTokenEncoder


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
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.encoder_type = encoder_type

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

        # 裁剪到 num_tokens（lite=65, full=265）
        tokens = all_tokens[:, :self.num_tokens, :]           # [B, num_tokens, feature_dim]

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
        return all_tokens[:, :self.num_tokens, :]

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

    # 用轻量替代模拟 backbone
    class DummyViT(nn.Module):
        """模拟 UNI2-H: 返回 [B, 265, 1536] token 序列"""
        def __init__(self):
            super().__init__()
            self.dummy = nn.Conv2d(3, 1536, 14, 14)

        def forward_features(self, x):
            # 伪造 token 序列
            b = x.size(0)
            return torch.randn(b, 265, 1536, device=x.device)

    dummy_backbone = DummyViT()
    model = OnlineTokenModel(dummy_backbone, num_tokens=65)
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
    print("  Forward test passed")
    print("=" * 60)
