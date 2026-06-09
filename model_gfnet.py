"""
model_gfnet.py - GFNet Token Encoder for PFMval.

Uses a learnable FFT filter to replace Transformer token mixing.
No external dependencies beyond PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalFilter(nn.Module):
    """Learnable residual FFT filter over the token dimension."""

    def __init__(self, seq_len: int, dim: int, init_scale: float = 0.02):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.complex_weight = nn.Parameter(
            torch.randn(seq_len // 2 + 1, dim, 2, dtype=torch.float32) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, d = x.shape
        if n != self.seq_len:
            raise ValueError(f"GlobalFilter expected seq_len={self.seq_len}, got {n}")
        if d != self.dim:
            raise ValueError(f"GlobalFilter expected dim={self.dim}, got {d}")

        x_freq = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight.unsqueeze(0)
        out = torch.fft.irfft(x_freq, n=n, dim=1, norm="ortho")
        return out.to(dtype=x.dtype)


class GFNetBlock(nn.Module):
    """GFNet block with residual FFT filter and residual FFN."""

    def __init__(self, seq_len: int, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.3):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(seq_len, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.filter(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GFNetTokenEncoder(nn.Module):
    """Drop-in replacement for LightweightTokenEncoder."""

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        seq_len: int = 65,
        n_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GFNetBlock(seq_len, hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return self.norm(x)


if __name__ == "__main__":
    print("=" * 60)
    print("GFNetTokenEncoder self-test")
    print("=" * 60)
    for seq_len in (65, 265):
        encoder = GFNetTokenEncoder(seq_len=seq_len)
        params = sum(p.numel() for p in encoder.parameters())
        filter_params = sum(p.numel() for p in encoder.blocks[0].filter.parameters())
        x = torch.randn(4, seq_len, 1536)
        with torch.no_grad():
            y = encoder(x)
        assert y.shape == (4, 1536), y.shape
        print(f"seq_len={seq_len}: params={params:,}, filter={filter_params:,}, output={tuple(y.shape)}")
    print("GFNetTokenEncoder self-test passed")
