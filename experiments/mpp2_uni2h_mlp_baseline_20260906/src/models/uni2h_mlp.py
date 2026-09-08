from __future__ import annotations

import torch.nn as nn


class MPPMLPHead(nn.Module):
    """Frozen UNI2-h CLS [in_dim] -> two-layer MLP -> target outputs."""

    def __init__(self, in_dim: int = 1536, hidden: int = 1024, out_dim: int = 30, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.head(x)


def build_uni2h_mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> MPPMLPHead:
    return MPPMLPHead(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
