from __future__ import annotations

from pathlib import Path

import torch

from .protocol import Predictor
from .uni2h_mlp import build_uni2h_mlp


def build_model(model_id: str, in_dim: int, hidden: int, out_dim: int, dropout: float):
    if model_id == "uni2h_mlp":
        return build_uni2h_mlp(in_dim, hidden, out_dim, dropout)
    raise ValueError(
        f"unknown model_id={model_id!r}. Add src/models/<id>.py and register it here."
    )


def load_predictor(model_id: str, in_dim: int, hidden: int, out_dim: int, dropout: float, device) -> Predictor:
    model = build_model(model_id, in_dim, hidden, out_dim, dropout).to(device)
    return Predictor(model_id=model_id, model=model, in_dim=in_dim, out_dim=out_dim)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"checkpoint is missing model_state_dict: {path}")
    return payload
