"""Dynamic H/C/B regression model used by the full-field Phase2 package."""
from __future__ import annotations

from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from errors import ConfigError

ARMS = ("point", "spatial", "no_b", "spatial_no_b")
ARM_FLAGS = {"point": False, "spatial": True, "no_b": False, "spatial_no_b": False}

def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="none")

def make_center_dropout_mask(shape: Sequence[int], dropout_p: float, generator: torch.Generator | None = None, *, device=None, dtype=None) -> torch.Tensor:
    if not 0 <= float(dropout_p) < 1:
        raise ConfigError("dropout 必须在 [0, 1)")
    one = torch.ones(tuple(shape), device=device, dtype=dtype)
    if not dropout_p:
        return one
    return torch.bernoulli(one * (1 - float(dropout_p)), generator=generator) / (1 - float(dropout_p))

def apply_center_dropout_mask(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden
    if hidden.shape != mask.shape:
        raise ValueError("中心 dropout 掩码必须与 H 输出同形")
    return hidden * mask

class SoftlinkModel(nn.Module):
    """H maps frozen features, C reads pathways, B is a zero-initialised residual."""
    def __init__(self, config: dict, *, use_spatial: bool = False):
        super().__init__()
        data, cfg = config["data"], config["model"]
        self.input_dim, self.output_dim = int(data["input_dim"]), int(data["output_dim"])
        self.hidden_dim = int(cfg["hidden_dim"])
        self.dropout_p, self.use_spatial = float(cfg.get("dropout", 0.0)), bool(use_spatial)
        self.shared = nn.Linear(self.input_dim, self.hidden_dim, bias=bool(cfg.get("shared_bias", True)))
        self.point_head = nn.Linear(self.hidden_dim, self.output_dim, bias=bool(cfg.get("readout_bias", True)))
        self.spatial_head = nn.Linear(self.hidden_dim, self.output_dim, bias=bool(cfg.get("spatial_bias", True)))
        nn.init.zeros_(self.spatial_head.weight)
        if self.spatial_head.bias is not None:
            nn.init.zeros_(self.spatial_head.bias)
        if not self.use_spatial:
            for parameter in self.spatial_head.parameters():
                parameter.requires_grad_(False)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return gelu_exact(self.shared(features))

    def spatial_delta(self, h_center: torch.Tensor, h_neighbors: torch.Tensor | None, neighbor_weights: torch.Tensor | None, neighbor_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_spatial or h_neighbors is None or neighbor_weights is None or not h_neighbors.numel():
            u = torch.zeros_like(h_center)
            return torch.zeros((h_center.shape[0], self.output_dim), device=h_center.device, dtype=h_center.dtype), u
        weights = neighbor_weights * (neighbor_mask.to(neighbor_weights.dtype) if neighbor_mask is not None else 1)
        u = (weights[..., None] * (h_neighbors - h_center[:, None, :])).sum(1)
        return self.spatial_head(u), u

    def forward(self, center_features: torch.Tensor, neighbor_features: torch.Tensor | None = None, neighbor_weights: torch.Tensor | None = None, neighbor_mask: torch.Tensor | None = None, center_dropout_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        h = self.encode(center_features)
        point = self.point_head(apply_center_dropout_mask(h, center_dropout_mask))
        hn = None
        if self.use_spatial and neighbor_features is not None and neighbor_features.numel():
            b, k, d = neighbor_features.shape
            hn = self.encode(neighbor_features.reshape(b * k, d)).reshape(b, k, self.hidden_dim)
            if neighbor_mask is not None:
                hn = hn * neighbor_mask[..., None].to(hn.dtype)
        delta, u = self.spatial_delta(h, hn, neighbor_weights, neighbor_mask)
        return {"y_hat": point + delta, "y_point": point, "delta": delta, "h_center": h, "h_neighbors": hn, "u": u, "z": None}

def build_model(config: dict, arm: str) -> SoftlinkModel:
    if arm not in ARM_FLAGS:
        raise ConfigError(f"只支持 point/spatial 及 no_b 对照，当前={arm!r}")
    return SoftlinkModel(config, use_spatial=ARM_FLAGS[arm])

def predict(model: SoftlinkModel, center_features: torch.Tensor, neighbor_features: torch.Tensor | None = None, neighbor_weights: torch.Tensor | None = None, neighbor_mask: torch.Tensor | None = None) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(center_features, neighbor_features, neighbor_weights, neighbor_mask)["y_hat"]
    finally:
        model.train(was_training)

def count_parameters(model: nn.Module) -> dict[str, int]:
    parts = {name: int(p.numel()) for name, p in model.named_parameters()}
    return {"total": sum(parts.values()), "by_name": parts}

def assert_spatial_zero_initialized(model: SoftlinkModel) -> None:
    if torch.count_nonzero(model.spatial_head.weight).item() or (model.spatial_head.bias is not None and torch.count_nonzero(model.spatial_head.bias).item()):
        raise ConfigError("B 必须零初始化")
