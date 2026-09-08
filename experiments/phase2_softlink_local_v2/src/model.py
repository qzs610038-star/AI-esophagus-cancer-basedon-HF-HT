"""h / z / point head / direct B. Dimensions come from config, never hardcoded 256."""

from __future__ import annotations

import inspect
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from errors import ConfigError

ARMS = ("point", "relation", "spatial", "joint")
ARM_FLAGS = {
    "point": (False, False),
    "relation": (True, False),
    "spatial": (False, True),
    "joint": (True, True),
}


def gelu_exact(values: torch.Tensor) -> torch.Tensor:
    return F.gelu(values, approximate="none")


def make_center_dropout_mask(
    shape: Sequence[int],
    dropout_p: float,
    generator: torch.Generator | None = None,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if dropout_p < 0 or dropout_p >= 1:
        raise ConfigError(f"dropout 必须在 [0,1)，当前={dropout_p}")
    ones = torch.ones(tuple(shape), device=device, dtype=dtype)
    if dropout_p == 0:
        return ones
    keep_p = 1.0 - float(dropout_p)
    kept = torch.bernoulli(ones * keep_p, generator=generator)
    return kept / keep_p


def apply_center_dropout_mask(h_center: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return h_center
    if mask.shape != h_center.shape:
        raise ValueError(f"Dropout 掩码形状 {tuple(mask.shape)} 与中心 h {tuple(h_center.shape)} 不一致")
    return h_center * mask


class SoftlinkModel(nn.Module):
    def __init__(
        self,
        config: dict,
        *,
        use_relation: bool,
        use_spatial: bool,
    ):
        super().__init__()
        data = config["data"]
        model_cfg = config["model"]
        if model_cfg.get("activation") != "gelu_exact":
            raise ConfigError("只实现 gelu_exact")
        self.input_dim = int(data["input_dim"])
        self.output_dim = int(data["output_dim"])
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.relation_dim = int(model_cfg["relation_dim"])
        self.dropout_p = float(model_cfg["dropout"])
        self.norm_eps = float(model_cfg["normalization_epsilon"])
        self.use_relation = bool(use_relation)
        self.use_spatial = bool(use_spatial)
        self.shared = nn.Linear(self.input_dim, self.hidden_dim, bias=bool(model_cfg["shared_bias"]))
        self.point_head = nn.Linear(self.hidden_dim, self.output_dim, bias=bool(model_cfg["readout_bias"]))
        self.relation_head = None
        if self.use_relation:
            self.relation_head = nn.Linear(
                self.hidden_dim, self.relation_dim, bias=bool(model_cfg["relation_bias"])
            )
        self.spatial_head = None
        if self.use_spatial:
            if model_cfg.get("spatial_initialization") != "zeros":
                raise ConfigError("只实现 spatial_initialization=zeros")
            self.spatial_head = nn.Linear(
                self.hidden_dim, self.output_dim, bias=bool(model_cfg["spatial_bias"])
            )
            nn.init.zeros_(self.spatial_head.weight)
            if self.spatial_head.bias is not None:
                nn.init.zeros_(self.spatial_head.bias)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return gelu_exact(self.shared(features))

    def encode_indexed(self, unique_features: torch.Tensor, inverse_index: torch.Tensor) -> torch.Tensor:
        return self.encode(unique_features)[inverse_index]

    def relation_z(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if self.relation_head is None:
            return None
        projected = self.relation_head(hidden)
        return F.normalize(projected, p=2, dim=-1, eps=self.norm_eps)

    def spatial_delta(
        self,
        h_center: torch.Tensor,
        h_neighbors: torch.Tensor | None,
        neighbor_weights: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_head is None or h_neighbors is None or neighbor_weights is None or h_neighbors.numel() == 0:
            zeros = torch.zeros_like(h_center)
            return self.spatial_head(zeros) if self.spatial_head is not None else zeros.new_zeros(
                h_center.shape[0], self.output_dim
            ), zeros
        weights = neighbor_weights
        if neighbor_mask is not None:
            weights = weights * neighbor_mask.to(dtype=weights.dtype)
        u = (weights.unsqueeze(-1) * (h_neighbors - h_center.unsqueeze(1))).sum(dim=1)
        return self.spatial_head(u), u

    def forward(
        self,
        center_features: torch.Tensor,
        neighbor_features: torch.Tensor | None = None,
        neighbor_weights: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        center_dropout_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        h_center = self.encode(center_features)
        h_for_point = apply_center_dropout_mask(h_center, center_dropout_mask)
        y_point = self.point_head(h_for_point)
        h_neighbors = None
        if self.use_spatial and neighbor_features is not None and neighbor_features.numel() > 0:
            batch, k_max, dim = neighbor_features.shape
            h_neighbors = self.encode(neighbor_features.reshape(batch * k_max, dim)).reshape(batch, k_max, self.hidden_dim)
            if neighbor_mask is not None:
                h_neighbors = h_neighbors * neighbor_mask.unsqueeze(-1).to(dtype=h_neighbors.dtype)
        delta, u = self.spatial_delta(h_center, h_neighbors, neighbor_weights, neighbor_mask)
        if not self.use_spatial:
            delta = torch.zeros_like(y_point)
            u = torch.zeros_like(h_center)
        y_hat = y_point + delta
        z = self.relation_z(h_center) if self.use_relation else None
        return {
            "y_hat": y_hat,
            "y_point": y_point,
            "delta": delta,
            "h_center": h_center,
            "h_neighbors": h_neighbors,
            "u": u,
            "z": z,
        }


def build_model(config: dict, arm: str) -> SoftlinkModel:
    if arm not in ARM_FLAGS:
        raise ConfigError(f"未知实验臂 {arm!r}，允许 {list(ARM_FLAGS)}")
    use_relation, use_spatial = ARM_FLAGS[arm]
    return SoftlinkModel(config, use_relation=use_relation, use_spatial=use_spatial)


def predict(
    model: SoftlinkModel,
    center_features: torch.Tensor,
    neighbor_features: torch.Tensor | None = None,
    neighbor_weights: torch.Tensor | None = None,
    neighbor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frozen-graph inference. Labels are not part of this interface."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(
                center_features,
                neighbor_features=neighbor_features,
                neighbor_weights=neighbor_weights,
                neighbor_mask=neighbor_mask,
                center_dropout_mask=None,
            )
        return output["y_hat"]
    finally:
        model.train(was_training)


def forward_parameter_names() -> tuple[str, ...]:
    return tuple(inspect.signature(SoftlinkModel.forward).parameters)


def count_parameters(model: nn.Module) -> dict[str, int]:
    named = {name: int(param.numel()) for name, param in model.named_parameters()}
    return {"total": int(sum(named.values())), "by_name": named}


def assert_spatial_zero_initialized(model: SoftlinkModel) -> None:
    if model.spatial_head is None:
        return
    if float(model.spatial_head.weight.detach().abs().sum().cpu()) != 0.0:
        raise ConfigError("直接B必须零初始化")
