"""Warm-start point predictor and zero-initialised spatial residual head."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpatialDeltaProvider(Protocol):
    """Structural type implemented by :class:`src.spatial.SpatialGraph`."""

    def spatial_delta(self, hidden: Tensor) -> Tensor: ...


class SharedProjection(nn.Module):
    """Stage-one H: biased linear projection followed by exact GELU."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.GELU(approximate="none")

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(self.linear(features))


class RegressionHead(nn.Module):
    """Stage-one C: dropout followed by a biased pathway readout."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 30, dropout: float = 0.3) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(float(dropout))
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(
        self,
        hidden: Tensor,
        *,
        dropout_generator: torch.Generator | None = None,
        dropout_mask: Tensor | None = None,
    ) -> Tensor:
        if dropout_generator is not None and dropout_mask is not None:
            raise ValueError("provide either dropout_generator or dropout_mask, not both")
        if not self.training or self.dropout.p == 0.0:
            dropped = hidden
        elif dropout_mask is not None:
            if dropout_mask.dtype != torch.bool or dropout_mask.shape != hidden.shape:
                raise ValueError("dropout_mask must be boolean with the same shape as hidden")
            dropped = hidden * dropout_mask.to(device=hidden.device, dtype=hidden.dtype)
            dropped = dropped / (1.0 - self.dropout.p)
        elif dropout_generator is not None:
            keep = torch.rand(
                hidden.shape,
                dtype=hidden.dtype,
                device=hidden.device,
                generator=dropout_generator,
            ) >= self.dropout.p
            dropped = hidden * keep / (1.0 - self.dropout.p)
        else:
            dropped = self.dropout(hidden)
        return self.linear(dropped)


@dataclass(frozen=True)
class WarmstartOutput:
    """Named model outputs used by training and label-free prediction."""

    prediction: Tensor
    point_prediction: Tensor
    correction: Tensor
    hidden: Tensor
    spatial_delta: Tensor


class SpatialWarmstartModel(nn.Module):
    """Complete inherited H/C predictor plus a zero-initialised B matrix.

    ``use_spatial=False`` is the point-only arm.  Spatial arms can provide an
    already-computed ``spatial_delta`` or a graph.  Passing the graph ensures
    that the delta is recomputed from the current H during joint optimisation.
    """

    def __init__(
        self,
        *,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        output_dim: int = 30,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(int(input_dim), int(hidden_dim), int(output_dim)) < 1:
            raise ValueError("model dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.shared = SharedProjection(self.input_dim, self.hidden_dim)
        self.regression = RegressionHead(self.hidden_dim, self.output_dim, dropout)
        self.B = nn.Parameter(torch.zeros(self.output_dim, self.hidden_dim))

    def forward(
        self,
        features: Tensor,
        *,
        spatial_delta: Tensor | None = None,
        graph: SpatialDeltaProvider | None = None,
        use_spatial: bool = True,
        dropout_generator: torch.Generator | None = None,
        dropout_mask: Tensor | None = None,
    ) -> WarmstartOutput:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [points, {self.input_dim}]")
        if graph is not None and spatial_delta is not None:
            raise ValueError("provide either graph or spatial_delta, not both")

        hidden = self.shared(features)
        point_prediction = self.regression(
            hidden,
            dropout_generator=dropout_generator,
            dropout_mask=dropout_mask,
        )
        if not use_spatial:
            delta = torch.zeros_like(hidden)
        elif graph is not None:
            delta = graph.spatial_delta(hidden)
        elif spatial_delta is not None:
            delta = spatial_delta
        else:
            raise ValueError("spatial prediction requires graph or spatial_delta")

        if delta.shape != hidden.shape:
            raise ValueError(
                f"spatial_delta must have shape {tuple(hidden.shape)}, got {tuple(delta.shape)}"
            )
        correction = F.linear(delta, self.B)
        return WarmstartOutput(
            prediction=point_prediction + correction,
            point_prediction=point_prediction,
            correction=correction,
            hidden=hidden,
            spatial_delta=delta,
        )


@dataclass(frozen=True)
class SourceLoadReport:
    """Audit record for the exact stage-one parameters copied into H/C."""

    loaded_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]


_SOURCE_TO_MODEL_KEYS = {
    "shared.linear.weight": "shared.linear.weight",
    "shared.linear.bias": "shared.linear.bias",
    "regression.linear.weight": "regression.linear.weight",
    "regression.linear.bias": "regression.linear.bias",
}


def _checkpoint_state(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, PathLike)):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"stage-one checkpoint not found: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    else:
        loaded = checkpoint
    if not isinstance(loaded, Mapping):
        raise TypeError("stage-one checkpoint must contain a state mapping")
    for wrapper_key in ("state_dict", "model_state_dict"):
        wrapped = loaded.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            loaded = wrapped
            break
    if not isinstance(loaded, Mapping):
        raise TypeError("stage-one checkpoint state must be a mapping")
    return {str(key).removeprefix("module."): value for key, value in loaded.items()}


def load_stage1_hc(
    model: SpatialWarmstartModel,
    checkpoint: Mapping[str, Any] | str | PathLike[str],
) -> SourceLoadReport:
    """Strictly copy source H/C while leaving the new B at its current value.

    All four H/C tensors are mandatory and shape checked before any parameter
    is mutated.  Only explicitly named non-H/C prefixes may be ignored; this
    prevents a missing regression head from being hidden by ``strict=False``.
    """

    if not isinstance(model, SpatialWarmstartModel):
        raise TypeError("model must be a SpatialWarmstartModel")
    state = _checkpoint_state(checkpoint)
    required = tuple(_SOURCE_TO_MODEL_KEYS)
    missing = tuple(key for key in required if key not in state)
    if missing:
        raise KeyError(f"missing required stage-one H/C keys: {missing}")

    ignored = tuple(
        sorted(
            key
            for key in state
            if key not in _SOURCE_TO_MODEL_KEYS
            and key.startswith("teacher.")
        )
    )
    unexpected = tuple(
        sorted(key for key in state if key not in _SOURCE_TO_MODEL_KEYS and key not in ignored)
    )
    if unexpected:
        raise KeyError(f"unexpected stage-one checkpoint keys: {unexpected}")

    model_state = model.state_dict()
    prepared: dict[str, Tensor] = {}
    for source_key, model_key in _SOURCE_TO_MODEL_KEYS.items():
        value = state[source_key]
        if not isinstance(value, Tensor):
            raise TypeError(f"stage-one value for {source_key} is not a tensor")
        expected = model_state[model_key]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"shape mismatch for {source_key}: expected {tuple(expected.shape)}, "
                f"got {tuple(value.shape)}"
            )
        prepared[model_key] = value.detach().to(device=expected.device, dtype=expected.dtype)

    with torch.no_grad():
        for model_key, value in prepared.items():
            model_state[model_key].copy_(value)
    return SourceLoadReport(loaded_keys=required, ignored_keys=ignored)
