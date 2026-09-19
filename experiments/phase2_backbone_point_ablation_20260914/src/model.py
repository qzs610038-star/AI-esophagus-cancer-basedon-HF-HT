"""The shared d→256→30 point-regression head used by every backbone."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from errors import ConfigError, NonFiniteDataError


class PointRegressor(nn.Module):
    """Two-layer point head with externally controlled training dropout."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        output_dim: int = 30,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or output_dim < 1:
            raise ConfigError("input_dim、hidden_dim、output_dim 必须为正整数")
        if not 0.0 <= float(dropout_p) < 1.0:
            raise ConfigError("dropout_p 必须位于 [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_p = float(dropout_p)
        self.projection = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.readout = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(
        self,
        features: torch.Tensor,
        dropout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features 必须为 [B,{self.input_dim}]，实际={tuple(features.shape)}"
            )
        hidden = F.gelu(self.projection(features), approximate="none")
        if dropout_mask is not None:
            if dropout_mask.shape != hidden.shape:
                raise ValueError(
                    f"dropout_mask 形状 {tuple(dropout_mask.shape)} 与 hidden {tuple(hidden.shape)} 不一致"
                )
            hidden = hidden * dropout_mask
        return self.readout(hidden)


def _cpu_device_context():
    try:
        return torch.device("cpu")
    except Exception:
        return nullcontext()


def _validate_output_state(
    output_state: Mapping[str, torch.Tensor], hidden_dim: int, output_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(output_state) != {"weight", "bias"}:
        raise ConfigError("历史输出层必须且只能包含 weight 与 bias")
    weight = torch.as_tensor(output_state["weight"]).detach().cpu().float()
    bias = torch.as_tensor(output_state["bias"]).detach().cpu().float()
    if tuple(weight.shape) != (int(output_dim), int(hidden_dim)):
        raise ConfigError(
            f"输出层 weight 应为 {(output_dim, hidden_dim)}，实际={tuple(weight.shape)}"
        )
    if tuple(bias.shape) != (int(output_dim),):
        raise ConfigError(f"输出层 bias 应为 {(output_dim,)}，实际={tuple(bias.shape)}")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise NonFiniteDataError("历史输出层初值含非有限值")
    return weight, bias


def build_paired_regressor(
    input_dim: int,
    *,
    seed: int,
    output_state: Mapping[str, torch.Tensor],
    hidden_dim: int = 256,
    output_dim: int = 30,
    dropout_p: float = 0.3,
    model_seed_offset: int = 0,
) -> PointRegressor:
    """Construct on CPU with the legacy seed, then copy only the shared output layer."""

    weight, bias = _validate_output_state(output_state, hidden_dim, output_dim)
    global_state = torch.get_rng_state()
    try:
        with _cpu_device_context():
            torch.manual_seed(int(seed) + int(model_seed_offset))
            model = PointRegressor(
                input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout_p=dropout_p,
            )
        with torch.no_grad():
            model.readout.weight.copy_(weight)
            model.readout.bias.copy_(bias)
        return model.cpu()
    finally:
        torch.set_rng_state(global_state)


def make_dropout_generator(seed: int, *, offset: int = 200_000) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(offset))
    return generator


def next_dropout_mask(
    generator: torch.Generator,
    batch_size: int,
    hidden_dim: int,
    dropout_p: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not 0.0 <= float(dropout_p) < 1.0:
        raise ConfigError("dropout_p 必须位于 [0, 1)")
    keep_p = 1.0 - float(dropout_p)
    if keep_p == 1.0:
        return torch.ones((int(batch_size), int(hidden_dim)), dtype=dtype)
    base = torch.full((int(batch_size), int(hidden_dim)), keep_p, dtype=dtype)
    return torch.bernoulli(base, generator=generator) / keep_p


def predict(model: PointRegressor, features: torch.Tensor) -> torch.Tensor:
    """Inference does not accept labels and never applies dropout."""

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return model(features, dropout_mask=None)
    finally:
        model.train(was_training)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
