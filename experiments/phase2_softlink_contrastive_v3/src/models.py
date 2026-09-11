"""Model primitives for the Phase2 soft-link contrastive v4 experiment.

The LoRA adapter deliberately wraps a fused timm-style ``attention.qkv``
linear layer.  It adapts only Q and V with separate factors; K remains an
exact frozen slice of the original layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SharedProjection(nn.Module):
    """The shared 1536 -> 256 projection with the specified exact GELU."""

    def __init__(self, input_dim: int = 1536, embedding_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))


class RegressionHead(nn.Module):
    """Dropout regression head; contrastive embeddings bypass this dropout."""

    def __init__(self, input_dim: int = 256, output_dim: int = 30, dropout: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, h: Tensor) -> Tensor:
        return self.linear(self.dropout(h))


class PathwayTeacher(nn.Module):
    """Trainable pathway MLP T: 30 -> 128 -> 256."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 128, embedding_dim: int = 256) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, r: Tensor) -> Tensor:
        return self.output(self.activation(self.input(r)))


def _cls_from_backbone_output(output: Any) -> Tensor:
    """Accept conventional timm tensor/dict outputs while retaining CLS only."""
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls", "cls_token", "features", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("backbone dictionary output has no recognised CLS feature")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError("backbone must return a Tensor, sequence, or recognised dictionary")
    return output[:, 0] if output.ndim == 3 else output


class Stage1Model(nn.Module):
    """Stage-one image backbone, shared representation, regression, and T."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_dim: int = 1536,
        embedding_dim: int = 256,
        output_dim: int = 30,
        dropout: float = 0.3,
        teacher: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.shared = SharedProjection(input_dim, embedding_dim)
        self.regression = RegressionHead(embedding_dim, output_dim, dropout)
        self.teacher = teacher if teacher is not None else PathwayTeacher(output_dim, 128, embedding_dim)

    def forward(self, x: Tensor, teacher_inputs: Tensor | None = None) -> dict[str, Tensor]:
        # Historical frozen CLS caches use this timm-style path; keep online
        # extraction on the same representation contract when it is available.
        forward_features = getattr(self.backbone, "forward_features", None)
        backbone_output = forward_features(x) if callable(forward_features) else self.backbone(x)
        cls = _cls_from_backbone_output(backbone_output)
        h = self.shared(cls)
        result = {"cls": cls, "h": h, "pred": self.regression(h)}
        if teacher_inputs is not None:
            result["teacher_embedding"] = self.teacher(teacher_inputs)
        return result


class IndependentQVLoRA(nn.Module):
    """Frozen fused QKV linear plus independent low-rank Q and V increments."""

    def __init__(self, base_qkv: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError("IndependentQVLoRA only wraps nn.Linear qkv layers")
        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError("fused qkv must have exactly 3 * in_features outputs")
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base_qkv = base_qkv
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.dropout = nn.Dropout(float(dropout))
        dim, input_dim = base_qkv.in_features, base_qkv.in_features
        self.q_A = nn.Parameter(torch.empty(self.rank, input_dim))
        self.q_B = nn.Parameter(torch.zeros(dim, self.rank))
        self.v_A = nn.Parameter(torch.empty(self.rank, input_dim))
        self.v_B = nn.Parameter(torch.zeros(dim, self.rank))
        nn.init.kaiming_uniform_(self.q_A, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_A, a=5**0.5)
        for parameter in self.base_qkv.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base_qkv.in_features

    @property
    def out_features(self) -> int:
        return self.base_qkv.out_features

    @property
    def bias(self) -> Tensor | None:
        return self.base_qkv.bias

    @property
    def weight(self) -> Tensor:
        return self.base_qkv.weight

    def delta_weight(self, component: str) -> Tensor:
        """Return the actual scaled increment for Q, K, V, or fused QKV."""
        component = component.lower()
        if component == "q":
            return self.scaling * (self.q_B @ self.q_A)
        if component == "v":
            return self.scaling * (self.v_B @ self.v_A)
        if component == "k":
            return self.weight.new_zeros((self.in_features, self.in_features))
        if component == "qkv":
            return torch.cat((self.delta_weight("q"), self.delta_weight("k"), self.delta_weight("v")), dim=0)
        raise ValueError("component must be one of q, k, v, qkv")

    def forward(self, x: Tensor) -> Tensor:
        base = self.base_qkv(x)
        if not self.enabled:
            return base
        adapted = self.dropout(x)
        q_delta = F.linear(F.linear(adapted, self.q_A), self.q_B) * self.scaling
        v_delta = F.linear(F.linear(adapted, self.v_A), self.v_B) * self.scaling
        zeros = torch.zeros_like(q_delta)
        return base + torch.cat((q_delta, zeros, v_delta), dim=-1)


def _blocks(backbone: nn.Module) -> Sequence[nn.Module]:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("backbone must expose timm-style .blocks")
    return blocks


def inject_independent_qv_lora(
    backbone: nn.Module,
    block_indices: Iterable[int] = (20, 21, 22, 23),
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
) -> nn.Module:
    """Freeze a backbone and replace selected ``blocks[i].attn.qkv`` layers.

    The operation is intentionally in-place and returns ``backbone`` for
    convenient chaining.  Re-injection is rejected so independent factors
    cannot silently be replaced during a run.
    """
    selected = tuple(int(index) for index in block_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("block_indices must be a non-empty sequence of unique indices")
    blocks = _blocks(backbone)
    if min(selected) < 0 or max(selected) >= len(blocks):
        raise IndexError("a requested LoRA block index is outside backbone.blocks")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for index in selected:
        attention = getattr(blocks[index], "attn", None)
        qkv = getattr(attention, "qkv", None)
        if isinstance(qkv, IndependentQVLoRA):
            raise ValueError(f"block {index} already has an IndependentQVLoRA adapter")
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"block {index} lacks a fused nn.Linear attn.qkv")
        attention.qkv = IndependentQVLoRA(qkv, rank=rank, alpha=alpha, dropout=dropout)
    return backbone


def _lora_wrappers(backbone: nn.Module) -> list[tuple[int, IndependentQVLoRA]]:
    found: list[tuple[int, IndependentQVLoRA]] = []
    for index, block in enumerate(_blocks(backbone)):
        adapter = getattr(getattr(block, "attn", None), "qkv", None)
        if isinstance(adapter, IndependentQVLoRA):
            found.append((index, adapter))
    return found


@contextmanager
def lora_disabled(backbone: nn.Module):
    """Temporarily expose the exact frozen-base forward without a model copy."""
    wrappers = [adapter for _, adapter in _lora_wrappers(backbone)]
    previous = [adapter.enabled for adapter in wrappers]
    try:
        for adapter in wrappers:
            adapter.enabled = False
        yield backbone
    finally:
        for adapter, enabled in zip(wrappers, previous):
            adapter.enabled = enabled


@torch.no_grad()
def lora_delta_report(backbone: nn.Module) -> dict[int, dict[str, float]]:
    """Frobenius DeltaW/base-W ratios for each adapted block and Q/K/V slice."""
    report: dict[int, dict[str, float]] = {}
    for index, adapter in _lora_wrappers(backbone):
        dim = adapter.in_features
        base = adapter.weight
        ratios: dict[str, float] = {}
        for name, start in (("q", 0), ("k", dim), ("v", 2 * dim)):
            denominator = torch.linalg.vector_norm(base[start : start + dim]).item()
            numerator = torch.linalg.vector_norm(adapter.delta_weight(name)).item()
            ratios[name] = 0.0 if denominator == 0.0 else numerator / denominator
        report[index] = ratios
    return report


def count_trainable_parameters(module: nn.Module) -> int:
    """Number of scalar parameters that optimisers are allowed to update."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def assert_lora_contract(backbone: nn.Module, block_indices: Iterable[int] | None = None) -> None:
    """Raise ``AssertionError`` if a backbone violates the v4 LoRA contract."""
    selected = None if block_indices is None else set(int(index) for index in block_indices)
    wrappers = dict(_lora_wrappers(backbone))
    if not wrappers:
        raise AssertionError("no independent Q/V LoRA wrappers were found")
    if selected is not None and set(wrappers) != selected:
        raise AssertionError("adapted blocks do not match the requested block indices")
    allowed_suffixes = {"q_A", "q_B", "v_A", "v_B"}
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad and name.split(".")[-1] not in allowed_suffixes:
            raise AssertionError(f"non-LoRA parameter is trainable: {name}")
    for index, adapter in wrappers.items():
        if adapter.q_A.data_ptr() == adapter.v_A.data_ptr() or adapter.q_B.data_ptr() == adapter.v_B.data_ptr():
            raise AssertionError(f"block {index} shares Q and V LoRA factors")
        if adapter.delta_weight("k").abs().sum().item() != 0.0:
            raise AssertionError(f"block {index} has a non-zero K increment")
        if any(parameter.requires_grad for parameter in adapter.base_qkv.parameters()):
            raise AssertionError(f"block {index} has trainable base QKV parameters")


class Stage2Head(nn.Module):
    """Stage-two point/spatial regression head with explicit shared-H policy."""

    _MODES = {"point", "spatial"}
    _SHARED_MODES = {"freeze", "reset", "continue"}

    def __init__(
        self,
        shared: nn.Module | None = None,
        *,
        mode: str = "point",
        input_dim: int = 256,
        output_dim: int = 30,
        dropout: float = 0.3,
        shared_mode: str = "freeze",
    ) -> None:
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {sorted(self._MODES)}")
        self.shared = shared
        self.mode = mode
        self.regression = RegressionHead(input_dim, output_dim, dropout)
        self.B = nn.Parameter(torch.zeros(output_dim, input_dim)) if mode == "spatial" else None
        self.shared_mode: str | None = None
        self.set_shared_mode(shared_mode)

    def set_shared_mode(self, mode: str) -> "Stage2Head":
        """Set H policy: freeze inherited H, reset it, or continue training it."""
        if mode not in self._SHARED_MODES:
            raise ValueError(f"shared mode must be one of {sorted(self._SHARED_MODES)}")
        self.shared_mode = mode
        if self.shared is None:
            return self
        if mode == "reset":
            for child in self.shared.modules():
                reset = getattr(child, "reset_parameters", None)
                if callable(reset):
                    reset()
        for parameter in self.shared.parameters():
            parameter.requires_grad_(mode != "freeze")
        return self

    def forward(
        self,
        features: Tensor,
        spatial_delta: Tensor | None = None,
        spatial_weights: Tensor | None = None,
    ) -> Tensor:
        h = self.shared(features) if self.shared is not None else features
        prediction = self.regression(h)
        if self.mode == "point":
            return prediction
        if spatial_delta is None:
            raise ValueError("spatial_delta is required for a spatial Stage2Head")
        if spatial_delta.ndim == h.ndim + 1:
            if spatial_weights is None:
                spatial_delta = spatial_delta.sum(dim=-2)
            else:
                spatial_delta = (spatial_delta * spatial_weights.unsqueeze(-1)).sum(dim=-2)
        if spatial_delta.shape != h.shape:
            raise ValueError("spatial_delta must have the same final shape as h")
        return prediction + F.linear(spatial_delta, self.B)
