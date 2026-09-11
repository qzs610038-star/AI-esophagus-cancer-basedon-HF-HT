"""Small numerical diagnostics used by the Phase2 v4 training logger."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import lora_delta_report


def delta_w_ratios(backbone: nn.Module) -> dict[int, dict[str, float]]:
    """Return per-block actual Q/K/V DeltaW-to-base-W Frobenius ratios."""
    return lora_delta_report(backbone)


def off_diagonal_cosine_statistics(embeddings: Tensor) -> dict[str, float]:
    """Mean and variance of pairwise cosine similarities excluding self-pairs."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must have at least two rows")
    values = (F.normalize(embeddings.float(), dim=-1) @ F.normalize(embeddings.float(), dim=-1).transpose(0, 1))
    mask = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    off_diagonal = values[mask]
    return {"mean": off_diagonal.mean().item(), "variance": off_diagonal.var(unbiased=False).item()}


def covariance_effective_rank(embeddings: Tensor, eps: float = 1e-12) -> float:
    """Effective rank (tr C)^2 / tr(C^2), returning 0 for a degenerate C."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [samples, features]")
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / max(embeddings.shape[0] - 1, 1)
    trace = torch.trace(covariance)
    squared_trace = torch.trace(covariance @ covariance)
    if squared_trace.abs().item() <= eps:
        return 0.0
    return ((trace.square() / squared_trace).item())


def representation_statistics(embeddings: Tensor) -> dict[str, float]:
    """Combined cosine, variance, and covariance-effective-rank summary."""
    return {**off_diagonal_cosine_statistics(embeddings), "effective_rank": covariance_effective_rank(embeddings)}


def gradient_norms(parameters: nn.Module | Mapping[str, Tensor] | Iterable[tuple[str, Tensor]]) -> dict[str, float]:
    """L2 gradient norm by name; parameters without a gradient are omitted."""
    if isinstance(parameters, nn.Module):
        iterator = parameters.named_parameters()
    elif isinstance(parameters, Mapping):
        iterator = parameters.items()
    else:
        iterator = parameters
    result: dict[str, float] = {}
    for name, parameter in iterator:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            result[str(name)] = torch.linalg.vector_norm(gradient.detach().float()).item()
    return result


def gradient_cosine(first: Tensor, second: Tensor, eps: float = 1e-12) -> float:
    """Cosine between two flattened gradients; 0 for a zero-norm gradient."""
    first, second = first.detach().float().reshape(-1), second.detach().float().reshape(-1)
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator.item() <= eps:
        return 0.0
    return (torch.dot(first, second) / denominator).item()
