"""Small, shape-checked PyTorch components for the Phase 3 local pipeline.

These modules are deliberately data-agnostic: they accept already prepared
bag features and do not read manifests, labels, or server resources.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _require_rank(name: str, value: Tensor, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have shape rank {rank}, got {tuple(value.shape)}")


def _validate_mask(features: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    if mask.shape != features.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(features.shape[:2])}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a bool tensor")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every bag must contain at least one unmasked instance")
    return mask


def mean_mil_pool(features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Masked mean pooling for a ``[batch, instances, features]`` bag."""
    _require_rank("features", features, 3)
    mask = _validate_mask(features, mask)
    weight = mask.unsqueeze(-1).to(features.dtype)
    return (features * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)


class AttentionMILPooling(nn.Module):
    """Attention MIL pooling; returns pooled vectors, optionally attention."""

    def __init__(self, feature_dim: int, attention_dim: Optional[int] = None) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        attention_dim = attention_dim or max(1, feature_dim // 2)
        self.score = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    def forward(self, features: Tensor, mask: Optional[Tensor] = None, return_attention: bool = False):
        _require_rank("features", features, 3)
        mask = _validate_mask(features, mask)
        logits = self.score(features).squeeze(-1).masked_fill(~mask, torch.finfo(features.dtype).min)
        attention = torch.softmax(logits, dim=1)
        pooled = torch.sum(features * attention.unsqueeze(-1), dim=1)
        return (pooled, attention) if return_attention else pooled


FusionMode = Literal["h_and_e", "pathway", "concat", "dual_linear", "adaptive_gated", "low_rank"]


class MILBagClassifier(nn.Module):
    """Unified light bag classifier for all requested Phase 3 fusion baselines."""

    def __init__(self, image_dim: int, pathway_dim: int = 30, hidden_dim: int = 128,
                 fusion: FusionMode = "concat", pooling: Literal["mean", "attention"] = "attention",
                 low_rank_dim: int = 16, num_outputs: int = 1) -> None:
        super().__init__()
        if image_dim <= 0 or pathway_dim <= 0 or hidden_dim <= 0 or num_outputs <= 0:
            raise ValueError("dimensions and num_outputs must be positive")
        if fusion not in {"h_and_e", "pathway", "concat", "dual_linear", "adaptive_gated", "low_rank"}:
            raise ValueError(f"unknown fusion mode: {fusion}")
        self.image_dim, self.pathway_dim, self.hidden_dim, self.fusion = image_dim, pathway_dim, hidden_dim, fusion
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        self.pathway_encoder = nn.Linear(pathway_dim, hidden_dim)
        if fusion == "concat":
            self.fuser = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "dual_linear":
            self.image_branch, self.pathway_branch = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        elif fusion == "adaptive_gated":
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "low_rank":
            if low_rank_dim <= 0:
                raise ValueError("low_rank_dim must be positive")
            self.image_rank = nn.Linear(hidden_dim, low_rank_dim, bias=False)
            self.pathway_rank = nn.Linear(hidden_dim, low_rank_dim, bias=False)
            self.rank_out = nn.Linear(low_rank_dim, hidden_dim)
        self.pool = AttentionMILPooling(hidden_dim) if pooling == "attention" else None
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"unknown pooling mode: {pooling}")
        self.classifier = nn.Linear(hidden_dim, num_outputs)

    def encode_bag(self, image_features: Optional[Tensor] = None, pathway_features: Optional[Tensor] = None,
                   mask: Optional[Tensor] = None):
        source = image_features if image_features is not None else pathway_features
        if source is None:
            raise ValueError("at least one of image_features or pathway_features is required")
        _require_rank("input features", source, 3)
        if self.fusion != "pathway":
            if image_features is None or image_features.shape[-1] != self.image_dim:
                raise ValueError(f"image_features must end in {self.image_dim} for fusion={self.fusion}")
        if self.fusion != "h_and_e":
            if pathway_features is None or pathway_features.shape != source.shape[:2] + (self.pathway_dim,):
                raise ValueError(f"pathway_features must have shape [B, N, {self.pathway_dim}]")
        if image_features is not None and image_features.shape[:2] != source.shape[:2]:
            raise ValueError("image_features and pathway_features must share [B, N]")
        image = F.gelu(self.image_encoder(image_features)) if image_features is not None else None
        pathway = F.gelu(self.pathway_encoder(pathway_features)) if pathway_features is not None else None
        if self.fusion == "h_and_e": fused = image
        elif self.fusion == "pathway": fused = pathway
        elif self.fusion == "concat": fused = F.gelu(self.fuser(torch.cat([image, pathway], dim=-1)))
        elif self.fusion == "dual_linear": fused = F.gelu(self.image_branch(image) + self.pathway_branch(pathway))
        elif self.fusion == "adaptive_gated":
            gate = torch.sigmoid(self.gate(torch.cat([image, pathway], dim=-1)))
            fused = gate * image + (1.0 - gate) * pathway
        else: fused = image + pathway + self.rank_out(self.image_rank(image) * self.pathway_rank(pathway))
        if self.pool is None:
            pooled, attention = mean_mil_pool(fused, mask), None
        else:
            pooled, attention = self.pool(fused, mask, return_attention=True)
        return pooled, attention

    def forward(self, image_features: Optional[Tensor] = None, pathway_features: Optional[Tensor] = None,
                mask: Optional[Tensor] = None, return_attention: bool = False):
        pooled, attention = self.encode_bag(image_features, pathway_features, mask)
        logits = self.classifier(pooled)
        return (logits, attention) if return_attention else logits


class GenericEqualParameterAdapter(nn.Module):
    """Capacity-matched generic adapter used as a non-pathway-identity control."""

    def __init__(self, pathway_dim: int = 30, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pathway_dim = pathway_dim
        self.net = nn.Sequential(nn.Linear(pathway_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, pathway_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: Tensor) -> Tensor:
        _require_rank("values", values, 3)
        if values.shape[-1] != self.pathway_dim:
            raise ValueError(f"values must end in {self.pathway_dim}")
        return values + self.net(values)


class PathwayViewResidualAdapter(nn.Module):
    """30-pathway identity token adapter over absolute/ordinal/prototype/uncertainty views.

    The final residual projection is zero-initialized, so construction is an
    exact identity map regardless of supplied auxiliary views.
    """

    def __init__(self, pathway_dim: int = 30, token_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pathway_dim = pathway_dim
        self.identity_tokens = nn.Parameter(torch.empty(pathway_dim, token_dim))
        nn.init.normal_(self.identity_tokens, std=0.02)
        self.view_net = nn.Sequential(nn.Linear(token_dim + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.view_net[-1].weight)
        nn.init.zeros_(self.view_net[-1].bias)

    def forward(self, absolute: Tensor, ordinal: Optional[Tensor] = None, prototype: Optional[Tensor] = None,
                uncertainty: Optional[Tensor] = None) -> Tensor:
        _require_rank("absolute", absolute, 3)
        if absolute.shape[-1] != self.pathway_dim:
            raise ValueError(f"absolute must end in {self.pathway_dim}")
        views = [absolute]
        for name, view in (("ordinal", ordinal), ("prototype", prototype), ("uncertainty", uncertainty)):
            if view is None: view = torch.zeros_like(absolute)
            if view.shape != absolute.shape: raise ValueError(f"{name} must have shape {tuple(absolute.shape)}")
            views.append(view)
        stacked = torch.stack(views, dim=-1)  # [B, N, pathways, four views]
        tokens = self.identity_tokens.view(1, 1, self.pathway_dim, -1).expand(*absolute.shape[:2], -1, -1)
        residual = self.view_net(torch.cat([tokens, stacked], dim=-1)).squeeze(-1)
        return absolute + residual


class PathologyPathwayResidualAdapter(nn.Module):
    """Inject four-view pathway evidence into pathology features.

    ``epsilon`` and the output projection are zero-initialized, hence the
    initial output is exactly the input pathology representation ``h`` from the
    protocol equation. The 30 pathway identities remain distinct tokens.
    """

    def __init__(self, image_dim: int, pathway_dim: int = 30, token_dim: int = 24) -> None:
        super().__init__()
        if image_dim <= 0 or pathway_dim <= 0 or token_dim <= 0:
            raise ValueError("all dimensions must be positive")
        self.image_dim = image_dim
        self.pathway_dim = pathway_dim
        self.identity_tokens = nn.Parameter(torch.empty(pathway_dim, token_dim))
        nn.init.normal_(self.identity_tokens, std=0.02)
        self.gate = nn.Sequential(nn.Linear(token_dim + 4, token_dim), nn.GELU(), nn.Linear(token_dim, 1))
        self.output = nn.Linear(token_dim, image_dim, bias=False)
        self.epsilon = nn.Parameter(torch.zeros(()))

    def forward(self, image: Tensor, absolute: Tensor, ordinal: Tensor,
                prototype: Tensor, uncertainty: Tensor) -> Tensor:
        _require_rank("image", image, 3)
        if image.shape[-1] != self.image_dim:
            raise ValueError(f"image must end in {self.image_dim}")
        expected = image.shape[:2] + (self.pathway_dim,)
        for name, value in (("absolute", absolute), ("ordinal", ordinal),
                            ("prototype", prototype), ("uncertainty", uncertainty)):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        views = torch.stack((absolute, ordinal, prototype, uncertainty), dim=-1)
        tokens = self.identity_tokens.view(1, 1, self.pathway_dim, -1).expand(*image.shape[:2], -1, -1)
        gate = torch.sigmoid(self.gate(torch.cat((tokens, views), dim=-1)))
        confidence = (1.0 - uncertainty.clamp(0, 1)).unsqueeze(-1)
        pathway_message = (gate * confidence * tokens).sum(dim=2) / self.pathway_dim
        return image + self.epsilon * self.output(pathway_message)


class MultiSlideSetAttention(nn.Module):
    """Set-attention aggregation from per-slide embeddings to one patient embedding."""

    def __init__(self, feature_dim: int, attention_dim: Optional[int] = None) -> None:
        super().__init__()
        self.pool = AttentionMILPooling(feature_dim, attention_dim)

    def forward(self, slide_features: Tensor, slide_mask: Optional[Tensor] = None, return_attention: bool = False):
        return self.pool(slide_features, slide_mask, return_attention)


class MonotonicPCRMPRHead(nn.Module):
    """Joint logits ordered as ``[pCR, MPR]`` with pCR probability <= MPR."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mpr = nn.Linear(feature_dim, 1)
        self.gap = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2: raise ValueError("features must have batch and feature dimensions")
        mpr_logit = self.mpr(features)
        pcr_logit = mpr_logit - F.softplus(self.gap(features))
        return torch.cat([pcr_logit, mpr_logit], dim=-1)


class JointEndpointMILModel(nn.Module):
    """MIL encoder with the structural pCR/MPR monotonic output contract."""

    def __init__(self, image_dim: int, pathway_dim: int = 30, hidden_dim: int = 128,
                 fusion: FusionMode = "concat", pooling: Literal["mean", "attention"] = "attention") -> None:
        super().__init__()
        self.encoder = MILBagClassifier(image_dim, pathway_dim, hidden_dim, fusion, pooling, num_outputs=1)
        self.head = MonotonicPCRMPRHead(hidden_dim)

    def forward(self, image_features: Optional[Tensor], pathway_features: Optional[Tensor],
                mask: Optional[Tensor] = None, return_attention: bool = False):
        embedding, attention = self.encoder.encode_bag(image_features, pathway_features, mask)
        logits = self.head(embedding)
        return (logits, attention) if return_attention else logits


def inverse_slide_count_weights(slide_counts: Tensor) -> Tensor:
    """Return a per-patient slice weight of ``1 / number_of_slides``."""
    if slide_counts.ndim != 1: raise ValueError("slide_counts must be a rank-1 tensor")
    if not bool((slide_counts > 0).all()): raise ValueError("all slide_counts must be positive")
    return slide_counts.to(dtype=torch.get_default_dtype()).reciprocal()


def monotonic_joint_bce_loss(logits: Tensor, targets: Tensor,
                             sample_weights: Optional[Tensor] = None) -> Tensor:
    """Binary cross-entropy for logits ordered ``[pCR, MPR]``.

    Targets may use ``NaN`` for an unknown endpoint; unknown entries contribute
    no loss and are never coerced to the negative class.
    """
    if logits.ndim != 2 or logits.shape[-1] != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have shape [batch, 2]")
    known = torch.isfinite(targets)
    if not bool(known.any()):
        raise ValueError("at least one endpoint target must be known")
    known_targets = targets[known]
    if not bool(((known_targets == 0) | (known_targets == 1)).all()):
        raise ValueError("known endpoint targets must be binary")
    loss = F.binary_cross_entropy_with_logits(logits[known], known_targets, reduction="none")
    if sample_weights is None:
        return loss.mean()
    if sample_weights.shape != (logits.shape[0],) or not bool((sample_weights > 0).all()):
        raise ValueError("sample_weights must be a positive vector with one value per row")
    expanded = sample_weights[:, None].expand_as(logits)[known]
    return (loss * expanded).sum() / expanded.sum()
