"""Online UNI2-h + exact MPP MLP head model for the guarded LoRA route."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

from lora_utils import (
    freeze_all_parameters,
    get_lora_parameters,
    inject_lora_to_backbone,
)
from train_mpp_uni2h_mlp import MPPMLPHead


P0_LORA_BLOCKS = tuple(range(24))
P0_LORA_TARGETS = frozenset({"qkv", "proj"})


class OnlineMPPModel(nn.Module):
    """Raw image -> UNI2-h CLS -> the accepted two-layer ``MPPMLPHead``."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 1536,
        hidden_dim: int = 1024,
        output_dim: int = 30,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head = MPPMLPHead(
            in_dim=feature_dim,
            hidden=hidden_dim,
            out_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            shape = getattr(features, "shape", None)
            raise RuntimeError(f"UNI2-h backbone must return [B,D] CLS features, got {shape}")
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"UNI2-h feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape[-1]}"
            )
        return self.head(features)


def configure_p0_lora(
    backbone: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_blocks: Sequence[int] = P0_LORA_BLOCKS,
) -> int:
    """Freeze the base backbone and inject the approved qkv+proj adapters."""
    if rank != 8:
        raise ValueError("Phase 0/P0 is pre-registered for LoRA rank=8 only")
    if alpha != 16.0:
        raise ValueError("Phase 0/P0 is pre-registered for LoRA alpha=16 only")
    freeze_all_parameters(backbone)
    created = inject_lora_to_backbone(
        backbone,
        target_blocks=list(target_blocks),
        rank=rank,
        alpha=alpha,
        target_modules=set(P0_LORA_TARGETS),
        dropout=dropout,
    )
    if not created:
        raise RuntimeError("no LoRA modules were injected")
    return sum(parameter.numel() for parameter in get_lora_parameters(backbone))


def build_paired_optimizer(
    model: OnlineMPPModel,
    mode: str,
    head_lr: float = 1e-4,
    lora_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """Build paired S0/S1 optimizers and reject unintended trainable params."""
    if mode not in {"frozen", "lora"}:
        raise ValueError(f"unsupported MPP online mode: {mode}")
    head_parameters = list(model.head.parameters())
    lora_parameters = list(get_lora_parameters(model.backbone))
    allowed_ids = {id(parameter) for parameter in head_parameters}
    if mode == "lora":
        if not lora_parameters:
            raise ValueError("lora mode requires injected LoRA parameters")
        allowed_ids.update(id(parameter) for parameter in lora_parameters)
    elif lora_parameters:
        raise ValueError("frozen mode must not contain LoRA modules")

    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed_ids
    ]
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {unexpected[:10]}")

    duplicate_count = len(head_parameters) + len(lora_parameters) - len(
        {id(parameter) for parameter in head_parameters + lora_parameters}
    )
    if duplicate_count:
        raise ValueError("optimizer parameter groups contain duplicates")

    groups = [{"params": head_parameters, "lr": head_lr, "name": "mpp_head"}]
    if mode == "lora":
        groups.append({"params": lora_parameters, "lr": lora_lr, "name": "lora"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def load_accepted_mpp_head(
    model: OnlineMPPModel,
    checkpoint_path: str | Path,
) -> dict:
    """Load a frozen-baseline ``MPPMLPHead`` checkpoint into the online model."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("head checkpoint has no model_state_dict mapping")
    model.head.load_state_dict(state, strict=True)
    return checkpoint


def trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
