"""Self-contained offline UNI2-h construction and CLS extraction.

The experiment package never downloads weights.  ``checkpoint`` must resolve to
an existing ``pytorch_model.bin`` or to a directory containing exactly one such
file.  This keeps server execution tied to the configured read-only model cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


UNI2_MODEL_ID = "MahmoodLab/UNI2-h"
UNI2_FEATURE_DIM = 1536
UNI2_DEPTH = 24


def resolve_uni2_checkpoint(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"UNI2-h checkpoint path does not exist: {candidate}")
    matches = sorted(
        path for path in candidate.rglob("pytorch_model.bin")
        if ".incomplete" not in str(path)
    )
    if len(matches) != 1:
        raise RuntimeError(
            "UNI2-h checkpoint directory must contain exactly one complete "
            f"pytorch_model.bin; found {len(matches)} under {candidate}"
        )
    return matches[0].resolve()


def build_uni2_h() -> nn.Module:
    """Construct the official UNI2-h ViT without network access or weights."""
    try:
        from timm.layers import SwiGLUPacked
        from timm.models.vision_transformer import VisionTransformer
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise RuntimeError("timm is required to construct UNI2-h") from exc

    return VisionTransformer(
        img_size=224,
        patch_size=14,
        depth=UNI2_DEPTH,
        num_heads=24,
        init_values=1e-5,
        embed_dim=UNI2_FEATURE_DIM,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        global_pool="",
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )


def load_uni2_h(
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Path]:
    """Load a local UNI2-h checkpoint and freeze every base parameter."""
    checkpoint_path = resolve_uni2_checkpoint(checkpoint)
    model = build_uni2_h()
    state: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("UNI2-h checkpoint must contain a state-dict mapping")
    state = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if not str(key).startswith(("head.", "pre_logits."))
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    material_missing = [
        key for key in missing if not key.startswith(("head.", "pre_logits."))
    ]
    material_unexpected = [
        key for key in unexpected if not key.startswith(("head.", "pre_logits."))
    ]
    if material_missing or material_unexpected:
        raise RuntimeError(
            "UNI2-h checkpoint is structurally incompatible: "
            f"missing={material_missing[:8]}, unexpected={material_unexpected[:8]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if device is not None:
        model.to(device)
    return model, checkpoint_path


def forward_cls(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the exact CLS token used by the historical frozen cache."""
    output: Any = (
        backbone.forward_features(images)
        if hasattr(backbone, "forward_features")
        else backbone(images)
    )
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "features"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"UNI2-h output must be a tensor, got {type(output).__name__}")
    if output.ndim == 3:
        output = output[:, 0, :]
    if output.ndim != 2 or output.shape[-1] != UNI2_FEATURE_DIM:
        raise RuntimeError(
            f"UNI2-h CLS must be [B,{UNI2_FEATURE_DIM}], got {tuple(output.shape)}"
        )
    return output


def set_gradient_checkpointing(backbone: nn.Module, enabled: bool) -> None:
    """Enable timm checkpointing only when the installed model supports it."""
    if not enabled:
        return
    setter = getattr(backbone, "set_grad_checkpointing", None)
    if setter is None:
        raise RuntimeError("configured gradient checkpointing is unsupported")
    setter(True)
