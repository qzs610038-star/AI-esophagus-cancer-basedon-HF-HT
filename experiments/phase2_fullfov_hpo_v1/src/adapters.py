"""Strict offline encoder contracts for UNI2-h, UNI and Virchow2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from errors import ConfigError, NonFiniteDataError

# Keep offline snapshot failures within the package's established config-error API.
OfflineModelError = ConfigError


PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
MODEL_NAMES = ("uni2h", "uni", "virchow2")
EXPECTED_LAYOUTS = {
    "uni2h": (1536, 265, 0, 9, 256, 8),
    "uni": (1024, 197, 0, 1, 196, 0),
    "virchow2": (1280, 261, 0, 5, 256, 4),
}


@dataclass(frozen=True)
class TokenLayout:
    feature_dim: int
    total_tokens: int
    cls_index: int
    patch_start: int
    patch_count: int
    register_count: int

    def validate(self) -> None:
        if min(self.total_tokens, self.feature_dim, self.patch_count) <= 0:
            raise ConfigError("token 数、特征维数与 patch 数必须为正")
        if self.cls_index != 0 or self.patch_start != 1 + self.register_count:
            raise ConfigError("CLS/register/patch token 布局不符合固定合同")
        if self.patch_start + self.patch_count != self.total_tokens:
            raise ConfigError("patch token 边界与总 token 数不一致")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    repo_id: str
    revision: str | None
    architecture: str
    checkpoint_filename: str | None
    config_filename: str | None
    snapshot_path: str | None
    layout: TokenLayout
    extract_in_this_package: bool = True


def pool_tokens(tokens: torch.Tensor, layout: TokenLayout, *, include_mean_patch: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reject ambiguous output layouts and return CLS plus optional patch-token mean."""
    layout.validate()
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ConfigError("编码器必须返回三维 [B,T,d] token 张量，禁止猜测字段")
    if tuple(tokens.shape[1:]) != (layout.total_tokens, layout.feature_dim):
        raise ConfigError(f"token 形状不符合登记布局: actual={tuple(tokens.shape)}")
    if tokens.dtype != torch.float32:
        raise ConfigError(f"编码器输出必须为 float32，实际={tokens.dtype}")
    if not torch.isfinite(tokens).all():
        raise NonFiniteDataError("编码器 token 含 NaN/Inf")
    cls = tokens[:, layout.cls_index, :].contiguous()
    mean_patch = tokens[:, layout.patch_start:, :].mean(dim=1).contiguous() if include_mean_patch else None
    return cls, mean_patch


class EncoderAdapter:
    """Frozen FP32 encoder that always obtains complete token sequences."""

    def __init__(self, name: str, model: nn.Module, layout: TokenLayout, device: torch.device) -> None:
        self.name, self.model, self.layout, self.device = str(name), model.to(device=device, dtype=torch.float32), layout, device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, images: torch.Tensor, *, include_mean_patch: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ConfigError("images 必须是冻结预处理后的 [B,3,224,224]")
        self.model.eval()
        with torch.inference_mode():
            tokens = self.model.forward_features(images.to(self.device, dtype=torch.float32, non_blocking=False)) if hasattr(self.model, "forward_features") else None
        if isinstance(tokens, (tuple, list, dict)) or tokens is None:
            raise OfflineModelError("编码器未返回明确的完整 token 张量")
        return pool_tokens(tokens, self.layout, include_mean_patch=include_mean_patch)


def strict_load_state(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise OfflineModelError("本地权重不是非空 state_dict")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise OfflineModelError(f"本地权重严格加载失败: {exc}") from exc


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineModelError(f"无法读取模型文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OfflineModelError("模型清单根必须是 JSON object")
    return value


def load_model_specs(manifest_path: str | Path, *, require_local_files: bool = True) -> dict[str, ModelSpec]:
    payload = _read_json(Path(manifest_path))
    if payload.get("schema_version") != "1.0" or set(payload.get("models") or ()) != set(MODEL_NAMES):
        raise ConfigError("模型清单必须登记 schema 1.0 的 uni2h、uni、virchow2")
    specs: dict[str, ModelSpec] = {}
    for name in MODEL_NAMES:
        raw = payload["models"][name]
        layout = TokenLayout(int(raw["cls_dim"]), int(raw["total_tokens"]), int(raw["cls_index"]), int(raw["patch_start"]), int(raw["patch_count"]), int(raw["register_count"]))
        layout.validate()
        if (layout.feature_dim, layout.total_tokens, layout.cls_index, layout.patch_start, layout.patch_count, layout.register_count) != EXPECTED_LAYOUTS[name]:
            raise OfflineModelError(f"{name} token 布局偏离固定合同")
        revision = raw.get("revision")
        pinned = isinstance(revision, str) and PINNED_REVISION.fullmatch(revision)
        if require_local_files and not pinned:
            raise OfflineModelError(f"{name} 尚未登记固定 40 位 snapshot revision，不能提取新缓存")
        spec = ModelSpec(name, str(raw.get("role") or "candidate"), str(raw.get("repo_id")), revision, str(raw.get("architecture")), raw.get("checkpoint_filename"), raw.get("config_filename"), raw.get("snapshot_path"), layout, bool(raw.get("extract_in_this_package", True)))
        if require_local_files:
            _validate_local_snapshot(spec)
        specs[name] = spec
    return specs


def _validate_local_snapshot(spec: ModelSpec) -> tuple[Path, Path]:
    if not spec.snapshot_path or not spec.checkpoint_filename:
        raise OfflineModelError(f"{spec.name} 缺少本地 snapshot_path 或 checkpoint_filename")
    root = Path(spec.snapshot_path).resolve()
    if root.name != spec.revision or root.parent.name != "snapshots":
        raise OfflineModelError(f"{spec.name} snapshot_path 必须精确指向 snapshots/<revision>")
    checkpoint = root / str(spec.checkpoint_filename)
    config = root / str(spec.config_filename) if spec.config_filename else None
    files = [checkpoint] if config is None else [config, checkpoint]
    for item in files:
        if not item.is_file() or item.stat().st_size == 0 or item.suffix == ".incomplete":
            raise OfflineModelError(f"{spec.name} 本地快照文件无效: {item}")
    if checkpoint.stat().st_size < 1024 and b"git-lfs" in checkpoint.read_bytes()[:256]:
        raise OfflineModelError(f"{spec.name} 权重仍为 Git LFS 指针")
    return config or root, checkpoint


def _build_model(spec: ModelSpec, config_path: Path) -> nn.Module:
    try:
        import timm
        from timm.layers import SwiGLUPacked
    except ImportError as exc:
        raise OfflineModelError("缺少 timm 及其模型层，不能构造本地编码器") from exc
    if spec.name == "uni2h":
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, global_pool="", mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    if spec.name == "uni":
        return timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    raw = _read_json(config_path)
    if raw.get("architecture") != spec.architecture:
        raise OfflineModelError("Virchow2 config architecture 与清单不一致")
    args = dict(raw.get("model_args") or {})
    args.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, num_classes=0)
    return timm.create_model(spec.architecture, pretrained=False, **args)


def _load_checkpoint(spec: ModelSpec, checkpoint: Path) -> Mapping[str, torch.Tensor]:
    try:
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            value = load_file(str(checkpoint), device="cpu")
        else:
            value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise OfflineModelError(f"无法读取 {spec.name} 权重: {exc}") from exc
    if isinstance(value, Mapping) and set(value) == {"state_dict"}:
        value = value["state_dict"]
    return value


def load_local_encoder(spec: ModelSpec, device: str | torch.device) -> EncoderAdapter:
    """Load only the manifest-pinned local snapshot; this function has no network path."""
    if not spec.extract_in_this_package:
        raise ConfigError(f"{spec.name} 未标记为可提取；请先在模型清单登记本地快照")
    config_path, checkpoint = _validate_local_snapshot(spec)
    model = _build_model(spec, config_path)
    strict_load_state(model, _load_checkpoint(spec, checkpoint))
    return EncoderAdapter(spec.name, model, spec.layout, torch.device(device))
