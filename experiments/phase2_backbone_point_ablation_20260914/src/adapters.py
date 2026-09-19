"""Strict, local-only adapters for UNI and Virchow2 token encoders."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from errors import ConfigError, NonFiniteDataError, OfflineModelError


PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_MODEL_CONTRACTS = {
    "uni2h": {
        "role": "historical_reference_only",
        "repo_id": "MahmoodLab/UNI2-h",
        "revision": None,
        "architecture": "vit_giant_patch14_224",
        "cls_dim": 1536,
        "mean_patch_dim": 1536,
        "total_tokens": 265,
        "cls_index": 0,
        "patch_start": 9,
        "patch_count": 256,
        "register_count": 8,
        "extract_in_this_package": False,
    },
    "uni": {
        "role": "candidate",
        "repo_id": "MahmoodLab/UNI",
        "revision": "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
        "architecture": "vit_large_patch16_224",
        "checkpoint_filename": "pytorch_model.bin",
        "config_filename": "config.json",
        "cls_dim": 1024,
        "mean_patch_dim": 1024,
        "total_tokens": 197,
        "cls_index": 0,
        "patch_start": 1,
        "patch_count": 196,
        "register_count": 0,
        "extract_in_this_package": True,
    },
    "virchow2": {
        "role": "candidate",
        "repo_id": "paige-ai/Virchow2",
        "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
        "architecture": "vit_huge_patch14_224",
        "checkpoint_filename": "model.safetensors",
        "config_filename": "config.json",
        "cls_dim": 1280,
        "mean_patch_dim": 1280,
        "total_tokens": 261,
        "cls_index": 0,
        "patch_start": 5,
        "patch_count": 256,
        "register_count": 4,
        "extract_in_this_package": True,
    },
}


@dataclass(frozen=True)
class TokenLayout:
    total_tokens: int
    feature_dim: int
    cls_index: int
    patch_start: int
    patch_count: int
    register_count: int

    def validate(self) -> None:
        if min(
            self.total_tokens,
            self.feature_dim,
            self.patch_count,
        ) <= 0:
            raise ConfigError("token 数、特征维数与 patch 数必须为正")
        if self.cls_index != 0:
            raise ConfigError("本协议 CLS 索引必须为 0")
        if self.patch_start + self.patch_count != self.total_tokens:
            raise ConfigError("patch token 边界与总 token 数不一致")
        if self.patch_start != 1 + self.register_count:
            raise ConfigError("patch_start 必须跳过 CLS 与全部 register token")


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
    extract_in_this_package: bool


def pool_tokens(tokens: torch.Tensor, layout: TokenLayout) -> tuple[torch.Tensor, torch.Tensor]:
    layout.validate()
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ConfigError("编码器必须返回三维 [B,T,d] token 张量，禁止猜测二维输出")
    expected = (layout.total_tokens, layout.feature_dim)
    if tuple(tokens.shape[1:]) != expected:
        raise ConfigError(
            f"token 形状必须为 [B,{expected[0]},{expected[1]}]，实际={tuple(tokens.shape)}"
        )
    if tokens.dtype != torch.float32:
        raise ConfigError(f"编码器输出必须为 float32，实际={tokens.dtype}")
    if not torch.isfinite(tokens).all():
        raise NonFiniteDataError("编码器 token 含 NaN/Inf")
    cls = tokens[:, layout.cls_index, :].contiguous()
    patches = tokens[:, layout.patch_start :, :]
    if patches.shape[1] != layout.patch_count:
        raise ConfigError("patch token 数量与模型登记不一致")
    return cls, patches.mean(dim=1).contiguous()


class EncoderAdapter:
    """Freeze a local encoder and expose CLS/mean-patch separately."""

    def __init__(
        self, name: str, model: nn.Module, layout: TokenLayout, device: torch.device
    ) -> None:
        self.name = str(name)
        self.model = model.to(device=device, dtype=torch.float32)
        self.layout = layout
        self.device = device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ConfigError("images 必须为 [B,3,H,W]")
        self.model.eval()
        images = images.to(device=self.device, dtype=torch.float32, non_blocking=False)
        with torch.inference_mode():
            if not hasattr(self.model, "forward_features"):
                raise OfflineModelError("编码器缺少 forward_features，无法取得完整 token")
            tokens = self.model.forward_features(images)
        if isinstance(tokens, (tuple, list, dict)):
            raise OfflineModelError("编码器返回结构不明确，拒绝猜测 token 字段")
        return pool_tokens(tokens, self.layout)


def strict_load_state(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise OfflineModelError(f"本地权重严格加载失败: {exc}") from exc


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineModelError(f"无法读取模型清单 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OfflineModelError("模型清单根必须是 JSON object")
    return payload


def load_model_specs(
    manifest_path: str | Path, *, require_local_files: bool = True
) -> dict[str, ModelSpec]:
    payload = _read_json(Path(manifest_path).resolve())
    if payload.get("schema_version") != "1.0":
        raise ConfigError("模型清单 schema_version 必须为 1.0")
    raw_models = payload.get("models") or {}
    if set(raw_models) != {"uni2h", "uni", "virchow2"}:
        raise ConfigError("模型清单必须且只能登记 uni2h、uni、virchow2")
    specs: dict[str, ModelSpec] = {}
    for name, raw in raw_models.items():
        for field, expected in EXPECTED_MODEL_CONTRACTS[name].items():
            if raw.get(field) != expected:
                raise OfflineModelError(
                    f"{name}.{field} 偏离固定合同: expected={expected!r}, actual={raw.get(field)!r}"
                )
        layout = TokenLayout(
            int(raw["total_tokens"]),
            int(raw["cls_dim"]),
            int(raw["cls_index"]),
            int(raw["patch_start"]),
            int(raw["patch_count"]),
            int(raw["register_count"]),
        )
        layout.validate()
        revision = raw.get("revision")
        extract = bool(raw.get("extract_in_this_package"))
        if extract and not (isinstance(revision, str) and PINNED_REVISION.fullmatch(revision)):
            raise OfflineModelError(f"{name} revision 必须固定为 40 位 commit SHA")
        if name == "uni2h" and extract:
            raise ConfigError("UNI2-h 禁止由本包提取")
        spec = ModelSpec(
            name=name,
            role=str(raw.get("role")),
            repo_id=str(raw.get("repo_id")),
            revision=revision,
            architecture=str(raw.get("architecture")),
            checkpoint_filename=raw.get("checkpoint_filename"),
            config_filename=raw.get("config_filename"),
            snapshot_path=raw.get("snapshot_path"),
            layout=layout,
            extract_in_this_package=extract,
        )
        if extract and require_local_files:
            _validate_local_snapshot(spec)
        specs[name] = spec
    return specs


def _validate_local_snapshot(spec: ModelSpec) -> tuple[Path, Path]:
    if not spec.snapshot_path:
        raise OfflineModelError(f"{spec.name} 尚未登记 snapshot_path")
    snapshot = Path(spec.snapshot_path)
    expected_suffix = Path("snapshots") / str(spec.revision)
    if expected_suffix.as_posix().lower() not in snapshot.as_posix().lower() or snapshot.name != spec.revision:
        raise OfflineModelError(
            f"{spec.name} snapshot_path 必须指向固定 revision 的 snapshots/<commit>"
        )
    config_path = snapshot / str(spec.config_filename)
    checkpoint_path = snapshot / str(spec.checkpoint_filename)
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise OfflineModelError(f"{spec.name} 本地文件缺失: {path}")
        if path.suffix == ".incomplete" or path.stat().st_size == 0:
            raise OfflineModelError(f"{spec.name} 本地文件未完成: {path}")
    if checkpoint_path.stat().st_size < 1024:
        prefix = checkpoint_path.read_bytes()[:256]
        if b"git-lfs" in prefix or b"version https://" in prefix:
            raise OfflineModelError(f"{spec.name} 权重仍是 LFS 指针: {checkpoint_path}")
    return config_path, checkpoint_path


def _build_model(spec: ModelSpec, config_path: Path) -> nn.Module:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment precheck covers this
        raise OfflineModelError("缺少 timm，不能构造本地编码器") from exc

    if spec.name == "uni":
        expected_architecture = "vit_large_patch16_224"
        if spec.architecture != expected_architecture:
            raise OfflineModelError("UNI architecture 与官方固定构造不一致")
        return timm.create_model(
            expected_architecture,
            pretrained=False,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
    if spec.name == "virchow2":
        config_payload = _read_json(config_path)
        if config_payload.get("architecture") != spec.architecture:
            raise OfflineModelError("Virchow2 config architecture 与清单不一致")
        args = dict(config_payload.get("model_args") or {})
        try:
            from timm.layers import SwiGLUPacked
        except ImportError as exc:
            raise OfflineModelError("当前 timm 缺少 Virchow2 所需 SwiGLUPacked") from exc
        args.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        return timm.create_model(spec.architecture, pretrained=False, **args)
    raise ConfigError(f"本包不构造编码器 {spec.name!r}")


def _load_checkpoint(spec: ModelSpec, checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    if spec.name == "virchow2":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise OfflineModelError("缺少 safetensors，无法读取 Virchow2 权重") from exc
        state = load_file(str(checkpoint_path), device="cpu")
    else:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise OfflineModelError(f"无法读取 {spec.name} 权重: {exc}") from exc
        if isinstance(state, dict) and set(state) == {"state_dict"}:
            state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise OfflineModelError(f"{spec.name} 权重不是非空 state_dict")
    return state


def load_local_encoder(spec: ModelSpec, device: str | torch.device) -> EncoderAdapter:
    """Never contacts Hugging Face; config and weights must already be local."""

    if not spec.extract_in_this_package or spec.name not in {"uni", "virchow2"}:
        raise ConfigError(f"{spec.name} 不是本包可提取的新候选模型")
    config_path, checkpoint_path = _validate_local_snapshot(spec)
    model = _build_model(spec, config_path)
    state = _load_checkpoint(spec, checkpoint_path)
    strict_load_state(model, state)
    return EncoderAdapter(spec.name, model, spec.layout, torch.device(device))
