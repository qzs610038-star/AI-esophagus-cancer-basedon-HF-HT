"""Strict encoder output adapters. Unit tests cover the three output modes without weights."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

from errors import ConfigError, NonFiniteDataError, OfflineModelError
from transforms import HOPTIMUS_RGB_V1, IMAGENET_RGB_V1


PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
TRAIN_MODELS = ("hoptimus0", "hoptimus1", "phikonv2")
OUTPUT_MODES = ("timm_tokens_cls", "timm_final_embedding", "transformers_last_hidden_cls")
EXPECTED_CONTRACTS = {
    "hoptimus0": {
        "repo_id": "bioptimus/H-optimus-0",
        "revision": "b145cc1e6c6b30d3251aa8b1f844e6974188a743",
        "loader_type": "timm_local_snapshot",
        "output_mode": "timm_final_embedding",
        "feature_dim": 1536,
        "normalization_profile": HOPTIMUS_RGB_V1,
        "checkpoint_filename": "pytorch_model.bin",
        "config_filename": "config.json",
        "official_expected_mpp": 0.5,
        "gated": True,
        "access_note": "auto_gated_apache2; 401/403 is access denial not a network retry",
    },
    "hoptimus1": {
        "repo_id": "bioptimus/H-optimus-1",
        "revision": "3592cb220dec7a150c5d7813fb56e68bd57473b9",
        "loader_type": "timm_local_snapshot",
        "output_mode": "timm_final_embedding",
        "feature_dim": 1536,
        "normalization_profile": HOPTIMUS_RGB_V1,
        "checkpoint_filename": "model.safetensors",
        "config_filename": "config.json",
        "official_expected_mpp": 0.5,
        "gated": True,
        "access_note": "manual_gated_cc_by_nc_nd; failure must not block the other two models",
    },
    "phikonv2": {
        "repo_id": "owkin/phikon-v2",
        "revision": "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd",
        "loader_type": "transformers_local_snapshot",
        "output_mode": "transformers_last_hidden_cls",
        "feature_dim": 1024,
        "normalization_profile": IMAGENET_RGB_V1,
        "checkpoint_filename": "model.safetensors",
        "config_filename": "config.json",
        "official_expected_mpp": None,
        "gated": False,
        "access_note": "owkin_non_commercial; do not copy weights into the package or ordinary returns",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    loader_type: str
    output_mode: str
    feature_dim: int
    normalization_profile: str
    checkpoint_filename: str
    config_filename: str
    snapshot_path: str | None
    official_expected_mpp: float | None
    gated: bool
    architecture: str | None = None
    cls_index: int = 0
    license_note: str | None = None
    allow_patterns: tuple[str, ...] = ()
    extract_in_this_package: bool = True


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineModelError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OfflineModelError("JSON 根必须是 object")
    return payload


def load_model_specs(manifest_path: str | Path, *, require_local_files: bool = True) -> dict[str, ModelSpec]:
    payload = _read_json(Path(manifest_path).resolve())
    if payload.get("schema_version") != "1.0":
        raise ConfigError("模型清单 schema_version 必须为 1.0")
    raw_models = payload.get("models") or {}
    if set(raw_models) != set(TRAIN_MODELS):
        raise ConfigError("模型清单必须且只能登记 hoptimus0、hoptimus1、phikonv2")
    specs: dict[str, ModelSpec] = {}
    for name in TRAIN_MODELS:
        raw = raw_models[name]
        expected = EXPECTED_CONTRACTS[name]
        for field, value in expected.items():
            if field == "access_note":
                continue
            if raw.get(field) != value:
                raise OfflineModelError(f"{name}.{field} 偏离固定合同: expected={value!r}, actual={raw.get(field)!r}")
        revision = str(raw["revision"])
        if not PINNED_REVISION.fullmatch(revision):
            raise OfflineModelError(f"{name} revision 必须为 40 位 commit SHA")
        spec = ModelSpec(
            name=name,
            repo_id=str(raw["repo_id"]),
            revision=revision,
            loader_type=str(raw["loader_type"]),
            output_mode=str(raw["output_mode"]),
            feature_dim=int(raw["feature_dim"]),
            normalization_profile=str(raw["normalization_profile"]),
            checkpoint_filename=str(raw["checkpoint_filename"]),
            config_filename=str(raw["config_filename"]),
            snapshot_path=raw.get("snapshot_path"),
            official_expected_mpp=raw.get("official_expected_mpp"),
            gated=bool(raw.get("gated")),
            architecture=raw.get("architecture"),
            cls_index=int(raw.get("cls_index", 0)),
            license_note=raw.get("license_note"),
            allow_patterns=tuple(raw.get("allow_patterns") or ()),
            extract_in_this_package=bool(raw.get("extract_in_this_package", True)),
        )
        if spec.output_mode not in OUTPUT_MODES:
            raise ConfigError(f"{name} 未知 output_mode={spec.output_mode}")
        if require_local_files:
            validate_local_snapshot(spec)
        specs[name] = spec
    return specs


def validate_local_snapshot(spec: ModelSpec) -> tuple[Path, Path]:
    if not spec.snapshot_path:
        raise OfflineModelError(f"{spec.name} 缺少本地 snapshot_path")
    root = Path(spec.snapshot_path).resolve()
    if root.name != spec.revision or root.parent.name != "snapshots":
        raise OfflineModelError(f"{spec.name} snapshot_path 必须精确指向 snapshots/<revision>")
    checkpoint = root / spec.checkpoint_filename
    config = root / spec.config_filename
    for item in (checkpoint, config):
        if not item.is_file() or item.stat().st_size == 0 or item.suffix == ".incomplete":
            raise OfflineModelError(f"{spec.name} 本地快照文件无效: {item}")
    if checkpoint.stat().st_size < 1024 and b"git-lfs" in checkpoint.read_bytes()[:256]:
        raise OfflineModelError(f"{spec.name} 权重仍为 Git LFS 指针")
    return config, checkpoint


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise ConfigError(f"编码器输出必须是张量，实际={type(value).__name__}")


def adapt_encoder_output(raw: Any, spec: ModelSpec, *, batch_size: int) -> torch.Tensor:
    """Map a raw encoder return value onto the selected [B, d] feature. No guessing."""
    mode = spec.output_mode
    if mode == "timm_final_embedding":
        tensor = _as_tensor(raw)
        if tensor.ndim != 2:
            raise ConfigError(f"H-optimus 必须返回二维 [B,d]，实际 rank={tensor.ndim}；拒绝猜测 token")
        features = tensor
    elif mode == "transformers_last_hidden_cls":
        hidden = getattr(raw, "last_hidden_state", None)
        if hidden is None and isinstance(raw, Mapping):
            hidden = raw.get("last_hidden_state")
        if hidden is None:
            raise ConfigError("Phikon-v2 缺少 last_hidden_state，拒绝猜测替代输出")
        tensor = _as_tensor(hidden)
        if tensor.ndim != 3:
            raise ConfigError(f"Phikon last_hidden_state 必须为三维 [B,T,d]，实际 rank={tensor.ndim}")
        features = tensor[:, int(spec.cls_index), :]
    elif mode == "timm_tokens_cls":
        tensor = _as_tensor(raw)
        if tensor.ndim != 3:
            raise ConfigError("timm_tokens_cls 要求三维 token 张量 [B,T,d]")
        features = tensor[:, int(spec.cls_index), :]
    else:
        raise ConfigError(f"未实现的 output_mode={mode}")
    if features.shape[0] != int(batch_size):
        raise ConfigError(f"特征 batch 维与输入不一致: expected={batch_size}, actual={tuple(features.shape)}")
    if features.shape[-1] != int(spec.feature_dim):
        raise ConfigError(f"特征维数与清单不一致: expected={spec.feature_dim}, actual={tuple(features.shape)}")
    if features.dtype != torch.float32:
        features = features.float()
    if not torch.isfinite(features).all():
        raise NonFiniteDataError("编码器特征含 NaN/Inf")
    return features.detach().to(device="cpu", dtype=torch.float32).contiguous()


class EncoderAdapter:
    def __init__(self, spec: ModelSpec, model: nn.Module, device: torch.device) -> None:
        self.spec = spec
        self.model = model.to(device=device, dtype=torch.float32)
        self.device = device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ConfigError("images 必须是冻结预处理后的 [B,3,224,224]")
        self.model.eval()
        batch = images.to(self.device, dtype=torch.float32, non_blocking=False)
        with torch.inference_mode():
            if self.spec.output_mode == "transformers_last_hidden_cls":
                raw = self.model(pixel_values=batch)
            else:
                raw = self.model(batch)
        return adapt_encoder_output(raw, self.spec, batch_size=int(images.shape[0]))


def strict_load_state(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise OfflineModelError("本地权重不是非空 state_dict")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise OfflineModelError(f"本地权重严格加载失败: {exc}") from exc


def _load_checkpoint(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    try:
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            value = load_file(str(checkpoint), device="cpu")
        else:
            value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise OfflineModelError(f"无法读取权重 {checkpoint}: {exc}") from exc
    if isinstance(value, Mapping) and set(value) == {"state_dict"}:
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise OfflineModelError("权重文件不是 state_dict")
    return value


def _timm_kwargs(config_payload: dict, spec: ModelSpec) -> dict:
    args = dict(config_payload.get("model_args") or {})
    args.setdefault("pretrained", False)
    args.setdefault("num_classes", 0)
    args.setdefault("init_values", 1e-5)
    args.setdefault("dynamic_img_size", False)
    args["pretrained"] = False
    args["num_classes"] = 0
    if spec.architecture and "model_name" in args:
        args.pop("model_name", None)
    return args


def _build_timm_model(spec: ModelSpec, config_path: Path) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise OfflineModelError("缺少 timm，不能构造 H-optimus") from exc
    payload = _read_json(config_path)
    architecture = spec.architecture or payload.get("architecture")
    if not architecture:
        raise OfflineModelError(f"{spec.name} 本地 config.json 没有 architecture，不能猜测结构")
    kwargs = _timm_kwargs(payload, spec)
    try:
        return timm.create_model(str(architecture), **kwargs)
    except TypeError:
        kwargs.pop("init_values", None)
        try:
            return timm.create_model(str(architecture), **kwargs)
        except Exception as exc:
            raise OfflineModelError(f"{spec.name} 无法按本地 config 构造 timm 模型: {exc}") from exc
    except Exception as exc:
        raise OfflineModelError(f"{spec.name} 无法按本地 config 构造 timm 模型: {exc}") from exc


def _build_transformers_model(snapshot: Path) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise OfflineModelError("缺少 transformers，不能加载 Phikon-v2") from exc
    try:
        return AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    except Exception as exc:
        raise OfflineModelError(f"Phikon-v2 本地严格加载失败: {exc}") from exc


def load_local_encoder(spec: ModelSpec, device: str | torch.device) -> EncoderAdapter:
    if not spec.extract_in_this_package:
        raise ConfigError(f"{spec.name} 未标记为可提取")
    config_path, checkpoint = validate_local_snapshot(spec)
    snapshot = Path(spec.snapshot_path).resolve()
    if spec.loader_type == "timm_local_snapshot":
        model = _build_timm_model(spec, config_path)
        strict_load_state(model, _load_checkpoint(checkpoint))
    elif spec.loader_type == "transformers_local_snapshot":
        model = _build_transformers_model(snapshot)
    else:
        raise ConfigError(f"未知 loader_type={spec.loader_type}")
    return EncoderAdapter(spec, model, torch.device(device))


def fake_spec(**overrides: Any) -> ModelSpec:
    """Test helper: a complete spec with no snapshot requirement."""
    values = dict(
        name="hoptimus0",
        repo_id="bioptimus/H-optimus-0",
        revision="b" * 40,
        loader_type="timm_local_snapshot",
        output_mode="timm_final_embedding",
        feature_dim=1536,
        normalization_profile=HOPTIMUS_RGB_V1,
        checkpoint_filename="pytorch_model.bin",
        config_filename="config.json",
        snapshot_path=None,
        official_expected_mpp=0.5,
        gated=True,
    )
    values.update(overrides)
    return ModelSpec(**values)


def last_hidden_state_output(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(last_hidden_state=tensor)
