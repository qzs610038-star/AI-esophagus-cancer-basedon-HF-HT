"""Explicit extraction entrypoints for the three frozen encoders."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from adapters import MODEL_NAMES, ModelSpec, load_local_encoder, load_model_specs
from errors import IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache
from input_data import IdentityRecord, ensure_common_identity_manifest, identities, select_dataset_rows
from transforms import FULL_FOV_PROTOCOL, SUPPORTED_PROTOCOLS, build_transform, describe_transform, validate_protocols


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_image_batch(rows: Sequence[IdentityRecord], transform) -> torch.Tensor:
    tensors = []
    for row in rows:
        with Image.open(row.image_path) as image:
            image.load()
            if image.size != (row.width_px, row.height_px):
                raise IdentityMismatchError(f"图像像素尺寸在共同清单建立后发生变化: {row.identity_key}")
            tensors.append(transform(image))
    return torch.stack(tensors).contiguous().float()


def _runtime(device: torch.device, started: float) -> dict:
    return {"elapsed_seconds": time.perf_counter() - started, "device": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else str(device), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" and torch.cuda.is_available() else None, "software": {"python": platform.python_version(), "pytorch": torch.__version__, "timm": _version("timm"), "pillow": _version("Pillow")}}


def _oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def cache_path(config: dict, *, model: str, protocol: str, dataset: str) -> Path:
    return Path(config["paths"]["feature_caches_root"]) / config["experiment_id"] / model / protocol / dataset


def cache_contract(config: dict, spec: ModelSpec, *, protocol: str, dataset: str, manifest_path: Path) -> dict:
    snapshot = Path(spec.snapshot_path).resolve() if spec.snapshot_path else None
    checkpoint = (snapshot / str(spec.checkpoint_filename)).resolve() if snapshot and spec.checkpoint_filename else None
    return {"experiment_id": config["experiment_id"], "protocol_version": config["protocol_version"], "model": spec.name, "repo_id": spec.repo_id, "revision": spec.revision, "architecture": spec.architecture, "snapshot_path": None if snapshot is None else str(snapshot), "checkpoint_path": None if checkpoint is None else str(checkpoint), "preprocessing": describe_transform(protocol), "dataset": dataset, "common_identity_manifest": str(manifest_path.resolve()), "precision": {"dtype": "float32", "amp": False, "tf32": False}}


def extract_one_model(spec: ModelSpec, rows: Sequence[IdentityRecord], output_dir: str | Path, *, protocol: str, device: str | torch.device, batch_sizes: Sequence[int], include_mean_patch: bool = False, adapter_factory: Callable = load_local_encoder, metadata_context: dict | None = None) -> dict:
    """Extract CLS features for one immutable model/protocol/dataset tuple."""
    if protocol not in SUPPORTED_PROTOCOLS or not rows or not batch_sizes or any(int(size) < 1 for size in batch_sizes):
        raise ValueError("协议、身份清单或 batch_sizes 非法")
    select_dataset_rows(rows, "all", protocol=protocol)  # Reject non-square full-fov rows before writes.
    keys = identities(rows)
    if len(keys) != len(set(keys)):
        raise IdentityMismatchError("共同身份清单含重复身份")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前环境不可用")
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats(torch_device)
    adapter, transform, failures, started = adapter_factory(spec, torch_device), build_transform(protocol), [], time.perf_counter()
    for batch_size in map(int, batch_sizes):
        writer = FeatureCacheWriter(output_dir, n_rows=len(rows), feature_dim=spec.layout.feature_dim, identities=keys, include_mean_patch=include_mean_patch)
        try:
            for start in range(0, len(rows), batch_size):
                image_batch = _load_image_batch(rows[start:start + batch_size], transform)
                cls, mean = adapter.encode(image_batch, include_mean_patch=include_mean_patch)
                if start == 0:
                    second_cls, second_mean = adapter.encode(image_batch, include_mean_patch=include_mean_patch)
                    if not torch.equal(cls, second_cls) or (include_mean_patch and not torch.equal(mean, second_mean)):
                        raise RuntimeError("冻结 eval 编码器的重复推理不一致")
                writer.write_rows(start, cls.detach().cpu().numpy(), None if mean is None else mean.detach().cpu().numpy())
            runtime = _runtime(torch_device, started)
            location = writer.finalize({**dict(metadata_context or {}), "token_layout": asdict(spec.layout), "actual_batch_size": batch_size, "fallback_failures": failures, "runtime": runtime})
            return {"status": "complete", "model": spec.name, "protocol": protocol, "cache_dir": str(location), "n_rows": len(rows), "feature_dim": spec.layout.feature_dim, "actual_batch_size": batch_size, "runtime": runtime}
        except Exception as exc:
            writer.abort()
            if _oom(exc):
                failures.append({"batch_size": batch_size, "reason": "cuda_out_of_memory"})
                if torch_device.type == "cuda": torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"{spec.name} 在所有 batch size 下均内存不足: {failures}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, path)


def prepare_feature_caches(config: dict, *, device: str = "cuda", models: Sequence[str] | None = None, protocols: Sequence[str] | None = None, combinations: Sequence[tuple[str, str]] | None = None, datasets: Sequence[str] = ("all",), adapter_factory: Callable = load_local_encoder) -> dict:
    """Prepare or verify caches.  Calls are explicit; no action is implicit at import time."""
    configured_protocols = validate_protocols(config.get("preprocessing", {}).get("protocols"))
    if combinations is not None and (models is not None or protocols is not None):
        raise ValueError("combinations 与 models/protocols 不能同时指定")
    if combinations is None:
        selected_models = tuple(models or config["execution"]["models"])
        selected_protocols = tuple(protocols or configured_protocols)
        if not selected_models or any(name not in MODEL_NAMES for name in selected_models) or len(set(selected_models)) != len(selected_models):
            raise ValueError("models 必须是无重复的 uni2h/uni/virchow2 子集")
        if not selected_protocols or any(name not in SUPPORTED_PROTOCOLS for name in selected_protocols) or len(set(selected_protocols)) != len(selected_protocols):
            raise ValueError("protocols 必须是无重复的已登记协议子集")
        selected_pairs = tuple((model, protocol) for model in selected_models for protocol in selected_protocols)
    else:
        selected_pairs = tuple((str(model), str(protocol)) for model, protocol in combinations)
        if not selected_pairs or len(set(selected_pairs)) != len(selected_pairs):
            raise ValueError("combinations 必须是无重复的非空 (model, protocol) 对")
        if any(model not in MODEL_NAMES or protocol not in SUPPORTED_PROTOCOLS for model, protocol in selected_pairs):
            raise ValueError("combinations 含未登记的 model 或 protocol")
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets 必须为非空且无重复")
    manifest_path, all_rows = ensure_common_identity_manifest(config, require_images=True)
    specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=True)
    batches = [int(config["feature_extraction"]["initial_batch_size"]), *[int(value) for value in config["feature_extraction"].get("fallback_batch_sizes", [])]]
    include_mean = bool(config["feature_extraction"].get("save_mean_patch", False))
    entries = []
    for model, protocol in selected_pairs:
        spec = specs[model]
        for dataset in datasets:
            rows = select_dataset_rows(all_rows, dataset, protocol=protocol)
            target = cache_path(config, model=model, protocol=protocol, dataset=dataset)
            contract = cache_contract(config, spec, protocol=protocol, dataset=dataset, manifest_path=manifest_path)
            if target.exists():
                cache = load_feature_cache(target, expected_identities=identities(rows), expected_dim=spec.layout.feature_dim, expected_metadata=contract, require_mean_patch=False)
                entries.append({"status": "reused_complete_cache", "model": model, "protocol": protocol, "dataset": dataset, "cache_dir": cache["cache_dir"], "n_rows": len(rows), "feature_dim": spec.layout.feature_dim,
                                "mean_patch_available": "mean_patch" in cache})
            else:
                entries.append({**extract_one_model(spec, rows, target, protocol=protocol, device=device, batch_sizes=batches, include_mean_patch=include_mean, adapter_factory=adapter_factory, metadata_context=contract), "dataset": dataset})
    root = Path(config["paths"]["feature_caches_root"]) / config["experiment_id"]
    registry = {"schema_version": "2.0", "status": "complete", "experiment_id": config["experiment_id"], "protocol_version": config["protocol_version"], "common_identity_manifest": str(manifest_path.resolve()), "entries": entries}
    _atomic_json(root / "feature_caches.json", registry)
    return {**registry, "registry_path": str((root / "feature_caches.json").resolve())}


def main() -> int:
    from config import load_config
    parser = argparse.ArgumentParser(description="准备或核对全视野 Phase2 特征缓存")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*"); parser.add_argument("--protocols", nargs="*"); parser.add_argument("--datasets", nargs="*", default=["all"])
    args = parser.parse_args(); os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps(prepare_feature_caches(load_config(args.config), device=args.device, models=args.models, protocols=args.protocols, datasets=args.datasets), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
