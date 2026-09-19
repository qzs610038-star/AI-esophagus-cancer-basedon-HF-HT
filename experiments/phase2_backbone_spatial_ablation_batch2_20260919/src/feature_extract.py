"""Explicit extraction for the three new encoders. No mean-patch or unused tokens."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from config import GEOMETRY_PROTOCOL, TRAIN_MODELS, preprocess_profile
from errors import IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache
from input_data import IdentityRecord, ensure_common_identity_manifest, identities, select_dataset_rows
from model_adapters import ModelSpec, load_local_encoder, load_model_specs, validate_local_snapshot
from transforms import build_transform, describe_transform


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


def _software() -> dict:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": _version("torchvision"),
        "timm": _version("timm"),
        "transformers": _version("transformers"),
        "huggingface_hub": _version("huggingface_hub"),
        "pillow": _version("Pillow"),
    }


def _runtime(device: torch.device, started: float) -> dict:
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else str(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" and torch.cuda.is_available() else None,
        "software": _software(),
    }


def _oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def cache_path(config: dict, *, model: str, dataset: str = "all") -> Path:
    profile = preprocess_profile(model)
    return Path(config["paths"]["feature_caches_root"]) / config["experiment_id"] / model / profile / dataset


def cache_contract(config: dict, spec: ModelSpec, *, dataset: str, manifest_path: Path) -> dict:
    snapshot = Path(spec.snapshot_path).resolve() if spec.snapshot_path else None
    checkpoint = (snapshot / spec.checkpoint_filename).resolve() if snapshot else None
    transform_meta = describe_transform(GEOMETRY_PROTOCOL, spec.normalization_profile)
    return {
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "model": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "loader_type": spec.loader_type,
        "output_mode": spec.output_mode,
        "feature_dim": spec.feature_dim,
        "snapshot_path": None if snapshot is None else str(snapshot),
        "checkpoint_filename": spec.checkpoint_filename,
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "normalization_profile": spec.normalization_profile,
        "image_normalization_object": "image_input_before_encoder",
        "image_normalization_mean": transform_meta["mean"],
        "image_normalization_std": transform_meta["std"],
        "uses_auto_image_processor": False,
        "official_expected_mpp": spec.official_expected_mpp,
        "project_mpp_status": "unverified",
        "dataset": dataset,
        "common_identity_manifest": str(manifest_path.resolve()),
        "identity_schema": "patient_id|source_group|spot_id",
        "split_id": config["data"]["split_id"],
        "label_version": config["data"]["label_version"],
        "precision": {"dtype": "float32", "amp": False, "tf32": False},
        "uses_auto_image_processor": False,
    }


def extract_one_model(
    spec: ModelSpec,
    rows: Sequence[IdentityRecord],
    output_dir: str | Path,
    *,
    device: str | torch.device,
    batch_sizes: Sequence[int],
    adapter_factory: Callable = load_local_encoder,
    metadata_context: dict | None = None,
) -> dict:
    if spec.output_mode not in {"timm_final_embedding", "transformers_last_hidden_cls"}:
        raise ValueError("本批提取只允许 H-optimus 二维 embedding 或 Phikon CLS")
    if not rows or not batch_sizes or any(int(size) < 1 for size in batch_sizes):
        raise ValueError("身份清单或 batch_sizes 非法")
    select_dataset_rows(rows, "all", protocol=GEOMETRY_PROTOCOL)
    keys = identities(rows)
    if len(keys) != len(set(keys)):
        raise IdentityMismatchError("共同身份清单含重复身份")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前环境不可用")
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(torch_device)
    adapter = adapter_factory(spec, torch_device)
    transform = build_transform(GEOMETRY_PROTOCOL, spec.normalization_profile)
    failures, started = [], time.perf_counter()
    for batch_size in map(int, batch_sizes):
        writer = FeatureCacheWriter(output_dir, n_rows=len(rows), feature_dim=spec.feature_dim, identities=keys)
        try:
            for start in range(0, len(rows), batch_size):
                image_batch = _load_image_batch(rows[start:start + batch_size], transform)
                features = adapter.encode(image_batch)
                if start == 0:
                    second = adapter.encode(image_batch)
                    if not torch.equal(features, second):
                        raise RuntimeError("冻结 eval 编码器的重复推理不一致")
                writer.write_rows(start, features.numpy())
            runtime = _runtime(torch_device, started)
            location = writer.finalize({
                **dict(metadata_context or {}),
                "actual_batch_size": batch_size,
                "fallback_failures": failures,
                "runtime": runtime,
                "software": runtime["software"],
            })
            return {
                "status": "complete",
                "model": spec.name,
                "preprocess_profile": preprocess_profile(spec.name),
                "cache_dir": str(location),
                "n_rows": len(rows),
                "feature_dim": spec.feature_dim,
                "actual_batch_size": batch_size,
                "runtime": runtime,
            }
        except Exception as exc:
            writer.abort()
            if _oom(exc):
                failures.append({"batch_size": batch_size, "reason": "cuda_out_of_memory"})
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"{spec.name} 在所有 batch size 下均内存不足: {failures}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cache_entry_succeeded(entry: dict) -> bool:
    return entry.get("status") in {"complete", "reused_complete_cache"}


def _merge_registry_entries(registry_path: Path, current: list[dict], experiment_id: str) -> list[dict]:
    indexed: dict[str, dict] = {}
    if registry_path.is_file():
        previous = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        if previous.get("schema_version") != "batch2-spatial-v1" or previous.get("experiment_id") != experiment_id:
            raise RuntimeError(f"已有特征登记与当前实验不兼容: {registry_path}")
        indexed.update({str(item.get("model")): item for item in previous.get("entries", []) if item.get("model") in TRAIN_MODELS})
    indexed.update({str(item["model"]): item for item in current})
    return [indexed[name] for name in TRAIN_MODELS if name in indexed]


def prepare_feature_caches(
    config: dict,
    *,
    device: str = "cuda",
    models: Sequence[str] | None = None,
    adapter_factory: Callable = load_local_encoder,
) -> dict:
    selected = tuple(models or TRAIN_MODELS)
    if not selected or any(name not in TRAIN_MODELS for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("models 必须是无重复的 hoptimus0/hoptimus1/phikonv2 子集")
    manifest_path, all_rows = ensure_common_identity_manifest(config, require_images=True)
    specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=False)
    batches = [int(config["feature_extraction"]["initial_batch_size"]), *[int(v) for v in config["feature_extraction"].get("fallback_batch_sizes", [])]]
    entries = []
    rows = select_dataset_rows(all_rows, "all", protocol=GEOMETRY_PROTOCOL)
    for model in selected:
        spec = specs[model]
        target = cache_path(config, model=model, dataset="all")
        contract = cache_contract(config, spec, dataset="all", manifest_path=manifest_path)
        try:
            validate_local_snapshot(spec)
            if target.exists():
                cache = load_feature_cache(
                    target,
                    expected_identities=identities(rows),
                    expected_dim=spec.feature_dim,
                    expected_metadata=contract,
                )
                entries.append({
                    "status": "reused_complete_cache",
                    "model": model,
                    "preprocess_profile": preprocess_profile(model),
                    "dataset": "all",
                    "cache_dir": cache["cache_dir"],
                    "n_rows": len(rows),
                    "feature_dim": spec.feature_dim,
                })
            else:
                entries.append({
                    **extract_one_model(
                        spec, rows, target, device=device, batch_sizes=batches,
                        adapter_factory=adapter_factory, metadata_context=contract,
                    ),
                    "dataset": "all",
                })
        except Exception as exc:
            entries.append({
                "status": "failed",
                "model": model,
                "preprocess_profile": preprocess_profile(model),
                "dataset": "all",
                "cache_dir": str(target.resolve()),
                "feature_dim": spec.feature_dim,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    root = Path(config["paths"]["feature_caches_root"]) / config["experiment_id"]
    registry_path = root / "feature_caches.json"
    merged_entries = _merge_registry_entries(registry_path, entries, config["experiment_id"])
    selected_success = sum(_cache_entry_succeeded(item) for item in entries)
    selected_status = "completed" if selected_success == len(entries) else ("partial" if selected_success else "failed")
    complete_models = {item["model"] for item in merged_entries if _cache_entry_succeeded(item)}
    overall_status = "complete" if complete_models == set(TRAIN_MODELS) else ("partial" if complete_models else "failed")
    registry = {
        "schema_version": "batch2-spatial-v1",
        "status": overall_status,
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "common_identity_manifest": str(manifest_path.resolve()),
        "entries": merged_entries,
        "return_policy": "server_feature_caches_excluded_from_local_result_copy",
    }
    _atomic_json(registry_path, registry)
    return {
        **registry,
        "status": selected_status,
        "overall_status": overall_status,
        "selected_models": list(selected),
        "selected_entries": entries,
        "registry_path": str(registry_path.resolve()),
    }


def main() -> int:
    from config import load_config
    parser = argparse.ArgumentParser(description="准备或核对本批特征缓存；不训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result = prepare_feature_caches(load_config(args.config), device=args.device, models=args.models)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
