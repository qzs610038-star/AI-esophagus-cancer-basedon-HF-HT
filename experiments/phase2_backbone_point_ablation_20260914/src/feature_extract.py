"""Sequential, FP32 extraction of CLS and auxiliary mean-patch caches."""

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

from adapters import ModelSpec, load_local_encoder, load_model_specs
from config import load_config
from data import IdentityRecord, build_common_identity_manifest, load_common_manifest
from errors import IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache
from transforms import build_shared_transform, describe_transform


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_image_batch(rows: Sequence[IdentityRecord], transform) -> torch.Tensor:
    tensors = []
    for row in rows:
        with Image.open(row.image_path) as image:
            image.load()
            if image.size != (int(row.width_px), int(row.height_px)):
                raise IdentityMismatchError(
                    f"图像像素尺寸在共同清单建立后发生变化: {row.identity_key}; "
                    f"expected={(row.width_px, row.height_px)}, actual={image.size}"
                )
            tensors.append(transform(image))
    return torch.stack(tensors, dim=0).contiguous().float()


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _runtime_metadata(device: torch.device, started: float) -> dict:
    peak = None
    device_name = str(device)
    if device.type == "cuda" and torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated(device))
        device_name = torch.cuda.get_device_name(device)
    return {
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_bytes": peak,
        "device": device_name,
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "timm": _package_version("timm"),
            "pillow": _package_version("Pillow"),
            "safetensors": _package_version("safetensors"),
        },
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
    """Extract one candidate model; labels are intentionally absent from this API."""

    if not rows:
        raise IdentityMismatchError("共同身份清单为空")
    if not batch_sizes or any(int(value) < 1 for value in batch_sizes):
        raise ValueError("batch_sizes 必须是正整数序列")
    identities = [row.identity_key for row in rows]
    if len(set(identities)) != len(identities):
        raise IdentityMismatchError("共同身份清单含重复身份")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但 torch.cuda.is_available() 为 False")
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(torch_device)
    adapter = adapter_factory(spec, torch_device)
    transform = build_shared_transform()
    started = time.perf_counter()
    failures: list[dict] = []

    for batch_size in [int(value) for value in batch_sizes]:
        writer = FeatureCacheWriter(
            output_dir,
            n_rows=len(rows),
            feature_dim=spec.layout.feature_dim,
            identities=identities,
        )
        try:
            start = 0
            repeat_checked = False
            while start < len(rows):
                end = min(start + batch_size, len(rows))
                images = _load_image_batch(rows[start:end], transform)
                cls_tensor, mean_tensor = adapter.encode(images)
                if not repeat_checked:
                    repeat_cls, repeat_mean = adapter.encode(images)
                    if not torch.equal(cls_tensor, repeat_cls) or not torch.equal(mean_tensor, repeat_mean):
                        raise RuntimeError("编码器 eval 重复推理结果不完全一致")
                    repeat_checked = True
                cls = cls_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                mean_patch = mean_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                writer.write_rows(start, cls, mean_patch)
                start = end
            runtime = _runtime_metadata(torch_device, started)
            cache_dir = writer.finalize(
                {
                    **dict(metadata_context or {}),
                    "model": spec.name,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "architecture": spec.architecture,
                    "snapshot_path": spec.snapshot_path,
                    "token_layout": asdict(spec.layout),
                    "preprocessing": describe_transform(),
                    "precision": {"dtype": "float32", "amp": False, "tf32": False},
                    "actual_batch_size": batch_size,
                    "fallback_failures": failures,
                    "identity_schema": "patient_id|source_group|spot_id",
                    "runtime": runtime,
                }
            )
            return {
                "status": "complete",
                "model": spec.name,
                "cache_dir": str(cache_dir),
                "actual_batch_size": batch_size,
                "n_rows": len(rows),
                "feature_dim": spec.layout.feature_dim,
                "runtime": runtime,
                "fallback_failures": failures,
            }
        except Exception as exc:
            writer.abort()
            if _is_cuda_oom(exc):
                failures.append({"batch_size": batch_size, "reason": "cuda_out_of_memory"})
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"{spec.name} 在全部降级 batch size 下均发生 CUDA OOM: {failures}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_feature_caches(
    config: dict,
    *,
    device: str,
    inspection_report: str | Path,
    models: Sequence[str] = ("uni", "virchow2"),
) -> dict:
    expected_models = [str(value) for value in config["execution"]["train_models"]]
    if list(models) != expected_models:
        raise ValueError(
            f"正式特征准备必须同时按固定顺序处理 {expected_models}；"
            "中断后重跑会复用已完成缓存"
        )
    inspection_path = Path(inspection_report).resolve()
    try:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityMismatchError(f"必须先提供可读取的 UNI2-h 核查报告: {exc}") from exc
    expected_total = sum(int(value) for value in config["data"]["expected_counts"].values())
    references = inspection.get("references") or []
    if not (
        inspection.get("status") == "ok"
        and inspection.get("read_only") is True
        and int(inspection.get("resolved_identities", -1)) == expected_total
        and int(inspection.get("missing_count", -1)) == 0
        and any(int(item.get("seed", -1)) == 42 and item.get("status") == "historical_matched_reference_reused" for item in references)
    ):
        raise IdentityMismatchError("UNI2-h 核查报告未证明共同身份完整且 seed42 历史参照可复用")
    feature_root = (
        Path(config["paths"]["feature_caches_root"])
        / config["experiment_id"]
    )
    common_dir = feature_root / "common_identity_v1"
    common_manifest = common_dir / "common_identity_manifest.csv"
    counts = config["data"]["expected_counts"]
    if common_manifest.is_file():
        rows = load_common_manifest(common_manifest, expected_counts=counts, require_images=True)
    else:
        rows = build_common_identity_manifest(
            config["inputs"]["split_manifest"],
            config["paths"]["image_root"],
            common_manifest,
            mpp_id=config["data"]["mpp_id"],
            external_patient=config["data"]["external_patient"],
            expected_counts=counts,
        )
    specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=True)
    registry = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "common_identity_manifest": str(common_manifest.resolve()),
        "common_identity_count": len(rows),
        "identity_counts": {key: int(value) for key, value in counts.items()},
        "uni2h_inspection_report": str(inspection_path),
        "uni2h_inspection_summary": {
            key: inspection.get(key)
            for key in (
                "status",
                "read_only",
                "expected_identities",
                "resolved_identities",
                "missing_count",
                "ambiguous_count",
                "shape_counts",
                "source_dtype_counts",
                "historical_compute_precision_fully_recorded",
                "mean_patch_policy",
                "references",
            )
        },
        "entries": [
            {
                "model": "uni2h",
                "status": "historical_legacy_file_cache_reused_not_reextracted",
                "revision": None,
                "cls": {
                    "paths": [
                        config["paths"]["uni2h_partner_cache_root"],
                        config["paths"]["uni2h_flat_cache_root"],
                    ],
                    "dimension": 1536,
                    "observed_source_dtype_counts": inspection.get("source_dtype_counts"),
                    "historical_compute_precision_fully_recorded": False,
                    "role": "historical_reference_primary_input",
                    "format": "registered_per_spot_pt",
                },
                "mean_patch": {
                    "status": "not_required; available_only_for_exact_[265,1536]",
                    "absence_triggers_reextraction": False,
                },
                "historical_preprocessing_fully_recorded": False,
                "inspection_report": str(inspection_path),
            }
        ],
        "status": "running",
    }
    registry_path = feature_root / "feature_caches.json"
    _atomic_json(registry_path, registry)
    identities = [row.identity_key for row in rows]
    batch_sizes = [
        int(config["feature_extraction"]["initial_batch_size"]),
        *[int(value) for value in config["feature_extraction"]["fallback_batch_sizes"]],
    ]
    for model_name in models:
        spec = specs[model_name]
        output = feature_root / model_name / config["feature_extraction"]["feature_version"]
        if output.is_dir():
            cache = load_feature_cache(
                output,
                expected_identities=identities,
                expected_dim=spec.layout.feature_dim,
            )
            metadata = cache["metadata"]
            if (
                metadata.get("revision") != spec.revision
                or metadata.get("preprocessing") != describe_transform()
                or metadata.get("experiment_id") != config["experiment_id"]
                or metadata.get("protocol_version") != config["protocol_version"]
                or Path(str(metadata.get("common_identity_manifest") or "")).resolve()
                != common_manifest.resolve()
            ):
                raise IdentityMismatchError(f"既有 {model_name} 缓存协议与当前清单不匹配")
            result = {
                "status": "reused_complete_cache",
                "model": model_name,
                "cache_dir": str(output.resolve()),
                "n_rows": len(rows),
                "feature_dim": spec.layout.feature_dim,
            }
        else:
            result = extract_one_model(
                spec,
                rows,
                output,
                device=device,
                batch_sizes=batch_sizes,
                metadata_context={
                    "experiment_id": config["experiment_id"],
                    "protocol_version": config["protocol_version"],
                    "common_identity_manifest": str(common_manifest.resolve()),
                    "identity_counts": {
                        key: int(value) for key, value in counts.items()
                    },
                },
            )
        cache_metadata = json.loads((output / "cache_meta.json").read_text(encoding="utf-8"))
        result.update(
            repo_id=spec.repo_id,
            revision=spec.revision,
            snapshot_path=spec.snapshot_path,
            preprocessing=cache_metadata.get("preprocessing"),
            runtime=cache_metadata.get("runtime"),
            token_layout=cache_metadata.get("token_layout"),
            precision=cache_metadata.get("precision"),
            cls={
                "path": str((output / "cls.npy").resolve()),
                "dimension": spec.layout.feature_dim,
                "dtype": "float32",
                "role": "primary_training_input",
            },
            mean_patch={
                "path": str((output / "mean_patch.npy").resolve()),
                "dimension": spec.layout.feature_dim,
                "dtype": "float32",
                "role": "auxiliary_not_training_input",
            },
        )
        registry["entries"].append(result)
        _atomic_json(registry_path, registry)
    registry["status"] = "complete"
    _atomic_json(registry_path, registry)
    return {**registry, "registry_path": str(registry_path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 UNI / Virchow2 共同 CLS 与 mean-patch 缓存")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inspection-report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config(args.config)
    result = prepare_feature_caches(
        config,
        device=args.device,
        inspection_report=args.inspection_report,
        models=("uni", "virchow2"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
