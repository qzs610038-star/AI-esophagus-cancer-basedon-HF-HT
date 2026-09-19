"""Explicit online download stage for pinned gated Hugging Face snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from adapters import load_local_encoder, load_model_specs
from config import load_config
from errors import OfflineModelError
from run_io import utc_now, write_json


def record_snapshot(manifest_path: str | Path, model_name: str, snapshot_path: str | Path) -> None:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if model_name not in {"uni", "virchow2"}:
        raise OfflineModelError("下载阶段只允许登记 uni 或 virchow2")
    model = payload.get("models", {}).get(model_name) or {}
    revision = model.get("revision")
    snapshot = Path(snapshot_path).resolve()
    try:
        commit_directory = snapshot.parts[snapshot.parts.index("snapshots") + 1]
    except (ValueError, IndexError) as exc:
        raise OfflineModelError("snapshot 路径必须包含 snapshots/<revision>") from exc
    if commit_directory != revision or snapshot.name != revision:
        raise OfflineModelError(
            f"下载结果 revision 不匹配: expected={revision}, actual={snapshot.name}"
        )
    for field in ("config_filename", "checkpoint_filename"):
        candidate = snapshot / str(model.get(field))
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise OfflineModelError(f"下载结果缺少或未完成: {candidate}")
    model["snapshot_path"] = str(snapshot)
    model["download_status"] = "files_present_unverified"
    model["strict_load_status"] = "pending"
    model["snapshot_registered_at"] = utc_now()
    model.pop("verification_error", None)
    write_json(path, payload)


def _record_verification_status(
    manifest_path: str | Path,
    model_name: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    model = payload["models"][model_name]
    model["strict_load_status"] = status
    if status == "passed":
        model["download_status"] = "complete"
    model["verified_at"] = utc_now()
    if error is None:
        model.pop("verification_error", None)
    else:
        model["verification_error"] = error
    write_json(path, payload)


def _verify_and_record(config: dict, model_name: str, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    try:
        result = _verify_registered(config, model_name, verify_device)
    except Exception as exc:
        _record_verification_status(
            manifest_path,
            model_name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _record_verification_status(manifest_path, model_name, status="passed")
    return result


def _verify_registered(config: dict, model_name: str, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    refreshed = load_model_specs(manifest_path, require_local_files=True)[model_name]
    adapter = load_local_encoder(refreshed, verify_device)
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    cls, mean_patch = adapter.encode(sample)
    if cls.shape != (1, refreshed.layout.feature_dim) or mean_patch.shape != cls.shape:
        raise OfflineModelError("严格加载后的零样本输出形状不符合登记")
    cls_shape = list(cls.shape)
    mean_shape = list(mean_patch.shape)
    del cls, mean_patch, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "repo_id": refreshed.repo_id,
        "revision": refreshed.revision,
        "snapshot_path": refreshed.snapshot_path,
        "checkpoint_filename": refreshed.checkpoint_filename,
        "strict_load": "passed",
        "sample_cls_shape": cls_shape,
        "sample_mean_patch_shape": mean_shape,
        "verified_at": utc_now(),
    }


def register_cached_model(config: dict, model_name: str, *, verify_device: str) -> dict:
    """Register a cache tree copied offline; no Hugging Face network call is made."""

    manifest_path = Path(config["inputs"]["model_manifest"])
    spec = load_model_specs(manifest_path, require_local_files=False)[model_name]
    repo_directory = "models--" + spec.repo_id.replace("/", "--")
    snapshot = Path(config["paths"]["hf_home"]) / "hub" / repo_directory / "snapshots" / str(spec.revision)
    record_snapshot(manifest_path, model_name, snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return _verify_and_record(config, model_name, verify_device)


def download_model(config: dict, model_name: str, *, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    specs = load_model_specs(manifest_path, require_local_files=False)
    spec = specs[model_name]
    if not spec.revision:
        raise OfflineModelError(f"{model_name} 尚未固定 revision")
    os.environ["HF_HOME"] = str(Path(config["paths"]["hf_home"]).resolve())
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise OfflineModelError("缺少 huggingface_hub；请按操作卡定向安装") from exc
    allow_patterns = [
        str(spec.config_filename),
        str(spec.checkpoint_filename),
        "README.md",
    ]
    snapshot = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        allow_patterns=allow_patterns,
        cache_dir=str(Path(config["paths"]["hf_home"]) / "hub"),
        token=True,
    )
    record_snapshot(manifest_path, model_name, snapshot)

    # After the online call returns, the verification path is strictly local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    return _verify_and_record(config, model_name, verify_device)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载固定版本的 UNI / Virchow2；不会启动提取或训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=("uni", "virchow2"))
    parser.add_argument("--verify-device", default="cuda")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    results = []
    failure = None
    for model_name in args.model or ("uni", "virchow2"):
        try:
            if args.register_only:
                results.append(register_cached_model(config, model_name, verify_device=args.verify_device))
            else:
                results.append(download_model(config, model_name, verify_device=args.verify_device))
        except Exception as exc:
            failure = {
                "model": model_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            break
    report = {
        "status": "failed" if failure else "complete",
        "download_only": not args.register_only,
        "register_only": args.register_only,
        "feature_extraction_started": False,
        "training_started": False,
        "hashes_used": False,
        "entries": results,
        "failure": failure,
    }
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
