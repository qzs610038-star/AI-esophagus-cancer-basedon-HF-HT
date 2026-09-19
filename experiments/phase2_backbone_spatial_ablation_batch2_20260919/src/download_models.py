"""Pinned Hugging Face snapshot download with independent per-model failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from config import load_config
from errors import AccessDeniedError, OfflineModelError
from model_adapters import TRAIN_MODELS, load_local_encoder, load_model_specs
from run_io import utc_now, write_json
from transforms import GEOMETRY_PROTOCOL, build_transform


def _is_access_denied(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"GatedRepoError", "HfHubHTTPError", "LocalEntryNotFoundError"}:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or name == "GatedRepoError":
            return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    text = str(exc).lower()
    return any(token in text for token in (" 401", "401 ", "403", "gated", "restricted", "access denied", "unauthorized"))


def record_snapshot(manifest_path: str | Path, model_name: str, snapshot_path: str | Path) -> None:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if model_name not in TRAIN_MODELS:
        raise OfflineModelError("下载阶段只允许登记 hoptimus0 / hoptimus1 / phikonv2")
    model = payload.get("models", {}).get(model_name) or {}
    revision = model.get("revision")
    snapshot = Path(snapshot_path).resolve()
    try:
        commit_directory = snapshot.parts[snapshot.parts.index("snapshots") + 1]
    except (ValueError, IndexError) as exc:
        raise OfflineModelError("snapshot 路径必须包含 snapshots/<revision>") from exc
    if commit_directory != revision or snapshot.name != revision:
        raise OfflineModelError(f"下载结果 revision 不匹配: expected={revision}, actual={snapshot.name}")
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


def _record_verification_status(manifest_path: str | Path, model_name: str, *, status: str, error: str | None = None, extra: dict | None = None) -> None:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    model = payload["models"][model_name]
    model["strict_load_status"] = status
    if status == "passed":
        model["download_status"] = "complete"
    model["verified_at"] = utc_now()
    if extra:
        model.update(extra)
    if error is None:
        model.pop("verification_error", None)
    else:
        model["verification_error"] = error
    write_json(path, payload)


def _verify_registered(config: dict, model_name: str, verify_device: str) -> dict:
    from importlib import metadata as importlib_metadata

    def _ver(name: str) -> str | None:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest_path = Path(config["inputs"]["model_manifest"])
    spec = load_model_specs(manifest_path, require_local_files=True)[model_name]
    adapter = load_local_encoder(spec, verify_device)
    image = Image.new("RGB", (224, 224), (12, 34, 56))
    tensor = build_transform(GEOMETRY_PROTOCOL, spec.normalization_profile)(image).unsqueeze(0)
    features = adapter.encode(tensor)
    expected = (1, spec.feature_dim)
    if tuple(features.shape) != expected:
        raise OfflineModelError(f"严格加载后形状不符合登记: expected={expected}, actual={tuple(features.shape)}")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise OfflineModelError("严格加载后的特征必须是有限 float32")
    result = {
        "model": model_name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "snapshot_path": spec.snapshot_path,
        "checkpoint_filename": spec.checkpoint_filename,
        "loader_type": spec.loader_type,
        "output_mode": spec.output_mode,
        "strict_load": "passed",
        "sample_feature_shape": list(features.shape),
        "sample_dtype": str(features.dtype).replace("torch.", ""),
        "software": {
            "python": __import__("platform").python_version(),
            "pytorch": torch.__version__,
            "torchvision": _ver("torchvision"),
            "timm": _ver("timm"),
            "transformers": _ver("transformers"),
            "huggingface_hub": _ver("huggingface_hub"),
        },
        "feature_extraction_started": False,
        "training_started": False,
        "verified_at": utc_now(),
    }
    del features, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _verify_and_record(config: dict, model_name: str, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    try:
        result = _verify_registered(config, model_name, verify_device)
    except Exception as exc:
        _record_verification_status(manifest_path, model_name, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _record_verification_status(
        manifest_path, model_name, status="passed",
        extra={"sample_feature_shape": result["sample_feature_shape"], "software": result["software"]},
    )
    return result


def register_cached_model(config: dict, model_name: str, *, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    spec = load_model_specs(manifest_path, require_local_files=False)[model_name]
    repo_directory = "models--" + spec.repo_id.replace("/", "--")
    snapshot = Path(config["paths"]["hf_home"]) / "hub" / repo_directory / "snapshots" / str(spec.revision)
    record_snapshot(manifest_path, model_name, snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return _verify_and_record(config, model_name, verify_device)


def download_model(config: dict, model_name: str, *, verify_device: str) -> dict:
    manifest_path = Path(config["inputs"]["model_manifest"])
    spec = load_model_specs(manifest_path, require_local_files=False)[model_name]
    os.environ["HF_HOME"] = str(Path(config["paths"]["hf_home"]).resolve())
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise OfflineModelError("缺少 huggingface_hub；请按操作卡定向安装，启动器不会联网安装") from exc
    allow = list(spec.allow_patterns) or [spec.config_filename, spec.checkpoint_filename, "README.md"]
    try:
        snapshot = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=allow,
            cache_dir=str(Path(config["paths"]["hf_home"]) / "hub"),
            token=True,
        )
    except Exception as exc:
        if _is_access_denied(exc):
            raise AccessDeniedError(
                f"{model_name} 访问被拒绝或仓库仍 gated（401/403）。这不是可重试的网络抖动；"
                "请在浏览器完成该仓库条款/审批后单独重跑该模型。不得把 token 写入命令或日志。"
            ) from exc
        raise
    record_snapshot(manifest_path, model_name, snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return _verify_and_record(config, model_name, verify_device)


def run_models(config: dict, model_names: list[str], *, verify_device: str, register_only: bool) -> dict:
    results, failures = [], []
    for model_name in model_names:
        try:
            if register_only:
                results.append(register_cached_model(config, model_name, verify_device=verify_device))
            else:
                results.append(download_model(config, model_name, verify_device=verify_device))
        except Exception as exc:
            failures.append({
                "model": model_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "access_denied": isinstance(exc, AccessDeniedError) or _is_access_denied(exc),
            })
    return {
        "status": "complete" if not failures else ("partial" if results else "failed"),
        "download_only": not register_only,
        "register_only": register_only,
        "feature_extraction_started": False,
        "training_started": False,
        "hashes_used": False,
        "entries": results,
        "failures": failures,
        "note": "一个模型失败不会取消已成功模型的登记；请按模型单独重跑。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="下载固定 revision 的三个新编码器；不提取特征、不训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=(*TRAIN_MODELS, "all"))
    parser.add_argument("--verify-device", default="cuda")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    requested = []
    for value in args.model or ["all"]:
        if value == "all":
            requested.extend(TRAIN_MODELS)
        else:
            requested.append(value)
    # Preserve order, drop duplicates, keep independent attempts.
    models = list(dict.fromkeys(requested))
    report = run_models(load_config(args.config), models, verify_device=args.verify_device, register_only=args.register_only)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
