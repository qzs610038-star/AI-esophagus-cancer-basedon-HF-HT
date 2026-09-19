"""Read-only server readiness report; it never installs, downloads, or creates a run."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch

from adapters import load_model_specs
from config import load_config
from run_io import utc_now, write_json


def _import_status(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {"status": "ok", "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def build_environment_report(config: dict) -> dict:
    checks: list[dict] = []
    configured_python = Path(config["python_interpreter"])
    current_python = Path(sys.executable)
    checks.append(
        {
            "name": "absolute_python",
            "status": "ok" if str(current_python).casefold() == str(configured_python).casefold() else "error",
            "configured": str(configured_python),
            "actual": str(current_python),
        }
    )
    package_root = Path(config["_package_root"]).resolve()
    expected_code_directory = (
        Path(config["paths"]["code_root"]) / config["experiment_id"]
    ).resolve()
    checks.append(
        {
            "name": "code_directory",
            "status": "ok" if str(package_root).casefold() == str(expected_code_directory).casefold() else "error",
            "expected": str(expected_code_directory),
            "actual": str(package_root),
        }
    )
    for module in ("numpy", "PIL", "torch", "torchvision", "timm", "safetensors", "huggingface_hub"):
        checks.append({"name": f"import:{module}", **_import_status(module)})
    cuda_ok = bool(torch.cuda.is_available())
    checks.append(
        {
            "name": "cuda",
            "status": "ok" if cuda_ok else "error",
            "available": cuda_ok,
            "device": torch.cuda.get_device_name(0) if cuda_ok else None,
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda_ok else None,
        }
    )
    for key in ("code_root", "runs_root", "weights_root", "feature_caches_root", "hf_home", "image_root", "labels_root", "historical_reference_root"):
        path = Path(config["paths"][key])
        exists = path.exists()
        free = None
        try:
            free = shutil.disk_usage(path if exists else path.anchor).free
        except OSError:
            pass
        checks.append({"name": f"path:{key}", "status": "ok" if exists else "error", "path": str(path), "free_bytes": free})
    for name, path in config["inputs"].items():
        if name in {"model_manifest", "baseline_reference_manifest", "split_manifest", "split_info", "zscore_manifest", "normalization"}:
            checks.append({"name": f"input:{name}", "status": "ok" if Path(path).is_file() else "error", "path": str(path)})
    feature_registry = (
        Path(config["paths"]["feature_caches_root"])
        / config["experiment_id"]
        / "feature_caches.json"
    )
    feature_registry_status = "error"
    feature_registry_error = None
    if feature_registry.is_file():
        try:
            feature_payload = json.loads(feature_registry.read_text(encoding="utf-8-sig"))
            candidate_models = {
                str(item.get("model"))
                for item in feature_payload.get("entries", [])
                if item.get("status") in {"complete", "reused_complete_cache"}
            }
            if feature_payload.get("status") != "complete" or not {"uni", "virchow2"}.issubset(candidate_models):
                raise ValueError("登记未同时包含完整的 UNI 与 Virchow2")
            feature_registry_status = "ok"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            feature_registry_error = str(exc)
    checks.append(
        {
            "name": "feature_registry",
            "status": feature_registry_status,
            "path": str(feature_registry),
            "error": feature_registry_error,
        }
    )
    try:
        manifest_payload = json.loads(
            Path(config["inputs"]["model_manifest"]).read_text(encoding="utf-8-sig")
        )
        specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=False)
        for name in ("uni", "virchow2"):
            snapshot = Path(specs[name].snapshot_path) if specs[name].snapshot_path else None
            registration = manifest_payload["models"][name]
            files_present = bool(snapshot and (snapshot / str(specs[name].checkpoint_filename)).is_file())
            ready = files_present and registration.get("strict_load_status") == "passed"
            checks.append(
                {
                    "name": f"model_snapshot:{name}",
                    "status": "ok" if ready else "error",
                    "revision": specs[name].revision,
                    "snapshot_path": specs[name].snapshot_path,
                    "files_present": files_present,
                    "download_status": registration.get("download_status"),
                    "strict_load_status": registration.get("strict_load_status"),
                }
            )
    except Exception as exc:
        checks.append({"name": "model_manifest", "status": "error", "error": str(exc)})
    return {
        "status": "ok" if all(item["status"] == "ok" for item in checks) else "error",
        "checked_at": utc_now(),
        "read_only": True,
        "downloads_started": False,
        "runs_created": False,
        "platform": platform.platform(),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查服务器运行环境与全部前置资产")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_environment_report(load_config(args.config))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
