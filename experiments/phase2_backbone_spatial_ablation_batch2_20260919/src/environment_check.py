"""Read-only environment report. Never installs, downloads, or trains."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

from config import TRAIN_MODELS, load_config
from model_adapters import load_model_specs
from run_io import utc_now, write_json


def _import_status(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {"status": "ok", "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"status": "missing", "error": str(exc)}


def build_environment_report(config: dict, *, require_server_assets: bool = False) -> dict:
    checks: list[dict] = []
    configured_python = Path(config["python_interpreter"])
    checks.append({
        "name": "configured_python_recorded",
        "status": "ok" if str(configured_python) else "error",
        "configured": str(configured_python),
        "actual": str(Path(sys.executable)),
        "match": str(Path(sys.executable)).casefold() == str(configured_python).casefold(),
    })
    for module in ("numpy", "PIL", "torch", "torchvision", "timm", "safetensors", "huggingface_hub", "transformers", "pandas"):
        item = {"name": f"import:{module}", **_import_status(module)}
        if module == "transformers" and item["status"] != "ok":
            item["status"] = "unverified"
            item["note"] = "Phikon-v2 需要 transformers；请在服务器显式安装后用下载预检固定兼容版本，不要改 torch"
        checks.append(item)
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        checks.append({
            "name": "cuda",
            "status": "ok" if cuda_ok else "unverified",
            "available": cuda_ok,
            "device": torch.cuda.get_device_name(0) if cuda_ok else None,
        })
    except Exception as exc:
        checks.append({"name": "cuda", "status": "unverified", "error": str(exc)})
    for key in ("runs_root", "weights_root", "feature_caches_root", "hf_home", "image_root", "labels_root"):
        path = Path(config["paths"][key])
        exists = path.exists()
        free = None
        try:
            free = shutil.disk_usage(path if exists else path.anchor).free
        except OSError:
            pass
        checks.append({
            "name": f"path:{key}",
            "status": "ok" if exists else ("error" if require_server_assets else "unverified"),
            "path": str(path),
            "exists": exists,
            "free_bytes": free,
        })
    for name, path in config["inputs"].items():
        checks.append({"name": f"input:{name}", "status": "ok" if Path(path).is_file() else "error", "path": str(path)})
    try:
        specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=False)
        for name in TRAIN_MODELS:
            snapshot = Path(specs[name].snapshot_path) if specs[name].snapshot_path else None
            files_present = bool(snapshot and (snapshot / specs[name].checkpoint_filename).is_file())
            checks.append({
                "name": f"model_snapshot:{name}",
                "status": "ok" if files_present else "unverified",
                "revision": specs[name].revision,
                "snapshot_path": specs[name].snapshot_path,
                "files_present": files_present,
                "strict_model_load_status": "not_run_by_environment_check",
            })
    except Exception as exc:
        checks.append({"name": "model_manifest", "status": "error", "error": str(exc)})
    statuses = {item["status"] for item in checks}
    if "error" in statuses:
        overall = "error"
    elif "unverified" in statuses or "missing" in statuses:
        overall = "warn"
    else:
        overall = "ok"
    return {
        "status": overall,
        "checked_at": utc_now(),
        "read_only": True,
        "downloads_started": False,
        "runs_created": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checks": checks,
        "note": "本检查不安装依赖、不下载模型、不训练。transformers 的兼容版本以服务器严格加载预检为准。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查解释器、依赖和目录；不安装")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-server-assets", action="store_true")
    args = parser.parse_args()
    report = build_environment_report(load_config(args.config), require_server_assets=args.require_server_assets)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"ok", "warn"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
