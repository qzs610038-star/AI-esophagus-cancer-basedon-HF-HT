"""Offline, sampled loading checks. Never train, download, or modify inputs."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def first_file(root, suffixes):
    root = Path(root)
    if root.is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    visited = 0
    for directory, children, files in os.walk(root, followlinks=False):
        children.sort()
        if len(Path(directory).relative_to(root).parts) >= 4:
            children[:] = []
        visited += 1
        if visited > 300:
            raise RuntimeError(f"Sample search limit (300 directories) reached: {root}")
        # Walk only the configured data/model subdirectory, never the shared cache root.
        for name in sorted(files):
            if Path(name).suffix.lower() in suffixes:
                return Path(directory) / name
    raise FileNotFoundError(f"No sample {suffixes} within depth 4: {root}")


def tensor_info(value):
    import torch
    tensors = []
    def collect(obj, prefix=""):
        if len(tensors) >= 5:
            return
        if isinstance(obj, torch.Tensor):
            sample = obj.detach().reshape(-1)[:128]
            finite = bool(torch.isfinite(sample).all())
            if not finite:
                raise ValueError(f"Non-finite values in sampled tensor: {prefix}")
            tensors.append({"key": prefix, "shape": list(obj.shape), "dtype": str(obj.dtype), "sample_finite": finite})
        elif isinstance(obj, dict):
            for key, item in obj.items():
                collect(item, str(key))
                if len(tensors) >= 5:
                    break
    collect(value)
    if not tensors:
        raise ValueError("Loaded object contains no directly discoverable tensor")
    return tensors


def load_sample(path, kind, expected_dim=None):
    path = Path(path)
    result = {"sample_path": str(path), "size_bytes": path.stat().st_size}
    if kind in ("tensor", "weights"):
        import torch
        # Never fall back to arbitrary pickle execution for a failed checkpoint load.
        value = torch.load(path, map_location="cpu", weights_only=True)
        result["tensors"] = tensor_info(value)
        result["mapping_keys"] = len(value) if isinstance(value, dict) else None
        if expected_dim is not None:
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[-1] != expected_dim:
                raise ValueError(f"Expected direct tensor with last dimension {expected_dim}; got {result['tensors']}")
        del value
    elif kind == "image":
        from PIL import Image
        import numpy as np
        with Image.open(path) as image:
            image.load()
            result.update(image_size=list(image.size), mode=image.mode)
            pixels = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
            result["resized_shape"] = list(pixels.shape)
    elif kind == "csv":
        import pandas as pd
        table = pd.read_csv(path, nrows=5)
        if table.empty:
            raise ValueError("CSV contains no data rows")
        result.update(sample_rows=len(table), column_count=len(table.columns), columns=list(table.columns)[:40])
    elif kind == "json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        result["type"] = type(obj).__name__
    elif kind == "h5":
        import h5py
        with h5py.File(path, "r") as handle:
            arrays = []
            def inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if obj.size:
                        _ = obj[tuple(slice(0, 1) for _ in obj.shape)] if obj.shape else obj[()]
                    arrays.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
                    if len(arrays) == 3:
                        return True
            handle.visititems(inspect)
            if not arrays:
                raise ValueError("HDF5 file contains no datasets")
            result["datasets"] = arrays
    return result


def probe(check):
    kind = check["kind"]
    if kind == "runtime":
        if platform.system() != "Windows":
            raise RuntimeError("This package targets the PFMval Windows server")
        return {"platform": platform.platform(), "python": sys.version, "executable": sys.executable}
    if kind == "module":
        module = importlib.import_module(check["module"])
        return {"version": getattr(module, "__version__", "unknown"), "module_file": getattr(module, "__file__", None)}
    if kind == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable; torch={torch.__version__}, compiled_cuda={torch.version.cuda}")
        with torch.inference_mode():
            x = torch.ones((16, 16), device="cuda")
            result = x @ x
            torch.cuda.synchronize()
            if float(result[0, 0]) != 16.0:
                raise ValueError("CUDA matrix multiplication returned an unexpected value")
        return {"torch": torch.__version__, "compiled_cuda": torch.version.cuda,
                "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                             "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())], "tested_device": 0}
    if kind == "directory":
        path = Path(check["path"])
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        return {"path": str(path), "exists": True, "load_level": "directory_only"}
    if kind == "cache_candidates":
        # Match the two layouts documented in dataset_mpp_manifest.py.
        errors = []
        for candidate in check["candidates"]:
            try:
                path = first_file(candidate, {".pt"})
            except FileNotFoundError as exc:
                errors.append(str(exc))
                continue
            return load_sample(path, "tensor", 1536)
        raise FileNotFoundError("No recorded cache layout yielded a sample: " + " | ".join(errors))
    suffixes = {"image": {".png", ".jpg", ".jpeg"}, "csv": {".csv"}, "tensor": {".pt", ".pth"},
                "weights": {".bin", ".pth", ".pt"}, "json": {".json"}, "h5": {".h5", ".hdf5", ".h5ad"}}
    path = first_file(check["path"], suffixes[kind])
    return load_sample(path, kind, check.get("expected_dim"))


def write_report(run_dir, records, complete):
    counts = {status: sum(r["status"] == status for r in records) for status in ("PASS", "WARN", "FAIL")}
    report = {"diagnostic_only": True, "complete": complete, "counts": counts, "checks": records,
              "limitations": ["Sample loading only; not full-data coverage or image-label alignment validation",
                              "Weights deserialized on CPU; no complete backbone construction or inference",
                              "No training, downloads, input writes, hashes, or scientific result registration"]}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PFMval 服务器抽样加载检查", "", f"完成：{complete}；PASS={counts['PASS']}，WARN={counts['WARN']}，FAIL={counts['FAIL']}",
             "", "PASS 仅表示该项抽样成功，不等于完整训练就绪。权重仅验证 CPU 反序列化，未做完整模型推理。", "",
             "| 检查 | 状态 | 来源 | 说明 |", "|---|---|---|---|"]
    for item in records:
        detail = item.get("error", json.dumps(item.get("detail", {}), ensure_ascii=False))
        lines.append(f"| {item['label']} | {item['status']} | {item['source']} | {detail.replace('|', '/').replace(chr(10), ' ')[:600]} |")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Any library-created scratch/cache goes to this run, not into original input caches.
    os.environ["HF_HOME"] = str(args.run_dir / "runtime_cache" / "huggingface")
    os.environ["TORCH_HOME"] = str(args.run_dir / "runtime_cache" / "torch")
    sys.dont_write_bytecode = True
    records = []
    write_report(args.run_dir, records, False)
    for check in config["checks"]:
        started = time.monotonic()
        print(f"Checking: {check['label']}", flush=True)
        item = {key: check[key] for key in ("id", "label", "source", "required")}
        try:
            item.update(status="PASS", detail=probe(check))
        except Exception as exc:
            item.update(status="FAIL" if check["required"] else "WARN", error=f"{type(exc).__name__}: {exc}")
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        item["seconds"] = round(time.monotonic() - started, 3)
        records.append(item)
        write_report(args.run_dir, records, False)
        print(f"[{item['status']}] {check['label']}", flush=True)
        gc.collect()
    counts = write_report(args.run_dir, records, True)
    print(f"Report: {args.run_dir / 'REPORT.md'}; {counts}", flush=True)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
