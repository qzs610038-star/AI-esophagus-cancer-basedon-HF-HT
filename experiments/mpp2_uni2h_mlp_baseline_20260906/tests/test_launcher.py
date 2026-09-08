"""Exercise the public launcher with synthetic local inputs. No repository imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

PACKAGE = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_synthetic(root: Path) -> dict:
    names = ["tls", "tgfb"]
    patients = ["HYZ15040"]
    splits = root / "splits"
    labels = root / "labels"
    cache = root / "cache"
    flat = root / "flat"
    for folder in (splits, labels, cache, flat):
        folder.mkdir(parents=True)
    rows = []
    for index, (x, y, split) in enumerate([(0, 0, "train"), (1, 0, "train"), (0, 1, "train"), (1, 1, "internal_val")]):
        stem = f"patch_x{x}_y{y}"
        rows.append({"mpp_id": 2, "patient": "HYZ15040", "patch_stem": stem, "x": x, "y": y, "split": split, "block_id": "b"})
        pt = cache / "MPP2_UNI" / "HYZ15040" / f"{stem}.pt"
        pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1536), pt)
        split_dir = "train" if split == "train" else "val"
        csv = labels / split_dir / "HYZ15040" / "HYZ15040_ssGSEA_zscore.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        if csv.exists():
            table = pd.read_csv(csv)
        else:
            table = pd.DataFrame(columns=["barcode", *names])
        table.loc[len(table)] = [stem, 0.1 * index, 0.2 * index]
        table.to_csv(csv, index=False)
    ext = flat / "2" / "XZY"
    ext.mkdir(parents=True)
    torch.save(torch.ones(1536), ext / "patch_x3_y3.pt")
    ext_csv = labels / "external" / "XZY" / "XZY_ssGSEA_zscore_by_group_2_train.csv"
    ext_csv.parent.mkdir(parents=True)
    pd.DataFrame([{"barcode": "patch_x3_y3", "tls": 0.3, "tgfb": 0.4}]).to_csv(ext_csv, index=False)
    pd.DataFrame(rows).to_csv(splits / "split_manifest.csv", index=False)
    _write_json(splits / "split_info.json", {"leakage_pairs": 0})
    _write_json(splits / "zscore_manifest.json", {"fit_patients": patients, "pathway_names": names})
    _write_json(splits / "zscore_params_from_train.json", {"pathways": {name: {"mean": 0, "std": 1} for name in names}})
    return {
        "splits_dir": str(splits),
        "labels_root": str(labels),
        "cache_root": str(cache),
        "flat_cache_root": str(flat),
        "accepted_checkpoint": str(root / "missing.pth"),
        "data_manifest_id": "synthetic-local",
    }


class LauncherTest(unittest.TestCase):
    def test_isolated_smoke_and_code_replacement(self):
        scratch = Path(os.environ.get("PFMVAL_TEST_ROOT", str(PACKAGE / "tests" / "work")))
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="独立包 验收 ", dir=scratch))
        package = root / "code" / "mpp2_pack"
        shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        synth = _make_synthetic(root / "synth")
        config = json.loads((package / "config.json").read_text(encoding="utf-8"))
        config["python_interpreter"] = sys.executable
        config["runs_root"] = str(root / "runs")
        config["inputs"] = synth
        config["parameters"]["train_patients"] = ["HYZ15040"]
        config["parameters"]["hidden_dim"] = 8
        config["parameters"]["batch_size"] = 2
        config_file = package / "本地 配置.json"
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        (package / "run_mode.json").write_text(json.dumps({"run_kind": "smoke"}), encoding="utf-8")

        def launch(python_override=None):
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(package / "run.ps1"), "-Config", str(config_file),
            ]
            if python_override:
                command += ["-PythonInterpreter", python_override]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            return subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=180)

        success = launch()
        self.assertEqual(success.returncode, 0, success.stderr.decode("utf-8", "replace")[-2000:])
        first = next((root / "runs" / "mpp2_uni2h_mlp_baseline_20260906").iterdir())
        record = json.loads((first / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "succeeded")
        metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["run_kind"], "smoke")
        self.assertEqual(metrics["selection"]["metric"], "val_loss")
        pred = pd.read_csv(first / "raw" / "predictions_external.csv")
        self.assertIn("patient_id", pred.columns)
        self.assertIn("pred_tls", pred.columns)
        original = {str(path.relative_to(first)): path.read_bytes() for path in first.rglob("*") if path.is_file()}

        package.rename(root / "previous_code")
        shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        meta = json.loads((package / "package.json").read_text(encoding="utf-8"))
        meta.update(code_version="v002", args=["--fail"])
        (package / "package.json").write_text(json.dumps(meta), encoding="utf-8")
        (package / "run_mode.json").write_text(json.dumps({"run_kind": "smoke"}), encoding="utf-8")
        failure = launch()
        self.assertNotEqual(failure.returncode, 0)
        directories = list((root / "runs" / "mpp2_uni2h_mlp_baseline_20260906").iterdir())
        self.assertEqual(len(directories), 2)
        second = next(item for item in directories if item != first)
        self.assertIn("RuntimeError", (second / "logs" / "errors.log").read_text(encoding="utf-8"))
        missing = launch(str(root / "missing_python.exe"))
        self.assertNotEqual(missing.returncode, 0)
        self.assertEqual(original, {str(path.relative_to(first)): path.read_bytes() for path in first.rglob("*") if path.is_file()})
        print(f"Verification artifacts: {root}")


if __name__ == "__main__":
    unittest.main()
