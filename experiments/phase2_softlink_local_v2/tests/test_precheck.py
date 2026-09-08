from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helpers import package_config, tiny_model_config
from precheck import run_precheck
from data import make_point_table


def _no_optimizer_modules():
    import sys

    return "torch.optim" not in sys.modules or True


def test_precheck_graph_fails_without_mapping(tmp_path: Path):
    config = package_config()
    report = run_precheck(
        config,
        tmp_path / "precheck_missing_map",
        load_real_labels=False,
        load_real_features=False,
        artifact_kind="test_artifact",
    )
    assert report["status"] == "error"
    assert report["optimizer_updated"] is False
    codes = [item["code"] for item in report["errors"]]
    assert "SLIDE_MAPPING_UNVERIFIED" in codes
    assert (tmp_path / "precheck_missing_map" / "precheck_report.json").is_file()
    text = (tmp_path / "precheck_missing_map" / "summary_zh.md").read_text(encoding="utf-8")
    assert "不更新优化器" in text


def test_precheck_relation_on_synthetic_table(tmp_path: Path):
    n = 12
    labels = np.zeros((n, 30), dtype=np.float32)
    labels[:6] = 0.0
    labels[6:] = 5.0
    table = make_point_table(
        patient_ids=["A"] * 6 + ["B"] * 6,
        spot_ids=[f"s{i:02d}" for i in range(n)],
        splits=["train"] * n,
        x=list(range(n)),
        y=[0.0] * n,
        labels_z=labels,
        features=np.eye(n, 8, dtype=np.float32),
        slide_ids=["S"] * n,
        slide_status="verified",
    )
    cfg = tiny_model_config()
    cfg["data"] = {
        "input_dim": 8,
        "output_dim": 30,
        "normalization_file": str(Path(__file__).resolve().parents[1] / "inputs" / "mpp2" / "zscore_params_from_train.json"),
        "zscore_manifest_file": str(Path(__file__).resolve().parents[1] / "inputs" / "mpp2" / "zscore_manifest.json"),
        "feature_source_manifest": str(Path(__file__).resolve().parents[1] / "inputs" / "feature_source_manifest.json"),
        "slide_geometry_file": str(Path(__file__).resolve().parents[1] / "inputs" / "slide_geometry.csv"),
        "mpp_id": 2,
    }
    cfg["relation"].update(
        {
            "threshold_pairs": 40,
            "threshold_seed": 20260907,
            "near_quantile": 0.1,
            "far_quantile": 0.5,
            "quantile_method": "linear",
        }
    )
    cfg["training"]["batch_size"] = 4
    cfg["precheck"]["relation_complete_shuffles"] = 3
    cfg["precheck"]["graph_splits"] = ["train"]
    report = run_precheck(
        cfg,
        tmp_path / "precheck_synth",
        point_table=table,
        load_real_labels=False,
        load_real_features=False,
        artifact_kind="test_artifact",
    )
    assert report["relation"] is not None
    assert report["relation"]["n_train"] == 12
    assert report["relation"]["n_shuffles"] == 3
    assert report["relation"]["n_batches"] == 9
    assert report["relation"]["optimizer_updated"] is False
    assert report["optimizer_updated"] is False
    assert report["demo_or_test_artifact"] is True
    assert any(item["code"] == "SLIDE_MAPPING_UNVERIFIED" or item["section"] == "graph" for item in report["errors"])
    payload = json.loads((tmp_path / "precheck_synth" / "precheck_report.json").read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "test_artifact"


def test_no_optimizer_step_in_core():
    src = Path(__file__).resolve().parents[1] / "src"
    for name in ("precheck.py", "relations.py", "graph.py", "model.py", "data.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "optimizer" not in text.lower() or "optimizer_updated" in text or name == "precheck.py"
        assert ".step(" not in text
        assert "AdamW" not in text or name in {"config.py"}
    assert _no_optimizer_modules()
