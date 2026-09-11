from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import _write_run_outputs


def _summary(root: Path, arm: str, best_pcc: float, pooled: float) -> dict:
    paths = {}
    for key in ("warmup", "warmup_bundle", "formal", "last", "model_bundle"):
        path = root / arm / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        paths[key] = str(path)
    step0 = {"update": 0, "patient_macro_pathway_pcc": .4, "flattened_pooled_pcc": .5,
             "pooled_pcc": .5, "zMSE": .8, "valid_patient_pathway_count": 2,
             "invalid_patient_pathway_count": 0}
    best = {**step0, "update": 10, "patient_macro_pathway_pcc": best_pcc,
            "flattened_pooled_pcc": pooled, "pooled_pcc": pooled, "zMSE": .6}
    return {"arm": arm, "seed": 42, "best_update": 10,
            "best_metrics": {"update": 10, "patient_macro_pathway_pcc": best_pcc,
                             "zMSE": .6, "arm": arm},
            "best_validation_metrics": best, "step0_metrics": step0, **paths}


def test_top_level_outputs_keep_both_pcc_and_predeclared_comparisons(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"x")
    config = {"experiment_id": "e", "batch_id": "b", "inputs": {"stage1_checkpoint": str(source)}}
    summaries = [
        _summary(tmp_path, "spatial_residual_only", .45, .52),
        _summary(tmp_path, "point_continue", .46, .51),
        _summary(tmp_path, "spatial_joint", .47, .53),
    ]
    _write_run_outputs(config, tmp_path, summaries)
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert selection["selected"]["arm"] == "spatial_joint"
    assert selection["trajectories"][0]["best_validation_metrics"]["flattened_pooled_pcc"] == .52
    assert set(selection["paired_comparisons"]) == {
        "spatial_residual_only_minus_step0", "point_continue_minus_step0",
        "spatial_joint_minus_point_continue",
    }
    assert metrics["paired_comparisons"]["spatial_joint_minus_point_continue"]["flattened_pooled_pcc"] == pytest.approx(.02)
