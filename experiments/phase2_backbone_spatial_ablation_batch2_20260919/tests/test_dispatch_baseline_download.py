from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze import coverage_records
from baseline_reference import load_baseline_reference, planned_baseline_task_ids
from dispatch import ACTIONS, action_exit_code, check_inputs, plan_train_tasks
from download_models import AccessDeniedError, _is_access_denied, run_models
from errors import BaselineReferenceError, ConfigError
from config import load_config


def test_nine_new_tasks_are_unique_spatial_heads():
    tasks = plan_train_tasks()
    assert len(tasks) == 9
    assert len({t.task_id for t in tasks}) == 9
    assert {t.model for t in tasks} == {"hoptimus0", "hoptimus1", "phikonv2"}
    assert {t.seed for t in tasks} == {45, 46, 47}
    assert all(t.arm == "spatial" and t.recipe == "frozen_spatial" for t in tasks)
    assert ACTIONS[1] == "check-inputs"
    assert "search-point" not in ACTIONS and "freeze" not in ACTIONS


def test_model_subset_plans_complete_seed_triples():
    tasks = plan_train_tasks(["hoptimus0", "phikonv2"])
    assert len(tasks) == 6
    assert [task.model for task in tasks] == ["hoptimus0"] * 3 + ["phikonv2"] * 3
    assert {task.seed for task in tasks if task.model == "phikonv2"} == {45, 46, 47}
    with pytest.raises(ConfigError, match="非空子集"):
        plan_train_tasks([])


def test_check_inputs_reads_packaged_baseline_without_training():
    result = check_inputs(load_config())
    assert result["status"] in {"pass", "warn"}
    assert result["baseline"]["n_tasks"] == 9
    assert result["baseline"]["do_not_retrain"] is True
    assert len(result["planned_train_tasks"]) == 9
    assert result["external_labels_used_for_selection"] is False


def test_baseline_rejects_non_spatial_or_missing_seed(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "inputs" / "baseline_reference_manifest.json"
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    payload["tasks"][0]["arm"] = "point"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BaselineReferenceError, match="spatial"):
        load_baseline_reference(bad, require_prediction_files=False)
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    payload["tasks"] = payload["tasks"][:-1]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BaselineReferenceError, match="不完整"):
        load_baseline_reference(missing, require_prediction_files=False)
    assert len(planned_baseline_task_ids()) == 9


def test_baseline_embedded_metrics_are_portable_without_original_paths(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "inputs" / "baseline_reference_manifest.json"
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    for task in payload["tasks"]:
        for field in ("task_record", "internal_metrics", "internal_predictions", "external_predictions", "external_metrics", "formal_checkpoint_metadata"):
            task[field] = str(tmp_path / "missing" / task["task_id"] / field)
    portable = tmp_path / "portable.json"
    portable.write_text(json.dumps(payload), encoding="utf-8")
    result = load_baseline_reference(portable)
    assert result["embedded_metrics_valid"] is True
    assert result["source_files_required"] is False
    assert result["tasks"][0]["files"]["task_record"]["exists"] is False
    with pytest.raises(BaselineReferenceError, match="基线路径缺失"):
        load_baseline_reference(portable, require_source_files=True)


def test_baseline_rejects_corrupt_embedded_metrics(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "inputs" / "baseline_reference_manifest.json"
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    payload["tasks"][0]["accepted_external_metrics"]["pooled_pcc"] = None
    bad = tmp_path / "bad_metrics.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BaselineReferenceError, match="pooled_pcc"):
        load_baseline_reference(bad)


def test_coverage_exposes_failed_and_missing_tasks():
    expected = [("hoptimus0", 45, "internal_val"), ("hoptimus0", 46, "internal_val")]
    actual = [{"model": "hoptimus0", "seed": 45, "split": "internal_val", "status": "succeeded"}]
    result = coverage_records(expected, actual)
    assert result["complete"] is False
    assert result["missing"][0]["seed"] == 46
    failed = coverage_records(
        [("hoptimus1", 47, "external_test")],
        [{"model": "hoptimus1", "seed": 47, "split": "external_test", "status": "failed"}],
    )
    assert failed["failed"]


class _Denied(Exception):
    def __init__(self) -> None:
        self.response = type("R", (), {"status_code": 403})()
        super().__init__("403 gated")


def test_access_denied_is_not_treated_as_retryable_network():
    assert _is_access_denied(_Denied()) is True
    assert _is_access_denied(RuntimeError("connection reset")) is False


def test_download_continues_after_one_model_fails(tmp_path: Path, monkeypatch):
    from download_models import download_model

    calls = []

    def fake_download(config, model_name, *, verify_device):
        calls.append(model_name)
        if model_name == "hoptimus1":
            raise AccessDeniedError("hoptimus1 gated")
        return {"model": model_name, "strict_load": "passed"}

    monkeypatch.setattr("download_models.download_model", fake_download)
    report = run_models({"unused": True}, ["hoptimus0", "hoptimus1", "phikonv2"], verify_device="cpu", register_only=False)
    assert calls == ["hoptimus0", "hoptimus1", "phikonv2"]
    assert report["status"] == "partial"
    assert [item["model"] for item in report["entries"]] == ["hoptimus0", "phikonv2"]
    assert report["failures"][0]["model"] == "hoptimus1"
    assert report["feature_extraction_started"] is False
    assert report["training_started"] is False


def test_partial_exit_semantics_are_action_specific():
    assert action_exit_code("train-spatial", "partial") == 2
    assert action_exit_code("prepare-features", "partial") == 2
    assert action_exit_code("external-eval", "partial") == 2
    assert action_exit_code("analyze-local", "partial") == 0
