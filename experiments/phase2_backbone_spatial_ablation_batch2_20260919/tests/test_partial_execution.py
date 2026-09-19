from __future__ import annotations

import json
from pathlib import Path

from dispatch import plan_train_tasks
from external_eval import evaluate_external
from stage_runtime import execute_training_batch


def test_training_records_hoptimus1_failure_and_continues_to_phikon(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(config, task, **kwargs):
        calls.append(task.model)
        if task.model == "hoptimus1":
            raise RuntimeError("cache unavailable")
        return {"task_id": task.task_id, "status": "completed", "attempt": {"cache_dir": f"cache/{task.model}", "result": {}}}

    monkeypatch.setattr("stage_runtime._run_task", fake_run)
    run_dir = tmp_path / "batch" / "train-spatial_run"
    run_dir.mkdir(parents=True)
    result = execute_training_batch(
        {"experiment_id": "exp", "selection": {"formal_start_epoch": 1}},
        plan_train_tasks(),
        run_dir=run_dir,
        weights_dir=tmp_path / "weights",
        device="cpu",
    )
    assert calls == ["hoptimus0"] * 3 + ["hoptimus1"] * 3 + ["phikonv2"] * 3
    assert result["status"] == "partial"
    assert result["completed"] == 6
    assert result["failed"] == 3
    assert result["complete_comparison"] is False
    for task in plan_train_tasks(["hoptimus1"]):
        record = tmp_path / "batch" / "tasks" / f"{task.task_id}.json"
        assert json.loads(record.read_text(encoding="utf-8-sig"))["attempts"][-1]["status"] == "failed"


def test_external_eval_records_all_missing_tasks_instead_of_stopping_early(tmp_path: Path, monkeypatch):
    calls = []

    def missing_formal(batch, task):
        calls.append(task.model)
        raise RuntimeError("formal checkpoint unavailable")

    monkeypatch.setattr("external_eval._formal_attempt", missing_formal)
    run_dir = tmp_path / "batch" / "external-eval_run"
    run_dir.mkdir(parents=True)
    result = evaluate_external({}, run_dir=run_dir, device="cpu")
    assert calls == ["hoptimus0"] * 3 + ["hoptimus1"] * 3 + ["phikonv2"] * 3
    assert result["status"] == "failed"
    assert result["failed"] == 9
    manifest = json.loads((run_dir / "external_predictions.json").read_text(encoding="utf-8-sig"))
    assert len(manifest["failures"]) == 9
    assert manifest["entries"] == []
