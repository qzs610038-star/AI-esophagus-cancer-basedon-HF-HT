from __future__ import annotations

import json
from pathlib import Path

from dispatch import plan_train_tasks
from external_eval import evaluate_external
from stage_runtime import _attempt_directory_name, _run_task, execute_training_batch


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


def test_training_attempt_uses_short_output_directory(tmp_path: Path, monkeypatch):
    import train
    from run_io import write_json

    task = plan_train_tasks()[0]
    monkeypatch.setattr("stage_runtime.model_config", lambda config, model: {})
    monkeypatch.setattr("stage_runtime._development_table", lambda config, model_name: (object(), "cache", []))

    def fake_train_arm(config, arm, seed, run_dir, *, checkpoint_dir, point_table, device):
        write_json(run_dir / "raw" / "optimizer_effective.json", {"optimizer": "Adam"})
        checkpoint_dir.mkdir(parents=True)
        return {"status": "completed"}

    monkeypatch.setattr(train, "train_arm", fake_train_arm)
    run_dir = tmp_path / "batch" / "train-spatial_20260927_164130_317_fb9bff68"
    run_dir.mkdir(parents=True)
    record_path = tmp_path / "batch" / "tasks" / f"{task.task_id}.json"
    write_json(record_path, {"task_id": task.task_id, "spec": vars(task), "attempts": [{"attempt_id": f"{task.task_id}__attempt01", "status": "failed"}]})
    result = _run_task({}, task, run_dir=run_dir, weights_dir=tmp_path / "weights", device="cpu", table_cache={})
    attempt = result["attempt"]
    assert result["status"] == "completed"
    assert attempt["attempt_id"] == f"{task.task_id}__attempt02"
    assert Path(attempt["run_dir"]).name == "hoptimus0_s45_a02"
    assert (Path(attempt["run_dir"]) / "raw" / "optimizer_effective.json").is_file()
    assert [a["status"] for a in json.loads(record_path.read_text(encoding="utf-8"))["attempts"]] == ["failed", "completed"]

    server_run = Path(r"D:\AIPatho\qzs\runs\phase2_backbone_spatial_ablation_batch2_20260919\20260927_154318_475_2242f53b\train-spatial_20260927_164130_317_fb9bff68")
    paths = [str(server_run / "raw" / _attempt_directory_name(t, 2) / "raw" / "optimizer_effective.json") for t in plan_train_tasks()]
    assert len(set(paths)) == 9
    assert max(map(len, paths)) < 240


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
