import json

from scripts import check_project_state


def _check_registry(tmp_path, monkeypatch, experiments):
    registry_path = tmp_path / "experiments" / "experiment_registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(
        json.dumps({"version": "1.0", "updated_at": "2026-10-15", "experiments": experiments}),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_project_state, "PROJECT_ROOT", tmp_path)
    checker = check_project_state.Checker()
    check_project_state.check_experiment_registry_schema(checker)
    return checker.results


def test_registry_check_accepts_legacy_runs_and_independent_packages(tmp_path, monkeypatch):
    results = _check_registry(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "legacy-run", "status": "done", "script": "train.py",
                "run_dir": "runs/legacy-run", "selection_metric": "val_loss_min",
                "created_at": "2026-07-01", "started_at": "2026-07-02",
                "completed_at": "2026-07-03",
            },
            {
                "id": "independent-package", "status": "done",
                "experiment_package": "experiments/independent-package",
                "result_id": "independent-result", "registered_at": "2026-09-19",
            },
            {
                "id": "pending-return", "status": "returned_pending_review",
                "experiment_package": "experiments/pending-return",
                "result_id": "pending-result", "registered_at": "2026-09-20",
            },
        ],
    )

    assert not [item for item in results if item[0] in {"FAIL", "WARN"}]


def test_registry_check_still_reports_missing_identity_and_legacy_run_fields(tmp_path, monkeypatch):
    results = _check_registry(
        tmp_path,
        monkeypatch,
        [
            {"status": "planned", "experiment_package": "experiments/unnamed"},
            {"id": "broken-legacy-run", "status": "done", "run_dir": "runs/broken"},
        ],
    )

    failures = [message for level, _, message in results if level == "FAIL"]
    warnings = [message for level, _, message in results if level == "WARN"]
    assert "missing field: id" in failures
    assert "missing legacy run field: script" in warnings
    assert "missing legacy run field: completed_at" in warnings
