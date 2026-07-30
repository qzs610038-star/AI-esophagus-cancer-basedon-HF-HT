import contextlib
import json

from deploy import pfmval_ops


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def test_experiment_refresh_never_scans_docs_or_rewrites_project_guide(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    _write_json(root / "experiments" / "experiment_registry.json", {"experiments": []})
    _write_json(root / "project_state" / "scientific_records.jsonl", {})
    _write_json(root / "project_state" / "document_registry.json", {"documents": []})
    guide = root / "PROJECT_GUIDE.md"
    guide.write_text("original guide\n", encoding="utf-8")
    before_registry = (root / "project_state" / "document_registry.json").read_bytes()
    before_guide = guide.read_bytes()
    monkeypatch.setattr(pfmval_ops, "PROJECT_ROOT", root)
    monkeypatch.setattr(pfmval_ops, "project_decision_summaries", lambda _root: {"status": "already_projected"})
    monkeypatch.setattr(pfmval_ops, "refresh_experiment_views", lambda _root: {"dashboard_sha256": "x", "progress_sha256": "y"})
    monkeypatch.setattr(pfmval_ops, "sync_state", lambda _root: {"state_revision": 1})
    monkeypatch.setattr(pfmval_ops, "state_lock", lambda _root: contextlib.nullcontext())
    monkeypatch.setattr("sys.argv", ["pfmval_ops.py", "views", "refresh", "--experiments"])

    assert pfmval_ops.main() == 0
    assert (root / "project_state" / "document_registry.json").read_bytes() == before_registry
    assert guide.read_bytes() == before_guide


def test_docs_scan_never_mutates_experiment_or_scientific_records(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    experiment = root / "experiments" / "experiment_registry.json"
    science = root / "project_state" / "scientific_records.jsonl"
    _write_json(experiment, {"experiments": [{"id": "exp-a"}]})
    science.parent.mkdir(parents=True, exist_ok=True)
    science.write_text('{"record_id":"r"}\n', encoding="utf-8")
    monkeypatch.setattr(pfmval_ops, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        pfmval_ops,
        "scan_documents",
        lambda _root: {"summary": {"total": 0}, "documents": []},
    )
    before_experiment, before_science = experiment.read_bytes(), science.read_bytes()
    monkeypatch.setattr("sys.argv", ["pfmval_ops.py", "docs", "scan", "--write"])

    assert pfmval_ops.main() == 0
    assert experiment.read_bytes() == before_experiment
    assert science.read_bytes() == before_science
