import json
import subprocess
from pathlib import Path

import pytest

from scripts import pfmval_state as state_module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def modern_directive(summary: str = "manual delivery") -> dict[str, object]:
    return {
        "event_type": "directive",
        "directive_id": "DIR-20260919-001",
        "issued_at": "2026-09-19T12:00:00+08:00",
        "summary": summary,
        "scope": "experiment_delivery",
        "topic": "manual_package_delivery",
        "status": "active",
        "source": "explicit_user_instruction",
    }


def test_identical_directive_replay_does_not_reactivate_completed_instruction() -> None:
    declaration = modern_directive()
    completion = {
        "event_type": "status_update",
        "directive_id": declaration["directive_id"],
        "status": "completed",
        "changed_at": "2026-09-19T13:00:00+08:00",
    }
    folded = state_module.fold_directives([declaration, completion, dict(declaration)])
    assert folded[declaration["directive_id"]]["status"] == "completed"
    assert folded[declaration["directive_id"]]["changed_at"] == completion["changed_at"]


def make_document_root(root: Path) -> Path:
    write_json(root / "project_state" / "current_state.json", {
        "state_revision": 7,
        "active_plans": {},
        "active_skills": {},
        "pending_plan_reviews": {},
    })
    write_json(root / "experiments" / "experiment_registry.json", {
        "experiments": [
            {"id": "declared", "experiment_package": "experiments/declared_package"},
            {"id": "reported", "report_path": "experiments/results/reported/run/analysis/report.md"},
        ]
    })
    (root / "experiments" / "declared_package").mkdir(parents=True)
    (root / "experiments" / "declared_package" / "README.md").write_text("# declared\n", encoding="utf-8")
    (root / "experiments" / "undeclared_package").mkdir(parents=True)
    (root / "experiments" / "undeclared_package" / "README.md").write_text("# undeclared\n", encoding="utf-8")
    report = root / "experiments" / "results" / "reported" / "run" / "analysis" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# registered report\n", encoding="utf-8")
    hidden = root / "experiments" / "results" / "hidden" / "run" / "analysis" / "report.md"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("# unregistered report\n", encoding="utf-8")
    (root / "03审计报告").mkdir(parents=True)
    (root / "03审计报告" / "new_audit.docx").write_bytes(b"not parsed")
    (root / "maintenance_logs").mkdir(parents=True)
    (root / "maintenance_logs" / "new_log.md").write_text("# log\n", encoding="utf-8")
    receipt = root / "project_state" / "governance" / "receipt.pdf"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"not parsed")
    write_json(root / "project_state" / "document_registry.json", {
        "documents": [{
            "doc_id": "manual-governance-receipt",
            "path": "project_state/governance/receipt.pdf",
            "category": "人工收据",
            "scope": "manual_governance",
            "authority": "normative",
            "lifecycle": "superseded",
            "availability": "local_only",
            "verified_at": "2026-09-01T00:00:00+00:00",
            "state_revision": 1,
            "supersedes": ["old-receipt"],
            "superseded_by": ["new-receipt"],
            "truth_sources": ["project_state/directives.jsonl"],
            "related_docs": ["new-receipt"],
            "manual_note": "retain this governance decision",
        }]
    })
    return root


def test_read_directive_events_accepts_modern_optional_fields_and_idempotent_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    event = modern_directive()
    directives = root / "project_state" / "directives.jsonl"
    directives.parent.mkdir(parents=True)
    directives.write_text("\n".join(json.dumps(event) for _ in range(2)) + "\n", encoding="utf-8")

    events = state_module.read_directive_events(root)

    assert events == [event, event]
    assert "effective_from_revision" not in events[0]
    assert "affected_files" not in events[0]
    assert "supersedes" not in events[0]
    assert directives.read_text(encoding="utf-8").count("DIR-20260919-001") == 2


def test_read_directive_events_rejects_conflicting_duplicate_directive(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    directives = root / "project_state" / "directives.jsonl"
    directives.parent.mkdir(parents=True)
    directives.write_text(
        json.dumps(modern_directive()) + "\n" + json.dumps(modern_directive("different")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate directive id conflict"):
        state_module.read_directive_events(root)


def test_directive_schema_accepts_actual_modern_records_and_validates_optional_legacy_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = root / "project_state" / "schemas" / "directives.schema.json"
    events = state_module.read_directive_events(root)

    assert any(
        event.get("event_type") == "directive"
        and "effective_from_revision" not in event
        for event in events
    )
    for index, event in enumerate(events, 1):
        assert state_module.validate_against_schema(event, schema, f"actual directive {index}")

    invalid_values = {
        "effective_from_revision": "seven",
        "affected_files": "README.md",
        "supersedes": ["DIR-1", "DIR-1"],
    }
    for field, value in invalid_values.items():
        event = modern_directive()
        event[field] = value
        with pytest.raises(ValueError, match="schema violation"):
            state_module.validate_against_schema(event, schema, f"invalid {field}")


def test_scan_documents_preserves_curated_governance_and_discovers_only_registered_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_document_root(tmp_path / "repo")
    monkeypatch.setattr(state_module.hashlib, "sha1", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("scan must not hash IDs")))

    entries = {item["path"]: item for item in state_module.scan_documents(root)["documents"]}

    receipt = entries["project_state/governance/receipt.pdf"]
    assert receipt["doc_id"] == "manual-governance-receipt"
    assert receipt["lifecycle"] == "superseded"
    assert receipt["authority"] == "normative"
    assert receipt["scope"] == "manual_governance"
    assert receipt["availability"] == "local_only"
    assert receipt["related_docs"] == ["new-receipt"]
    assert receipt["manual_note"] == "retain this governance decision"
    for path in ("03审计报告/new_audit.docx", "maintenance_logs/new_log.md"):
        assert entries[path]["lifecycle"] == "draft"
        assert entries[path]["authority"] == "reference"
        assert entries[path]["freshness"] == "review_due"
    assert "experiments/declared_package/README.md" in entries
    assert "experiments/undeclared_package/README.md" not in entries
    assert "experiments/results/reported/run/analysis/report.md" in entries
    assert "experiments/results/hidden/run/analysis/report.md" not in entries


def test_scan_documents_honors_existing_parent_coverage_and_git_index_only(tmp_path: Path) -> None:
    root = make_document_root(tmp_path / "repo")
    handoff = root / "团队项目进度与结论" / "handoff.md"
    handoff.parent.mkdir(parents=True)
    handoff.write_text("# handoff\n", encoding="utf-8")
    covered = root / "团队项目进度与结论" / "copied_result.md"
    covered.write_text("# covered copy\n", encoding="utf-8")
    independent = root / "团队项目进度与结论" / "independent_result.md"
    independent.write_text("# independent\n", encoding="utf-8")
    missing_parent_child = root / "团队项目进度与结论" / "missing_parent_child.md"
    missing_parent_child.write_text("# must remain visible\n", encoding="utf-8")
    cache_readme = root / "团队项目进度与结论" / "package" / ".pytest_cache" / "README.md"
    cache_readme.parent.mkdir(parents=True)
    cache_readme.write_text("# cache\n", encoding="utf-8")
    tracked = root / "maintenance_logs" / "tracked.md"
    tracked.write_text("# tracked\n", encoding="utf-8")
    untracked = root / "maintenance_logs" / "untracked.md"
    untracked.write_text("# untracked\n", encoding="utf-8")
    guide = root / "01_指南与解读" / "guide.docx"
    guide.parent.mkdir(parents=True)
    guide.write_bytes(b"docx placeholder")
    slides = root / "02_组会汇报" / "slides.pdf"
    slides.parent.mkdir(parents=True)
    slides.write_bytes(b"pdf placeholder")
    team_pdf = root / "团队项目进度与结论" / "handoff.pdf"
    team_pdf.write_bytes(b"pdf placeholder")

    registry_path = root / "project_state" / "document_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["documents"].extend([
        {
            "doc_id": "handoff-parent",
            "path": "团队项目进度与结论/handoff.md",
            "category": "团队共享材料",
            "scope": "handoff",
            "authority": "reference",
            "lifecycle": "active",
            "availability": "local_only",
            "verified_at": "2026-09-01T00:00:00+00:00",
            "state_revision": 1,
            "supersedes": [],
            "superseded_by": [],
            "truth_sources": [],
            "covered_documents": ["团队项目进度与结论/copied_result.md"],
        },
        {
            "doc_id": "independent-copy",
            "path": "团队项目进度与结论/independent_result.md",
            "category": "团队共享材料",
            "scope": "independent",
            "authority": "reference",
            "lifecycle": "historical",
            "availability": "local_only",
            "verified_at": "2026-09-01T00:00:00+00:00",
            "state_revision": 1,
            "supersedes": [],
            "superseded_by": [],
            "truth_sources": [],
        },
        {
            "doc_id": "missing-parent",
            "path": "团队项目进度与结论/missing_parent.md",
            "category": "团队共享材料",
            "scope": "missing",
            "authority": "reference",
            "lifecycle": "missing",
            "availability": "local_only",
            "verified_at": "2026-09-01T00:00:00+00:00",
            "state_revision": 1,
            "supersedes": [],
            "superseded_by": [],
            "truth_sources": [],
            "covered_documents": ["团队项目进度与结论/missing_parent_child.md"],
        },
    ])
    write_json(registry_path, registry)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "maintenance_logs/tracked.md"], cwd=root, check=True, capture_output=True)

    entries = {item["path"]: item for item in state_module.scan_documents(root)["documents"]}

    assert "团队项目进度与结论/copied_result.md" not in entries
    assert entries["团队项目进度与结论/independent_result.md"]["doc_id"] == "independent-copy"
    assert "团队项目进度与结论/missing_parent_child.md" in entries
    assert "团队项目进度与结论/package/.pytest_cache/README.md" not in entries
    assert entries["maintenance_logs/tracked.md"]["availability"] == "tracked"
    assert entries["maintenance_logs/untracked.md"]["availability"] == "local_only"
    for path in (
        "01_指南与解读/guide.docx",
        "02_组会汇报/slides.pdf",
        "团队项目进度与结论/handoff.pdf",
    ):
        assert entries[path]["lifecycle"] == "draft"
        assert entries[path]["authority"] == "reference"
        assert entries[path]["freshness"] == "review_due"
