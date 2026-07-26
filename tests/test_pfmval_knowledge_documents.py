import json
from pathlib import Path

import pytest

from deploy import pfmval_ops
from scripts.pfmval_knowledge import (
    build_document_profile,
    confirm_project_fact,
    list_project_facts,
    preview_project_fact,
    review_document_freshness,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write_json(
        root / "experiments" / "experiment_registry.json",
        {
            "version": 1,
            "experiments": [
                {
                    "id": "exp-a",
                    "result_id": "res-a",
                    "evidence_status": "accepted",
                    "status": "done",
                },
                {
                    "id": "exp-failed",
                    "result_id": "res-failed",
                    "evidence_status": "rejected",
                    "status": "failed",
                },
            ],
        },
    )
    return root


def _fact(fact_id: str, statement: str, *, supersedes=None) -> dict:
    return {
        "fact_id": fact_id,
        "kind": "team_decision",
        "statement": statement,
        "source": "user:meeting",
        "observed_at": "2026-07-27T00:00:00+00:00",
        "status": "active",
        "supersedes": supersedes or [],
    }


def test_r10_t01_project_fact_preview_confirmation_correction_and_conflict(tmp_path):
    root = _root(tmp_path)
    facts_path = root / "project_state" / "project_facts.jsonl"
    preview = preview_project_fact(root, _fact("fact-1", "Use route A"))
    assert preview["status"] == "confirmation_required"
    assert not facts_path.exists()

    first = confirm_project_fact(
        root,
        _fact("fact-1", "Use route A"),
        confirmation_source="explicit_user_confirmation",
    )
    assert first["status"] == "recorded"
    corrected = confirm_project_fact(
        root,
        _fact("fact-2", "Use route B", supersedes=["fact-1"]),
        confirmation_source="explicit_user_confirmation",
    )
    assert corrected["status"] == "recorded"
    facts = list_project_facts(root)
    assert [item["fact_id"] for item in facts] == ["fact-1", "fact-2"]
    assert facts[0]["statement"] == "Use route A"

    with pytest.raises(ValueError, match="conflict"):
        confirm_project_fact(
            root,
            _fact("fact-3", "Use route C"),
            confirmation_source="explicit_user_confirmation",
        )


def test_r10_t02_meeting_brief_keeps_accepted_explore_and_failed_distinct(tmp_path):
    root = _root(tmp_path)
    artifact = build_document_profile(
        root,
        profile="meeting_brief",
        payload={
            "document_id": "meeting-1",
            "title": "组会简述",
            "accepted_result_refs": ["exp-a:res-a"],
            "explore_refs": ["EXPLORE-local-1"],
            "failed_refs": ["exp-failed:res-failed"],
            "recent_work": ["completed local refactor"],
            "difficulties": ["no server operation"],
            "next_steps": ["review"],
        },
    )
    assert "已接纳证据" in artifact["body"]
    assert "探索材料（非 accepted）" in artifact["body"]
    assert "失败/否定材料" in artifact["body"]
    assert artifact["profile"] == "meeting_brief"


def test_r10_t03_research_path_preserves_negative_stop_and_uncertainty(tmp_path):
    artifact = build_document_profile(
        _root(tmp_path),
        profile="research_path",
        payload={
            "document_id": "path-1",
            "title": "研究路径",
            "confirmed_conclusions": ["A"],
            "interpretations": [
                {"statement": "B", "uncertainty": "limited evidence"}
            ],
            "negative_routes": [
                {"route": "C", "stop_reason": "no difference"}
            ],
            "open_hypotheses": ["D"],
        },
    )
    assert "limited evidence" in artifact["body"]
    assert "no difference" in artifact["body"]


def test_r10_t04_learning_guide_is_interface_only_and_does_not_touch_old_docs(
    tmp_path,
):
    root = _root(tmp_path)
    old = root / "01_指南与解读" / "old.md"
    old.parent.mkdir(parents=True)
    old.write_text("# old\n", encoding="utf-8")
    before = old.read_bytes()
    artifact = build_document_profile(
        root,
        profile="learning_guide",
        payload={
            "document_id": "guide-interface-1",
            "title": "接口",
            "topic": "metrics",
            "audience": "student",
            "prerequisites": ["Python"],
            "source_refs": [
                {
                    "commit": "a" * 40,
                    "file": "pfmval_core/metrics.py",
                    "symbol": "compute_metrics",
                }
            ],
            "experiment_refs": ["exp-a:res-a"],
            "last_verified_commit": "a" * 40,
            "supersedes": [],
        },
    )
    assert artifact["status"] == "interface_reserved"
    assert old.read_bytes() == before


def test_r10_t05_paper_output_stops_at_template_pending_without_draft(tmp_path):
    artifact = build_document_profile(
        _root(tmp_path),
        profile="paper_output",
        payload={
            "document_id": "paper-1",
            "title": "论文材料接口",
            "template_ref": None,
            "material_refs": ["experiment:exp-a", "result:res-a"],
            "claim_refs": [],
            "section_mapping": {},
            "output_path": "papers/pending.md",
            "review_status": "not_started",
        },
    )
    assert artifact["status"] == "template_pending"
    assert "body" not in artifact


def test_r10_t06_paper_source_gate_rejects_external_material(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="external material"):
        build_document_profile(
            root,
            profile="paper_output",
            payload={
                "document_id": "paper-external",
                "title": "论文材料接口",
                "template_ref": None,
                "material_refs": ["external:https-example"],
                "claim_refs": [],
                "section_mapping": {},
                "output_path": "papers/pending.md",
                "review_status": "not_started",
            },
        )


def test_r10_t07_freshness_review_only_returns_review_due(tmp_path):
    root = _root(tmp_path)
    document_registry = root / "project_state" / "document_registry.json"
    _write_json(
        document_registry,
        {
            "schema_version": "1.0",
            "documents": [
                {
                    "doc_id": "doc-1",
                    "path": "guide.md",
                    "lifecycle": "active",
                    "freshness": "fresh",
                    "dependency_fingerprints": {"source": "old"},
                }
            ],
        },
    )
    before = document_registry.read_bytes()
    preview = review_document_freshness(
        root,
        document_id="doc-1",
        current_fingerprints={"source": "new"},
    )
    assert preview["freshness"] == "review_due"
    assert preview["lifecycle"] == "active"
    assert preview["writes"] == 0
    assert document_registry.read_bytes() == before


def test_r10_t08_fact_replay_is_noop_and_experiment_registry_is_isolated(tmp_path):
    root = _root(tmp_path)
    experiment_path = root / "experiments" / "experiment_registry.json"
    before = experiment_path.read_bytes()
    fact = _fact("fact-1", "Use route A")
    first = confirm_project_fact(
        root,
        fact,
        confirmation_source="explicit_user_confirmation",
    )
    second = confirm_project_fact(
        root,
        fact,
        confirmation_source="explicit_user_confirmation",
    )
    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    assert experiment_path.read_bytes() == before


def test_r10_t09_knowledge_module_has_no_external_runtime_dependency():
    source = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "pfmval_knowledge.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "requests",
        "urllib",
        "http://",
        "https://",
        "subprocess",
        "elabftw",
        "quarto",
    )
    assert all(item not in source.lower() for item in forbidden)


def test_r10_local_profile_cli_fixture_does_not_write_without_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _root(tmp_path)
    schema_target = (
        root / "project_state" / "schemas" / "document_profile.schema.json"
    )
    schema_target.parent.mkdir(parents=True)
    source_schema = (
        Path(__file__).resolve().parent.parent
        / "project_state"
        / "schemas"
        / "document_profile.schema.json"
    )
    schema_target.write_text(
        source_schema.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload = root / "paper.json"
    _write_json(
        payload,
        {
            "document_id": "paper-cli",
            "title": "论文接口",
            "template_ref": None,
            "material_refs": ["experiment:exp-a", "result:res-a"],
            "claim_refs": [],
            "section_mapping": {},
            "output_path": "papers/pending.md",
            "review_status": "not_started",
        },
    )
    before = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(pfmval_ops, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pfmval_ops.py",
            "knowledge",
            "profile",
            "--profile",
            "paper_output",
            "--input",
            str(payload),
        ],
    )
    assert pfmval_ops.main() == 0
    assert "template_pending" in capsys.readouterr().out
    after = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
