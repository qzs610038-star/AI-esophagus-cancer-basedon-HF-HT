import json
from pathlib import Path

import pytest

from deploy import pfmval_ops
from scripts.pfmval_science import (
    adjudicate_fact_candidates,
    list_scientific_records,
    record_science_decision,
    record_scientific_entry,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _root(tmp_path: Path, *, accepted: bool = True) -> Path:
    root = tmp_path / "repo"
    _write_json(
        root / "experiments" / "experiment_registry.json",
        {
            "version": 1,
            "experiments": [
                {
                    "id": "exp-a",
                    "result_id": "res-a",
                    "evidence_status": "accepted" if accepted else "pending",
                }
            ],
        },
    )
    _write_json(
        root / "project_state" / "workspace_registry.json",
        {"schema_version": "1.0", "workspaces": []},
    )
    _write_json(
        root / "project_state" / "document_registry.json",
        {"schema_version": "1.0", "documents": []},
    )
    return root


def test_p0c_t01_progressive_layers_cannot_promote_idea_or_explore(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="cannot be accepted evidence"):
        record_scientific_entry(
            root,
            {
                "record_id": "idea-1",
                "record_type": "idea",
                "created_at": "2026-07-27T00:00:00+00:00",
                "source_refs": ["user:question"],
                "payload": {
                    "question": "Does X help?",
                    "hypothesis": "X may help",
                    "next_action": "local explore",
                    "evidence_status": "accepted",
                },
            },
        )


def test_p0c_t02_formal_claim_is_traceable_to_accepted_result(tmp_path):
    root = _root(tmp_path)
    result = record_scientific_entry(
        root,
        {
            "record_id": "claim-1",
            "record_type": "claim",
            "created_at": "2026-07-27T00:00:00+00:00",
            "source_refs": ["result:res-a"],
            "experiment_id": "exp-a",
            "result_id": "res-a",
            "payload": {
                "statement": "X supports Y",
                "evidence_direction": "supports",
                "uncertainty": "single accepted protocol",
            },
        },
    )
    assert result["status"] == "recorded"
    claim = list_scientific_records(root, record_type="claim")[0]
    assert claim["experiment_id"] == "exp-a"
    assert claim["result_id"] == "res-a"

    pending_root = _root(tmp_path / "pending", accepted=False)
    with pytest.raises(ValueError, match="accepted result"):
        record_scientific_entry(
            pending_root,
            {
                "record_id": "claim-2",
                "record_type": "claim",
                "created_at": "2026-07-27T00:00:00+00:00",
                "source_refs": ["result:res-a"],
                "experiment_id": "exp-a",
                "result_id": "res-a",
                "payload": {
                    "statement": "unsupported",
                    "evidence_direction": "supports",
                    "uncertainty": "unknown",
                },
            },
        )


def test_p0c_t03_negative_result_and_stop_rule_are_append_only(tmp_path):
    root = _root(tmp_path)
    negative = {
        "record_id": "negative-1",
        "record_type": "negative_result",
        "created_at": "2026-07-27T00:00:00+00:00",
        "source_refs": ["experiment:exp-a"],
        "experiment_id": "exp-a",
        "result_id": "res-a",
        "payload": {
            "route": "identical retry",
            "outcome": "no material difference",
            "stop_condition": "do not retry unchanged protocol",
            "continue_condition": "new internal-validation hypothesis",
        },
    }
    record_scientific_entry(root, negative)
    record_scientific_entry(
        root,
        {
            "record_id": "idea-next",
            "record_type": "idea",
            "created_at": "2026-07-27T00:00:01+00:00",
            "source_refs": ["negative:negative-1"],
            "payload": {
                "question": "What changed?",
                "hypothesis": "new factor",
                "next_action": "review",
            },
        },
    )
    records = list_scientific_records(root)
    assert records[0]["payload"]["stop_condition"] == (
        "do not retry unchanged protocol"
    )
    assert len(records) == 2


def test_p0c_t04_fact_conflict_adjudication_is_deterministic_and_preserves_conflict():
    candidates = [
        {
            "candidate_id": "view-1",
            "fact_domain": "scientific_measurement",
            "source_type": "generated_view",
            "value": 0.8,
        },
        {
            "candidate_id": "result-1",
            "fact_domain": "scientific_measurement",
            "source_type": "accepted_result",
            "value": 0.5,
        },
        {
            "candidate_id": "result-2",
            "fact_domain": "scientific_measurement",
            "source_type": "accepted_result",
            "value": 0.6,
        },
    ]
    first = adjudicate_fact_candidates(candidates)
    second = adjudicate_fact_candidates(list(reversed(candidates)))
    assert first == second
    assert first["status"] == "conflict"
    assert first["authoritative_candidate_ids"] == ["result-1", "result-2"]
    assert first["discarded_candidate_ids"] == ["view-1"]


def test_p0c_t05_explanation_separates_observation_interpretation_recommendation(
    tmp_path,
):
    root = _root(tmp_path)
    record_scientific_entry(
        root,
        {
            "record_id": "explanation-1",
            "record_type": "explanation",
            "created_at": "2026-07-27T00:00:00+00:00",
            "source_refs": ["result:res-a"],
            "experiment_id": "exp-a",
            "result_id": "res-a",
            "payload": {
                "observation": "PCC was 0.5",
                "interpretation": "effect is uncertain",
                "recommendation": "review another accepted attempt",
                "unit": "correlation",
                "statistic": "Pearson PCC",
                "text_description": "A scalar correlation from accepted result res-a.",
            },
        },
    )
    payload = list_scientific_records(root)[0]["payload"]
    assert set(
        ("observation", "interpretation", "recommendation")
    ).issubset(payload)


def test_p0c_t06_implementation_has_no_external_runtime_dependency():
    source = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "pfmval_science.py"
    ).read_text(encoding="utf-8")
    forbidden = ("requests", "urllib", "http://", "https://", "subprocess")
    assert all(item not in source for item in forbidden)


def test_p0c_t07_replay_is_noop_and_invalid_write_is_atomic(tmp_path):
    root = _root(tmp_path)
    entry = {
        "record_id": "idea-1",
        "record_type": "idea",
        "created_at": "2026-07-27T00:00:00+00:00",
        "source_refs": ["user:question"],
        "payload": {
            "question": "Q",
            "hypothesis": "H",
            "next_action": "review",
        },
    }
    first = record_scientific_entry(root, entry)
    second = record_scientific_entry(root, entry)
    path = root / "project_state" / "scientific_records.jsonl"
    before = path.read_bytes()
    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    with pytest.raises(ValueError):
        record_scientific_entry(root, {**entry, "payload": {}})
    assert path.read_bytes() == before


def test_p0c_t08_scientific_records_do_not_write_other_registries(tmp_path):
    root = _root(tmp_path)
    protected = [
        root / "experiments" / "experiment_registry.json",
        root / "project_state" / "workspace_registry.json",
        root / "project_state" / "document_registry.json",
    ]
    before = {path: path.read_bytes() for path in protected}
    record_scientific_entry(
        root,
        {
            "record_id": "idea-isolated",
            "record_type": "idea",
            "created_at": "2026-07-27T00:00:00+00:00",
            "source_refs": ["user:question"],
            "payload": {
                "question": "Q",
                "hypothesis": "H",
                "next_action": "review",
            },
        },
    )
    assert {path: path.read_bytes() for path in protected} == before


def test_science_decision_v1_single_result_is_one_atomic_append_only_receipt(
    tmp_path,
):
    """The public decision API projects three records from one approved input."""
    root = _root(tmp_path)
    decision = {
        "schema_version": "science_decision_v1",
        "decision_id": "decision-single-1",
        "created_at": "2026-07-30T00:00:00+00:00",
        "experiment_id": "exp-a",
        "result_binding": {
            "kind": "single_result",
            "result_id": "res-a",
            "acceptance_event_id": "accept-res-a",
        },
        "user_confirmation_ref": "user:DIR-20260730-002",
        "decision_summary": {
            "primary_metric": "external_xzy_pcc",
            "control_value": 0.6549,
            "treatment_value": 0.6527,
            "delta": -0.0022,
            "direction": "no_improvement",
            "stop_rule": "do_not_retry_unchanged_protocol",
            "uncertainty": "single paired comparison",
            "open_questions": ["new hypothesis required"],
        },
        "claim": {"statement": "Huber does not improve PCC"},
        "negative_result": {
            "route": "identical retry",
            "outcome": "no improvement",
            "stop_condition": "do not retry unchanged protocol",
            "continue_condition": "new hypothesis",
        },
        "explanation": {
            "observation": "delta PCC is negative",
            "interpretation": "no material benefit",
            "recommendation": "close without retry",
            "unit": "correlation",
            "statistic": "Pearson PCC",
            "text_description": "accepted paired comparison",
        },
    }

    first = record_science_decision(root, decision)
    record_path = root / "project_state" / "scientific_records.jsonl"
    before_replay = record_path.read_bytes()
    second = record_science_decision(root, decision)

    assert first["status"] == "recorded"
    assert first["receipt"]["decision_id"] == "decision-single-1"
    assert len(first["receipt"]["projected_record_ids"]) == 3
    assert second["status"] == "already_recorded"
    assert record_path.read_bytes() == before_replay
    assert {
        record["record_type"] for record in list_scientific_records(root)
    } == {"claim", "negative_result", "explanation"}


def test_p0c_local_cli_fixture_uses_only_repository_files(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _root(tmp_path)
    monkeypatch.setattr(pfmval_ops, "PROJECT_ROOT", root)
    entry_path = root / "idea.json"
    _write_json(
        entry_path,
        {
            "record_id": "idea-cli",
            "record_type": "idea",
            "created_at": "2026-07-27T00:00:00+00:00",
            "source_refs": ["user:question"],
            "payload": {
                "question": "Q",
                "hypothesis": "H",
                "next_action": "review",
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["pfmval_ops.py", "science", "record", "--input", str(entry_path)],
    )
    assert pfmval_ops.main() == 0
    monkeypatch.setattr(
        "sys.argv",
        ["pfmval_ops.py", "science", "list", "--record-type", "idea"],
    )
    assert pfmval_ops.main() == 0
    assert "idea-cli" in capsys.readouterr().out
