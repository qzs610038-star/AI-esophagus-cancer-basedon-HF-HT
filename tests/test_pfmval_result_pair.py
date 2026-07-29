import json
from pathlib import Path

import pytest

from scripts.pfmval_result_pair import finalize_result_pair
from scripts.pfmval_views import build_experiment_progress
from scripts.finalize_experiment import build_dashboard


CONTROL_ID = "W001-A003-result-R004"
TREATMENT_ID = "W001-A004-result-R002"
PAIR_ID = "PAIR-W001-A003-A004"
EXPERIMENT_ID = "mpp2_huber_loss_paired_v001_20260728"
CONTROL_SHA = "a" * 64
TREATMENT_SHA = "b" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
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
                    "id": EXPERIMENT_ID,
                    "workspace_id": "W001",
                    "status": "done",
                    "evidence_status": "accepted",
                    "result_id": TREATMENT_ID,
                    "supersedes_results": [CONTROL_ID],
                    "next_action": "validate_attempt",
                }
            ],
        },
    )
    _write_jsonl(
        root / "project_state" / "result_import_events.jsonl",
        [
            {
                "event_type": "RESULT_IMPORT_VERIFIED",
                "event_id": f"import-{CONTROL_ID}",
                "experiment_id": EXPERIMENT_ID,
                "workspace_id": "W001",
                "attempt_id": "A003",
                "job_id": "W001-A003",
                "result_id": CONTROL_ID,
                "bundle_sha256": CONTROL_SHA,
                "approval_id": "approval-1",
                "critical_contract_sha256": "c" * 64,
                "status": "success",
            },
            {
                "event_type": "RESULT_IMPORT_VERIFIED",
                "event_id": f"import-{TREATMENT_ID}",
                "experiment_id": EXPERIMENT_ID,
                "workspace_id": "W001",
                "attempt_id": "A004",
                "job_id": "W001-A004",
                "result_id": TREATMENT_ID,
                "bundle_sha256": TREATMENT_SHA,
                "approval_id": "approval-1",
                "critical_contract_sha256": "c" * 64,
                "status": "success",
            },
        ],
    )
    _write_jsonl(
        root / "project_state" / "scientific_records.jsonl",
        [
            {
                "record_id": "claim-1",
                "record_type": "claim",
                "experiment_id": EXPERIMENT_ID,
                "result_id": TREATMENT_ID,
            },
            {
                "record_id": "negative-1",
                "record_type": "negative_result",
                "experiment_id": EXPERIMENT_ID,
                "result_id": TREATMENT_ID,
            },
            {
                "record_id": "explanation-1",
                "record_type": "explanation",
                "experiment_id": EXPERIMENT_ID,
                "result_id": TREATMENT_ID,
            },
        ],
    )
    return root


def _input() -> dict:
    return {
        "schema_version": "1.0",
        "pair_id": PAIR_ID,
        "experiment_id": EXPERIMENT_ID,
        "workspace_id": "W001",
        "accepted_at": "2026-07-29T13:20:03+00:00",
        "confirmation_source": "explicit_user_confirmation:G14-1-2-3",
        "member_results": [
            {
                "role": "control",
                "label": "MSE",
                "result_id": CONTROL_ID,
                "bundle_sha256": CONTROL_SHA,
            },
            {
                "role": "treatment",
                "label": "Huber(delta=1)",
                "result_id": TREATMENT_ID,
                "bundle_sha256": TREATMENT_SHA,
            },
        ],
        "scientific_record_ids": [
            "claim-1",
            "negative-1",
            "explanation-1",
        ],
        "decision_summary": {
            "primary_metric": "external_xzy_pooled_pcc",
            "control_value": 0.6548958191,
            "treatment_value": 0.6526906630,
            "delta": -0.0022051561,
            "evidence_direction": "refutes",
            "statement": "Huber(delta=1) did not improve external XZY pooled PCC.",
            "descriptive_improved_pathways": 17,
            "descriptive_total_pathways": 30,
            "uncertainty": "cluster bootstrap unavailable",
        },
        "next_action": "closed_no_retry",
    }


def test_pair_finalize_preserves_both_members_and_writes_acceptance_event(tmp_path):
    root = _root(tmp_path)

    result = finalize_result_pair(root, _input())

    assert result["status"] == "finalized"
    registry = json.loads(
        (root / "experiments" / "experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )
    experiment = registry["experiments"][0]
    assert experiment["result_id"] == TREATMENT_ID
    assert experiment["paired_result"]["pair_id"] == PAIR_ID
    assert experiment["result_ids"] == [CONTROL_ID, TREATMENT_ID]
    assert experiment["supersedes_results"] == []
    assert experiment["next_action"] == "closed_no_retry"
    assert experiment["decision_summary"]["delta"] == -0.0022051561

    events = [
        json.loads(line)
        for line in (
            root / "project_state" / "result_pair_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event_type"] == "PAIR_RESULT_ACCEPTED"
    assert events[0]["pair_id"] == PAIR_ID
    assert [item["role"] for item in events[0]["member_results"]] == [
        "control",
        "treatment",
    ]
    assert events[0]["confirmation_source"].startswith(
        "explicit_user_confirmation"
    )


def test_pair_finalize_is_idempotent_and_rejects_same_id_with_new_content(
    tmp_path,
):
    root = _root(tmp_path)
    first = finalize_result_pair(root, _input())
    before = {
        path: path.read_bytes()
        for path in (
            root / "experiments" / "experiment_registry.json",
            root / "project_state" / "result_pair_events.jsonl",
        )
    }

    second = finalize_result_pair(root, _input())

    assert first["status"] == "finalized"
    assert second["status"] == "already_finalized"
    assert {path: path.read_bytes() for path in before} == before

    changed = _input()
    changed["decision_summary"]["statement"] = "Different scientific decision."
    with pytest.raises(ValueError, match="different acceptance content"):
        finalize_result_pair(root, changed)
    assert {path: path.read_bytes() for path in before} == before


def test_pair_finalize_validates_both_imports_before_any_write(tmp_path):
    root = _root(tmp_path)
    import_path = root / "project_state" / "result_import_events.jsonl"
    imports = import_path.read_text(encoding="utf-8").splitlines()
    import_path.write_text(imports[0] + "\n", encoding="utf-8")
    registry_path = root / "experiments" / "experiment_registry.json"
    registry_before = registry_path.read_bytes()

    with pytest.raises(ValueError, match="verified import"):
        finalize_result_pair(root, _input())

    assert registry_path.read_bytes() == registry_before
    assert not (
        root / "project_state" / "result_pair_events.jsonl"
    ).exists()


def test_pair_decision_is_visible_in_experiment_progress(tmp_path):
    root = _root(tmp_path)
    finalize_result_pair(root, _input())
    registry = json.loads(
        (root / "experiments" / "experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )

    progress = build_experiment_progress(registry)
    dashboard = build_dashboard(registry)

    assert PAIR_ID in progress
    assert "ΔPCC=-0.0022" in progress
    assert "Huber(delta=1) did not improve" in progress
    assert "closed_no_retry" in progress
    assert "pair_ΔPCC=-0.0022" in dashboard
