from __future__ import annotations

from pathlib import Path

import pytest

from dispatch import _validate_feature_registry_entry, execute_plan, plan_scope
from errors import IdentityMismatchError


def test_default_scope_is_exactly_seed42_reference_then_two_candidates():
    units = plan_scope("seed42")
    assert [(unit.kind, unit.model, unit.seed) for unit in units] == [
        ("reference", "uni2h", 42),
        ("train", "uni", 42),
        ("external", "uni", 42),
        ("train", "virchow2", 42),
        ("external", "virchow2", 42),
    ]


def test_remaining_scope_contains_only_43_and_44_and_is_separate():
    units = plan_scope("remaining")
    assert {unit.seed for unit in units} == {43, 44}
    assert 42 not in {unit.seed for unit in units}
    assert len([unit for unit in units if unit.kind == "train"]) == 4


def test_first_failure_marks_every_later_unit_not_run():
    calls = []

    def executor(unit):
        calls.append(unit.run_id)
        if unit.kind == "train" and unit.model == "uni":
            raise RuntimeError("synthetic failure")
        return {"status": "succeeded"}

    records = execute_plan(plan_scope("seed42"), executor)
    assert [record["status"] for record in records] == [
        "succeeded",
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert calls == ["reference_42_uni2h", "train_42_uni"]
    assert [record["exit_code"] for record in records] == [0, 1, None, None, None]
    assert records[0]["started_at"] and records[0]["ended_at"]
    assert records[1]["started_at"] and records[1]["ended_at"]
    assert records[2]["started_at"] is None and records[2]["ended_at"] is None


def test_unknown_or_combined_scope_is_rejected():
    for scope in ("all", "42,43,44", "seed43"):
        try:
            plan_scope(scope)
        except ValueError:
            pass
        else:
            raise AssertionError(f"scope {scope!r} should be rejected")


def test_complete_feature_registry_must_contain_the_exact_candidate_cache(tmp_path: Path):
    cache = (tmp_path / "uni" / "v1").resolve()
    registry = {
        "status": "complete",
        "entries": [
            {
                "model": "uni",
                "status": "complete",
                "cache_dir": str(cache),
                "cls": {"path": str(cache / "cls.npy"), "role": "primary_training_input"},
            }
        ],
    }
    assert _validate_feature_registry_entry(registry, "uni", cache)["model"] == "uni"
    with pytest.raises(IdentityMismatchError, match="virchow2"):
        _validate_feature_registry_entry(registry, "virchow2", tmp_path / "virchow2" / "v1")
    registry["entries"][0]["cache_dir"] = str(tmp_path / "elsewhere")
    with pytest.raises(IdentityMismatchError, match="路径"):
        _validate_feature_registry_entry(registry, "uni", cache)
