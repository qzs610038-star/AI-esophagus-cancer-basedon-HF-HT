"""Atomic acceptance of paired governed experiment results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.pfmval_state import state_lock


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_ROLES = ("control", "treatment")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL entry is not an object: {path}:{line_number}")
        values.append(value)
    return values


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical(dict(value)) + "\n" for value in values).encode(
        "utf-8"
    )


def _write_temporary(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary_name


def _restore(path: Path, original: bytes | None) -> None:
    if original is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    temporary_name = _write_temporary(path, original)
    os.replace(temporary_name, path)


def _write_pair_transaction(
    registry_path: Path,
    event_path: Path,
    registry: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
) -> None:
    registry_original = (
        registry_path.read_bytes() if registry_path.exists() else None
    )
    event_original = event_path.read_bytes() if event_path.exists() else None
    registry_temporary = _write_temporary(
        registry_path,
        _json_bytes(registry),
    )
    event_temporary = _write_temporary(
        event_path,
        _jsonl_bytes(events),
    )
    registry_replaced = False
    event_replaced = False
    try:
        os.replace(registry_temporary, registry_path)
        registry_replaced = True
        os.replace(event_temporary, event_path)
        event_replaced = True
    except Exception:
        if registry_replaced:
            _restore(registry_path, registry_original)
        if event_replaced:
            _restore(event_path, event_original)
        for temporary_name in (registry_temporary, event_temporary):
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def _require_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")


def _validate_input(value: Mapping[str, Any]) -> None:
    _require_fields(
        value,
        {
            "schema_version",
            "pair_id",
            "experiment_id",
            "workspace_id",
            "accepted_at",
            "confirmation_source",
            "member_results",
            "scientific_record_ids",
            "decision_summary",
            "next_action",
        },
        "result pair",
    )
    if value["schema_version"] != "1.0":
        raise ValueError("result pair schema_version must be 1.0")
    if not str(value["pair_id"]).strip():
        raise ValueError("result pair_id must be non-empty")
    if not str(value["confirmation_source"]).startswith(
        "explicit_user_confirmation"
    ):
        raise ValueError("pair acceptance requires explicit user confirmation")
    members = value["member_results"]
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("result pair requires exactly two member results")
    roles = [str(member.get("role", "")) for member in members]
    if roles != list(PAIR_ROLES):
        raise ValueError("result pair roles must be control then treatment")
    result_ids = [str(member.get("result_id", "")) for member in members]
    if not all(result_ids) or len(set(result_ids)) != 2:
        raise ValueError("result pair member result_ids must be non-empty and unique")
    for member in members:
        if not SHA256_RE.fullmatch(str(member.get("bundle_sha256", ""))):
            raise ValueError("result pair member requires bundle SHA-256")
        if not str(member.get("label", "")).strip():
            raise ValueError("result pair member requires label")
    scientific_ids = value["scientific_record_ids"]
    if (
        not isinstance(scientific_ids, list)
        or not scientific_ids
        or len(scientific_ids) != len(set(scientific_ids))
    ):
        raise ValueError("result pair requires unique scientific record ids")
    decision = value["decision_summary"]
    if not isinstance(decision, Mapping):
        raise ValueError("result pair decision_summary must be an object")
    _require_fields(
        decision,
        {
            "primary_metric",
            "control_value",
            "treatment_value",
            "delta",
            "evidence_direction",
            "statement",
            "uncertainty",
        },
        "decision summary",
    )
    control = float(decision["control_value"])
    treatment = float(decision["treatment_value"])
    delta = float(decision["delta"])
    if abs((treatment - control) - delta) > 1e-9:
        raise ValueError("decision summary delta does not equal treatment-control")
    if decision["evidence_direction"] not in {
        "supports",
        "refutes",
        "unresolved",
    }:
        raise ValueError("decision summary has invalid evidence_direction")


def _stable_event_content(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in event.items()
        if key not in {"registry_before_sha256", "registry_after_sha256"}
    }


def finalize_result_pair(
    root: Path,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate two imported results and atomically accept their paired decision."""
    normalized = json.loads(_canonical(dict(acceptance)))
    _validate_input(normalized)
    root = root.resolve()
    registry_path = root / "experiments" / "experiment_registry.json"
    import_path = root / "project_state" / "result_import_events.jsonl"
    scientific_path = root / "project_state" / "scientific_records.jsonl"
    pair_event_path = root / "project_state" / "result_pair_events.jsonl"

    with state_lock(root):
        registry = _read_json(registry_path)
        imports = _read_jsonl(import_path)
        scientific_records = _read_jsonl(scientific_path)
        pair_events = _read_jsonl(pair_event_path)
        experiment = next(
            (
                item
                for item in registry.get("experiments", [])
                if item.get("id") == normalized["experiment_id"]
            ),
            None,
        )
        if experiment is None:
            raise ValueError(
                f"unknown experiment id: {normalized['experiment_id']}"
            )
        if experiment.get("workspace_id") != normalized["workspace_id"]:
            raise ValueError("result pair workspace does not match Registry")

        imports_by_result = {
            str(event.get("result_id", "")): event for event in imports
        }
        enriched_members = []
        approval_ids: set[str] = set()
        contract_hashes: set[str] = set()
        for member in normalized["member_results"]:
            result_id = str(member["result_id"])
            imported = imports_by_result.get(result_id)
            if (
                imported is None
                or imported.get("event_type") != "RESULT_IMPORT_VERIFIED"
                or imported.get("status") != "success"
            ):
                raise ValueError(
                    f"result pair member has no successful verified import: {result_id}"
                )
            if imported.get("experiment_id") != normalized["experiment_id"]:
                raise ValueError("verified import experiment mismatch")
            if imported.get("workspace_id") != normalized["workspace_id"]:
                raise ValueError("verified import workspace mismatch")
            if imported.get("bundle_sha256") != member["bundle_sha256"]:
                raise ValueError("verified import bundle SHA mismatch")
            approval_ids.add(str(imported.get("approval_id", "")))
            contract_hashes.add(
                str(imported.get("critical_contract_sha256", ""))
            )
            enriched_members.append(
                {
                    **member,
                    "attempt_id": imported.get("attempt_id"),
                    "job_id": imported.get("job_id"),
                    "import_event_id": imported.get("event_id"),
                }
            )
        if len(approval_ids) != 1 or "" in approval_ids:
            raise ValueError("paired imports do not share one approval")
        if len(contract_hashes) != 1 or not all(
            SHA256_RE.fullmatch(value) for value in contract_hashes
        ):
            raise ValueError("paired imports do not share one critical contract")

        scientific_by_id = {
            str(record.get("record_id", "")): record
            for record in scientific_records
        }
        for record_id in normalized["scientific_record_ids"]:
            record = scientific_by_id.get(str(record_id))
            if record is None:
                raise ValueError(
                    f"pair acceptance scientific record is missing: {record_id}"
                )
            if record.get("experiment_id") != normalized["experiment_id"]:
                raise ValueError("scientific record experiment mismatch")

        pair = {
            "schema_version": "1.0",
            "pair_id": normalized["pair_id"],
            "workspace_id": normalized["workspace_id"],
            "member_results": enriched_members,
            "approval_id": next(iter(approval_ids)),
            "critical_contract_sha256": next(iter(contract_hashes)),
        }
        event = {
            "schema_version": "1.0",
            "event_id": f"accept-{normalized['pair_id']}",
            "event_type": "PAIR_RESULT_ACCEPTED",
            "recorded_at": normalized["accepted_at"],
            "experiment_id": normalized["experiment_id"],
            **pair,
            "scientific_record_ids": list(normalized["scientific_record_ids"]),
            "decision_summary": dict(normalized["decision_summary"]),
            "confirmation_source": normalized["confirmation_source"],
        }
        existing = next(
            (
                item
                for item in pair_events
                if item.get("event_id") == event["event_id"]
                or item.get("pair_id") == event["pair_id"]
            ),
            None,
        )
        if existing is not None:
            if _stable_event_content(existing) != _stable_event_content(event):
                raise ValueError(
                    "pair_id is already bound to different acceptance content"
                )
            return {"status": "already_finalized", "event": existing}

        registry_before_sha256 = _sha256_json(registry)
        experiment["paired_result"] = pair
        experiment["result_ids"] = [
            member["result_id"] for member in enriched_members
        ]
        experiment["supersedes_results"] = [
            result_id
            for result_id in experiment.get("supersedes_results", [])
            if result_id not in experiment["result_ids"]
        ]
        experiment["decision_summary"] = dict(normalized["decision_summary"])
        experiment["conclusion"] = normalized["decision_summary"]["statement"]
        experiment["status"] = "done"
        experiment["evidence_status"] = "accepted"
        experiment["provenance_complete"] = True
        experiment["next_action"] = normalized["next_action"]
        experiment["accepted_at"] = normalized["accepted_at"]
        experiment["updated_at"] = normalized["accepted_at"]
        registry["updated_at"] = normalized["accepted_at"]
        registry_after_sha256 = _sha256_json(registry)
        event["registry_before_sha256"] = registry_before_sha256
        event["registry_after_sha256"] = registry_after_sha256
        _write_pair_transaction(
            registry_path,
            pair_event_path,
            registry,
            [*pair_events, event],
        )
        return {"status": "finalized", "event": event}
