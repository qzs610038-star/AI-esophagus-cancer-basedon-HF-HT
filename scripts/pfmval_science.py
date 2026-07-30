"""Local append-only records for scientific reasoning and maintenance."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping


RECORD_TYPES = {
    "idea",
    "explore",
    "candidate",
    "claim",
    "negative_result",
    "conflict",
    "explanation",
}
LIGHTWEIGHT_TYPES = {"idea", "explore", "candidate"}
DOMAIN_AUTHORITY = {
    "intent_authorization": ("user_directive",),
    "identity_lifecycle": ("current_state", "registry"),
    "scientific_measurement": ("accepted_result",),
    "future_plan": ("active_plan",),
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _records_path(root: Path) -> Path:
    return root / "project_state" / "scientific_records.jsonl"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid scientific JSONL at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"scientific record at line {line_number} is not an object"
            )
        records.append(record)
    return records


def _write_records_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(_canonical(dict(record)) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _accepted_experiment(
    root: Path,
    *,
    experiment_id: str,
    result_id: str,
) -> Mapping[str, Any]:
    registry = json.loads(
        (root / "experiments" / "experiment_registry.json").read_text(
            encoding="utf-8"
        )
    )
    experiment = next(
        (
            item
            for item in registry.get("experiments", [])
            if item.get("id") == experiment_id
        ),
        None,
    )
    paired_result = (
        experiment.get("paired_result", {})
        if isinstance(experiment, Mapping)
        else {}
    )
    accepted_result_ids = {
        str(experiment.get("result_id", ""))
        if isinstance(experiment, Mapping)
        else "",
        str(paired_result.get("pair_id", "")),
        *(
            str(value)
            for value in (
                experiment.get("result_ids", [])
                if isinstance(experiment, Mapping)
                else []
            )
        ),
    }
    if (
        experiment is None
        or result_id not in accepted_result_ids
        or experiment.get("evidence_status") != "accepted"
    ):
        raise ValueError(
            "formal scientific record must reference an accepted result"
        )
    return experiment


def _validate_record(root: Path, entry: Mapping[str, Any]) -> None:
    required = {
        "record_id",
        "record_type",
        "created_at",
        "source_refs",
        "payload",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(
            f"scientific record missing fields: {', '.join(missing)}"
        )
    if not str(entry["record_id"]).strip():
        raise ValueError("scientific record_id must be non-empty")
    record_type = str(entry["record_type"])
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unsupported scientific record_type: {record_type}")
    if not isinstance(entry["source_refs"], list) or not entry["source_refs"]:
        raise ValueError("scientific record requires source_refs")
    payload = entry["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("scientific payload must be an object")

    if record_type in LIGHTWEIGHT_TYPES:
        if payload.get("evidence_status") == "accepted":
            raise ValueError(
                f"{record_type} cannot be accepted evidence"
            )
        required_payload = {
            "idea": {"question", "hypothesis", "next_action"},
            "explore": {"explore_id", "input_scope", "observation", "limits"},
            "candidate": {
                "promotion_reason",
                "decision_condition",
                "estimated_cost",
            },
        }[record_type]
        missing_payload = sorted(required_payload - set(payload))
        if missing_payload:
            raise ValueError(
                f"{record_type} payload missing: {', '.join(missing_payload)}"
            )

    if record_type in {"claim", "explanation"}:
        experiment_id = str(entry.get("experiment_id", ""))
        result_id = str(entry.get("result_id", ""))
        if not experiment_id or not result_id:
            raise ValueError(
                f"{record_type} requires experiment_id and result_id"
            )
        _accepted_experiment(
            root,
            experiment_id=experiment_id,
            result_id=result_id,
        )

    if record_type == "claim":
        claim_fields = {
            "statement",
            "evidence_direction",
            "uncertainty",
        }
        missing_claim = sorted(claim_fields - set(payload))
        if missing_claim:
            raise ValueError(
                f"claim payload missing: {', '.join(missing_claim)}"
            )
        if payload["evidence_direction"] not in {
            "supports",
            "refutes",
            "unresolved",
        }:
            raise ValueError("claim has invalid evidence_direction")

    if record_type == "negative_result":
        negative_fields = {
            "route",
            "outcome",
            "stop_condition",
            "continue_condition",
        }
        missing_negative = sorted(negative_fields - set(payload))
        if missing_negative:
            raise ValueError(
                "negative_result payload missing: "
                + ", ".join(missing_negative)
            )

    if record_type == "explanation":
        explanation_fields = {
            "observation",
            "interpretation",
            "recommendation",
            "unit",
            "statistic",
            "text_description",
        }
        missing_explanation = sorted(explanation_fields - set(payload))
        if missing_explanation:
            raise ValueError(
                "explanation payload missing: "
                + ", ".join(missing_explanation)
            )


def record_scientific_entry(
    root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and append one record without touching other registries."""
    normalized = json.loads(_canonical(dict(entry)))
    _validate_record(root, normalized)
    path = _records_path(root)
    records = _read_records(path)
    existing = next(
        (
            item
            for item in records
            if item.get("record_id") == normalized["record_id"]
        ),
        None,
    )
    if existing is not None:
        if _canonical(existing) != _canonical(normalized):
            raise ValueError(
                "record_id is already bound to different scientific content"
            )
        return {"status": "already_recorded", "record": existing}
    _write_records_atomic(path, [*records, normalized])
    return {"status": "recorded", "record": normalized}


def _decision_schema_path(root: Path) -> Path:
    return root / "project_state" / "schemas" / "science_decision_v1.schema.json"


def _decision_projection(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create the three append-only scientific records from one decision package."""
    binding = dict(decision["result_binding"])
    common = {
        "created_at": decision["created_at"],
        "source_refs": [
            f"decision:{decision['decision_id']}",
            f"result:{binding['result_id']}",
            f"acceptance:{binding['acceptance_event_id']}",
            str(decision["user_confirmation_ref"]),
        ],
        "experiment_id": decision["experiment_id"],
        "result_id": binding["result_id"],
    }
    summary = dict(decision["decision_summary"])
    evidence_direction = {
        "improvement": "supports",
        "no_improvement": "refutes",
        "unresolved": "unresolved",
    }.get(str(summary["direction"]), "unresolved")
    claim = {
        **common,
        "record_id": f"{decision['decision_id']}:claim",
        "record_type": "claim",
        "payload": {
            **dict(decision["claim"]),
            "evidence_direction": evidence_direction,
            "uncertainty": summary["uncertainty"],
            "decision_summary": summary,
        },
    }
    negative = {
        **common,
        "record_id": f"{decision['decision_id']}:negative_result",
        "record_type": "negative_result",
        "payload": {**dict(decision["negative_result"]), "decision_summary": summary},
    }
    explanation = {
        **common,
        "record_id": f"{decision['decision_id']}:explanation",
        "record_type": "explanation",
        "payload": {**dict(decision["explanation"]), "decision_summary": summary},
    }
    return [claim, negative, explanation]


def _decision_receipt(decision: Mapping[str, Any], projections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    projected = [dict(item) for item in projections]
    digest = hashlib.sha256(_canonical(projected).encode("utf-8")).hexdigest()
    return {
        "decision_id": str(decision["decision_id"]),
        "schema_version": "science_decision_v1",
        "result_binding": dict(decision["result_binding"]),
        "projected_record_ids": [str(item["record_id"]) for item in projected],
        "projection_sha256": digest,
    }


def record_science_decision(root: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one decision package then atomically append its three projections."""
    normalized = json.loads(_canonical(dict(decision)))
    from scripts.pfmval_state import validate_against_schema

    validate_against_schema(normalized, _decision_schema_path(root), "science_decision_v1")
    projections = _decision_projection(normalized)
    for entry in projections:
        _validate_record(root, entry)
    path = _records_path(root)
    records = _read_records(path)
    existing = {str(item.get("record_id")): item for item in records}
    ids = [str(item["record_id"]) for item in projections]
    present = [item_id for item_id in ids if item_id in existing]
    receipt = _decision_receipt(normalized, projections)
    if present:
        if len(present) != len(ids) or any(
            _canonical(existing[item["record_id"]]) != _canonical(item)
            for item in projections
        ):
            raise ValueError("decision_id is already bound to different scientific content")
        return {"status": "already_recorded", "receipt": receipt}
    _write_records_atomic(path, [*records, *projections])
    return {"status": "recorded", "receipt": receipt}


def list_scientific_records(
    root: Path,
    *,
    record_type: str | None = None,
) -> list[dict[str, Any]]:
    records = _read_records(_records_path(root))
    if record_type is None:
        return records
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unsupported scientific record_type: {record_type}")
    return [
        record
        for record in records
        if record.get("record_type") == record_type
    ]


def adjudicate_fact_candidates(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply deterministic within-domain authority without erasing conflicts."""
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: str(item.get("candidate_id", "")),
    )
    if not ordered:
        raise ValueError("fact adjudication requires candidates")
    domains = {str(item.get("fact_domain", "")) for item in ordered}
    if len(domains) != 1:
        raise ValueError("fact candidates must share one fact_domain")
    domain = next(iter(domains))
    allowed_sources = DOMAIN_AUTHORITY.get(domain)
    if not allowed_sources:
        raise ValueError(f"unsupported fact_domain: {domain}")
    authoritative = [
        item
        for item in ordered
        if item.get("source_type") in allowed_sources
    ]
    discarded = [
        item
        for item in ordered
        if item.get("source_type") not in allowed_sources
    ]
    if not authoritative:
        return {
            "fact_domain": domain,
            "status": "no_authoritative_candidate",
            "authoritative_candidate_ids": [],
            "discarded_candidate_ids": [
                str(item["candidate_id"]) for item in discarded
            ],
            "value": None,
        }
    values = {_canonical(item.get("value")) for item in authoritative}
    if len(values) == 1:
        status = "resolved"
        selected = authoritative[0]
    else:
        authoritative_ids = {
            str(item["candidate_id"]) for item in authoritative
        }
        superseding = [
            item
            for item in authoritative
            if authoritative_ids
            - {str(item["candidate_id"])}
            <= {str(value) for value in item.get("supersedes", [])}
        ]
        if len(superseding) == 1:
            status = "resolved_by_supersedes"
            selected = superseding[0]
        else:
            status = "conflict"
            selected = None
    return {
        "fact_domain": domain,
        "status": status,
        "authoritative_candidate_ids": [
            str(item["candidate_id"]) for item in authoritative
        ],
        "discarded_candidate_ids": [
            str(item["candidate_id"]) for item in discarded
        ],
        "value": selected.get("value") if selected else None,
        "conflicting_values": (
            [item.get("value") for item in authoritative]
            if status == "conflict"
            else []
        ),
    }
