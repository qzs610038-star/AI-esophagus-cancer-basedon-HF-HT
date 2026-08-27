"""Scoped W004 Stage 2 history reconciliation and four-arm acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from scripts.pfmval_result_bundle import validate_result_bundle_v1
from scripts.pfmval_science import _decision_projection
from scripts.pfmval_state import state_lock, validate_against_schema


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODES = {"check", "apply"}


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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


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
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL entry is not an object: {path}:{line_number}")
        values.append(value)
    return values


def _append_jsonl_bytes(
    original: bytes | None,
    records: Iterable[Mapping[str, Any]],
) -> bytes:
    payload = original or b""
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    return payload + "".join(
        _canonical(dict(record)) + "\n" for record in records
    ).encode("utf-8")


def _git_bytes(repo: Path, ref: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_json(repo: Path, ref: str, relative: str) -> dict[str, Any]:
    value = json.loads(_git_bytes(repo, ref, relative).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"historical JSON root is not an object: {relative}")
    return value


def _git_jsonl(repo: Path, ref: str, relative: str) -> list[dict[str, Any]]:
    values = []
    for raw in _git_bytes(repo, ref, relative).decode("utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"historical JSONL entry is not an object: {relative}")
        values.append(value)
    return values


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "migration_id",
        "history_ref",
        "experiment_id",
        "workspace_id",
        "protocol_revision",
        "approval_id",
        "source_commit",
        "confirmation_source",
        "evidence_scope",
        "critical_contract",
        "historical_sources",
        "bundles",
        "result_group",
        "decision",
        "limitations",
        "receipt_path",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"W004 reconciliation manifest missing: {missing}")
    if manifest["schema_version"] != "1.0":
        raise ValueError("W004 reconciliation schema_version must be 1.0")
    if manifest["workspace_id"] != "W004":
        raise ValueError("W004 reconciliation is scoped only to W004")
    if int(manifest["protocol_revision"]) != 2:
        raise ValueError("W004 reconciliation requires protocol revision 2")
    if manifest["evidence_scope"] != "COMPATIBILITY_ONLY":
        raise ValueError("W004 evidence scope must remain COMPATIBILITY_ONLY")
    if not str(manifest["confirmation_source"]).startswith(
        "explicit_user_confirmation:"
    ):
        raise ValueError("W004 reconciliation requires explicit confirmation")
    bundles = manifest["bundles"]
    if not isinstance(bundles, list) or [item.get("arm") for item in bundles] != [
        "A001",
        "A002",
        "A003",
        "A004",
    ]:
        raise ValueError("W004 reconciliation requires ordered A001-A004 bundles")
    if len({item.get("result_id") for item in bundles}) != 4:
        raise ValueError("W004 reconciliation result_ids must be unique")
    for item in bundles:
        if not SHA256_RE.fullmatch(str(item.get("bundle_sha256", ""))):
            raise ValueError(f"invalid bundle SHA-256 for {item.get('arm')}")
    contract_sha = str(manifest["critical_contract"].get("canonical_sha256", ""))
    if not SHA256_RE.fullmatch(contract_sha):
        raise ValueError("invalid W004 critical contract SHA-256")
    group = manifest["result_group"]
    if not str(group.get("group_id", "")).strip():
        raise ValueError("W004 result group_id must be non-empty")
    if manifest["decision"].get("result_binding", {}).get("kind") != "grouped_result":
        raise ValueError("W004 decision must use grouped_result binding")


def _load_history(
    history_repo: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ref = str(manifest["history_ref"])
    paths = manifest["historical_sources"]
    experiment_id = str(manifest["experiment_id"])
    workspace_id = str(manifest["workspace_id"])

    registry = _git_json(history_repo, ref, paths["experiment_registry"])
    experiments = [
        item
        for item in registry.get("experiments", [])
        if item.get("id") == experiment_id
    ]
    if len(experiments) != 1:
        raise ValueError("historical W004 experiment must occur exactly once")

    workspace_registry = _git_json(
        history_repo, ref, paths["workspace_registry"]
    )
    workspaces = [
        item
        for item in workspace_registry.get("workspaces", [])
        if item.get("workspace_id") == workspace_id
    ]
    if len(workspaces) != 1:
        raise ValueError("historical W004 workspace must occur exactly once")

    approvals = [
        item
        for item in _git_jsonl(history_repo, ref, paths["approvals"])
        if item.get("experiment_id") == experiment_id
        or item.get("workspace_id") == workspace_id
    ]
    attempts = [
        item
        for item in _git_jsonl(history_repo, ref, paths["attempt_events"])
        if item.get("experiment_id") == experiment_id
        or item.get("workspace_id") == workspace_id
    ]
    imports = [
        item
        for item in _git_jsonl(history_repo, ref, paths["result_import_events"])
        if item.get("experiment_id") == experiment_id
        or item.get("workspace_id") == workspace_id
    ]
    wanted_science = set(paths["scientific_record_ids"])
    scientific = [
        item
        for item in _git_jsonl(history_repo, ref, paths["scientific_records"])
        if item.get("record_id") in wanted_science
    ]
    if len(approvals) != 2:
        raise ValueError(f"expected 2 historical W004 approvals, got {len(approvals)}")
    if len(attempts) != 15:
        raise ValueError(f"expected 15 historical W004 attempt events, got {len(attempts)}")
    if len(imports) != 1 or imports[0].get("attempt_id") != "A001":
        raise ValueError("historical W004 import truth must contain only A001")
    if {item.get("record_id") for item in scientific} != wanted_science:
        raise ValueError("historical W004 scientific explore records are incomplete")

    contract = manifest["critical_contract"]
    contract_raw = _git_bytes(history_repo, ref, contract["contract_path"])
    contract_input_raw = _git_bytes(history_repo, ref, contract["input_path"])
    contract_value = json.loads(contract_raw.decode("utf-8"))
    contract_input = json.loads(contract_input_raw.decode("utf-8"))
    expected_contract_sha = str(contract["canonical_sha256"])
    if _sha256_json(contract_input) != expected_contract_sha:
        raise ValueError("historical W004 critical contract input SHA-256 mismatch")
    if contract_value.get("canonical_sha256") != expected_contract_sha:
        raise ValueError("historical W004 critical contract binding mismatch")
    without_digest = {
        key: value
        for key, value in contract_value.items()
        if key != "canonical_sha256"
    }
    if _canonical(without_digest) != _canonical(contract_input):
        raise ValueError("historical W004 contract does not match its canonical input")

    jobs = {
        item["arm"]: _git_json(
            history_repo, ref, str(item["history_job_path"])
        )
        for item in manifest["bundles"]
    }
    return {
        "experiment": experiments[0],
        "workspace": workspaces[0],
        "approvals": approvals,
        "attempts": attempts,
        "imports": imports,
        "scientific": scientific,
        "contract_raw": contract_raw,
        "contract_input_raw": contract_input_raw,
        "jobs": jobs,
    }


def _metric_means(bundle_path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    metric_artifact = next(
        (item for item in result["artifacts"] if item.get("kind") == "metrics"),
        None,
    )
    if metric_artifact is None:
        raise ValueError("W004 result bundle has no metrics artifact")
    relative = Path(*PurePosixPath(str(metric_artifact["path"])).parts)
    metrics = _read_json(bundle_path / relative)
    if metrics.get("raw_mpp2_claim") is not False:
        raise ValueError("W004 metrics raw_mpp2_claim must be false")
    if metrics.get("evidence_status") != "COMPATIBILITY_ONLY/pending_review":
        raise ValueError("W004 metrics evidence boundary mismatch")
    rows = metrics.get("per_fit")
    if not isinstance(rows, list):
        raise ValueError("W004 metrics per_fit must be an array")
    output: dict[str, Any] = {}
    for task in ("pCR", "MPR"):
        selected = [
            row
            for row in rows
            if str(row.get("task", "")).lower() == task.lower()
        ]
        if len(selected) != 20:
            raise ValueError(f"W004 {task} must contain 20 per-fit records")
        output[task] = {
            metric: statistics.fmean(
                float(row["test"][metric]) for row in selected
            )
            for metric in ("auc", "auprc", "brier")
        }
    return output


def _validate_bundles(
    evidence_root: Path,
    history: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members = []
    for spec in manifest["bundles"]:
        bundle_path = evidence_root / Path(
            *PurePosixPath(str(spec["bundle_path"])).parts
        )
        report = validate_result_bundle_v1(evidence_root, bundle_path)
        if report["bundle_sha256"] != spec["bundle_sha256"]:
            raise ValueError(f"W004 {spec['arm']} bundle SHA-256 mismatch")
        result = _read_json(bundle_path / "result.json")
        for field in ("result_id", "job_id", "attempt_id"):
            if result.get(field) != spec[field]:
                raise ValueError(f"W004 {spec['arm']} {field} mismatch")
        expected = {
            "experiment_id": manifest["experiment_id"],
            "workspace_id": manifest["workspace_id"],
            "approval_id": manifest["approval_id"],
            "source_commit": manifest["source_commit"],
            "critical_contract_sha256": manifest["critical_contract"][
                "canonical_sha256"
            ],
            "status": "success",
        }
        for field, value in expected.items():
            if result.get(field) != value:
                raise ValueError(f"W004 {spec['arm']} {field} mismatch")
        job_artifact = next(
            (
                item
                for item in result["artifacts"]
                if item.get("kind") == "job_manifest"
            ),
            None,
        )
        if job_artifact is None:
            raise ValueError(f"W004 {spec['arm']} has no embedded job manifest")
        relative_job = Path(
            *PurePosixPath(str(job_artifact["path"])).parts
        )
        embedded_job = _read_json(bundle_path / relative_job)
        if _canonical(embedded_job) != _canonical(history["jobs"][spec["arm"]]):
            raise ValueError(
                f"W004 {spec['arm']} embedded job does not match history"
            )
        members.append(
            {
                "arm": spec["arm"],
                "role": spec["role"],
                "label": spec["label"],
                "result_id": spec["result_id"],
                "job_id": spec["job_id"],
                "attempt_id": spec["attempt_id"],
                "bundle_path": spec["bundle_path"],
                "bundle_sha256": spec["bundle_sha256"],
                "metrics": _metric_means(bundle_path, result),
            }
        )
    return members


def _validate_contrasts(
    members: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    by_arm = {str(member["arm"]): member for member in members}
    for contrast in manifest["result_group"]["contrasts"]:
        treatment_arm, control_arm = str(contrast["contrast"]).split("-", 1)
        task = str(contrast["task"])
        observed_control = float(by_arm[control_arm]["metrics"][task]["auc"])
        observed_treatment = float(
            by_arm[treatment_arm]["metrics"][task]["auc"]
        )
        observed_delta = observed_treatment - observed_control
        if abs(observed_control - float(contrast["control_value"])) > 1e-12:
            raise ValueError(f"W004 {contrast['contrast']} control AUC mismatch")
        if abs(observed_treatment - float(contrast["treatment_value"])) > 1e-12:
            raise ValueError(f"W004 {contrast['contrast']} treatment AUC mismatch")
        if abs(observed_delta - float(contrast["delta"])) > 1e-12:
            raise ValueError(f"W004 {contrast['contrast']} delta AUC mismatch")


def _find_applied_state(
    root: Path,
    manifest: Mapping[str, Any],
) -> bool:
    registry_path = root / "experiments" / "experiment_registry.json"
    receipt_path = root / Path(
        *PurePosixPath(str(manifest["receipt_path"])).parts
    )
    registry = _read_json(registry_path)
    matching = [
        item
        for item in registry.get("experiments", [])
        if item.get("id") == manifest["experiment_id"]
    ]
    if not matching and not receipt_path.exists():
        return False
    if len(matching) != 1 or not receipt_path.exists():
        raise ValueError("partial W004 reconciliation state requires manual review")
    group_id = str(
        (matching[0].get("result_group") or {}).get("group_id", "")
    )
    receipt = _read_json(receipt_path)
    if (
        group_id != manifest["result_group"]["group_id"]
        or receipt.get("migration_id") != manifest["migration_id"]
        or receipt.get("status") != "applied"
    ):
        raise ValueError("existing W004 reconciliation conflicts with manifest")
    return True


def _missing_records(
    current: Iterable[Mapping[str, Any]],
    incoming: Iterable[Mapping[str, Any]],
    *,
    key: str,
    label: str,
) -> list[dict[str, Any]]:
    current_by_key = {str(item.get(key, "")): dict(item) for item in current}
    missing = []
    for item in incoming:
        item_key = str(item.get(key, ""))
        if not item_key:
            raise ValueError(f"{label} has empty {key}")
        existing = current_by_key.get(item_key)
        if existing is None:
            missing.append(dict(item))
            current_by_key[item_key] = dict(item)
        elif _canonical(existing) != _canonical(item):
            raise ValueError(f"conflicting {label} {key}: {item_key}")
    return missing


def _temporary_payload(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return name


def _restore(path: Path, original: bytes | None) -> None:
    if original is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    temporary = _temporary_payload(path, original)
    os.replace(temporary, path)


def _write_transaction(payloads: Mapping[Path, bytes]) -> None:
    originals = {
        path: path.read_bytes() if path.exists() else None for path in payloads
    }
    temporaries = {
        path: _temporary_payload(path, payload)
        for path, payload in payloads.items()
    }
    replaced: list[Path] = []
    try:
        for path, temporary in temporaries.items():
            os.replace(temporary, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            _restore(path, originals[path])
        for temporary in temporaries.values():
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _build_payloads(
    root: Path,
    manifest: Mapping[str, Any],
    history: Mapping[str, Any],
    members: list[dict[str, Any]],
    recorded_at: str,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    registry_path = root / "experiments" / "experiment_registry.json"
    workspace_path = root / "project_state" / "workspace_registry.json"
    approval_path = root / "project_state" / "experiment_approvals.jsonl"
    attempt_path = root / "project_state" / "attempt_events.jsonl"
    import_path = root / "project_state" / "result_import_events.jsonl"
    science_path = root / "project_state" / "scientific_records.jsonl"
    group_event_path = root / "project_state" / "result_group_events.jsonl"

    registry = _read_json(registry_path)
    registry_before_sha256 = _sha256_json(registry)
    if any(
        item.get("id") == manifest["experiment_id"]
        for item in registry.get("experiments", [])
    ):
        raise ValueError("W004 experiment already exists without migration receipt")

    historical_experiment = dict(history["experiment"])
    decision = {**dict(manifest["decision"]), "created_at": recorded_at}
    summary = dict(decision["decision_summary"])
    summary["statement"] = decision["claim"]["statement"]

    historical_import = dict(history["imports"][0])
    current_import_events = _read_jsonl(import_path)
    historical_import_missing = _missing_records(
        current_import_events,
        [historical_import],
        key="event_id",
        label="historical result import",
    )
    import_by_arm: dict[str, dict[str, Any]] = {}
    current_events = []
    for member in members:
        base = {
            "recorded_at": recorded_at,
            "result_id": member["result_id"],
            "job_id": member["job_id"],
            "experiment_id": manifest["experiment_id"],
            "workspace_id": manifest["workspace_id"],
            "attempt_id": member["attempt_id"],
            "approval_id": manifest["approval_id"],
            "critical_contract_sha256": manifest["critical_contract"][
                "canonical_sha256"
            ],
            "bundle_sha256": member["bundle_sha256"],
            "bundle_hash_profile": manifest["result_group"][
                "bundle_hash_profile"
            ],
            "status": "success",
            "evidence_scope": manifest["evidence_scope"],
        }
        if member["arm"] == "A001":
            event = {
                "event_type": "RESULT_IMPORT_REVERIFIED",
                "event_id": f"reverify-{member['result_id']}-v001",
                **base,
                "previous_import_event_id": historical_import["event_id"],
                "previous_bundle_sha256": historical_import["bundle_sha256"],
            }
        else:
            event = {
                "event_type": "RESULT_IMPORT_VERIFIED",
                "event_id": f"import-{member['result_id']}",
                **base,
            }
        current_events.append(event)
        import_by_arm[member["arm"]] = event

    import_missing = _missing_records(
        [*current_import_events, *historical_import_missing],
        current_events,
        key="event_id",
        label="current result import",
    )
    enriched_members = [
        {
            **member,
            "import_event_id": import_by_arm[member["arm"]]["event_id"],
        }
        for member in members
    ]
    result_ids = [member["result_id"] for member in enriched_members]
    group = {
        "schema_version": "1.0",
        "group_id": manifest["result_group"]["group_id"],
        "member_result_ids": result_ids,
        "member_results": enriched_members,
        "approval_id": manifest["approval_id"],
        "critical_contract_sha256": manifest["critical_contract"][
            "canonical_sha256"
        ],
        "evidence_scope": manifest["evidence_scope"],
        "bundle_hash_profile": manifest["result_group"][
            "bundle_hash_profile"
        ],
        "contrasts": list(manifest["result_group"]["contrasts"]),
    }

    historical_experiment.update(
        {
            "status": "done",
            "protocol_revision": 2,
            "active_attempt_id": None,
            "result_group": group,
            "result_ids": result_ids,
            "evidence_status": "accepted",
            "evidence_scope": "COMPATIBILITY_ONLY",
            "raw_mpp2_claim": False,
            "provenance_complete": True,
            "decision_summary": summary,
            "conclusion": decision["claim"]["statement"],
            "limitations": list(manifest["limitations"]),
            "next_action": "open_new_patient_independent_protocol",
            "accepted_at": recorded_at,
            "updated_at": recorded_at,
        }
    )
    historical_experiment.pop("result_id", None)
    registry["experiments"].append(historical_experiment)
    registry["updated_at"] = recorded_at
    registry_after_sha256 = _sha256_json(registry)

    workspace_registry = _read_json(workspace_path)
    workspaces = [
        item
        for item in workspace_registry.get("workspaces", [])
        if item.get("workspace_id") == manifest["workspace_id"]
    ]
    if len(workspaces) != 1:
        raise ValueError("current W004 workspace must occur exactly once")
    workspace = workspaces[0]
    if workspace.get("experiment_id") != manifest["experiment_id"]:
        raise ValueError("current W004 workspace experiment binding mismatch")
    historical_host = history["workspace"].get("hosts", {}).get("local", {})
    current_host = workspace.get("hosts", {}).get("local", {})
    for field in ("branch", "relative_path", "current_source_commit"):
        if current_host.get(field) != historical_host.get(field):
            raise ValueError(f"current W004 local host {field} mismatch")
    workspace.update(
        {
            "protocol_revision": 2,
            "status": "shadow",
            "active_attempt_id": None,
            "lease": None,
            "last_verified_at": recorded_at,
            "close_eligibility": "blocked_pending_worktree_review",
        }
    )
    workspace_registry["updated_at"] = recorded_at

    approvals = _read_jsonl(approval_path)
    approval_missing = _missing_records(
        approvals,
        history["approvals"],
        key="approval_id",
        label="historical approval",
    )
    attempts = _read_jsonl(attempt_path)
    attempt_missing = _missing_records(
        attempts,
        history["attempts"],
        key="event_id",
        label="historical attempt event",
    )
    science = _read_jsonl(science_path)
    history_science_missing = _missing_records(
        science,
        history["scientific"],
        key="record_id",
        label="historical scientific record",
    )
    projections = _decision_projection(decision)
    projection_missing = _missing_records(
        [*science, *history_science_missing],
        projections,
        key="record_id",
        label="grouped science decision",
    )

    group_events = _read_jsonl(group_event_path)
    group_event = {
        "schema_version": "1.0",
        "event_id": manifest["result_group"]["acceptance_event_id"],
        "event_type": "GROUPED_RESULT_ACCEPTED",
        "recorded_at": recorded_at,
        "experiment_id": manifest["experiment_id"],
        "workspace_id": manifest["workspace_id"],
        **group,
        "decision_summary": summary,
        "scientific_record_ids": [
            item["record_id"] for item in projections
        ],
        "confirmation_source": manifest["confirmation_source"],
        "registry_before_sha256": registry_before_sha256,
        "registry_after_sha256": registry_after_sha256,
    }
    group_missing = _missing_records(
        group_events,
        [group_event],
        key="event_id",
        label="result group event",
    )

    receipt = {
        "schema_version": "1.0",
        "migration_id": manifest["migration_id"],
        "status": "applied",
        "recorded_at": recorded_at,
        "experiment_id": manifest["experiment_id"],
        "workspace_id": manifest["workspace_id"],
        "group_id": group["group_id"],
        "confirmation_source": manifest["confirmation_source"],
        "evidence_scope": manifest["evidence_scope"],
        "history_ref": manifest["history_ref"],
        "restored_counts": {
            "approvals": len(history["approvals"]),
            "attempt_events": len(history["attempts"]),
            "historical_import_events": len(history["imports"]),
            "historical_scientific_records": len(history["scientific"]),
        },
        "verification": {
            "bundle_hash_profile": manifest["result_group"][
                "bundle_hash_profile"
            ],
            "bundle_sha256": {
                member["arm"]: member["bundle_sha256"]
                for member in enriched_members
            },
            "critical_contract_sha256": manifest["critical_contract"][
                "canonical_sha256"
            ],
            "embedded_historical_job_matches": 4,
            "dispatch_job_files_restored": False,
        },
        "transaction_state": {
            "registry_before_sha256": registry_before_sha256,
            "registry_after_sha256": registry_after_sha256,
        },
        "result_ids": result_ids,
        "limitations": list(manifest["limitations"]),
    }

    contract = manifest["critical_contract"]
    receipt_path = root / Path(
        *PurePosixPath(str(manifest["receipt_path"])).parts
    )
    payloads = {
        registry_path: _json_bytes(registry),
        workspace_path: _json_bytes(workspace_registry),
        approval_path: _append_jsonl_bytes(
            approval_path.read_bytes() if approval_path.exists() else None,
            approval_missing,
        ),
        attempt_path: _append_jsonl_bytes(
            attempt_path.read_bytes() if attempt_path.exists() else None,
            attempt_missing,
        ),
        import_path: _append_jsonl_bytes(
            import_path.read_bytes() if import_path.exists() else None,
            [*historical_import_missing, *import_missing],
        ),
        science_path: _append_jsonl_bytes(
            science_path.read_bytes() if science_path.exists() else None,
            [*history_science_missing, *projection_missing],
        ),
        group_event_path: _append_jsonl_bytes(
            group_event_path.read_bytes() if group_event_path.exists() else None,
            group_missing,
        ),
        root
        / Path(*PurePosixPath(str(contract["contract_path"])).parts): history[
            "contract_raw"
        ],
        root
        / Path(*PurePosixPath(str(contract["input_path"])).parts): history[
            "contract_input_raw"
        ],
        receipt_path: _json_bytes(receipt),
    }
    return payloads, receipt


def reconcile_w004(
    root: Path,
    manifest_path: Path,
    *,
    mode: str,
    evidence_root: Path | None = None,
    history_repo: Path | None = None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Check or atomically apply the explicitly scoped W004 reconciliation."""
    if mode not in MODES:
        raise ValueError("W004 reconciliation mode must be check or apply")
    root = root.resolve()
    evidence_root = (evidence_root or root).resolve()
    history_repo = (history_repo or evidence_root).resolve()
    manifest = _read_json(manifest_path.resolve())
    _validate_manifest(manifest)
    history = _load_history(history_repo, manifest)
    members = _validate_bundles(evidence_root, history, manifest)
    _validate_contrasts(members, manifest)
    decision_for_validation = {
        **dict(manifest["decision"]),
        "created_at": recorded_at or "check-only",
    }
    validate_against_schema(
        decision_for_validation,
        evidence_root
        / "project_state"
        / "schemas"
        / "science_decision_v1.schema.json",
        "science_decision_v1",
    )
    summary = decision_for_validation["decision_summary"]
    if abs(
        float(summary["treatment_value"])
        - float(summary["control_value"])
        - float(summary["delta"])
    ) > 1e-12:
        raise ValueError("W004 grouped decision delta is inconsistent")

    base_report = {
        "migration_id": manifest["migration_id"],
        "validated_bundle_count": len(members),
        "historical_approval_count": len(history["approvals"]),
        "historical_attempt_event_count": len(history["attempts"]),
        "historical_import_event_count": len(history["imports"]),
        "historical_scientific_record_count": len(history["scientific"]),
        "historical_job_match_count": len(history["jobs"]),
        "group_id": manifest["result_group"]["group_id"],
        "evidence_scope": manifest["evidence_scope"],
    }
    if _find_applied_state(root, manifest):
        return {"status": "already_applied", **base_report}
    if mode == "check":
        return {"status": "ready", **base_report}

    applied_at = recorded_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    with state_lock(root):
        if _find_applied_state(root, manifest):
            return {"status": "already_applied", **base_report}
        payloads, receipt = _build_payloads(
            root,
            manifest,
            history,
            members,
            applied_at,
        )
        _write_transaction(payloads)
    return {"status": "applied", **base_report, "receipt": receipt}
