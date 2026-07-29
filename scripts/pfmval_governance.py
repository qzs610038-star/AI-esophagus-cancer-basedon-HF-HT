"""Local workflow-governance primitives for PFMval.

The module owns deterministic local registries and validation only.  It does
not contact a server, run training, create worktrees, or remove assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


WORKSPACE_SCHEMA_VERSION = "1.0"
WORKSPACE_ID_RE = re.compile(r"^W([0-9]{3,})$")
ATTEMPT_ID_RE = re.compile(r"^A([0-9]{3,})$")
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
E1_CHANGE_CLASSES = {
    "registered_path",
    "path_separator",
    "encoding",
    "interpreter_bootstrap",
    "launcher_argument_compatibility",
    "logging",
    "exit_code",
    "packaging",
    "gitee_return",
}
JOB_V2_COMMANDS = {
    "state_preflight",
    "cache_parity",
    "standard_training",
    "mpp_pathway_ridge_calibration",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"JSONL event must be an object at {path}:{line_number}")
        events.append(event)
    return events


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _refresh_experiment_views(root: Path) -> None:
    from scripts.pfmval_views import refresh_experiment_views

    refresh_experiment_views(root)


def _update_experiment_lifecycle_view(
    root: Path,
    *,
    experiment_id: str,
    updates: Mapping[str, Any],
) -> None:
    registry_path = root / "experiments" / "experiment_registry.json"
    registry = _read_json(registry_path)
    experiment = next(
        (
            item
            for item in registry.get("experiments", [])
            if item.get("id") == experiment_id
        ),
        None,
    )
    if experiment is None:
        raise ValueError(f"unknown experiment for view refresh: {experiment_id}")
    now = _utc_now()
    experiment.update(dict(updates))
    experiment["updated_at"] = now
    registry["updated_at"] = now
    _write_json_atomic(registry_path, registry)
    _refresh_experiment_views(root)


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False))).replace("\\", "/")


def _workspace_registry_path(root: Path) -> Path:
    return root / "project_state" / "workspace_registry.json"


def _registered_local_worktree(
    root: Path,
    *,
    path: Path,
    branch: str,
    source_commit: str,
    require_source_commit: bool = True,
) -> Path:
    """Return a verified local linked-worktree path without contacting a remote."""
    if not branch:
        raise ValueError("external local worktree requires an explicit branch")
    if require_source_commit and not source_commit:
        raise ValueError("external local worktree requires a source_commit")
    completed = subprocess.run(
        ["git", "-C", str(root), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    records: list[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
    if current:
        records.append(current)
    resolved = path.resolve(strict=True)
    match = next(
        (
            item
            for item in records
            if _normalized_path(Path(item.get("path", ""))) == _normalized_path(resolved)
        ),
        None,
    )
    if match is None:
        raise ValueError("external path is not a local Git worktree of the registered repository")
    if match.get("branch") != branch:
        raise ValueError("external local worktree branch does not match requested branch")
    if source_commit and match.get("head") != source_commit:
        raise ValueError("external local worktree HEAD does not match source_commit")
    return resolved


def initialize_workspace_registry(root: Path) -> Dict[str, Any]:
    path = _workspace_registry_path(root)
    if path.exists():
        return _read_json(path)
    registry = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "next_workspace_number": 1,
        "workspaces": [],
    }
    _write_json_atomic(path, registry)
    return registry


def allocate_workspace(
    root: Path,
    *,
    experiment_id: str,
    display_name: str,
    local_relative_path: str,
    branch: str = "",
    source_commit: str = "",
    protocol_revision: int = 1,
    workspace_kind: str = "experiment",
) -> Dict[str, Any]:
    if workspace_kind not in {"experiment", "governance_maintenance"}:
        raise ValueError("invalid workspace_kind")
    registry_path = _workspace_registry_path(root)
    registry = initialize_workspace_registry(root)
    if any(
        item.get("experiment_id") == experiment_id
        for item in registry.get("workspaces", [])
    ):
        raise ValueError(f"experiment {experiment_id} already has workspace")

    existing_numbers = []
    for item in registry.get("workspaces", []):
        match = WORKSPACE_ID_RE.fullmatch(str(item.get("workspace_id", "")))
        if match:
            existing_numbers.append(int(match.group(1)))
    workspace_number = max(
        [int(registry.get("next_workspace_number", 1)) - 1, *existing_numbers],
        default=0,
    ) + 1
    workspace_id = f"W{workspace_number:03d}"

    declared_path = Path(local_relative_path)
    if declared_path.is_absolute():
        absolute_path = _registered_local_worktree(
            root,
            path=declared_path,
            branch=branch,
            source_commit=source_commit,
            require_source_commit=workspace_kind == "experiment",
        )
        stored_path = str(absolute_path)
    else:
        if ".." in declared_path.parts:
            raise ValueError("workspace path is outside registered root")
        absolute_path = (root / declared_path).resolve(strict=False)
        if not absolute_path.is_relative_to(root.resolve(strict=False)):
            raise ValueError("workspace path is outside registered root")
        stored_path = declared_path.as_posix()
    if absolute_path.name != workspace_id:
        raise ValueError(
            f"workspace physical path must end with allocated id {workspace_id}"
        )
    if source_commit and not FULL_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("workspace source_commit must be a full 40-character SHA")

    now = _utc_now()
    workspace = {
        "workspace_kind": workspace_kind,
        "workspace_id": workspace_id,
        "display_name": display_name,
        "experiment_id": experiment_id,
        "protocol_revision": protocol_revision,
        "status": "shadow",
        "active_attempt_id": None,
        "lease": None,
        "opened_at": now,
        "last_verified_at": None,
        "close_eligibility": "not_evaluated",
        "retention_policy": "retain_until_explicit_close_approval",
        "hosts": {
            "local": {
                "host_scope": "local",
                "branch": branch,
                "path_id": "local_experiment_workspaces",
                "relative_path": stored_path,
                "current_source_commit": source_commit,
            }
        },
    }
    registry["workspaces"].append(workspace)
    registry["next_workspace_number"] = workspace_number + 1
    registry["updated_at"] = now
    _write_json_atomic(registry_path, registry)
    return workspace


def scan_workspace_registry(
    root: Path,
    *,
    host_scope: str,
    observed_worktrees: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    if host_scope not in {"local", "server"}:
        raise ValueError("host_scope must be local or server")
    registry = initialize_workspace_registry(root)
    observed = [dict(item) for item in observed_worktrees]
    observed_by_path = {
        _normalized_path(Path(str(item["path"]))): item for item in observed
    }
    registered_paths: set[str] = set()
    rows = []
    failures = []
    for workspace in registry.get("workspaces", []):
        host = workspace.get("hosts", {}).get(host_scope)
        if not host:
            continue
        declared_path = Path(str(host["relative_path"]))
        absolute_path = (root / declared_path).resolve(strict=False)
        if not declared_path.is_absolute() and not absolute_path.is_relative_to(root.resolve(strict=False)):
            failures.append(
                {
                    "workspace_id": workspace["workspace_id"],
                    "severity": "FAIL",
                    "reason": "registered path escapes root",
                }
            )
            continue
        normalized = _normalized_path(absolute_path)
        registered_paths.add(normalized)
        actual = observed_by_path.get(normalized)
        rows.append(
            {
                "workspace_id": workspace["workspace_id"],
                "experiment_id": workspace["experiment_id"],
                "path": str(absolute_path),
                "observed": actual is not None,
                "branch": actual.get("branch", "") if actual else "",
                "head": actual.get("head", "") if actual else "",
                "dirty": bool(actual.get("dirty", False)) if actual else None,
            }
        )
    unregistered = [
        {
            "path": str(item["path"]),
            "severity": "WARN",
            "reason": "unregistered worktree observed in shadow scan",
        }
        for item in observed
        if _normalized_path(Path(str(item["path"]))) not in registered_paths
    ]
    status = "FAIL" if failures else ("WARN" if unregistered else "PASS")
    return {
        "host_scope": host_scope,
        "status": status,
        "workspaces": rows,
        "unregistered": unregistered,
        "failures": failures,
    }


def assert_workspace_locus(
    workspace: Mapping[str, Any],
    *,
    cwd: Path,
    branch: str,
    head: str,
    dirty: bool,
) -> None:
    local = workspace.get("hosts", {}).get("local")
    if not local:
        raise ValueError("workspace has no local host record")
    registered = Path(str(local["relative_path"]))
    if not registered.is_absolute():
        registered = cwd.parents[len(registered.parts) - 1] / registered
    if _normalized_path(cwd) != _normalized_path(registered):
        raise ValueError("cwd does not match registered workspace")
    expected_branch = str(local.get("branch", ""))
    if expected_branch and branch != expected_branch:
        raise ValueError("branch does not match registered workspace")
    expected_head = str(local.get("current_source_commit", ""))
    if (
        workspace.get("workspace_kind", "experiment")
        != "governance_maintenance"
        and expected_head
        and head != expected_head
    ):
        raise ValueError("HEAD does not match registered workspace")
    if dirty:
        raise ValueError("workspace is dirty")


def resolve_attempt_run_root(
    run_root: Path,
    *,
    workspace_id: str,
    attempt_id: str,
) -> Path:
    if not WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise ValueError("invalid workspace_id")
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("invalid attempt_id")
    base = run_root.resolve(strict=False)
    target = (base / workspace_id / attempt_id).resolve(strict=False)
    if not target.is_relative_to(base):
        raise ValueError("attempt run root escapes registered root")
    return target


def acquire_workspace_lease(
    root: Path,
    *,
    workspace_id: str,
    experiment_id: str,
    attempt_id: str,
    source_commit: str,
    critical_contract_sha256: str,
    holder: str,
    pid: int,
    run_root: Path,
) -> Dict[str, Any]:
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("invalid attempt_id")
    if not FULL_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("lease source_commit must be a full 40-character SHA")
    if not SHA256_RE.fullmatch(critical_contract_sha256):
        raise ValueError("lease critical contract must be a SHA-256 digest")
    registry_path = _workspace_registry_path(root)
    registry = initialize_workspace_registry(root)
    workspace = next(
        (
            item
            for item in registry.get("workspaces", [])
            if item.get("workspace_id") == workspace_id
        ),
        None,
    )
    if workspace is None:
        raise ValueError(f"unknown workspace_id: {workspace_id}")
    if workspace.get("experiment_id") != experiment_id:
        raise ValueError("workspace experiment_id mismatch")
    if workspace.get("lease") is not None:
        raise ValueError("workspace already has an active lease")
    now = _utc_now()
    lease = {
        "workspace_id": workspace_id,
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "source_commit": source_commit,
        "critical_contract_sha256": critical_contract_sha256,
        "holder": holder,
        "pid": int(pid),
        "run_root": str(run_root.resolve(strict=False)),
        "acquired_at": now,
        "heartbeat_at": now,
        "release_policy": "explicit_only",
    }
    workspace["lease"] = lease
    workspace["active_attempt_id"] = attempt_id
    registry["updated_at"] = now
    _write_json_atomic(registry_path, registry)
    return lease


def release_workspace_lease(
    root: Path,
    *,
    workspace_id: str,
    attempt_id: str,
) -> None:
    registry_path = _workspace_registry_path(root)
    registry = initialize_workspace_registry(root)
    workspace = next(
        (
            item
            for item in registry.get("workspaces", [])
            if item.get("workspace_id") == workspace_id
        ),
        None,
    )
    if workspace is None:
        raise ValueError(f"unknown workspace_id: {workspace_id}")
    lease = workspace.get("lease")
    if lease is None:
        raise ValueError("workspace has no active lease")
    if lease.get("attempt_id") != attempt_id:
        raise ValueError("attempt_id does not own the active lease")
    workspace["lease"] = None
    workspace["active_attempt_id"] = None
    registry["updated_at"] = _utc_now()
    _write_json_atomic(registry_path, registry)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_critical_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "experiment_id",
        "protocol_revision",
        "source_commit",
        "critical_code",
        "resolved_hyperparameters",
        "loss_contract",
        "regularization_contract",
        "feature_flags",
        "data_manifest_id",
        "data_manifest_sha256",
        "input_artifacts",
        "selection_policy",
        "external_eval_policy",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"critical contract missing fields: {', '.join(missing)}")
    source_commit = str(payload["source_commit"])
    if not FULL_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a full 40-character SHA")
    if not SHA256_RE.fullmatch(str(payload["data_manifest_sha256"])):
        raise ValueError("data_manifest_sha256 must be a SHA-256 digest")
    for artifact in payload["input_artifacts"]:
        if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError("every critical input artifact requires SHA-256")
    contract = json.loads(_canonical_json_bytes(dict(payload)).decode("utf-8"))
    contract["canonical_sha256"] = hashlib.sha256(
        _canonical_json_bytes(contract)
    ).hexdigest()
    return contract


def register_governed_experiment(
    root: Path,
    *,
    experiment_id: str,
    display_name: str,
    phase: str,
    run_limit: int,
    critical_contract: Mapping[str, Any],
    local_relative_path: str,
    workspace_branch: str = "",
) -> Dict[str, Any]:
    if run_limit < 1:
        raise ValueError("run_limit must be an explicit positive integer")
    if phase not in {"preflight", "smoke", "formal"}:
        raise ValueError("invalid experiment phase")
    if critical_contract.get("experiment_id") != experiment_id:
        raise ValueError("critical contract experiment_id mismatch")
    if not SHA256_RE.fullmatch(
        str(critical_contract.get("canonical_sha256", ""))
    ):
        raise ValueError("critical contract has no canonical SHA-256")
    registry_path = root / "experiments" / "experiment_registry.json"
    registry = _read_json(registry_path)
    if any(
        item.get("id") == experiment_id
        for item in registry.get("experiments", [])
    ):
        raise ValueError(f"experiment already registered: {experiment_id}")
    workspace = allocate_workspace(
        root,
        experiment_id=experiment_id,
        display_name=display_name,
        local_relative_path=local_relative_path,
        branch=workspace_branch,
        source_commit=str(critical_contract["source_commit"]),
        protocol_revision=int(critical_contract["protocol_revision"]),
    )
    now = _utc_now()
    experiment = {
        "id": experiment_id,
        "display_name": display_name,
        "family": "governed_v3",
        "status": "planned",
        "phase": phase,
        "workspace_id": workspace["workspace_id"],
        "protocol_revision": int(critical_contract["protocol_revision"]),
        "run_limit": int(run_limit),
        "critical_contract_sha256": critical_contract["canonical_sha256"],
        "source_commit": critical_contract["source_commit"],
        "evidence_status": "pending",
        "created_at": now,
        "next_action": "approve_protocol",
    }
    registry.setdefault("experiments", []).append(experiment)
    registry["updated_at"] = now
    _write_json_atomic(registry_path, registry)
    _refresh_experiment_views(root)
    return experiment


def approve_experiment_protocol(
    root: Path,
    *,
    approval_id: str,
    experiment_id: str,
    protocol_revision: int,
    phase: str,
    run_limit: int,
    critical_contract_sha256: str,
    adaptation_policy: str,
    source: str,
) -> Dict[str, Any]:
    if source != "explicit_user_instruction":
        raise ValueError("protocol approval must come from explicit_user_instruction")
    if run_limit < 1:
        raise ValueError("approval run_limit must be positive")
    registry = _read_json(root / "experiments" / "experiment_registry.json")
    experiment = next(
        (
            item
            for item in registry.get("experiments", [])
            if item.get("id") == experiment_id
        ),
        None,
    )
    if experiment is None:
        raise ValueError("approval references unknown experiment")
    comparisons = {
        "protocol_revision": protocol_revision,
        "phase": phase,
        "run_limit": run_limit,
        "critical_contract_sha256": critical_contract_sha256,
    }
    for field, expected in comparisons.items():
        if experiment.get(field) != expected:
            raise ValueError(f"approval mismatch for {field}")
    if adaptation_policy != "e1_allowlist_only":
        raise ValueError("unsupported adaptation policy")
    approvals_path = root / "project_state" / "experiment_approvals.jsonl"
    approvals = _read_jsonl(approvals_path)
    existing = next(
        (item for item in approvals if item.get("approval_id") == approval_id),
        None,
    )
    approval = {
        "event_type": "protocol_approval",
        "approval_id": approval_id,
        "experiment_id": experiment_id,
        "protocol_revision": protocol_revision,
        "phase": phase,
        "run_limit": run_limit,
        "critical_contract_sha256": critical_contract_sha256,
        "adaptation_policy": adaptation_policy,
        "source": source,
        "status": "active",
        "approved_at": _utc_now(),
    }
    if existing:
        comparable_existing = {
            key: value for key, value in existing.items() if key != "approved_at"
        }
        comparable_new = {
            key: value for key, value in approval.items() if key != "approved_at"
        }
        if comparable_existing != comparable_new:
            raise ValueError("approval_id is already bound to a different contract")
        return existing
    _append_jsonl(approvals_path, approval)
    _update_experiment_lifecycle_view(
        root,
        experiment_id=experiment_id,
        updates={
            "approval_id": approval_id,
            "next_action": "prepare_attempt",
        },
    )
    return approval


def revise_experiment_protocol(
    root: Path, *, experiment_id: str, critical_contract: Mapping[str, Any]
) -> Dict[str, Any]:
    """Atomically advance a planned experiment before any started attempt exists."""
    registry = _read_json(root / "experiments" / "experiment_registry.json")
    experiment = next((x for x in registry.get("experiments", []) if x.get("id") == experiment_id), None)
    if experiment is None:
        raise ValueError("revision references unknown experiment")
    revision = int(critical_contract.get("protocol_revision", -1))
    if critical_contract.get("experiment_id") != experiment_id or revision <= int(experiment.get("protocol_revision", 0)):
        raise ValueError("protocol revision must strictly advance the registered experiment")
    if critical_contract.get("source_commit") != experiment.get("source_commit"):
        raise ValueError("protocol revision may not change the registered source_commit")
    events = _read_jsonl(_attempt_log_path(root))
    if any(e.get("experiment_id") == experiment_id and e.get("event_type") != "ATTEMPT_PREPARED" for e in events):
        raise ValueError("protocol revision is forbidden after an attempt starts or terminates")
    workspaces = initialize_workspace_registry(root)
    workspace = next((x for x in workspaces.get("workspaces", []) if x.get("experiment_id") == experiment_id), None)
    if workspace is None:
        raise ValueError("protocol revision has no bound workspace")
    workspace["protocol_revision"] = revision
    workspaces["updated_at"] = _utc_now()
    _write_json_atomic(_workspace_registry_path(root), workspaces)
    _update_experiment_lifecycle_view(root, experiment_id=experiment_id, updates={
        "protocol_revision": revision,
        "critical_contract_sha256": critical_contract["canonical_sha256"],
        "approval_id": None,
        "next_action": "approve_protocol_revision",
    })
    return {"experiment_id": experiment_id, "protocol_revision": revision, "superseded_approval_id": experiment.get("approval_id")}


def _attempt_log_path(root: Path) -> Path:
    return root / "project_state" / "attempt_events.jsonl"


def _attempt_budget(
    approvals: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    approval_id: str,
) -> Dict[str, int]:
    approval = next(
        (
            item
            for item in approvals
            if item.get("approval_id") == approval_id
            and item.get("status") == "active"
        ),
        None,
    )
    if approval is None:
        raise ValueError("attempt references missing or inactive approval")
    prepared = {
        item["attempt_id"]: item
        for item in events
        if item.get("event_type") == "ATTEMPT_PREPARED"
        and item.get("approval_id") == approval_id
    }
    started_ids = {
        item["attempt_id"]
        for item in events
        if item.get("event_type") == "EXPERIMENT_STARTED"
        and item.get("approval_id") == approval_id
    }
    consumed = sum(
        int(prepared[attempt_id]["run_units"])
        for attempt_id in started_ids
        if attempt_id in prepared
    )
    reserved = sum(
        int(item["run_units"])
        for attempt_id, item in prepared.items()
        if attempt_id not in started_ids
    )
    run_limit = int(approval["run_limit"])
    return {
        "run_limit": run_limit,
        "consumed": consumed,
        "reserved": reserved,
        "remaining": run_limit - consumed - reserved,
    }


def prepare_attempt(
    root: Path,
    *,
    experiment_id: str,
    workspace_id: str,
    approval_id: str,
    run_units: int,
    source_commit: str,
    critical_contract_sha256: str,
) -> Dict[str, Any]:
    if run_units < 1:
        raise ValueError("run_units must be positive")
    if not FULL_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("attempt source_commit must be a full 40-character SHA")
    if not SHA256_RE.fullmatch(critical_contract_sha256):
        raise ValueError("attempt critical contract must be SHA-256")
    approvals = _read_jsonl(root / "project_state" / "experiment_approvals.jsonl")
    approval = next(
        (
            item
            for item in approvals
            if item.get("approval_id") == approval_id
            and item.get("status") == "active"
        ),
        None,
    )
    if approval is None:
        raise ValueError("attempt references missing or inactive approval")
    if approval.get("experiment_id") != experiment_id:
        raise ValueError("attempt experiment_id does not match approval")
    registry = _read_json(root / "experiments" / "experiment_registry.json")
    experiment = next((x for x in registry.get("experiments", []) if x.get("id") == experiment_id), None)
    if experiment is None or approval.get("protocol_revision") != experiment.get("protocol_revision"):
        raise ValueError("attempt approval is superseded by the current protocol revision")
    if approval.get("critical_contract_sha256") != critical_contract_sha256:
        raise ValueError("attempt critical contract does not match approval")
    workspaces = initialize_workspace_registry(root).get("workspaces", [])
    workspace = next(
        (item for item in workspaces if item.get("workspace_id") == workspace_id),
        None,
    )
    if workspace is None or workspace.get("experiment_id") != experiment_id:
        raise ValueError("attempt workspace binding mismatch")
    local_binding = workspace.get("hosts", {}).get("local", {})
    bound_commit = str(local_binding.get("current_source_commit", ""))
    if bound_commit and bound_commit != source_commit:
        raise ValueError("attempt source_commit does not match workspace binding")
    bound_branch = str(local_binding.get("branch", ""))
    events_path = _attempt_log_path(root)
    events = _read_jsonl(events_path)
    budget = _attempt_budget(approvals, events, approval_id=approval_id)
    if run_units > budget["remaining"]:
        raise ValueError("run budget exhausted")
    attempt_numbers = [
        int(match.group(1))
        for item in events
        if item.get("event_type") == "ATTEMPT_PREPARED"
        and item.get("workspace_id") == workspace_id
        and (match := ATTEMPT_ID_RE.fullmatch(str(item.get("attempt_id", ""))))
    ]
    attempt_id = f"A{max(attempt_numbers, default=0) + 1:03d}"
    event = {
        "event_type": "ATTEMPT_PREPARED",
        "event_id": f"prepare-{workspace_id}-{attempt_id}",
        "recorded_at": _utc_now(),
        "experiment_id": experiment_id,
        "workspace_id": workspace_id,
        "attempt_id": attempt_id,
        "job_id": f"{workspace_id}-{attempt_id}",
        "approval_id": approval_id,
        "protocol_revision": approval["protocol_revision"],
        "phase": approval["phase"],
        "run_units": int(run_units),
        "source_commit": source_commit,
        "critical_contract_sha256": critical_contract_sha256,
        "workspace_branch": bound_branch or f"automation/local/{workspace_id}/{attempt_id}",
    }
    _append_jsonl(events_path, event)
    _update_experiment_lifecycle_view(
        root,
        experiment_id=experiment_id,
        updates={
            "active_attempt_id": attempt_id,
            "next_action": "validate_attempt",
        },
    )
    return {
        **event,
        "budget": {
            **budget,
            "consumed_after_event": budget["consumed"],
            "remaining_after_prepare": budget["remaining"] - run_units,
        },
    }


def record_attempt_event(
    root: Path,
    *,
    workspace_id: str,
    attempt_id: str,
    event_id: str,
    event_type: str,
) -> Dict[str, Any]:
    allowed_events = {
        "PREFLIGHT_FAILED",
        "VALIDATED",
        "EXPERIMENT_STARTED",
        "ATTEMPT_FAILED",
        "RESULT_RETURNED",
        "IMPORTED",
        "REJECTED",
    }
    if event_type not in allowed_events:
        raise ValueError("unsupported attempt event")
    events_path = _attempt_log_path(root)
    events = _read_jsonl(events_path)
    existing = next(
        (item for item in events if item.get("event_id") == event_id),
        None,
    )
    approvals = _read_jsonl(root / "project_state" / "experiment_approvals.jsonl")
    prepared = next(
        (
            item
            for item in events
            if item.get("event_type") == "ATTEMPT_PREPARED"
            and item.get("workspace_id") == workspace_id
            and item.get("attempt_id") == attempt_id
        ),
        None,
    )
    if prepared is None:
        raise ValueError("attempt event has no prepared attempt")
    if existing:
        if (
            existing.get("workspace_id") != workspace_id
            or existing.get("attempt_id") != attempt_id
            or existing.get("event_type") != event_type
        ):
            raise ValueError("event_id is already bound to different content")
        budget = _attempt_budget(
            approvals,
            events,
            approval_id=str(prepared["approval_id"]),
        )
        return {"status": "already_recorded", "event": existing, "budget": budget}
    if event_type == "EXPERIMENT_STARTED" and any(
        item.get("event_type") == "EXPERIMENT_STARTED"
        and item.get("workspace_id") == workspace_id
        and item.get("attempt_id") == attempt_id
        for item in events
    ):
        raise ValueError("attempt already crossed EXPERIMENT_STARTED")
    event = {
        "event_type": event_type,
        "event_id": event_id,
        "recorded_at": _utc_now(),
        "experiment_id": prepared["experiment_id"],
        "workspace_id": workspace_id,
        "attempt_id": attempt_id,
        "job_id": prepared["job_id"],
        "approval_id": prepared["approval_id"],
        "run_units": prepared["run_units"],
    }
    _append_jsonl(events_path, event)
    updates_by_event = {
        "PREFLIGHT_FAILED": {
            "status": "planned",
            "next_action": "fix_preflight_without_consuming_run_units",
        },
        "VALIDATED": {
            "status": "planned",
            "next_action": "dispatch_or_start_attempt",
        },
        "EXPERIMENT_STARTED": {
            "status": "running",
            "next_action": "await_result",
        },
        "ATTEMPT_FAILED": {
            "status": "failed",
            "next_action": "review_failure_before_new_attempt",
        },
        "RESULT_RETURNED": {
            "status": "running",
            "next_action": "verify_and_import_result",
        },
        "IMPORTED": {
            "status": "done",
            "next_action": "review_close_readiness",
        },
        "REJECTED": {
            "status": "failed",
            "evidence_status": "rejected",
            "next_action": "review_rejection",
        },
    }
    _update_experiment_lifecycle_view(
        root,
        experiment_id=str(prepared["experiment_id"]),
        updates=updates_by_event[event_type],
    )
    updated_events = [*events, event]
    budget = _attempt_budget(
        approvals,
        updated_events,
        approval_id=str(prepared["approval_id"]),
    )
    return {"status": "recorded", "event": event, "budget": budget}


def record_e1_adaptation(
    root: Path,
    *,
    adaptation_record_id: str,
    workspace_id: str,
    attempt_id: str,
    previous_source_commit: str,
    new_source_commit: str,
    previous_dispatch_revision: int,
    new_dispatch_revision: int,
    critical_contract_before: str,
    critical_contract_after: str,
    changed_items: Iterable[Mapping[str, Any]],
    tests: Iterable[str],
) -> Dict[str, Any]:
    if not WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise ValueError("invalid workspace_id")
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("invalid attempt_id")
    if not FULL_COMMIT_RE.fullmatch(previous_source_commit) or not FULL_COMMIT_RE.fullmatch(
        new_source_commit
    ):
        raise ValueError("adaptation commits must be full 40-character SHAs")
    if not SHA256_RE.fullmatch(critical_contract_before) or not SHA256_RE.fullmatch(
        critical_contract_after
    ):
        raise ValueError("adaptation critical contracts must be SHA-256")
    if critical_contract_before != critical_contract_after:
        raise ValueError("critical contract changed; new approval is required")
    if new_dispatch_revision <= previous_dispatch_revision:
        raise ValueError("dispatch revision must increase")
    normalized_items = [dict(item) for item in changed_items]
    if not normalized_items:
        raise ValueError("adaptation must identify changed items")
    for item in normalized_items:
        path = str(item.get("path", "")).replace("\\", "/")
        change_class = str(item.get("change_class", ""))
        path_allowed = (
            path.startswith("deploy/")
            or path.startswith("automation/")
            or path in {"configs/server_paths.yaml", "path_registry.py"}
        )
        if change_class not in E1_CHANGE_CLASSES or not path_allowed:
            raise ValueError(
                f"change is not E1-allowlisted: {path} ({change_class})"
            )
    normalized_tests = [str(item) for item in tests if str(item).strip()]
    if not normalized_tests:
        raise ValueError("E1 adaptation requires targeted tests")
    path = root / "project_state" / "adaptation_records.jsonl"
    existing = next(
        (
            item
            for item in _read_jsonl(path)
            if item.get("adaptation_record_id") == adaptation_record_id
        ),
        None,
    )
    record = {
        "event_type": "e1_adaptation",
        "adaptation_record_id": adaptation_record_id,
        "workspace_id": workspace_id,
        "attempt_id": attempt_id,
        "previous_source_commit": previous_source_commit,
        "new_source_commit": new_source_commit,
        "previous_dispatch_revision": previous_dispatch_revision,
        "new_dispatch_revision": new_dispatch_revision,
        "critical_contract_sha256": critical_contract_before,
        "changed_items": normalized_items,
        "tests": normalized_tests,
        "status": "recorded",
        "recorded_at": _utc_now(),
    }
    if existing:
        comparable_existing = {
            key: value for key, value in existing.items() if key != "recorded_at"
        }
        comparable_new = {
            key: value for key, value in record.items() if key != "recorded_at"
        }
        if comparable_existing != comparable_new:
            raise ValueError(
                "adaptation_record_id is already bound to different content"
            )
        return existing
    _append_jsonl(path, record)
    return record


def build_job_v2(
    prepared_attempt: Mapping[str, Any],
    *,
    command_id: str,
    path_ids: Iterable[str],
    resolved_argv: Iterable[str],
    input_binding: Mapping[str, Any],
    adaptation_record_id: str | None = None,
    dispatch_revision: int = 1,
) -> Dict[str, Any]:
    workspace_id = str(prepared_attempt["workspace_id"])
    attempt_id = str(prepared_attempt["attempt_id"])
    job = {
        "schema_version": "2.0",
        "job_id": str(prepared_attempt["job_id"]),
        "experiment_id": str(prepared_attempt["experiment_id"]),
        "workspace_id": workspace_id,
        "attempt_id": attempt_id,
        "protocol_revision": int(prepared_attempt["protocol_revision"]),
        "approval_id": str(prepared_attempt["approval_id"]),
        "critical_contract_sha256": str(
            prepared_attempt["critical_contract_sha256"]
        ),
        "source_commit": str(prepared_attempt["source_commit"]),
        "phase": str(prepared_attempt["phase"]),
        "command_id": command_id,
        "path_ids": list(path_ids),
        "resolved_argv": list(resolved_argv),
        "input_binding": dict(input_binding),
        "run_units": int(prepared_attempt["run_units"]),
        "workspace_branch": str(prepared_attempt["workspace_branch"]),
        "dispatch_revision": int(dispatch_revision),
        "adaptation_record_id": adaptation_record_id,
        "created_at": _utc_now(),
        "dispatch_branch": str(prepared_attempt["workspace_branch"]),
        "result_branch": f"automation/server/{workspace_id}/{attempt_id}",
        "artifact_policy": {
            "critical": "sha256_required",
            "supporting": "size_inventory_default",
            "diagnostic": "non_evidence",
            "large_artifacts": "registered_path_size_sha256_only",
        },
    }
    validate_job_v2(job)
    return job


def validate_job_v2(job: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "job_id",
        "experiment_id",
        "workspace_id",
        "attempt_id",
        "protocol_revision",
        "approval_id",
        "critical_contract_sha256",
        "source_commit",
        "phase",
        "command_id",
        "path_ids",
        "resolved_argv",
        "input_binding",
        "run_units",
        "workspace_branch",
        "dispatch_revision",
        "created_at",
        "dispatch_branch",
        "result_branch",
        "artifact_policy",
    }
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"job v2 missing fields: {', '.join(missing)}")
    if job["schema_version"] != "2.0":
        raise ValueError("job v2 schema_version must be 2.0")
    if not WORKSPACE_ID_RE.fullmatch(str(job["workspace_id"])):
        raise ValueError("invalid workspace_id")
    if not ATTEMPT_ID_RE.fullmatch(str(job["attempt_id"])):
        raise ValueError("invalid attempt_id")
    if not FULL_COMMIT_RE.fullmatch(str(job["source_commit"])):
        raise ValueError("job source_commit must be a full 40-character SHA")
    if not SHA256_RE.fullmatch(str(job["critical_contract_sha256"])):
        raise ValueError("job critical contract must be SHA-256")
    if int(job["run_units"]) < 1:
        raise ValueError("job run_units must be positive")
    if int(job["protocol_revision"]) < 1 or int(job["dispatch_revision"]) < 1:
        raise ValueError("job revisions must be positive")
    if job["phase"] not in {"preflight", "smoke", "formal"}:
        raise ValueError("invalid job phase")
    if job["command_id"] not in JOB_V2_COMMANDS:
        raise ValueError("job command_id is not allowlisted")
    if not isinstance(job["resolved_argv"], list) or not job["resolved_argv"]:
        raise ValueError("job resolved_argv must be a non-empty list")
    if any(not isinstance(item, str) or "\x00" in item for item in job["resolved_argv"]):
        raise ValueError("job resolved_argv contains unsafe values")
    input_binding = job["input_binding"]
    if not isinstance(input_binding, Mapping):
        raise ValueError("job input_binding must be an object")
    if not SHA256_RE.fullmatch(str(input_binding.get("data_manifest_sha256", ""))):
        raise ValueError("job input binding requires data manifest SHA-256")
    for artifact in input_binding.get("artifacts", []):
        if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError("job input artifact requires SHA-256")
    workspace_id = str(job["workspace_id"])
    attempt_id = str(job["attempt_id"])
    if job["job_id"] != f"{workspace_id}-{attempt_id}":
        raise ValueError("job_id must bind workspace_id and attempt_id")
    if job["result_branch"] != f"automation/server/{workspace_id}/{attempt_id}":
        raise ValueError("result branch must be attempt-unique server-owned ref")
    execution_binding = job.get("execution_binding")
    if execution_binding is not None:
        if not isinstance(execution_binding, Mapping):
            raise ValueError("job execution_binding must be an object")
        if execution_binding.get("mode") != "source_plus_governance_bundle":
            raise ValueError("unsupported job execution binding mode")
        if not FULL_COMMIT_RE.fullmatch(str(execution_binding.get("governance_commit", ""))):
            raise ValueError("job execution binding requires governance_commit SHA")


def _safe_artifact_path(value: str) -> bool:
    path = value.replace("\\", "/")
    parts = Path(path).parts
    return bool(path) and not path.startswith("/") and ".." not in parts and "\x00" not in path


def build_result_v2(
    job: Mapping[str, Any],
    *,
    result_id: str,
    status: str,
    artifacts: Iterable[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    metric_artifact_ids: Iterable[str],
    large_artifacts: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    result = {
        "schema_version": "2.0",
        "result_id": result_id,
        "job_id": str(job["job_id"]),
        "experiment_id": str(job["experiment_id"]),
        "workspace_id": str(job["workspace_id"]),
        "attempt_id": str(job["attempt_id"]),
        "protocol_revision": int(job["protocol_revision"]),
        "approval_id": str(job["approval_id"]),
        "critical_contract_sha256": str(job["critical_contract_sha256"]),
        "source_commit": str(job["source_commit"]),
        "phase": str(job["phase"]),
        "status": status,
        "created_at": _utc_now(),
        "run_units": int(job["run_units"]),
        "artifacts": [
            {
                **dict(item),
                "source_attempt_id": str(job["attempt_id"]),
            }
            for item in artifacts
        ],
        "metrics": dict(metrics),
        "metric_artifact_ids": list(metric_artifact_ids),
        "large_artifacts": [dict(item) for item in large_artifacts],
    }
    validate_result_v2(result)
    return result


def validate_result_v2(result: Mapping[str, Any]) -> None:
    from scripts.pfmval_result_bundle import validate_result_manifest_v1

    validate_result_manifest_v1(Path(__file__).resolve().parents[1], result)


def record_result_import_v2(
    root: Path,
    result: Mapping[str, Any],
    *,
    bundle_sha256: str,
) -> Dict[str, Any]:
    validate_result_v2(result)
    if not SHA256_RE.fullmatch(bundle_sha256):
        raise ValueError("bundle_sha256 must be SHA-256")
    path = root / "project_state" / "result_import_events.jsonl"
    events = _read_jsonl(path)
    existing = next(
        (
            item
            for item in events
            if item.get("result_id") == result.get("result_id")
        ),
        None,
    )
    if existing:
        if existing.get("bundle_sha256") != bundle_sha256:
            raise ValueError("result_id is already bound to a different bundle SHA")
        return {"status": "already_recorded", "event": existing}
    event = {
        "event_type": "RESULT_IMPORT_VERIFIED",
        "event_id": f"import-{result['result_id']}",
        "recorded_at": _utc_now(),
        "result_id": result["result_id"],
        "job_id": result["job_id"],
        "experiment_id": result["experiment_id"],
        "workspace_id": result["workspace_id"],
        "attempt_id": result["attempt_id"],
        "approval_id": result["approval_id"],
        "critical_contract_sha256": result["critical_contract_sha256"],
        "bundle_sha256": bundle_sha256,
        "status": result["status"],
    }
    _append_jsonl(path, event)
    return {"status": "recorded", "event": event}
