"""Content-addressed execution bundle and recoverable server roundtrip v1.

The server-facing entry accepts one execution bundle directory.  All mutable
runtime state is written beside the bundle, never into the bundle itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Mapping

import yaml

from scripts.pfmval_result_bundle import (
    build_result_bundle_v1,
    publish_result_bundle_v1,
    validate_result_bundle_v1,
)
from scripts.pfmval_state import validate_against_schema_strict


EXECUTION_BUNDLE_SCHEMA_VERSION = "1.0"
FINGERPRINT_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "bundle_manifest.json"
FINGERPRINT_NAME = "server_fingerprint_v1.json"
FORBIDDEN_BUNDLE_PREFIXES = (
    "automation/results/",
    "histogene/",
    "egnv1/",
    "egnv2/",
)
RUNTIME_FILES = (
    "deploy/pfmval_ops.py",
    "path_registry.py",
    "scripts/pfmval_execution_roundtrip.py",
    "scripts/pfmval_governance.py",
    "scripts/pfmval_result_bundle.py",
    "scripts/pfmval_state.py",
    "project_state/schemas/attempt_event_v2.schema.json",
    "project_state/schemas/critical_contract.schema.json",
    "project_state/schemas/execution_bundle_v1.schema.json",
    "project_state/schemas/experiment_approval_v2.schema.json",
    "project_state/schemas/prediction_artifact_contract_v1.schema.json",
    "project_state/schemas/result_envelope_v2.schema.json",
    "project_state/schemas/server_fingerprint_v1.schema.json",
    "project_state/schemas/server_job_v2.schema.json",
    "project_state/schemas/workspace_registry.schema.json",
)

PREDICTION_ARTIFACT_COLUMNS = (
    "sample_id",
    "spatial_cluster_id",
    "pathway_id",
    "y_true",
    "y_pred",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def validate_prediction_artifact_contract(
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the declared, pre-training scientific prediction table contract."""
    expected = list(PREDICTION_ARTIFACT_COLUMNS)
    columns = contract.get("required_columns")
    if contract.get("schema_version") != "prediction_artifact_contract_v1":
        raise ValueError("unsupported prediction artifact contract schema_version")
    if not isinstance(columns, list) or any(not isinstance(item, str) for item in columns):
        raise ValueError("prediction artifact contract requires a string required_columns array")
    duplicates = sorted({item for item in columns if columns.count(item) > 1})
    missing = [item for item in expected if item not in columns]
    unexpected = sorted(set(columns) - set(expected))
    if duplicates or missing or unexpected:
        raise ValueError(
            "prediction artifact contract must declare exactly "
            f"{expected}; missing={missing}; duplicates={duplicates}; unexpected={unexpected}"
        )
    return {"schema_version": "prediction_artifact_contract_v1", "required_columns": expected}


def preflight_prediction_artifact_columns(
    contract: Mapping[str, Any],
    observed_columns: Iterable[str],
) -> Dict[str, Any]:
    """Report every column-contract violation together before a training start."""
    canonical = validate_prediction_artifact_contract(contract)
    observed = [str(item) for item in observed_columns]
    required = canonical["required_columns"]
    missing = [item for item in required if item not in observed]
    duplicates = sorted({item for item in observed if observed.count(item) > 1})
    return {
        "status": "ready" if not missing and not duplicates else "blocked",
        "required_columns": required,
        "observed_columns": observed,
        "missing_columns": missing,
        "duplicate_columns": duplicates,
        "valid": not missing and not duplicates,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(dict(value)))
    os.replace(temporary, path)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _safe_relative_path(raw: str) -> bool:
    if not raw or "\\" in raw or "\x00" in raw:
        return False
    path = PurePosixPath(raw)
    return not path.is_absolute() and ".." not in path.parts


def _find_one(
    values: Iterable[Mapping[str, Any]],
    *,
    description: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> Dict[str, Any]:
    matches = [dict(item) for item in values if predicate(item)]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {description}, observed {len(matches)}"
        )
    return matches[0]


def _contract_for_job(project_root: Path, job: Mapping[str, Any]) -> Dict[str, Any]:
    expected = str(job["critical_contract_sha256"])
    candidates = []
    for path in (
        project_root / "project_state" / "governance"
    ).glob("*critical_contract.json"):
        value = _read_json(path)
        if value.get("canonical_sha256") == expected:
            candidates.append(value)
    return _find_one(
        candidates,
        description="critical contract",
        predicate=lambda _item: True,
    )


def _attempt_for_job(
    project_root: Path,
    job: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = []
    governance_dir = project_root / "project_state" / "governance"
    for path in governance_dir.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            continue
        if (
            value.get("event_type") == "ATTEMPT_PREPARED"
            and value.get("job_id") == job["job_id"]
            and value.get("attempt_id") == job["attempt_id"]
        ):
            candidates.append(value)
    if candidates:
        candidates.sort(
            key=lambda item: (
                isinstance(item.get("budget"), Mapping),
                str(item.get("recorded_at", "")),
            ),
            reverse=True,
        )
        return candidates[0]
    return _find_one(
        _read_jsonl(project_root / "project_state" / "attempt_events.jsonl"),
        description="prepared attempt",
        predicate=lambda item: (
            item.get("job_id") == job["job_id"]
            and item.get("attempt_id") == job["attempt_id"]
            and item.get("event_type") == "ATTEMPT_PREPARED"
        ),
    )


def _write_payload(
    bundle_dir: Path,
    relative_path: str,
    payload: bytes,
    *,
    role: str,
    records: list[Dict[str, Any]],
) -> None:
    if not _safe_relative_path(relative_path):
        raise ValueError(f"unsafe bundle path: {relative_path}")
    destination = bundle_dir / Path(*PurePosixPath(relative_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    records.append(
        {
            "path": relative_path,
            "role": role,
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    )


def _write_json_payload(
    bundle_dir: Path,
    relative_path: str,
    value: Mapping[str, Any],
    *,
    role: str,
    records: list[Dict[str, Any]],
) -> None:
    _write_payload(
        bundle_dir,
        relative_path,
        _canonical_json_bytes(dict(value)),
        role=role,
        records=records,
    )


def build_execution_bundle_v1(
    project_root: Path,
    job_path: Path,
    bundle_dir: Path,
) -> Dict[str, Any]:
    """Build and verify the minimum immutable governance execution closure."""
    project_root = project_root.resolve()
    job_path = job_path.resolve()
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise ValueError("execution bundle output must be empty")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    job = _read_json(job_path)
    required_job_fields = {
        "job_id",
        "experiment_id",
        "workspace_id",
        "attempt_id",
        "approval_id",
        "source_commit",
        "critical_contract_sha256",
        "run_units",
        "resolved_argv",
    }
    missing = sorted(required_job_fields - set(job))
    if missing:
        raise ValueError(f"job lacks execution bundle fields: {missing}")

    experiment_registry = _read_json(
        project_root / "experiments" / "experiment_registry.json"
    )
    experiment = _find_one(
        experiment_registry.get("experiments", []),
        description="experiment snapshot",
        predicate=lambda item: item.get("id") == job["experiment_id"],
    )
    approval = _find_one(
        _read_jsonl(project_root / "project_state" / "experiment_approvals.jsonl"),
        description="active approval",
        predicate=lambda item: (
            item.get("approval_id") == job["approval_id"]
            and item.get("status") == "active"
        ),
    )
    contract = _contract_for_job(project_root, job)
    attempt = _attempt_for_job(project_root, job)
    workspace_registry = _read_json(
        project_root / "project_state" / "workspace_registry.json"
    )
    workspace = _find_one(
        workspace_registry.get("workspaces", []),
        description="workspace snapshot",
        predicate=lambda item: item.get("workspace_id") == job["workspace_id"],
    )

    path_registry = yaml.safe_load(
        (project_root / "configs" / "server_paths.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(path_registry, dict):
        raise ValueError("server path registry must be an object")
    required_path_ids = {
        *[str(item) for item in job.get("path_ids", [])],
        "server_automation_worktrees",
        "server_repo_worktree",
        "server_mpp_results",
    }
    available_paths = path_registry.get("paths", {})
    unknown = sorted(required_path_ids - set(available_paths))
    if unknown:
        raise ValueError(f"execution bundle references unknown path ids: {unknown}")
    path_snapshot = {
        "schema_version": str(path_registry.get("schema_version", "1.0")),
        "paths": {
            path_id: available_paths[path_id]
            for path_id in sorted(required_path_ids)
        },
    }

    records: list[Dict[str, Any]] = []
    snapshots = (
        ("job/job.json", job, "immutable_job"),
        ("governance/experiment.json", experiment, "experiment_snapshot"),
        ("governance/approval.json", approval, "active_approval"),
        ("governance/critical_contract.json", contract, "critical_contract"),
        ("governance/attempt.json", attempt, "attempt_budget"),
        ("governance/workspace.json", workspace, "workspace_binding"),
        ("governance/path_snapshot.json", path_snapshot, "path_id_snapshot"),
    )
    for relative_path, value, role in snapshots:
        _write_json_payload(
            bundle_dir,
            relative_path,
            value,
            role=role,
            records=records,
        )

    for source_relative in RUNTIME_FILES:
        source_path = project_root / Path(*PurePosixPath(source_relative).parts)
        if not source_path.is_file():
            raise ValueError(f"runtime dependency is missing: {source_relative}")
        _write_payload(
            bundle_dir,
            f"runtime/{source_relative}",
            source_path.read_bytes(),
            role="runner_dependency",
            records=records,
        )

    operation_card = (
        "param([Parameter(Mandatory=$false)][switch]$DryRun)\n"
        "$Bundle = $PSScriptRoot\n"
        "$Runner = Join-Path $Bundle 'runtime\\deploy\\pfmval_ops.py'\n"
        "$Arguments = @($Runner, 'job', 'execute-v1', '--bundle', $Bundle)\n"
        "if ($DryRun) { $Arguments += '--dry-run' }\n"
        "$Python = Get-Command python -ErrorAction SilentlyContinue\n"
        "$PythonPrefix = @()\n"
        "if ($null -eq $Python) {\n"
        "  $Python = Get-Command py -ErrorAction SilentlyContinue\n"
        "  $PythonPrefix = @('-3')\n"
        "}\n"
        "if ($null -eq $Python) { throw 'No Python launcher is available' }\n"
        "& $Python.Source @PythonPrefix @Arguments\n"
        "exit $LASTEXITCODE\n"
    ).encode("utf-8")
    _write_payload(
        bundle_dir,
        "run_server.ps1",
        operation_card,
        role="single_operator_entry",
        records=records,
    )

    identity = {
        "job_id": str(job["job_id"]),
        "source_commit": str(job["source_commit"]),
        "critical_contract_sha256": str(job["critical_contract_sha256"]),
    }
    records.sort(key=lambda item: item["path"])
    bundle_id = _sha256_bytes(
        _canonical_json_bytes({"identity": identity, "files": records})
    )
    binding = job.get("execution_binding", {})
    expected_parent = str(binding.get("governance_commit", ""))
    revision = int(job.get("dispatch_revision", 1))
    result_ref = str(
        job.get("result_branch")
        or f"automation/server/{job['workspace_id']}/{job['attempt_id']}"
    )
    manifest = {
        "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "identity": identity,
        "created_at": str(job.get("created_at", "")),
        "job_schema_version": str(job.get("schema_version", "")),
        "workspace_id": str(job["workspace_id"]),
        "attempt_id": str(job["attempt_id"]),
        "approval_id": str(job["approval_id"]),
        "run_units": int(job["run_units"]),
        "requirements": {
            "minimum_gpu_count": 1,
            "minimum_gpu_free_bytes": 1_000_000_000,
            "minimum_disk_free_bytes": 1_000_000_000,
        },
        "publish": {
            "remote_name": "gitee",
            "ref": result_ref,
            "revision_path": (
                f"automation/results/{job['workspace_id']}/"
                f"{job['attempt_id']}/result-R{revision:03d}"
            ),
            "expected_parent": expected_parent,
        },
        "files": records,
    }
    _write_json_atomic(bundle_dir / MANIFEST_NAME, manifest)
    return validate_execution_bundle_v1(bundle_dir)


def validate_execution_bundle_v1(bundle_dir: Path) -> Dict[str, Any]:
    """Verify bundle closure, byte hashes, identity bindings and exclusions."""
    bundle_dir = bundle_dir.resolve()
    manifest = _read_json(bundle_dir / MANIFEST_NAME)
    validate_against_schema_strict(
        manifest,
        bundle_dir
        / "runtime"
        / "project_state"
        / "schemas"
        / "execution_bundle_v1.schema.json",
        "execution bundle v1",
    )
    if manifest.get("schema_version") != EXECUTION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported execution bundle schema_version")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("execution bundle files must be a non-empty array")

    expected_paths = {MANIFEST_NAME}
    observed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("execution bundle file record must be an object")
        relative_path = str(record.get("path", ""))
        if not _safe_relative_path(relative_path):
            raise ValueError(f"unsafe execution bundle path: {relative_path}")
        if relative_path in observed_paths:
            raise ValueError("execution bundle paths must be unique")
        if any(relative_path.startswith(prefix) for prefix in FORBIDDEN_BUNDLE_PREFIXES):
            raise ValueError(
                f"forbidden non-execution content in bundle: {relative_path}"
            )
        observed_paths.add(relative_path)
        expected_paths.add(relative_path)
        path = bundle_dir / Path(*PurePosixPath(relative_path).parts)
        if not path.is_file():
            raise ValueError(f"execution bundle file is missing: {relative_path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"execution bundle size mismatch: {relative_path}")
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"execution bundle SHA-256 mismatch: {relative_path}")
    actual_paths = {
        item.relative_to(bundle_dir).as_posix()
        for item in bundle_dir.rglob("*")
        if item.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "execution bundle closure mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )

    job = _read_json(bundle_dir / "job" / "job.json")
    experiment = _read_json(
        bundle_dir / "governance" / "experiment.json"
    )
    approval = _read_json(bundle_dir / "governance" / "approval.json")
    contract = _read_json(
        bundle_dir / "governance" / "critical_contract.json"
    )
    attempt = _read_json(bundle_dir / "governance" / "attempt.json")
    workspace = _read_json(bundle_dir / "governance" / "workspace.json")
    identity = manifest.get("identity", {})
    expected_identity = {
        "job_id": str(job.get("job_id", "")),
        "source_commit": str(job.get("source_commit", "")),
        "critical_contract_sha256": str(
            job.get("critical_contract_sha256", "")
        ),
    }
    if identity != expected_identity:
        raise ValueError("execution bundle identity does not match immutable job")
    binding_checks = (
        (experiment.get("id"), job.get("experiment_id"), "experiment"),
        (approval.get("approval_id"), job.get("approval_id"), "approval"),
        (
            contract.get("canonical_sha256"),
            job.get("critical_contract_sha256"),
            "contract",
        ),
        (attempt.get("job_id"), job.get("job_id"), "attempt job"),
        (attempt.get("attempt_id"), job.get("attempt_id"), "attempt"),
        (workspace.get("workspace_id"), job.get("workspace_id"), "workspace"),
    )
    for observed, expected, label in binding_checks:
        if observed != expected:
            raise ValueError(f"execution bundle {label} binding mismatch")

    records_for_id = sorted(
        [dict(item) for item in records],
        key=lambda item: item["path"],
    )
    expected_bundle_id = _sha256_bytes(
        _canonical_json_bytes(
            {"identity": expected_identity, "files": records_for_id}
        )
    )
    if manifest.get("bundle_id") != expected_bundle_id:
        raise ValueError("execution bundle_id mismatch")
    return {
        "status": "valid",
        "bundle_id": expected_bundle_id,
        "bundle_sha256": expected_bundle_id,
        "file_count": len(records),
    }


def _command_output(command: list[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    output = (completed.stdout or completed.stderr).strip()
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output,
    }


def _default_probe() -> Dict[str, Any]:
    python_jsonschema = _command_output(
        [
            sys.executable,
            "-c",
            "import jsonschema; print(jsonschema.__version__)",
        ]
    )
    python_torch = _command_output(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'version':torch.__version__,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_version':torch.version.cuda,"
                "'gpu_count':torch.cuda.device_count(),"
                "'gpu_free_bytes':"
                "(torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0)}))"
            ),
        ]
    )
    try:
        torch_capabilities = json.loads(str(python_torch.get("output", "")))
    except json.JSONDecodeError:
        torch_capabilities = {}
    powershell = _command_output(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "$PSVersionTable.PSVersion.ToString()",
        ]
    )
    powershell_schema = _command_output(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "if ((Get-Command Test-Json).Parameters.ContainsKey('SchemaFile')) { exit 0 } else { exit 1 }",
        ]
    )
    schema_backends = []
    if python_jsonschema.get("available"):
        schema_backends.append("python-jsonschema")
    if powershell_schema.get("available"):
        schema_backends.append("powershell-test-json")
    disk = shutil.disk_usage(Path.cwd())
    return {
        "os": {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "powershell": powershell,
        "git": _command_output(["git", "--version"]),
        "python_candidates": [
            {
                "path": sys.executable,
                "version": platform.python_version(),
                "jsonschema": bool(python_jsonschema.get("available")),
                "torch": bool(python_torch.get("available")),
            }
        ],
        "schema_backends": schema_backends,
        "selected_schema_backend": schema_backends[0] if schema_backends else None,
        "pytorch": {
            "available": bool(python_torch.get("available")),
            "version": torch_capabilities.get("version"),
        },
        "cuda": {
            "available": bool(torch_capabilities.get("cuda_available")),
            "version": torch_capabilities.get("cuda_version"),
        },
        "gpu": {
            "count": int(torch_capabilities.get("gpu_count") or 0),
            "free_bytes": int(torch_capabilities.get("gpu_free_bytes") or 0),
        },
        "disk": {"free_bytes": disk.free},
        "path_ids": {},
        "git_line_endings": _command_output(
            ["git", "config", "--get", "core.autocrlf"]
        ),
    }


def ensure_server_fingerprint_v1(
    state_dir: Path,
    *,
    now: datetime | None = None,
    ttl_seconds: int = 3600,
    probe: Callable[[], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Reuse a valid fingerprint until expiry, then refresh it atomically."""
    if ttl_seconds <= 0:
        raise ValueError("fingerprint ttl_seconds must be positive")
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FINGERPRINT_NAME
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if path.exists():
        current = _read_json(path)
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "project_state"
            / "schemas"
            / "server_fingerprint_v1.schema.json"
        )
        validate_against_schema_strict(
            current,
            schema_path,
            "server fingerprint v1",
        )
        if (
            current.get("schema_version") == FINGERPRINT_SCHEMA_VERSION
            and _parse_time(str(current["expires_at"])) >= observed_now
            and current.get("content_sha256")
            == _sha256_bytes(
                _canonical_json_bytes(
                    {
                        key: value
                        for key, value in current.items()
                        if key
                        not in {
                            "schema_version",
                            "generated_at",
                            "expires_at",
                            "content_sha256",
                        }
                    }
                )
            )
        ):
            return {
                "status": "valid",
                "reused": True,
                "fingerprint_sha256": str(current["content_sha256"]),
                "path": str(path),
                "fingerprint": current,
            }

    capabilities = dict((probe or _default_probe)())

    def reject_sensitive_keys(value: Any, location: str = "fingerprint") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(
                    marker in lowered
                    for marker in (
                        "credential",
                        "password",
                        "secret",
                        "token",
                        "private_key",
                    )
                ):
                    raise ValueError(
                        f"fingerprint probe returned forbidden field: {location}.{key}"
                    )
                reject_sensitive_keys(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                reject_sensitive_keys(item, f"{location}[{index}]")

    reject_sensitive_keys(capabilities)
    generated_at = _format_time(observed_now)
    expires_at = _format_time(observed_now + timedelta(seconds=ttl_seconds))
    content_sha256 = _sha256_bytes(_canonical_json_bytes(capabilities))
    fingerprint = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "expires_at": expires_at,
        "content_sha256": content_sha256,
        **capabilities,
    }
    validate_against_schema_strict(
        fingerprint,
        Path(__file__).resolve().parents[1]
        / "project_state"
        / "schemas"
        / "server_fingerprint_v1.schema.json",
        "server fingerprint v1",
    )
    _write_json_atomic(path, fingerprint)
    return {
        "status": "valid",
        "reused": False,
        "fingerprint_sha256": content_sha256,
        "path": str(path),
        "fingerprint": fingerprint,
    }


PREFLIGHT_CHECKS = (
    ("bundle", "valid", "bundle_integrity", "E1"),
    ("source", "head_matches", "source_head", "E0"),
    ("source", "clean", "source_clean", "E1"),
    ("source", "object_available", "source_object", "E0"),
    ("source", "entrypoint_sha_matches", "entrypoint_sha", "E0"),
    ("bindings", "approval_matches", "approval_binding", "E1"),
    ("bindings", "contract_matches", "contract_binding", "E0"),
    ("bindings", "job_matches", "job_binding", "E1"),
    ("bindings", "data_manifest_matches", "data_manifest", "E0"),
    ("bindings", "split_matches", "split_manifest", "E0"),
    ("attempt", "root_absent", "attempt_root", "E1"),
    ("attempt", "lease_available", "lease", "E1"),
    ("attempt", "budget_available", "budget", "E1"),
    ("runtime", "schema_backend_available", "schema_backend", "E1"),
    ("runtime", "gpu_sufficient", "gpu_capacity", "E1"),
    ("runtime", "disk_sufficient", "disk_capacity", "E1"),
    ("publish", "fast_forward", "publish_fast_forward", "E1"),
    ("publish", "remote_identity_matches", "remote_identity", "E1"),
)


def aggregate_preflight_v2(
    *,
    bundle_manifest: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate every independent check and report all blockers in one pass."""
    failures = []
    passes = []
    warnings = []
    for group, field, check_id, layer in PREFLIGHT_CHECKS:
        group_value = observations.get(group, {})
        observed = (
            group_value.get(field)
            if isinstance(group_value, Mapping)
            else None
        )
        record = {
            "check_id": check_id,
            "layer": layer,
            "observed": observed,
        }
        if observed is True:
            passes.append(record)
        else:
            failures.append(
                {
                    **record,
                    "severity": "HARD_FAIL",
                    "message": f"{group}.{field} did not pass",
                }
            )
    prediction_observation = observations.get("prediction_contract")
    if isinstance(prediction_observation, Mapping):
        record = {
            "check_id": "prediction_columns",
            "layer": "E0",
            "observed": prediction_observation.get("valid"),
        }
        if prediction_observation.get("valid") is True:
            passes.append(record)
        else:
            failures.append(
                {
                    **record,
                    "severity": "HARD_FAIL",
                    "message": "prediction artifact column contract did not pass",
                    "missing_columns": list(prediction_observation.get("missing_columns", [])),
                    "duplicate_columns": list(prediction_observation.get("duplicate_columns", [])),
                }
            )
    line_endings = fingerprint.get("git_line_endings")
    if line_endings not in (None, "", {}):
        warnings.append(
            {
                "check_id": "host_line_endings",
                "layer": "E3",
                "severity": "WARN",
                "message": "Git line-ending behavior is recorded but is not an E0/E1 blocker",
                "observed": line_endings,
            }
        )
    safe_to_start = not failures
    return {
        "schema_version": "2.0",
        "bundle_id": str(bundle_manifest.get("bundle_id", "")),
        "fingerprint_sha256": str(
            fingerprint.get("content_sha256", "")
        ),
        "status": "ready" if safe_to_start else "blocked",
        "safe_to_start": safe_to_start,
        "passes": passes,
        "failures": failures,
        "warnings": warnings,
    }


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=["git", "-C", str(repo), *args],
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _path_from_snapshot(snapshot: Mapping[str, Any], path_id: str) -> Path:
    entry = snapshot.get("paths", {}).get(path_id)
    if not isinstance(entry, Mapping) or not entry.get("path"):
        raise KeyError(path_id)
    return Path(str(entry["path"]))


def inspect_preflight_observations(
    bundle_dir: Path,
    fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Collect read-only observations for all preflight groups."""
    bundle_dir = bundle_dir.resolve()
    try:
        validate_execution_bundle_v1(bundle_dir)
        bundle_valid = True
    except Exception:
        bundle_valid = False
    manifest = _read_json(bundle_dir / MANIFEST_NAME)
    job = _read_json(bundle_dir / "job" / "job.json")
    experiment = _read_json(bundle_dir / "governance" / "experiment.json")
    approval = _read_json(bundle_dir / "governance" / "approval.json")
    contract = _read_json(
        bundle_dir / "governance" / "critical_contract.json"
    )
    attempt = _read_json(bundle_dir / "governance" / "attempt.json")
    workspace = _read_json(bundle_dir / "governance" / "workspace.json")
    path_snapshot = _read_json(
        bundle_dir / "governance" / "path_snapshot.json"
    )
    source_root = _path_from_snapshot(
        path_snapshot,
        "server_automation_worktrees",
    ) / str(job["workspace_id"])
    source_git = _git(source_root, "rev-parse", "HEAD")
    source_status = _git(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    source_object = _git(
        source_root,
        "cat-file",
        "-e",
        f"{job['source_commit']}^{{commit}}",
    )
    argv = job.get("resolved_argv", [])
    entrypoint = (
        source_root / str(argv[1])
        if isinstance(argv, list) and len(argv) >= 2
        else source_root / "__missing_entrypoint__"
    )
    entrypoint_binding = next(
        (
            item
            for item in job.get("input_binding", {}).get("artifacts", [])
            if item.get("artifact_id") == "mpp2_training_entrypoint"
        ),
        None,
    )
    entrypoint_matches = bool(
        entrypoint_binding
        and entrypoint.is_file()
        and _sha256_file(entrypoint) == entrypoint_binding.get("sha256")
    )
    run_root = (
        _path_from_snapshot(path_snapshot, "server_mpp_results")
        / "roundtrip_runs"
        / str(job["workspace_id"])
        / str(job["attempt_id"])
    )
    remaining = int(attempt.get("budget", {}).get("remaining_after_prepare", 0))
    requirements = manifest.get("requirements", {})
    schema_backends = fingerprint.get("schema_backends", [])
    gpu = fingerprint.get("gpu", {})
    disk = fingerprint.get("disk", {})
    remote = _git(source_root, "remote", "get-url", "gitee")
    result_ref = str(manifest.get("publish", {}).get("ref", ""))
    try:
        remote_head = subprocess.run(
            ["git", "ls-remote", "--heads", "gitee", f"refs/heads/{result_ref}"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        remote_head = subprocess.CompletedProcess(
            args=["git", "ls-remote", "--heads", "gitee"],
            returncode=127,
            stdout="",
            stderr=str(exc),
        )
    remote_line = remote_head.stdout.strip().split()
    observed_remote_sha = remote_line[0] if remote_line else ""
    expected_parent = str(
        manifest.get("publish", {}).get("expected_parent", "")
    )
    return {
        "bundle": {"valid": bundle_valid},
        "source": {
            "head_matches": (
                source_git.returncode == 0
                and source_git.stdout.strip() == job["source_commit"]
            ),
            "clean": (
                source_status.returncode == 0
                and not source_status.stdout.strip()
            ),
            "object_available": source_object.returncode == 0,
            "entrypoint_sha_matches": entrypoint_matches,
        },
        "bindings": {
            "approval_matches": (
                approval.get("approval_id") == job.get("approval_id")
                and approval.get("status") == "active"
            ),
            "contract_matches": (
                contract.get("canonical_sha256")
                == job.get("critical_contract_sha256")
            ),
            "job_matches": (
                experiment.get("id") == job.get("experiment_id")
                and workspace.get("workspace_id") == job.get("workspace_id")
                and attempt.get("job_id") == job.get("job_id")
            ),
            "data_manifest_matches": (
                contract.get("data_manifest_id")
                == job.get("input_binding", {}).get("data_manifest_id")
            ),
            "split_matches": any(
                item.get("artifact_id") == "mpp2_split_manifest"
                and any(
                    bound.get("artifact_id") == "mpp2_split_manifest"
                    and bound.get("sha256") == item.get("sha256")
                    for bound in job.get("input_binding", {}).get(
                        "artifacts", []
                    )
                )
                for item in contract.get("input_artifacts", [])
            ),
        },
        "attempt": {
            "root_absent": not run_root.exists(),
            "lease_available": workspace.get("lease") is None,
            "budget_available": remaining >= int(job.get("run_units", 0)),
        },
        "runtime": {
            "schema_backend_available": bool(schema_backends),
            "gpu_sufficient": (
                int(gpu.get("count", 0))
                >= int(requirements.get("minimum_gpu_count", 1))
                and int(gpu.get("free_bytes", 0))
                >= int(requirements.get("minimum_gpu_free_bytes", 0))
            ),
            "disk_sufficient": int(disk.get("free_bytes", 0))
            >= int(requirements.get("minimum_disk_free_bytes", 0)),
        },
        "publish": {
            "fast_forward": (
                remote_head.returncode == 0
                and (
                    not observed_remote_sha
                    or observed_remote_sha == expected_parent
                )
            ),
            "remote_identity_matches": (
                remote.returncode == 0
                and remote.stdout.strip()
                and manifest.get("publish", {}).get("remote_name") == "gitee"
            ),
        },
    }


def run_preflight_v2(
    bundle_dir: Path,
    *,
    state_root: Path | None = None,
    now: datetime | None = None,
    ttl_seconds: int = 3600,
    probe: Callable[[], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Validate the bundle, reuse/refresh fingerprint, and aggregate all checks."""
    bundle_dir = bundle_dir.resolve()
    manifest = _read_json(bundle_dir / MANIFEST_NAME)
    root = (
        state_root.resolve()
        if state_root is not None
        else bundle_dir.parent / ".pfmval_execution_state"
    )
    state_dir = root / str(manifest["bundle_id"])
    path_snapshot = _read_json(
        bundle_dir / "governance" / "path_snapshot.json"
    )

    def probe_with_path_ids() -> Mapping[str, Any]:
        capabilities = dict((probe or _default_probe)())
        capabilities["path_ids"] = {
            path_id: str(entry.get("path", ""))
            for path_id, entry in path_snapshot.get("paths", {}).items()
            if isinstance(entry, Mapping)
        }
        return capabilities

    fingerprint_report = ensure_server_fingerprint_v1(
        state_dir,
        now=now,
        ttl_seconds=ttl_seconds,
        probe=probe_with_path_ids,
    )
    observations = inspect_preflight_observations(
        bundle_dir,
        fingerprint_report["fingerprint"],
    )
    preflight = aggregate_preflight_v2(
        bundle_manifest=manifest,
        fingerprint=fingerprint_report["fingerprint"],
        observations=observations,
    )
    _write_json_atomic(state_dir / "preflight_v2.json", preflight)
    return preflight


def _default_executor(context: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_dir = Path(str(context["bundle_dir"]))
    state_dir = Path(str(context["state_dir"]))
    job = _read_json(bundle_dir / "job" / "job.json")
    path_snapshot = _read_json(
        bundle_dir / "governance" / "path_snapshot.json"
    )
    source_root = _path_from_snapshot(
        path_snapshot,
        "server_automation_worktrees",
    ) / str(job["workspace_id"])
    command = [str(item) for item in job["resolved_argv"]]
    execution_environment = dict(os.environ)
    execution_environment.update(
        {
            "PFMVAL_EXECUTION_BUNDLE": str(bundle_dir),
            "PFMVAL_EXECUTION_STATE": str(state_dir),
            "PFMVAL_RESULT_SOURCE_BUNDLE": str(
                state_dir / "result_source_bundle"
            ),
        }
    )
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=execution_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (state_dir / "stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (state_dir / "stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    return {"returncode": completed.returncode}


def _default_result_packager(context: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_dir = Path(str(context["bundle_dir"]))
    state_dir = Path(str(context["state_dir"]))
    job = _read_json(bundle_dir / "job" / "job.json")
    source_bundle = state_dir / "result_source_bundle"
    target = state_dir / "result_bundle_v1"
    if target.exists() and (target / "result.json").is_file():
        return validate_result_bundle_v1(
            bundle_dir / "runtime",
            target,
        )
    return build_result_bundle_v1(
        bundle_dir / "runtime",
        source_bundle,
        target,
        result_id=f"{job['job_id']}-result-R001",
        artifact_retention="retain_with_immutable_result_bundle",
    )


def _default_result_publisher(context: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_dir = Path(str(context["bundle_dir"]))
    state_dir = Path(str(context["state_dir"]))
    manifest = _read_json(bundle_dir / MANIFEST_NAME)
    path_snapshot = _read_json(
        bundle_dir / "governance" / "path_snapshot.json"
    )
    repo_root = _path_from_snapshot(path_snapshot, "server_repo_worktree")
    publish = manifest["publish"]
    return publish_result_bundle_v1(
        repo_root,
        state_dir / "result_bundle_v1",
        remote_name="gitee",
        ref=str(publish["ref"]),
        revision_path=str(publish["revision_path"]),
        expected_parent=str(publish["expected_parent"]),
        commit_message=(
            f"server: publish {manifest['workspace_id']} "
            f"{manifest['attempt_id']} result bundle"
        ),
    )


def execute_roundtrip_v1(
    bundle_dir: Path,
    *,
    state_root: Path | None = None,
    preflight: Mapping[str, Any] | None = None,
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    result_packager: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    result_publisher: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Run or recover one immutable bundle without implicit retry."""
    bundle_dir = bundle_dir.resolve()
    manifest = _read_json(bundle_dir / MANIFEST_NAME)
    bundle_id = str(manifest["bundle_id"])
    root = (
        state_root.resolve()
        if state_root is not None
        else bundle_dir.parent / ".pfmval_execution_state"
    )
    state_dir = root / bundle_id
    state_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "bundle_dir": str(bundle_dir),
        "state_dir": str(state_dir),
        "manifest": manifest,
    }

    identity_dir = root / "identities"
    identity_path = identity_dir / f"{manifest['identity']['job_id']}.json"
    if identity_path.exists():
        identity = _read_json(identity_path)
        if identity.get("bundle_id") != bundle_id:
            raise ValueError("same job identity is already bound to a different bundle SHA")
    else:
        _write_json_atomic(
            identity_path,
            {
                "job_id": manifest["identity"]["job_id"],
                "bundle_id": bundle_id,
            },
        )

    published_path = state_dir / "publish_receipt.json"
    started_path = state_dir / "attempt_started.json"
    terminal_path = state_dir / "attempt_terminal.json"
    package_path = state_dir / "result_package_receipt.json"
    if published_path.exists():
        published = _read_json(published_path)
        if published.get("bundle_id") != bundle_id:
            raise ValueError("published receipt bundle SHA mismatch")
        return {
            "status": "ALREADY_PUBLISHED",
            "bundle_id": bundle_id,
            "run_units_consumed": int(
                _read_json(started_path).get("run_units", 0)
            ),
            "commit_sha": published.get("commit_sha"),
            "recovered_from_terminal": False,
        }

    if preflight is None:
        preflight = run_preflight_v2(bundle_dir, state_root=root)
    _write_json_atomic(state_dir / "preflight_v2.json", dict(preflight))
    if not preflight.get("safe_to_start") and not terminal_path.exists():
        return {
            "status": "BLOCKED",
            "bundle_id": bundle_id,
            "run_units_consumed": 0,
            "failures": list(preflight.get("failures", [])),
            "warnings": list(preflight.get("warnings", [])),
        }

    recovered_from_terminal = terminal_path.exists()
    if not recovered_from_terminal and started_path.exists():
        return {
            "status": "RUNNING",
            "bundle_id": bundle_id,
            "run_units_consumed": int(
                _read_json(started_path).get("run_units", 0)
            ),
            "recovered_from_terminal": False,
        }

    if not recovered_from_terminal:
        started = {
            "event_type": "EXPERIMENT_STARTED",
            "bundle_id": bundle_id,
            "job_id": manifest["identity"]["job_id"],
            "run_units": int(manifest.get("run_units", 0)),
            "recorded_at": _format_time(datetime.now(timezone.utc)),
        }
        _write_json_atomic(started_path, started)
        execution = dict((executor or _default_executor)(context))
        returncode = int(execution.get("returncode", -1))
        terminal = {
            "event_type": "EXPERIMENT_TERMINAL",
            "bundle_id": bundle_id,
            "job_id": manifest["identity"]["job_id"],
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "recorded_at": _format_time(datetime.now(timezone.utc)),
        }
        if execution.get("error"):
            terminal["error"] = str(execution["error"])
        _write_json_atomic(terminal_path, terminal)
    terminal = _read_json(terminal_path)
    run_units_consumed = int(_read_json(started_path).get("run_units", 0))
    if (
        terminal.get("status") != "completed"
        or int(terminal.get("returncode", -1)) != 0
    ):
        return {
            "status": "FAILED",
            "bundle_id": bundle_id,
            "run_units_consumed": run_units_consumed,
            "recovered_from_terminal": recovered_from_terminal,
        }

    if package_path.exists():
        package = _read_json(package_path)
    else:
        package = dict((result_packager or _default_result_packager)(context))
        _write_json_atomic(
            package_path,
            {"bundle_id": bundle_id, **package},
        )
    publish = dict((result_publisher or _default_result_publisher)(context))
    receipt = {
        "bundle_id": bundle_id,
        "result_bundle_sha256": package.get("bundle_sha256"),
        **publish,
    }
    _write_json_atomic(published_path, receipt)
    return {
        "status": "PUBLISHED",
        "bundle_id": bundle_id,
        "run_units_consumed": run_units_consumed,
        "commit_sha": publish.get("commit_sha"),
        "recovered_from_terminal": recovered_from_terminal,
    }
