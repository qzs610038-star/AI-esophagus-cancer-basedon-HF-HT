#!/usr/bin/env python3
"""Unified local/server maintenance CLI for PFMval.

This command never opens a direct connection to the server.  Job and result
envelopes are files intended to be transported by the configured Gitee remote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pfmval_state import (  # noqa: E402
    DIAGNOSTIC_COMMANDS,
    activate_mpp_repair_evidence,
    active_directives,
    active_mpp_repair,
    append_directive,
    build_mpp_path_index,
    build_result_envelope,
    create_diagnostic_request,
    create_exploration_session,
    directive_lifecycle_candidates,
    create_job_manifest,
    import_mpp_repair_evidence_from_git,
    import_result_bundle,
    import_w007_four_arm_results,
    exploration_cleanup_candidates,
    list_exploration_sessions,
    migrate_experiment_provenance,
    promote_exploration_script,
    promote_md_import_paths,
    record_diagnostic_outputs,
    read_json,
    run_allowlisted_diagnostic,
    safe_job_parameters,
    scan_documents,
    set_document_lifecycle,
    state_lock,
    sync_state,
    transition_directive,
    validate_diagnostic_state,
    validate_against_schema,
    validate_job_manifest,
    validate_server_paths,
    validate_state,
    verify_mpp_repair_server_assets,
    ValidationReport,
    write_json_atomic,
    write_text_atomic,
)
from path_registry import get_registered_path  # noqa: E402
from scripts.pfmval_governance import (  # noqa: E402
    JOB_V2_COMMANDS,
    allocate_workspace,
    approve_experiment_protocol,
    assert_workspace_locus,
    build_critical_contract,
    build_job_v2,
    build_result_v2,
    initialize_workspace_registry,
    prepare_attempt,
    record_attempt_event,
    record_result_import_v2,
    register_governed_experiment,
    revise_experiment_protocol,
    scan_workspace_registry,
    validate_job_v2,
    validate_result_v2,
)
from scripts.pfmval_result_bundle import (  # noqa: E402
    build_result_bundle_v1,
    publish_result_bundle_v1,
    record_result_bundle_import_v1,
    validate_result_bundle_v1,
)
from scripts.pfmval_result_pair import finalize_result_pair  # noqa: E402
from scripts.pfmval_execution_roundtrip import (  # noqa: E402
    build_execution_bundle_v1,
    execute_roundtrip_v1,
    run_preflight_v2,
    validate_execution_bundle_v1,
)
from scripts.pfmval_assets import (  # noqa: E402
    build_close_preview,
    build_shadow_inventory,
    compile_asset_policy,
    evaluate_asset_writes,
)
from scripts.pfmval_views import (  # noqa: E402
    build_project_guide,
    refresh_experiment_views,
)
from scripts.pfmval_science import (  # noqa: E402
    adjudicate_fact_candidates,
    list_scientific_records,
    project_decision_summaries,
    record_science_decision,
    record_scientific_entry,
)
from scripts.pfmval_knowledge import (  # noqa: E402
    build_document_profile,
    confirm_project_fact,
    list_project_facts,
    preview_project_fact,
    review_document_freshness,
)
from scripts.pfmval_workflows import (  # noqa: E402
    activate_workflow,
    apply_workflow_discovery,
    deprecate_workflow,
    discover_workflows,
    list_workflows,
    merge_workflows,
    register_workflow_manifest,
    review_workflow,
    revise_workflow,
    validate_workflow_catalog,
)


def parse_key_values(items: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"parameter must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if key in values:
            current = values[key]
            values[key] = current + [value] if isinstance(current, list) else [current, value]
        else:
            lowered = value.lower()
            if lowered == "true":
                values[key] = True
            elif lowered == "false":
                values[key] = False
            else:
                values[key] = value
    return values


def ensure_job_worktree(manifest: Dict[str, Any]) -> Path:
    workspace_id = str(manifest.get("workspace_id", ""))
    if not re.fullmatch(r"W[0-9]{3,}", workspace_id):
        raise ValueError("unsafe workspace_id for persistent worktree path")
    worktree_root = get_registered_path("server_experiment_worktrees").resolve()
    worktree = (worktree_root / workspace_id).resolve()
    if not worktree.is_relative_to(worktree_root):
        raise ValueError("job worktree escapes the registered automation root")
    if worktree.exists():
        completed = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        status = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if status.stdout.strip():
            raise ValueError(f"persistent workspace is dirty and cannot advance: {worktree}")
        if completed.stdout.strip() != manifest["source_commit"]:
            subprocess.run(
                ["git", "-C", str(worktree), "checkout", "--detach", manifest["source_commit"]],
                check=True,
            )
    else:
        worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), manifest["source_commit"]],
            cwd=PROJECT_ROOT,
            check=True,
        )
    status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if status.stdout.strip():
        raise ValueError(f"persistent workspace is dirty and cannot be reused: {worktree}")
    return worktree


def _git_text(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_artifact(manifest: Dict[str, Any], artifact_id: str) -> Dict[str, Any] | None:
    artifacts = manifest.get("input_binding", {}).get("artifacts", [])
    return next(
        (item for item in artifacts if item.get("artifact_id") == artifact_id),
        None,
    )


def _contains_option(argv: List[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def _validate_git_archive_bundle(bundle_root: Path, repo_root: Path, commit: str) -> None:
    """Prove a regular directory matches tracked Git blobs.

    Gitee archives on Windows may materialize a text blob with CRLF even when
    the canonical Git blob uses LF.  Do not delegate this decision to the
    host's Git filters: accept only that complete, text-only LF-to-CRLF
    materialization; every other byte difference remains a hard failure.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("governance archive commit must be a full 40-character SHA")
    listing = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", commit],
        check=True,
        capture_output=True,
    ).stdout
    for item in listing.split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, object_type, blob_id = metadata.split()
        if object_type != b"blob" or mode not in {b"100644", b"100755"}:
            raise ValueError("governance archive contains unsupported Git tree entry")
        relative_path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        candidate = (bundle_root / relative_path).resolve()
        if not candidate.is_relative_to(bundle_root.resolve()) or not candidate.is_file():
            raise ValueError(f"governance archive is missing tracked file: {relative_path.as_posix()}")
        expected_bytes = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", blob_id.decode("ascii")],
            check=True,
            capture_output=True,
        ).stdout
        candidate_bytes = candidate.read_bytes()
        if candidate_bytes == expected_bytes:
            continue
        is_crlf_materialization = (
            b"\x00" not in expected_bytes
            and b"\r\n" not in expected_bytes
            and candidate_bytes == expected_bytes.replace(b"\n", b"\r\n")
        )
        if not is_crlf_materialization:
            raise ValueError(f"governance archive file hash mismatch: {relative_path.as_posix()}")


def _validate_governance_execution_root(
    governance_root: Path,
    *,
    expected_governance: str,
    governance_repo: Path | None,
    governance_commit: str | None,
) -> None:
    """Accept either a clean Git worktree or a byte-verified Gitee archive."""
    if (governance_root / ".git").exists():
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_governance, "HEAD"],
            cwd=governance_root,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("governance bundle does not contain execution_binding baseline")
        if _git_text(governance_root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("governance bundle is dirty")
        return

    if governance_repo is None or governance_commit is None:
        raise ValueError("regular governance bundle requires --governance-repo and --governance-commit")
    governance_repo = governance_repo.resolve()
    if not governance_repo.is_dir():
        raise ValueError("governance archive repository is missing")
    ancestry = subprocess.run(
        ["git", "-C", str(governance_repo), "merge-base", "--is-ancestor", expected_governance, governance_commit],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("governance archive does not contain execution_binding baseline")
    _validate_git_archive_bundle(governance_root, governance_repo, governance_commit)


def _job_v2_runtime_command(
    manifest: Dict[str, Any],
    *,
    governance_root: Path,
    source_worktree: Path,
    run_directory: Path,
    entrypoint: Path,
    argv: List[str],
) -> List[str]:
    """Build the allowlisted command with runtime paths outside the source tree.

    The immutable manifest supplies scientific parameters.  This helper supplies
    only server path bindings, checks the protected split hash, and forces the
    output root below the external attempt directory.
    """
    command = [sys.executable, str(entrypoint), *argv[2:]]
    if entrypoint.name != "train_mpp_uni2h_mlp.py":
        return command

    protected_options = (
        "--mpp_root", "--cache_root", "--flat_cache_root", "--splits_root",
        "--manifest_labels_root", "--output_root",
    )
    if any(_contains_option(argv[2:], option) for option in protected_options):
        raise ValueError("job run-v2 MPP command must not override runtime path bindings")

    split_binding = _bound_artifact(manifest, "mpp2_split_manifest")
    split_path = source_worktree / "mpp_standard_splits" / "group_2" / "split_manifest.csv"
    if not isinstance(split_binding, dict) or not split_path.is_file():
        raise ValueError("job run-v2 requires the bound MPP2 split manifest")
    if _sha256_file(split_path) != split_binding.get("sha256"):
        raise ValueError("source MPP2 split manifest hash does not match the job binding")

    repair = active_mpp_repair(governance_root)
    data_manifest_id = manifest.get("input_binding", {}).get("data_manifest_id")
    if repair is None or data_manifest_id != repair.get("data_manifest_id"):
        raise ValueError("job run-v2 repaired-label manifest does not match the active governance evidence")
    registry_path = governance_root / "configs" / "server_paths.yaml"
    command.extend([
        "--mpp_root", str(get_registered_path("mpp_data_root", registry_path=registry_path, project_root=governance_root)),
        "--cache_root", str(get_registered_path("server_mpp_partner_cache", registry_path=registry_path, project_root=governance_root)),
        "--flat_cache_root", str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=governance_root)),
        "--splits_root", str((source_worktree / "mpp_standard_splits").resolve()),
        "--manifest_labels_root", str(repair["server_stage_path"]),
        "--output_root", str(run_directory),
    ])
    return command


def prepare_job_v2_execution(
    manifest: Dict[str, Any],
    *,
    governance_root: Path,
    source_worktree: Path,
    run_root: Path,
    governance_repo: Path | None = None,
    governance_commit: str | None = None,
) -> Dict[str, Any]:
    """Validate the split source/governance execution boundary without writing it.

    `source_worktree` holds immutable training code; governance state is read from
    `governance_root`; every attempt writes only below `run_root/W###/A###`.
    """
    if manifest.get("schema_version") != "2.0":
        raise ValueError("job run-v2 only accepts schema_version=2.0")
    source_worktree = source_worktree.resolve()
    validate_job_manifest(
        governance_root,
        manifest,
        require_head=False,
        source_git_root=source_worktree,
    )
    binding = manifest.get("execution_binding")
    if not isinstance(binding, dict):
        raise ValueError("job run-v2 requires an explicit execution_binding; regenerate the job after protocol approval")
    if binding.get("mode") != "source_plus_governance_bundle":
        raise ValueError("job run-v2 requires source_plus_governance_bundle binding")
    expected_governance = str(binding.get("governance_commit", ""))
    _validate_governance_execution_root(
        governance_root,
        expected_governance=expected_governance,
        governance_repo=governance_repo,
        governance_commit=governance_commit,
    )
    if _git_text(source_worktree, "rev-parse", "HEAD") != manifest["source_commit"]:
        raise ValueError("registered source worktree HEAD does not match job source_commit")
    if _git_text(source_worktree, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("registered source worktree is dirty")
    argv = list(manifest["resolved_argv"])
    if len(argv) < 2 or argv[0] != "python":
        raise ValueError("job run-v2 only permits a resolved Python entrypoint")
    entrypoint = (source_worktree / argv[1]).resolve()
    if not entrypoint.is_relative_to(source_worktree) or not entrypoint.is_file():
        raise ValueError("job run-v2 entrypoint is missing or escapes source worktree")
    entrypoint_binding = _bound_artifact(manifest, "mpp2_training_entrypoint")
    if not isinstance(entrypoint_binding, dict):
        raise ValueError("job run-v2 requires a bound training entrypoint hash")
    if _sha256_file(entrypoint) != entrypoint_binding.get("sha256"):
        raise ValueError("source training entrypoint hash does not match the job binding")
    target = (run_root.resolve() / manifest["workspace_id"] / manifest["attempt_id"]).resolve()
    if not target.is_relative_to(run_root.resolve()):
        raise ValueError("attempt run root escapes supplied run_root")
    if target.exists():
        raise ValueError("attempt run root already exists; implicit retry is forbidden")
    command = _job_v2_runtime_command(
        manifest,
        governance_root=governance_root,
        source_worktree=source_worktree,
        run_directory=target,
        entrypoint=entrypoint,
        argv=argv,
    )
    return {
        "governance_root": str(governance_root.resolve()),
        "source_worktree": str(source_worktree),
        "run_directory": str(target),
        "command": command,
    }


def write_job_v2_started_event(plan: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    """Create the server-returnable start evidence immediately before execution."""
    run_directory = Path(plan["run_directory"])
    run_directory.mkdir(parents=True, exist_ok=False)
    event = {
        "event_type": "EXPERIMENT_STARTED",
        "event_id": f"started-{manifest['job_id']}",
        "recorded_at": _utc_now(),
        "experiment_id": manifest["experiment_id"],
        "workspace_id": manifest["workspace_id"],
        "attempt_id": manifest["attempt_id"],
        "job_id": manifest["job_id"],
        "approval_id": manifest["approval_id"],
        "source_commit": manifest["source_commit"],
        "critical_contract_sha256": manifest["critical_contract_sha256"],
        "run_units": manifest["run_units"],
        "execution_binding": manifest["execution_binding"],
    }
    output = run_directory / "attempt_started.json"
    write_json_atomic(output, event)
    return output


def write_job_v2_terminal_event(
    plan: Dict[str, Any], manifest: Dict[str, Any], *, returncode: int, error: str | None = None,
) -> Path:
    """Record the terminal server-side status without changing experiment state."""
    event = {
        "event_type": "EXPERIMENT_TERMINAL",
        "event_id": f"terminal-{manifest['job_id']}",
        "recorded_at": _utc_now(),
        "experiment_id": manifest["experiment_id"],
        "workspace_id": manifest["workspace_id"],
        "attempt_id": manifest["attempt_id"],
        "job_id": manifest["job_id"],
        "approval_id": manifest["approval_id"],
        "source_commit": manifest["source_commit"],
        "critical_contract_sha256": manifest["critical_contract_sha256"],
        "run_units": manifest["run_units"],
        "execution_binding": manifest["execution_binding"],
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
    }
    if error:
        event["error"] = error
    output = Path(plan["run_directory"]) / "attempt_terminal.json"
    write_json_atomic(output, event)
    return output


def bound_job_parameter_argv(work_root: Path, manifest: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
    parameters = dict(manifest.get("parameters", {}))
    script = str(experiment.get("script", "")).replace("\\", "/")
    if script.endswith("train_mpp_uni2h_mlp.py"):
        repair = active_mpp_repair(work_root)
        if repair is None:
            raise ValueError("MPP training requires an explicitly activated repaired-label data manifest")
        if manifest.get("data_manifest_id") != repair.get("data_manifest_id"):
            raise ValueError(
                "MPP job data_manifest_id does not match active repaired labels: "
                f"job={manifest.get('data_manifest_id')} active={repair.get('data_manifest_id')}"
            )
        registry_path = work_root / "configs" / "server_paths.yaml"
        parameters.update({
            "mpp_root": str(get_registered_path("mpp_data_root", registry_path=registry_path, project_root=work_root)),
            # MPP2-5 training features live in the partner-style MPP{N}_UNI
            # cache. The flat cache remains reserved for external XZY lookup.
            "cache_root": str(get_registered_path("server_mpp_partner_cache", registry_path=registry_path, project_root=work_root)),
            "flat_cache_root": str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=work_root)),
            # Standard splits are committed assets and must come from the pinned
            # worktree, not from a mutable checkout at the registered server path.
            "splits_root": str((work_root / "mpp_standard_splits").resolve()),
            "manifest_labels_root": str(repair["server_stage_path"]),
            "output_root": str(get_registered_path("server_mpp_results", registry_path=registry_path, project_root=work_root)),
        })
    elif script.endswith("train_mpp_uni2h_lora.py"):
        repair = active_mpp_repair(work_root)
        if repair is None:
            raise ValueError("MPP2 LoRA requires an explicitly activated repaired-label data manifest")
        if manifest.get("data_manifest_id") != repair.get("data_manifest_id"):
            raise ValueError(
                "MPP2 LoRA data_manifest_id does not match active repaired labels: "
                f"job={manifest.get('data_manifest_id')} active={repair.get('data_manifest_id')}"
            )
        registry_path = work_root / "configs" / "server_paths.yaml"
        parameters.update({
            "data_manifest_id": str(repair["data_manifest_id"]),
            "manifest_labels_root": str(repair["server_stage_path"]),
            "mpp_root": str(get_registered_path("mpp_data_root", registry_path=registry_path, project_root=work_root)),
            "splits_root": str((work_root / "mpp_standard_splits").resolve()),
            "head_checkpoint": str(get_registered_path(
                "server_mpp2_frozen_baseline_checkpoint",
                registry_path=registry_path,
                project_root=work_root,
            )),
            "output_root": str(get_registered_path("server_mpp_results", registry_path=registry_path, project_root=work_root)),
        })
    elif script.endswith("scripts/fit_mpp2_pathway_ridge_calibration.py"):
        repair = active_mpp_repair(work_root)
        if repair is None:
            raise ValueError("MPP2 pathway Ridge calibration requires active repaired-label evidence")
        if manifest.get("data_manifest_id") != repair.get("data_manifest_id"):
            raise ValueError(
                "MPP2 pathway Ridge calibration data_manifest_id does not match active repaired labels: "
                f"job={manifest.get('data_manifest_id')} active={repair.get('data_manifest_id')}"
            )
        registry_path = work_root / "configs" / "server_paths.yaml"
        result_root = get_registered_path("server_mpp_results", registry_path=registry_path, project_root=work_root)
        parameters.update({
            "splits_root": str((work_root / "mpp_standard_splits").resolve()),
            "manifest_labels_root": str(repair["server_stage_path"]),
            "cache_root": str(get_registered_path("server_mpp_partner_cache", registry_path=registry_path, project_root=work_root)),
            "flat_cache_root": str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=work_root)),
            "head_checkpoint": str(get_registered_path("server_mpp2_frozen_baseline_checkpoint", registry_path=registry_path, project_root=work_root)),
            "output_dir": str(result_root / manifest["job_id"]),
        })
    elif script.endswith("scripts/check_mpp_online_cache_parity.py"):
        repair = active_mpp_repair(work_root)
        if repair is None:
            raise ValueError("MPP cache parity requires an active repaired-label data manifest")
        if manifest.get("data_manifest_id") != repair.get("data_manifest_id"):
            raise ValueError(
                "cache parity data_manifest_id does not match active repaired labels: "
                f"job={manifest.get('data_manifest_id')} active={repair.get('data_manifest_id')}"
            )
        registry_path = work_root / "configs" / "server_paths.yaml"
        result_root = get_registered_path(
            "server_mpp_results", registry_path=registry_path, project_root=work_root,
        )
        parameters.update({
            "data_manifest_id": str(repair["data_manifest_id"]),
            "manifest_labels_root": str(repair["server_stage_path"]),
            "resource_release_ack": "MPP1_3_4_5_IDLE",
            "splits_root": str((work_root / "mpp_standard_splits").resolve()),
            "mpp_root": str(get_registered_path("mpp_data_root", registry_path=registry_path, project_root=work_root)),
            "cache_root": str(get_registered_path("server_mpp_partner_cache", registry_path=registry_path, project_root=work_root)),
            "flat_cache_root": str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=work_root)),
            "output": str(result_root / "cache_parity" / manifest["job_id"] / "cache_parity.csv"),
        })
    return safe_job_parameters(parameters)


def command_state(args: argparse.Namespace) -> int:
    if args.state_command == "record-directive":
        with state_lock(PROJECT_ROOT):
            directive_id = append_directive(
                PROJECT_ROOT,
                summary=args.summary,
                scope=args.scope,
                topic=args.topic,
                supersedes=args.supersedes,
                affected_files=args.affected_file,
                related_experiment_ids=args.related_experiment_id,
                review_after=args.review_after,
                completion_evidence=args.completion_evidence,
            )
            state = sync_state(PROJECT_ROOT, force_revision=True)
        print(f"[PASS] recorded {directive_id}; state_revision={state['state_revision']}")
        return 0
    if args.state_command == "directives":
        candidates = directive_lifecycle_candidates(PROJECT_ROOT, older_than_days=args.older_than_days)
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        print("[INFO] Review-only output; no directive status was changed.")
        return 0
    if args.state_command == "directive":
        with state_lock(PROJECT_ROOT):
            event = transition_directive(
                PROJECT_ROOT,
                directive_id=args.directive_id,
                status=args.status,
                reason=args.reason,
                completion_evidence=args.evidence_ref,
                superseded_by=args.superseded_by,
            )
            state = sync_state(PROJECT_ROOT, force_revision=True)
        print(f"[PASS] transitioned {event['directive_id']} to {event['status']}; state_revision={state['state_revision']}")
        return 0
    if args.state_command == "sync":
        with state_lock(PROJECT_ROOT):
            if args.with_docs:
                # 原子化修复路径：先重扫文档哈希与 PROJECT_GUIDE，再同步状态，
                # 一条命令消除"改文件忘同步"导致的 start-check FAIL。
                write_json_atomic(
                    PROJECT_ROOT / "project_state" / "document_registry.json",
                    scan_documents(PROJECT_ROOT),
                )
                guide = build_project_guide(PROJECT_ROOT)
                write_text_atomic(PROJECT_ROOT / "PROJECT_GUIDE.md", guide)
            state = sync_state(PROJECT_ROOT, force_revision=args.force_revision)
        print(f"[PASS] state synchronized; revision={state['state_revision']}")
        return 0
    if args.state_command == "migrate":
        from scripts.finalize_experiment import generate_dashboard
        with state_lock(PROJECT_ROOT):
            registry_path = PROJECT_ROOT / "experiments" / "experiment_registry.json"
            registry = read_json(registry_path)
            migrate_experiment_provenance(registry)
            write_json_atomic(registry_path, registry)
            generate_dashboard(registry)
            # First sync creates CURRENT_STATE; the second scan then registers it.
            sync_state(PROJECT_ROOT, force_revision=True)
            write_json_atomic(PROJECT_ROOT / "project_state" / "document_registry.json", scan_documents(PROJECT_ROOT))
            state = sync_state(PROJECT_ROOT)
        print(f"[PASS] initial state/document/provenance migration complete; revision={state['state_revision']}")
        return 0
    if args.state_command == "validate":
        action = "general" if args.task == "server" else args.task
        host_scope = args.host_scope or ("server" if args.task == "server" else "local")
        report = validate_state(
            PROJECT_ROOT,
            strict=args.strict,
            task=action,
            host_scope=host_scope,
        )
        report.emit()
        return 0 if report.ok else 1
    raise ValueError(f"unknown state command: {args.state_command}")


def command_docs(args: argparse.Namespace) -> int:
    if args.docs_command == "guide":
        guide = build_project_guide(PROJECT_ROOT)
        guide_path = PROJECT_ROOT / "PROJECT_GUIDE.md"
        write_text_atomic(guide_path, guide)
        print(json.dumps({"project_guide_sha256": hashlib.sha256(guide.encode("utf-8")).hexdigest()}, ensure_ascii=False, indent=2))
        return 0
    if args.docs_command == "set-lifecycle":
        result = set_document_lifecycle(
            PROJECT_ROOT,
            doc_id=args.doc_id,
            lifecycle=args.lifecycle,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[INFO] run `state sync --with-docs` to complete the lifecycle loop")
        return 0
    if args.docs_command != "scan":
        raise ValueError(f"unknown docs command: {args.docs_command}")
    registry = scan_documents(PROJECT_ROOT)
    summary = registry["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_json_atomic(PROJECT_ROOT / "project_state" / "document_registry.json", registry)
        print("[PASS] wrote project_state/document_registry.json")
    else:
        print("[DRY RUN] use --write to persist the registry")
    return 0


def command_paths(args: argparse.Namespace) -> int:
    if args.paths_command == "build-index":
        index = build_mpp_path_index(PROJECT_ROOT)
        output = PROJECT_ROOT / "mpp_standard_splits" / "path_index.json"
        write_json_atomic(output, index)
        print(f"[PASS] wrote {output}")
        print(f"[INFO] files={index['summary']['file_count']} bytes={index['summary']['total_bytes']} labels_validated={index['labels_validated']}")
        return 0 if index["labels_validated"] else 2
    if args.paths_command == "validate":
        report = ValidationReport()
        task = "training" if args.training else args.task
        host_scope = args.host_scope or ("server" if args.task == "server" else "local")
        validate_server_paths(PROJECT_ROOT, report, task=task, host_scope=host_scope)
        report.emit()
        return 0 if report.ok else 1
    if args.paths_command == "promote-md-import":
        result = promote_md_import_paths(
            PROJECT_ROOT,
            ids=args.ids,
            required_on=args.required_on,
            write=args.write,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.write:
            print("[DRY RUN] use --write to persist the promotion")
        return 0
    raise ValueError(f"unknown paths command: {args.paths_command}")


def command_result(args: argparse.Namespace) -> int:
    if args.result_command == "import-w007-four-arm":
        result = import_w007_four_arm_results(
            PROJECT_ROOT,
            Path(args.quarantine_root).resolve(),
            write=args.write,
        )
    else:
        result = import_result_bundle(PROJECT_ROOT, Path(args.bundle).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_mpp(args: argparse.Namespace) -> int:
    if args.mpp_command == "repair" and args.repair_command == "import":
        result = import_mpp_repair_evidence_from_git(
            PROJECT_ROOT,
            git_ref=args.git_ref,
            evidence_path=args.evidence_path,
            expected_audit_sha256=args.expected_audit_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[BLOCKED] Repair evidence is verified but still requires an explicit gate-release directive")
        return 0
    if args.mpp_command == "repair" and args.repair_command == "activate":
        result = activate_mpp_repair_evidence(
            PROJECT_ROOT,
            evidence_id=args.evidence_id,
            directive_id=args.directive_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[PASS] Barcode repair gate released; formal training approval remains required")
        return 0
    raise ValueError("unknown MPP command")


def command_agent(args: argparse.Namespace) -> int:
    if args.task == "diagnostic":
        report = validate_diagnostic_state(PROJECT_ROOT)
        report.emit()
        if not report.ok:
            print("[BLOCKED] Resolve diagnostic safety-boundary failures before continuing.")
            return 1
        print("[PASS] Diagnostic start gate passed. Only allowlisted, non-evidence diagnostics are permitted.")
        return 0
    action = "general" if args.task == "server" else args.task
    host_scope = args.host_scope or ("server" if args.task == "server" else "local")
    report = validate_state(
        PROJECT_ROOT,
        strict=args.strict,
        task=action,
        host_scope=host_scope,
    )
    report.emit()
    if not report.ok:
        print("[BLOCKED] Resolve critical state conflicts before continuing.")
        return 1
    print("[PASS] Agent start gate passed. Read CURRENT_STATE.md before acting.")
    return 0


def command_diagnostic(args: argparse.Namespace) -> int:
    if args.diagnostic_command == "request":
        request_path = create_diagnostic_request(
            PROJECT_ROOT,
            diagnostic_id=args.diagnostic_id,
            source_commit=args.source_commit,
            command_id=args.command_id,
            source_branch=args.source_branch,
            return_branch=args.return_branch,
        )
        print(f"[PASS] wrote non-evidence diagnostic request: {request_path}")
        print("[INFO] Push the request through Gitee; no server command was executed.")
        return 0
    if args.diagnostic_command == "record":
        event = record_diagnostic_outputs(
            PROJECT_ROOT, diagnostic_id=args.diagnostic_id, outputs=args.output,
        )
        print(json.dumps(event, ensure_ascii=False, indent=2))
        print("[PASS] recorded non-evidence diagnostic output inventory and size.")
        return 0
    if args.diagnostic_command == "run-allowlisted":
        result = run_allowlisted_diagnostic(PROJECT_ROOT, diagnostic_id=args.diagnostic_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[PASS] completed fixed allowlisted diagnostic with UTF-8/LF output.")
        return 0
    raise ValueError(f"unknown diagnostic command: {args.diagnostic_command}")


def command_explore(args: argparse.Namespace) -> int:
    if args.explore_command == "new":
        manifest_path = create_exploration_session(PROJECT_ROOT, session_id=args.session_id, purpose=args.purpose)
        print(f"[PASS] created local-only exploration session: {manifest_path}")
        return 0
    if args.explore_command == "list":
        sessions = list_exploration_sessions(PROJECT_ROOT)
        print(json.dumps(sessions, ensure_ascii=False, indent=2))
        return 0
    if args.explore_command == "candidates":
        candidates = exploration_cleanup_candidates(PROJECT_ROOT, older_than_days=args.older_than_days)
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        print("[INFO] Review-only output; no files were deleted.")
        return 0
    if args.explore_command == "promote":
        target_path = promote_exploration_script(
            PROJECT_ROOT, source=args.script, target=args.target, directive_id=args.directive,
        )
        print(f"[PASS] wrote promotion candidate: {target_path}")
        print("[INFO] Register an experiment and dispatch smoke/formal separately; this is not evidence.")
        return 0
    raise ValueError(f"unknown explore command: {args.explore_command}")


def _observed_local_worktrees() -> List[Dict[str, Any]]:
    output = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for raw in [*output.splitlines(), ""]:
        if not raw:
            if current.get("path"):
                path = Path(str(current["path"]))
                status = subprocess.run(
                    ["git", "-C", str(path), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                ).stdout
                current["dirty"] = bool(status.strip())
                records.append(current)
            current = {}
            continue
        key, _, value = raw.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "detached":
            current["branch"] = "(detached)"
    return records


def command_workspace(args: argparse.Namespace) -> int:
    if args.workspace_command == "init":
        registry = initialize_workspace_registry(PROJECT_ROOT)
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return 0
    if args.workspace_command == "status":
        registry = initialize_workspace_registry(PROJECT_ROOT)
        if args.workspace_id:
            workspace = next(
                (
                    item
                    for item in registry.get("workspaces", [])
                    if item.get("workspace_id") == args.workspace_id
                ),
                None,
            )
            if workspace is None:
                raise ValueError(f"unknown workspace_id: {args.workspace_id}")
            print(json.dumps(workspace, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(registry, ensure_ascii=False, indent=2))
        return 0
    if args.workspace_command == "scan":
        if args.host != "local":
            raise ValueError(
                "server workspace scan requires a returned Gitee inventory; "
                "direct server observation is not allowed"
            )
        report = scan_workspace_registry(
            PROJECT_ROOT,
            host_scope="local",
            observed_worktrees=_observed_local_worktrees(),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["status"] == "FAIL" else 0
    if args.workspace_command == "check-locus":
        registry = initialize_workspace_registry(PROJECT_ROOT)
        workspace = next(
            (
                item
                for item in registry.get("workspaces", [])
                if item.get("workspace_id") == args.workspace_id
            ),
            None,
        )
        if workspace is None:
            raise ValueError(f"unknown workspace_id: {args.workspace_id}")
        cwd = Path(args.cwd).resolve()
        branch = subprocess.run(
            ["git", "-C", str(cwd), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(cwd), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
        )
        assert_workspace_locus(
            workspace,
            cwd=cwd,
            branch=branch,
            head=head,
            dirty=dirty,
        )
        print(f"[PASS] workspace locus verified: {args.workspace_id}")
        return 0
    if args.workspace_command == "close":
        if not args.preview:
            raise ValueError(
                "close execute is unavailable without a separate exact-target approval"
            )
        observed = (
            read_json(Path(args.observed_json).resolve())
            if args.observed_json
            else _observed_local_worktrees()
        )
        preview = build_close_preview(
            PROJECT_ROOT,
            workspace_id=args.workspace_id,
            observed_worktrees=observed,
        )
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0 if preview["status"] == "close_ready" else 1
    raise ValueError(f"unknown workspace command: {args.workspace_command}")


def command_asset(args: argparse.Namespace) -> int:
    if args.asset_command == "inventory":
        if args.mode != "shadow":
            raise ValueError(
                "asset inventory is authorized only in shadow mode"
            )
        inventory = build_shadow_inventory(PROJECT_ROOT)
        validate_against_schema(
            inventory,
            PROJECT_ROOT
            / "project_state"
            / "schemas"
            / "asset_shadow_inventory.schema.json",
            "asset shadow inventory",
        )
        if args.output:
            write_json_atomic(Path(args.output).resolve(), inventory)
            print(f"[PASS] wrote shadow inventory: {args.output}")
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        print("[INFO] No asset was moved, rewritten, unhashed, or unprotected.")
        return 0
    if args.asset_command == "validate":
        registry = read_json(
            PROJECT_ROOT / "project_state" / "asset_registry.json"
        )
        validate_against_schema(
            registry,
            PROJECT_ROOT
            / "project_state"
            / "schemas"
            / "asset_registry.schema.json",
            "asset registry",
        )
        policy = compile_asset_policy(PROJECT_ROOT)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "asset_count": len(registry.get("assets", [])),
                    "compiled_rule_count": len(policy["rules"]),
                    "mode": "shadow",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.asset_command == "policy":
        policy = compile_asset_policy(PROJECT_ROOT)
        report = evaluate_asset_writes(
            PROJECT_ROOT,
            policy,
            args.path,
            mode="shadow",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.asset_command == "lifecycle-preview":
        inventory = build_shadow_inventory(PROJECT_ROOT)
        preview = {
            "mode": "preview",
            "candidates": inventory["candidates"],
            "lifecycle_events_written": 0,
            "protection_changes": 0,
            "physical_moves": 0,
            "deletions": 0,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown asset command: {args.asset_command}")


def command_views(args: argparse.Namespace) -> int:
    if args.views_command != "refresh":
        raise ValueError(f"unknown views command: {args.views_command}")
    if not args.experiments:
        raise ValueError("views refresh requires --experiments; PROJECT_GUIDE uses docs guide")
    projection = project_decision_summaries(PROJECT_ROOT)
    hashes = refresh_experiment_views(PROJECT_ROOT)
    with state_lock(PROJECT_ROOT):
        state = sync_state(PROJECT_ROOT)
    hashes["decision_projection"] = projection
    hashes["current_state_revision"] = str(state["state_revision"])
    print(json.dumps(hashes, ensure_ascii=False, indent=2))
    return 0


def command_science(args: argparse.Namespace) -> int:
    if args.science_command == "decision":
        decision = read_json(Path(args.input).resolve())
        result = record_science_decision(PROJECT_ROOT, decision)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.science_command == "record":
        entry = read_json(Path(args.input).resolve())
        result = record_scientific_entry(PROJECT_ROOT, entry)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.science_command == "list":
        records = list_scientific_records(
            PROJECT_ROOT,
            record_type=args.record_type,
        )
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    if args.science_command == "adjudicate":
        candidates = read_json(Path(args.input).resolve())
        if not isinstance(candidates, list):
            raise ValueError("fact adjudication input must be a JSON array")
        result = adjudicate_fact_candidates(candidates)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown science command: {args.science_command}")


def command_knowledge(args: argparse.Namespace) -> int:
    if args.knowledge_command == "fact-preview":
        fact = read_json(Path(args.input).resolve())
        preview = preview_project_fact(PROJECT_ROOT, fact)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if args.knowledge_command == "fact-confirm":
        fact = read_json(Path(args.input).resolve())
        result = confirm_project_fact(
            PROJECT_ROOT,
            fact,
            confirmation_source=args.confirmation_source,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.knowledge_command == "fact-list":
        print(
            json.dumps(
                list_project_facts(PROJECT_ROOT),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.knowledge_command == "profile":
        payload = read_json(Path(args.input).resolve())
        artifact = build_document_profile(
            PROJECT_ROOT,
            profile=args.profile,
            payload=payload,
        )
        validate_against_schema(
            artifact,
            PROJECT_ROOT
            / "project_state"
            / "schemas"
            / "document_profile.schema.json",
            "document profile artifact",
        )
        if args.output:
            write_json_atomic(Path(args.output).resolve(), artifact)
            print(f"[PASS] wrote local document profile: {args.output}")
        print(json.dumps(artifact, ensure_ascii=False, indent=2))
        return 0
    if args.knowledge_command == "freshness":
        fingerprints = (
            read_json(Path(args.fingerprints).resolve())
            if args.fingerprints
            else None
        )
        preview = review_document_freshness(
            PROJECT_ROOT,
            document_id=args.document_id,
            current_fingerprints=fingerprints,
        )
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown knowledge command: {args.knowledge_command}")


def command_workflow(args: argparse.Namespace) -> int:
    if args.workflow_command == "register":
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "--error-unmatch",
                "--",
                args.manifest,
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.splitlines()
        result = register_workflow_manifest(
            PROJECT_ROOT,
            manifest_path=args.manifest,
            authorization_ref=args.authorization_ref,
            tracked_paths=tracked,
            active_authorization_refs=active_directives(PROJECT_ROOT),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.workflow_command == "discover":
        scope = read_json(Path(args.scope_manifest).resolve())
        selected_paths = scope.get("selected_paths", [])
        if not isinstance(selected_paths, list) or not all(
            isinstance(item, str) for item in selected_paths
        ):
            raise ValueError("workflow scope selected_paths must be a string array")
        tracked = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "--", *selected_paths],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.splitlines()
        discovery = discover_workflows(
            PROJECT_ROOT,
            selected_paths=selected_paths,
            tracked_paths=tracked,
        )
        result: Dict[str, Any] = discovery
        if args.record:
            result = {
                "discovery": discovery,
                "record": apply_workflow_discovery(PROJECT_ROOT, discovery),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.workflow_command == "list":
        print(
            json.dumps(
                list_workflows(PROJECT_ROOT),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.workflow_command == "review":
        result = review_workflow(PROJECT_ROOT, args.workflow_id)
    elif args.workflow_command == "activate":
        result = activate_workflow(
            PROJECT_ROOT,
            args.workflow_id,
            standardized=args.standardized,
        )
    elif args.workflow_command == "revise":
        result = revise_workflow(
            PROJECT_ROOT,
            args.workflow_id,
            new_workflow_id=args.new_workflow_id,
            version=args.version,
        )
    elif args.workflow_command == "merge":
        result = merge_workflows(
            PROJECT_ROOT,
            args.workflow_ids,
            new_workflow_id=args.new_workflow_id,
            purpose=args.purpose,
            version=args.version,
        )
    elif args.workflow_command == "deprecate":
        result = deprecate_workflow(
            PROJECT_ROOT,
            args.workflow_id,
            replacement_workflow_ids=args.replacement_workflow_ids,
        )
    elif args.workflow_command == "validate":
        result = validate_workflow_catalog(PROJECT_ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    else:
        raise ValueError(f"unknown workflow command: {args.workflow_command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "register":
        contract = read_json(Path(args.critical_contract).resolve())
        experiment = register_governed_experiment(
            PROJECT_ROOT,
            experiment_id=args.experiment_id,
            display_name=args.display_name,
            phase=args.phase,
            run_limit=args.run_limit,
            critical_contract=contract,
            local_relative_path=args.workspace_path or args.workspace_relative_path,
            workspace_branch=args.workspace_branch,
        )
        print(json.dumps(experiment, ensure_ascii=False, indent=2))
        return 0
    if args.experiment_command == "approve":
        approval = approve_experiment_protocol(
            PROJECT_ROOT,
            approval_id=args.approval_id,
            experiment_id=args.experiment_id,
            protocol_revision=args.protocol_revision,
            phase=args.phase,
            run_limit=args.run_limit,
            critical_contract_sha256=args.critical_contract_sha256,
            adaptation_policy="e1_allowlist_only",
            source="explicit_user_instruction",
        )
        print(json.dumps(approval, ensure_ascii=False, indent=2))
        return 0
    if args.experiment_command == "revise":
        revised = revise_experiment_protocol(PROJECT_ROOT, experiment_id=args.experiment_id,
            critical_contract=read_json(Path(args.critical_contract).resolve()))
        print(json.dumps(revised, ensure_ascii=False, indent=2))
        return 0
    if args.experiment_command == "show":
        registry = read_json(PROJECT_ROOT / "experiments" / "experiment_registry.json")
        experiment = next(
            (
                item
                for item in registry.get("experiments", [])
                if item.get("id") == args.experiment_id
            ),
            None,
        )
        if experiment is None:
            raise ValueError(f"unknown experiment_id: {args.experiment_id}")
        print(json.dumps(experiment, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown experiment command: {args.experiment_command}")


def command_attempt(args: argparse.Namespace) -> int:
    if args.attempt_command == "prepare":
        attempt = prepare_attempt(
            PROJECT_ROOT,
            experiment_id=args.experiment_id,
            workspace_id=args.workspace_id,
            approval_id=args.approval_id,
            run_units=args.run_units,
            source_commit=args.source_commit,
            critical_contract_sha256=args.critical_contract_sha256,
        )
        write_json_atomic(Path(args.output).resolve(), attempt)
        print(json.dumps(attempt, ensure_ascii=False, indent=2))
        print(f"[PASS] wrote prepared attempt: {args.output}")
        return 0
    if args.attempt_command == "event":
        event = record_attempt_event(
            PROJECT_ROOT,
            workspace_id=args.workspace_id,
            attempt_id=args.attempt_id,
            event_id=args.event_id,
            event_type=args.event_type,
        )
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown attempt command: {args.attempt_command}")


def command_governance(args: argparse.Namespace) -> int:
    if args.governance_command == "contract":
        payload = read_json(Path(args.input).resolve())
        contract = build_critical_contract(payload)
        write_json_atomic(Path(args.output).resolve(), contract)
        print(f"[PASS] wrote critical contract: {args.output}")
        return 0
    if args.governance_command == "job-v2":
        if args.job_v2_command == "build":
            prepared = read_json(Path(args.prepared).resolve())
            input_binding = read_json(Path(args.input_binding).resolve())
            resolved_argv = read_json(Path(args.resolved_argv).resolve())
            job = build_job_v2(
                prepared,
                command_id=args.command_id,
                path_ids=args.path_id,
                resolved_argv=resolved_argv,
                input_binding=input_binding,
                adaptation_record_id=args.adaptation_record_id,
                dispatch_revision=args.dispatch_revision,
            )
            job["execution_binding"] = {
                "mode": "source_plus_governance_bundle",
                "governance_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
                ).strip(),
            }
            validate_job_v2(job)
            write_json_atomic(Path(args.output).resolve(), job)
            print(f"[PASS] wrote local job v2 envelope: {args.output}")
            print("[INFO] No Gitee push, server command or training was executed.")
            return 0
        if args.job_v2_command == "validate":
            job = read_json(Path(args.manifest).resolve())
            validate_job_v2(job)
            print("[PASS] job v2 manifest is valid")
            return 0
    if args.governance_command == "result-v2":
        if args.result_v2_command == "build":
            job = read_json(Path(args.job).resolve())
            artifacts = read_json(Path(args.artifacts).resolve())
            metrics = read_json(Path(args.metrics).resolve())
            large_artifacts = (
                read_json(Path(args.large_artifacts).resolve())
                if args.large_artifacts
                else []
            )
            result = build_result_v2(
                job,
                result_id=args.result_id,
                status=args.status,
                artifacts=artifacts,
                metrics=metrics,
                metric_artifact_ids=args.metric_artifact_id,
                large_artifacts=large_artifacts,
            )
            write_json_atomic(Path(args.output).resolve(), result)
            print(f"[PASS] wrote local result v2 envelope: {args.output}")
            print("[INFO] No result was accepted into project evidence.")
            return 0
        result = read_json(Path(args.manifest).resolve())
        validate_result_v2(result)
        if args.result_v2_command == "validate":
            print("[PASS] result v2 envelope is valid")
            return 0
        if args.result_v2_command == "record-import":
            event = record_result_import_v2(
                PROJECT_ROOT,
                result,
                bundle_sha256=args.bundle_sha256,
            )
            print(json.dumps(event, ensure_ascii=False, indent=2))
            return 0
    if args.governance_command == "result-bundle-v1":
        if args.result_bundle_v1_command == "build":
            report = build_result_bundle_v1(
                PROJECT_ROOT,
                Path(args.source_bundle),
                Path(args.staging),
                result_id=args.result_id,
                artifact_retention=args.artifact_retention,
                created_at=args.created_at,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("[INFO] Packaging-only build; no training or result import ran.")
            return 0
        if args.result_bundle_v1_command == "validate":
            report = validate_result_bundle_v1(
                PROJECT_ROOT,
                Path(args.bundle),
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
    if args.governance_command == "execution-bundle-v1":
        if args.execution_bundle_v1_command == "build":
            report = build_execution_bundle_v1(
                PROJECT_ROOT,
                Path(args.job),
                Path(args.output),
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("[INFO] Bundle-only build; no dispatch or training ran.")
            return 0
        if args.execution_bundle_v1_command == "validate":
            report = validate_execution_bundle_v1(Path(args.bundle))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.result_bundle_v1_command == "record-import":
            event = record_result_bundle_import_v1(
                PROJECT_ROOT,
                Path(args.bundle),
                bundle_sha256=args.bundle_sha256,
            )
            print(json.dumps(event, ensure_ascii=False, indent=2))
            return 0
        if args.result_bundle_v1_command == "publish":
            report = publish_result_bundle_v1(
                PROJECT_ROOT,
                Path(args.bundle),
                remote_name=args.remote,
                ref=args.ref,
                revision_path=args.revision_path,
                expected_parent=args.expected_parent,
                commit_message=args.commit_message,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
    if args.governance_command == "result-pair":
        acceptance = read_json(Path(args.input).resolve())
        result = finalize_result_pair(PROJECT_ROOT, acceptance)
        views = refresh_experiment_views(PROJECT_ROOT)
        with state_lock(PROJECT_ROOT):
            state = sync_state(PROJECT_ROOT)
        print(
            json.dumps(
                {
                    **result,
                    "views": views,
                    "state_revision": state["state_revision"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise ValueError(f"unknown governance command: {args.governance_command}")


def command_job(args: argparse.Namespace) -> int:
    if args.job_command == "preflight-v2":
        report = run_preflight_v2(Path(args.bundle))
        if args.report:
            write_json_atomic(Path(args.report).resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["safe_to_start"] else 1

    if args.job_command == "execute-v1":
        if args.dry_run:
            report = run_preflight_v2(Path(args.bundle))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["safe_to_start"] else 1
        report = execute_roundtrip_v1(Path(args.bundle))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"PUBLISHED", "ALREADY_PUBLISHED"} else 1

    if args.job_command == "dispatch":
        parameters = parse_key_values(args.param)
        registry = read_json(PROJECT_ROOT / "experiments" / "experiment_registry.json")
        experiment = next((item for item in registry["experiments"] if item["id"] == args.experiment_id), None)
        is_mpp_task = args.command_id in {"cache_parity", "standard_training", "mpp_pathway_ridge_calibration"} and experiment and (
            str(experiment.get("family", "")).startswith("mpp") or "mpp" in str(experiment.get("script", "")).lower()
        )
        report = validate_state(
            PROJECT_ROOT,
            strict=True,
            task="training" if is_mpp_task else "general",
            host_scope="local",
        )
        if not report.ok:
            report.emit()
            raise ValueError("job dispatch blocked by current project state")
        manifest = create_job_manifest(
            PROJECT_ROOT,
            job_id=args.job_id,
            experiment_id=args.experiment_id,
            phase=args.phase,
            command_id=args.command_id,
            path_ids=args.path_id,
            parameters=parameters,
            approval_path=Path(args.approval).resolve() if args.approval else None,
            data_manifest_id=args.data_manifest_id,
            path_index_version=args.path_index_version,
        )
        output = Path(args.output).resolve() if args.output else PROJECT_ROOT / "automation" / "jobs" / args.job_id / "job.json"
        write_json_atomic(output, manifest)
        print(f"[PASS] wrote job envelope: {output}")
        print("[INFO] Push the containing branch through Gitee; no direct server connection was attempted.")
        return 0

    if args.job_command == "validate":
        manifest = read_json(Path(args.manifest).resolve())
        validate_job_manifest(PROJECT_ROOT, manifest, require_head=args.require_head)
        print("[PASS] job manifest is valid")
        return 0

    if args.job_command == "run":
        manifest_path = Path(args.manifest).resolve()
        manifest = read_json(manifest_path)
        validate_job_manifest(PROJECT_ROOT, manifest, require_head=False)
        work_root = ensure_job_worktree(manifest)
        source_state = read_json(work_root / "project_state" / "current_state.json")
        if int(source_state.get("state_revision", -1)) != int(manifest["state_revision"]):
            raise ValueError("job state_revision does not match the pinned source commit")
        registry = read_json(work_root / "experiments" / "experiment_registry.json")
        experiment = next(item for item in registry["experiments"] if item["id"] == manifest["experiment_id"])
        is_mpp_task = manifest["command_id"] in {"cache_parity", "standard_training", "mpp_pathway_ridge_calibration"} and (
            str(experiment.get("family", "")).startswith("mpp") or "mpp" in str(experiment.get("script", "")).lower()
        )
        report = validate_state(
            work_root,
            strict=True,
            task="training" if is_mpp_task else "general",
            host_scope="server",
        )
        report.emit()
        if not report.ok:
            print("[BLOCKED] Server job preflight failed")
            return 1
        if is_mpp_task:
            repair = active_mpp_repair(work_root)
            if repair is None:
                raise ValueError("MPP training has no active repaired-label evidence")
            verified = verify_mpp_repair_server_assets(work_root, repair)
            print(
                "[PASS] repaired-label staging revalidated: "
                f"assets={verified['verified_generated_assets']} audit={verified['audit_sha256']}"
            )
        if args.dry_run:
            print(f"[PASS] pinned detached-worktree dry-run complete: {work_root}")
            return 0
        if manifest["command_id"] == "state_preflight":
            print(f"[PASS] pinned worktree preflight complete: {work_root}")
            return 0
        script = experiment.get("script")
        if not script or not (work_root / script).exists():
            raise ValueError(f"registered training script is missing: {script}")
        parameter_argv = bound_job_parameter_argv(work_root, manifest, experiment)
        argument_string = subprocess.list2cmdline(parameter_argv)
        command = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(work_root / "deploy" / "run_experiment.ps1"),
            "-ExperimentId", manifest["experiment_id"],
            "-Script", script,
            "-Arguments", argument_string,
            "-CheckRegistry",
        ]
        print(f"[INFO] launching allowlisted experiment entry: {manifest['command_id']}")
        completed = subprocess.run(command, cwd=work_root, check=False)
        return int(completed.returncode)

    if args.job_command == "run-v2":
        manifest = read_json(Path(args.manifest).resolve())
        plan = prepare_job_v2_execution(
            manifest,
            governance_root=PROJECT_ROOT,
            source_worktree=Path(args.source_worktree),
            run_root=Path(args.run_root),
            governance_repo=Path(args.governance_repo) if args.governance_repo else None,
            governance_commit=args.governance_commit,
        )
        if args.dry_run:
            print(json.dumps({"status": "validated", **plan}, ensure_ascii=False, indent=2))
            return 0
        started_path = write_job_v2_started_event(plan, manifest)
        try:
            completed = subprocess.run(plan["command"], cwd=plan["source_worktree"], check=False)
            returncode = int(completed.returncode)
            terminal_path = write_job_v2_terminal_event(
                plan, manifest, returncode=returncode,
            )
        except OSError as exc:
            returncode = 127
            terminal_path = write_job_v2_terminal_event(
                plan, manifest, returncode=returncode, error=str(exc),
            )
        print(f"[INFO] server-returnable started event: {started_path}")
        print(f"[INFO] server-returnable terminal event: {terminal_path}")
        return returncode

    if args.job_command == "pack":
        job = read_json(Path(args.manifest).resolve())
        validate_job_manifest(
            PROJECT_ROOT,
            job,
            require_head=False,
            source_git_root=Path(args.source_worktree).resolve() if args.source_worktree else None,
        )
        metrics = read_json(Path(args.metrics_json).resolve()) if args.metrics_json else {}
        artifacts = [Path(item).resolve() for item in args.artifact]
        large_artifacts = [Path(item).resolve() for item in args.large_artifact]
        envelope = build_result_envelope(
            job=job,
            status=args.status,
            output_dir=Path(args.output).resolve(),
            artifact_paths=artifacts,
            metrics=metrics,
            large_artifact_paths=large_artifacts,
        )
        print(f"[PASS] result envelope created: {envelope['result_id']}")
        return 0

    if args.job_command == "import":
        result = import_result_bundle(PROJECT_ROOT, Path(args.bundle).resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unknown job command: {args.job_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PFMval state, Gitee job and result maintenance")
    sub = parser.add_subparsers(dest="command", required=True)

    state = sub.add_parser("state")
    state_sub = state.add_subparsers(dest="state_command", required=True)
    record = state_sub.add_parser("record-directive")
    record.add_argument("--summary", required=True)
    record.add_argument("--scope", required=True)
    record.add_argument("--topic", required=True)
    record.add_argument("--supersedes", action="append", default=[])
    record.add_argument("--affected-file", action="append", default=[])
    record.add_argument("--related-experiment-id", action="append", default=[])
    record.add_argument("--review-after")
    record.add_argument("--completion-evidence", action="append", default=[])
    directives = state_sub.add_parser("directives")
    directives.add_argument("--check-stale", action="store_true", required=True)
    directives.add_argument("--older-than-days", type=int, default=14)
    directive = state_sub.add_parser("directive")
    directive_sub = directive.add_subparsers(dest="directive_command", required=True)
    transition = directive_sub.add_parser("transition")
    transition.add_argument("--id", dest="directive_id", required=True)
    transition.add_argument("--status", choices=["completed", "superseded", "cancelled"], required=True)
    transition.add_argument("--reason", required=True)
    transition.add_argument("--evidence-ref", action="append", default=[])
    transition.add_argument("--superseded-by")
    sync = state_sub.add_parser("sync")
    sync.add_argument("--force-revision", action="store_true")
    sync.add_argument(
        "--with-docs",
        action="store_true",
        help="先重扫 document registry 与 PROJECT_GUIDE，再同步状态（原子化修复路径）",
    )
    state_sub.add_parser("migrate", help="one-time initial migration of provenance and document lifecycle")
    validate = state_sub.add_parser("validate")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument(
        "--task",
        choices=["general", "server", "training", "knowledge"],
        default="general",
    )
    validate.add_argument("--host-scope", choices=["local", "server"], default=None)

    docs = sub.add_parser("docs")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)
    scan = docs_sub.add_parser("scan")
    scan.add_argument("--write", action="store_true")
    set_lifecycle = docs_sub.add_parser(
        "set-lifecycle",
        help="显式设置文档 lifecycle_override（P0-1）；随后运行 state sync --with-docs",
    )
    set_lifecycle.add_argument("--doc-id", required=True)
    set_lifecycle.add_argument(
        "--lifecycle",
        choices=["draft", "pending_review", "approved_design", "active", "superseded", "historical"],
        required=True,
    )
    docs_sub.add_parser("guide")

    paths = sub.add_parser("paths")
    paths_sub = paths.add_subparsers(dest="paths_command", required=True)
    paths_sub.add_parser("build-index")
    path_validate = paths_sub.add_parser("validate")
    path_validate.add_argument("--training", action="store_true")
    path_validate.add_argument("--task", choices=["general", "server", "training"], default="general")
    path_validate.add_argument("--host-scope", choices=["local", "server"], default=None)
    promote = paths_sub.add_parser(
        "promote-md-import",
        help="将 md_import 中经用户核验的路径晋升到主表 paths（先 dry-run 后 --write）",
    )
    promote.add_argument("--ids", nargs="+", required=True)
    promote.add_argument("--required-on", choices=["server", "local", "both"], required=True)
    promote.add_argument("--write", action="store_true")

    result = sub.add_parser("result")
    result_sub = result.add_subparsers(dest="result_command", required=True)
    result_import = result_sub.add_parser("import")
    result_import.add_argument("--bundle", required=True)
    result_w007 = result_sub.add_parser("import-w007-four-arm")
    result_w007.add_argument(
        "--quarantine-root",
        default=str(PROJECT_ROOT / "project_state" / "inbox" / "W007" / "quarantine"),
    )
    result_w007.add_argument("--write", action="store_true")

    mpp = sub.add_parser("mpp")
    mpp_sub = mpp.add_subparsers(dest="mpp_command", required=True)
    repair = mpp_sub.add_parser("repair")
    repair_sub = repair.add_subparsers(dest="repair_command", required=True)
    repair_import = repair_sub.add_parser("import")
    repair_import.add_argument("--git-ref", required=True)
    repair_import.add_argument("--evidence-path", required=True)
    repair_import.add_argument("--expected-audit-sha256", required=True)
    repair_activate = repair_sub.add_parser("activate")
    repair_activate.add_argument("--evidence-id", required=True)
    repair_activate.add_argument("--directive-id", required=True)

    agent = sub.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    start = agent_sub.add_parser("start-check")
    start.add_argument("--strict", action="store_true")
    start.add_argument(
        "--task",
        choices=["general", "server", "training", "diagnostic", "knowledge"],
        default="general",
    )
    start.add_argument("--host-scope", choices=["local", "server"], default=None)

    diagnostic = sub.add_parser("diagnostic")
    diagnostic_sub = diagnostic.add_subparsers(dest="diagnostic_command", required=True)
    diagnostic_request = diagnostic_sub.add_parser("request")
    diagnostic_request.add_argument("--diagnostic-id", required=True)
    diagnostic_request.add_argument("--source-commit", required=True)
    diagnostic_request.add_argument("--source-branch", required=True)
    diagnostic_request.add_argument("--return-branch", default="automation/diagnostics")
    diagnostic_request.add_argument("--command-id", choices=sorted(DIAGNOSTIC_COMMANDS), required=True)
    diagnostic_record = diagnostic_sub.add_parser("record")
    diagnostic_record.add_argument("--diagnostic-id", required=True)
    diagnostic_record.add_argument("--output", action="append", required=True)
    diagnostic_run = diagnostic_sub.add_parser("run-allowlisted")
    diagnostic_run.add_argument("--diagnostic-id", required=True)

    explore = sub.add_parser("explore")
    explore_sub = explore.add_subparsers(dest="explore_command", required=True)
    explore_new = explore_sub.add_parser("new")
    explore_new.add_argument("--session-id", required=True)
    explore_new.add_argument("--purpose", required=True)
    explore_sub.add_parser("list")
    explore_candidates = explore_sub.add_parser("candidates")
    explore_candidates.add_argument("--older-than-days", type=int, default=30)
    explore_promote = explore_sub.add_parser("promote")
    explore_promote.add_argument("--script", required=True)
    explore_promote.add_argument("--target", required=True)
    explore_promote.add_argument("--directive", required=True)

    workspace = sub.add_parser("workspace")
    workspace_sub = workspace.add_subparsers(dest="workspace_command", required=True)
    workspace_sub.add_parser("init")
    workspace_status = workspace_sub.add_parser("status")
    workspace_status.add_argument("--workspace-id")
    workspace_scan = workspace_sub.add_parser("scan")
    workspace_scan.add_argument("--host", choices=["local", "server"], required=True)
    workspace_locus = workspace_sub.add_parser("check-locus")
    workspace_locus.add_argument("--workspace-id", required=True)
    workspace_locus.add_argument("--cwd", required=True)
    workspace_close = workspace_sub.add_parser("close")
    workspace_close.add_argument("--workspace-id", required=True)
    workspace_close.add_argument("--preview", action="store_true", required=True)
    workspace_close.add_argument("--observed-json")

    asset = sub.add_parser("asset")
    asset_sub = asset.add_subparsers(dest="asset_command", required=True)
    asset_inventory = asset_sub.add_parser("inventory")
    asset_inventory.add_argument("--mode", choices=["shadow"], required=True)
    asset_inventory.add_argument("--output")
    asset_sub.add_parser("validate")
    asset_policy = asset_sub.add_parser("policy")
    asset_policy.add_argument("--path", action="append", required=True)
    asset_sub.add_parser("lifecycle-preview")

    views = sub.add_parser("views")
    views_sub = views.add_subparsers(dest="views_command", required=True)
    views_refresh = views_sub.add_parser("refresh")
    views_refresh.add_argument("--experiments", action="store_true")

    science = sub.add_parser("science")
    science_sub = science.add_subparsers(
        dest="science_command",
        required=True,
    )
    science_record = science_sub.add_parser("record")
    science_record.add_argument("--input", required=True)
    science_decision = science_sub.add_parser("decision")
    science_decision.add_argument("--input", required=True)
    science_list = science_sub.add_parser("list")
    science_list.add_argument(
        "--record-type",
        choices=[
            "idea",
            "explore",
            "candidate",
            "claim",
            "negative_result",
            "conflict",
            "explanation",
        ],
    )
    science_adjudicate = science_sub.add_parser("adjudicate")
    science_adjudicate.add_argument("--input", required=True)

    knowledge = sub.add_parser("knowledge")
    knowledge_sub = knowledge.add_subparsers(
        dest="knowledge_command",
        required=True,
    )
    knowledge_fact_preview = knowledge_sub.add_parser("fact-preview")
    knowledge_fact_preview.add_argument("--input", required=True)
    knowledge_fact_confirm = knowledge_sub.add_parser("fact-confirm")
    knowledge_fact_confirm.add_argument("--input", required=True)
    knowledge_fact_confirm.add_argument(
        "--confirmation-source",
        choices=["explicit_user_confirmation"],
        required=True,
    )
    knowledge_sub.add_parser("fact-list")
    knowledge_profile = knowledge_sub.add_parser("profile")
    knowledge_profile.add_argument(
        "--profile",
        choices=[
            "project_fact",
            "meeting_brief",
            "research_path",
            "learning_guide",
            "paper_output",
            "lifecycle_review",
        ],
        required=True,
    )
    knowledge_profile.add_argument("--input", required=True)
    knowledge_profile.add_argument("--output")
    knowledge_freshness = knowledge_sub.add_parser("freshness")
    knowledge_freshness.add_argument("--document-id", required=True)
    knowledge_freshness.add_argument("--fingerprints")

    workflow = sub.add_parser("workflow")
    workflow_sub = workflow.add_subparsers(
        dest="workflow_command",
        required=True,
    )
    workflow_discover = workflow_sub.add_parser("discover")
    workflow_discover.add_argument("--scope-manifest", required=True)
    workflow_discover.add_argument("--record", action="store_true")
    workflow_register = workflow_sub.add_parser("register")
    workflow_register.add_argument("--manifest", required=True)
    workflow_register.add_argument("--authorization-ref", required=True)
    workflow_sub.add_parser("list")
    workflow_review = workflow_sub.add_parser("review")
    workflow_review.add_argument("--workflow-id", required=True)
    workflow_activate = workflow_sub.add_parser("activate")
    workflow_activate.add_argument("--workflow-id", required=True)
    workflow_activate.add_argument("--standardized", action="store_true")
    workflow_revise = workflow_sub.add_parser("revise")
    workflow_revise.add_argument("--workflow-id", required=True)
    workflow_revise.add_argument("--new-workflow-id", required=True)
    workflow_revise.add_argument("--version", required=True)
    workflow_merge = workflow_sub.add_parser("merge")
    workflow_merge.add_argument("--workflow-ids", nargs="+", required=True)
    workflow_merge.add_argument("--new-workflow-id", required=True)
    workflow_merge.add_argument("--purpose", required=True)
    workflow_merge.add_argument("--version", required=True)
    workflow_deprecate = workflow_sub.add_parser("deprecate")
    workflow_deprecate.add_argument("--workflow-id", required=True)
    workflow_deprecate.add_argument(
        "--replacement-workflow-ids",
        nargs="*",
        default=[],
    )
    workflow_sub.add_parser("validate")

    experiment = sub.add_parser("experiment")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_register = experiment_sub.add_parser("register")
    experiment_register.add_argument("--experiment-id", required=True)
    experiment_register.add_argument("--display-name", required=True)
    experiment_register.add_argument(
        "--phase",
        choices=["preflight", "smoke", "formal"],
        required=True,
    )
    experiment_register.add_argument("--run-limit", type=int, required=True)
    experiment_register.add_argument("--critical-contract", required=True)
    experiment_workspace = experiment_register.add_mutually_exclusive_group(required=True)
    experiment_workspace.add_argument("--workspace-relative-path")
    experiment_workspace.add_argument("--workspace-path")
    experiment_register.add_argument("--workspace-branch", default="")
    experiment_approve = experiment_sub.add_parser("approve")
    experiment_approve.add_argument("--approval-id", required=True)
    experiment_approve.add_argument("--experiment-id", required=True)
    experiment_approve.add_argument("--protocol-revision", type=int, required=True)
    experiment_approve.add_argument(
        "--phase",
        choices=["preflight", "smoke", "formal"],
        required=True,
    )
    experiment_approve.add_argument("--run-limit", type=int, required=True)
    experiment_approve.add_argument("--critical-contract-sha256", required=True)
    experiment_revise = experiment_sub.add_parser("revise")
    experiment_revise.add_argument("--experiment-id", required=True)
    experiment_revise.add_argument("--critical-contract", required=True)
    experiment_show = experiment_sub.add_parser("show")
    experiment_show.add_argument("--experiment-id", required=True)

    attempt = sub.add_parser("attempt")
    attempt_sub = attempt.add_subparsers(dest="attempt_command", required=True)
    attempt_prepare = attempt_sub.add_parser("prepare")
    attempt_prepare.add_argument("--experiment-id", required=True)
    attempt_prepare.add_argument("--workspace-id", required=True)
    attempt_prepare.add_argument("--approval-id", required=True)
    attempt_prepare.add_argument("--run-units", type=int, required=True)
    attempt_prepare.add_argument("--source-commit", required=True)
    attempt_prepare.add_argument("--critical-contract-sha256", required=True)
    attempt_prepare.add_argument("--output", required=True)
    attempt_event = attempt_sub.add_parser("event")
    attempt_event.add_argument("--workspace-id", required=True)
    attempt_event.add_argument("--attempt-id", required=True)
    attempt_event.add_argument("--event-id", required=True)
    attempt_event.add_argument(
        "--event-type",
        choices=[
            "PREFLIGHT_FAILED",
            "VALIDATED",
            "EXPERIMENT_STARTED",
            "ATTEMPT_FAILED",
            "RESULT_RETURNED",
            "IMPORTED",
            "REJECTED",
        ],
        required=True,
    )

    governance = sub.add_parser("governance")
    governance_sub = governance.add_subparsers(
        dest="governance_command",
        required=True,
    )
    governance_contract = governance_sub.add_parser("contract")
    governance_contract.add_argument("--input", required=True)
    governance_contract.add_argument("--output", required=True)
    governance_job = governance_sub.add_parser("job-v2")
    governance_job_sub = governance_job.add_subparsers(
        dest="job_v2_command",
        required=True,
    )
    governance_job_build = governance_job_sub.add_parser("build")
    governance_job_build.add_argument("--prepared", required=True)
    governance_job_build.add_argument("--input-binding", required=True)
    governance_job_build.add_argument("--resolved-argv", required=True)
    governance_job_build.add_argument("--command-id", choices=sorted(JOB_V2_COMMANDS), required=True)
    governance_job_build.add_argument("--path-id", action="append", default=[])
    governance_job_build.add_argument("--adaptation-record-id")
    governance_job_build.add_argument("--dispatch-revision", type=int, default=1)
    governance_job_build.add_argument("--output", required=True)
    governance_job_validate = governance_job_sub.add_parser("validate")
    governance_job_validate.add_argument("--manifest", required=True)
    governance_result = governance_sub.add_parser("result-v2")
    governance_result_sub = governance_result.add_subparsers(
        dest="result_v2_command",
        required=True,
    )
    governance_result_build = governance_result_sub.add_parser("build")
    governance_result_build.add_argument("--job", required=True)
    governance_result_build.add_argument("--result-id", required=True)
    governance_result_build.add_argument(
        "--status",
        choices=["success", "failed", "incomplete"],
        required=True,
    )
    governance_result_build.add_argument("--artifacts", required=True)
    governance_result_build.add_argument("--metrics", required=True)
    governance_result_build.add_argument(
        "--metric-artifact-id",
        action="append",
        default=[],
    )
    governance_result_build.add_argument("--large-artifacts")
    governance_result_build.add_argument("--output", required=True)
    governance_result_validate = governance_result_sub.add_parser("validate")
    governance_result_validate.add_argument("--manifest", required=True)
    governance_result_record = governance_result_sub.add_parser("record-import")
    governance_result_record.add_argument("--manifest", required=True)
    governance_result_record.add_argument("--bundle-sha256", required=True)
    governance_result_bundle = governance_sub.add_parser("result-bundle-v1")
    governance_result_bundle_sub = governance_result_bundle.add_subparsers(
        dest="result_bundle_v1_command",
        required=True,
    )
    result_bundle_build = governance_result_bundle_sub.add_parser("build")
    result_bundle_build.add_argument("--source-bundle", required=True)
    result_bundle_build.add_argument("--staging", required=True)
    result_bundle_build.add_argument("--result-id", required=True)
    result_bundle_build.add_argument("--artifact-retention", required=True)
    result_bundle_build.add_argument("--created-at")
    result_bundle_validate = governance_result_bundle_sub.add_parser("validate")
    result_bundle_validate.add_argument("--bundle", required=True)
    result_bundle_record = governance_result_bundle_sub.add_parser("record-import")
    result_bundle_record.add_argument("--bundle", required=True)
    result_bundle_record.add_argument("--bundle-sha256", required=True)
    result_bundle_publish = governance_result_bundle_sub.add_parser("publish")
    result_bundle_publish.add_argument("--bundle", required=True)
    result_bundle_publish.add_argument("--remote", choices=["gitee"], default="gitee")
    result_bundle_publish.add_argument("--ref", required=True)
    result_bundle_publish.add_argument("--revision-path", required=True)
    result_bundle_publish.add_argument("--expected-parent", required=True)
    result_bundle_publish.add_argument("--commit-message", required=True)
    governance_execution_bundle = governance_sub.add_parser(
        "execution-bundle-v1"
    )
    governance_execution_bundle_sub = governance_execution_bundle.add_subparsers(
        dest="execution_bundle_v1_command",
        required=True,
    )
    execution_bundle_build = governance_execution_bundle_sub.add_parser("build")
    execution_bundle_build.add_argument("--job", required=True)
    execution_bundle_build.add_argument("--output", required=True)
    execution_bundle_validate = governance_execution_bundle_sub.add_parser(
        "validate"
    )
    execution_bundle_validate.add_argument("--bundle", required=True)
    governance_result_pair = governance_sub.add_parser("result-pair")
    governance_result_pair_sub = governance_result_pair.add_subparsers(
        dest="result_pair_command",
        required=True,
    )
    result_pair_finalize = governance_result_pair_sub.add_parser("finalize")
    result_pair_finalize.add_argument("--input", required=True)

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    preflight_v2 = job_sub.add_parser("preflight-v2")
    preflight_v2.add_argument("--bundle", required=True)
    preflight_v2.add_argument("--report")
    execute_v1 = job_sub.add_parser("execute-v1")
    execute_v1.add_argument("--bundle", required=True)
    execute_v1.add_argument("--dry-run", action="store_true")
    dispatch = job_sub.add_parser("dispatch")
    dispatch.add_argument("--job-id", required=True)
    dispatch.add_argument("--experiment-id", required=True)
    dispatch.add_argument("--phase", choices=["preflight", "smoke", "formal"], required=True)
    dispatch.add_argument("--command-id", choices=["state_preflight", "cache_parity", "standard_training", "mpp_pathway_ridge_calibration"], required=True)
    dispatch.add_argument("--path-id", action="append", default=[])
    dispatch.add_argument("--param", action="append", default=[])
    dispatch.add_argument("--approval")
    dispatch.add_argument("--data-manifest-id")
    dispatch.add_argument("--path-index-version", default="1.0")
    dispatch.add_argument("--output")
    job_validate = job_sub.add_parser("validate")
    job_validate.add_argument("--manifest", required=True)
    job_validate.add_argument("--require-head", action="store_true", help="require current checkout HEAD to equal source_commit")
    run = job_sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--dry-run", action="store_true")
    run_v2 = job_sub.add_parser("run-v2")
    run_v2.add_argument("--manifest", required=True)
    run_v2.add_argument("--source-worktree", required=True)
    run_v2.add_argument("--run-root", required=True)
    run_v2.add_argument("--governance-repo")
    run_v2.add_argument("--governance-commit")
    run_v2.add_argument("--dry-run", action="store_true")
    pack = job_sub.add_parser("pack")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--status", choices=["success", "failed", "incomplete"], required=True)
    pack.add_argument("--artifact", action="append", default=[])
    pack.add_argument("--large-artifact", action="append", default=[], help="server-only file; record path, size and SHA-256 without copying")
    pack.add_argument("--metrics-json")
    pack.add_argument("--output", required=True)
    pack.add_argument("--source-worktree", help="pinned source worktree when packaging from a Gitee governance archive")
    job_import = job_sub.add_parser("import")
    job_import.add_argument("--bundle", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "state":
            return command_state(args)
        if args.command == "docs":
            return command_docs(args)
        if args.command == "paths":
            return command_paths(args)
        if args.command == "result":
            return command_result(args)
        if args.command == "mpp":
            return command_mpp(args)
        if args.command == "agent":
            return command_agent(args)
        if args.command == "diagnostic":
            return command_diagnostic(args)
        if args.command == "explore":
            return command_explore(args)
        if args.command == "workspace":
            return command_workspace(args)
        if args.command == "asset":
            return command_asset(args)
        if args.command == "views":
            return command_views(args)
        if args.command == "science":
            return command_science(args)
        if args.command == "knowledge":
            return command_knowledge(args)
        if args.command == "workflow":
            return command_workflow(args)
        if args.command == "experiment":
            return command_experiment(args)
        if args.command == "attempt":
            return command_attempt(args)
        if args.command == "governance":
            return command_governance(args)
        if args.command == "job":
            return command_job(args)
        raise ValueError(f"unknown command: {args.command}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
