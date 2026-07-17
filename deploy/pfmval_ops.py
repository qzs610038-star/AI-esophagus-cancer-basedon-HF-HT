#!/usr/bin/env python3
"""Unified local/server maintenance CLI for PFMval.

This command never opens a direct connection to the server.  Job and result
envelopes are files intended to be transported by the configured Gitee remote.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pfmval_state import (  # noqa: E402
    DIAGNOSTIC_COMMANDS,
    activate_mpp_repair_evidence,
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
    exploration_cleanup_candidates,
    list_exploration_sessions,
    migrate_experiment_provenance,
    promote_exploration_script,
    record_diagnostic_outputs,
    read_json,
    safe_job_parameters,
    scan_documents,
    state_lock,
    sync_state,
    transition_directive,
    validate_diagnostic_state,
    validate_job_manifest,
    validate_server_paths,
    validate_state,
    verify_mpp_repair_server_assets,
    ValidationReport,
    write_json_atomic,
)
from path_registry import get_registered_path  # noqa: E402


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
    job_id = str(manifest.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("unsafe job_id for worktree path")
    worktree_root = get_registered_path("server_automation_worktrees").resolve()
    worktree = (worktree_root / job_id).resolve()
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
        if completed.stdout.strip() != manifest["source_commit"]:
            raise ValueError(f"existing job worktree has the wrong commit: {worktree}")
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
        raise ValueError(f"job worktree is dirty and cannot be reused: {worktree}")
    return worktree


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
    raise ValueError(f"unknown paths command: {args.paths_command}")


def command_result(args: argparse.Namespace) -> int:
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
        print("[PASS] recorded non-evidence diagnostic output hashes.")
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


def command_job(args: argparse.Namespace) -> int:
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

    if args.job_command == "pack":
        job = read_json(Path(args.manifest).resolve())
        validate_job_manifest(PROJECT_ROOT, job, require_head=False)
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
    state_sub.add_parser("migrate", help="one-time initial migration of provenance and document lifecycle")
    validate = state_sub.add_parser("validate")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--task", choices=["general", "server", "training"], default="general")
    validate.add_argument("--host-scope", choices=["local", "server"], default=None)

    docs = sub.add_parser("docs")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)
    scan = docs_sub.add_parser("scan")
    scan.add_argument("--write", action="store_true")

    paths = sub.add_parser("paths")
    paths_sub = paths.add_subparsers(dest="paths_command", required=True)
    paths_sub.add_parser("build-index")
    path_validate = paths_sub.add_parser("validate")
    path_validate.add_argument("--training", action="store_true")
    path_validate.add_argument("--task", choices=["general", "server", "training"], default="general")
    path_validate.add_argument("--host-scope", choices=["local", "server"], default=None)

    result = sub.add_parser("result")
    result_sub = result.add_subparsers(dest="result_command", required=True)
    result_import = result_sub.add_parser("import")
    result_import.add_argument("--bundle", required=True)

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
    start.add_argument("--task", choices=["general", "server", "training", "diagnostic"], default="general")
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

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
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
    pack = job_sub.add_parser("pack")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--status", choices=["success", "failed", "incomplete"], required=True)
    pack.add_argument("--artifact", action="append", default=[])
    pack.add_argument("--large-artifact", action="append", default=[], help="server-only file; record path, size and SHA-256 without copying")
    pack.add_argument("--metrics-json")
    pack.add_argument("--output", required=True)
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
        if args.command == "job":
            return command_job(args)
        raise ValueError(f"unknown command: {args.command}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
