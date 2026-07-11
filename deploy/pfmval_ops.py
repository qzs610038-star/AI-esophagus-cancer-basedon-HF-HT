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
    append_directive,
    build_mpp_path_index,
    build_result_envelope,
    create_job_manifest,
    import_result_bundle,
    migrate_experiment_provenance,
    read_json,
    safe_job_parameters,
    scan_documents,
    state_lock,
    sync_state,
    validate_job_manifest,
    validate_server_paths,
    validate_state,
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
        registry_path = work_root / "configs" / "server_paths.yaml"
        parameters.update({
            "mpp_root": str(get_registered_path("mpp_data_root", registry_path=registry_path, project_root=work_root)),
            "cache_root": str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=work_root)),
            "flat_cache_root": str(get_registered_path("server_mpp_flat_cache", registry_path=registry_path, project_root=work_root)),
            # Standard splits are committed assets and must come from the pinned
            # worktree, not from a mutable checkout at the registered server path.
            "splits_root": str((work_root / "mpp_standard_splits").resolve()),
            "output_root": str(get_registered_path("server_mpp_results", registry_path=registry_path, project_root=work_root)),
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
            )
            state = sync_state(PROJECT_ROOT, force_revision=True)
        print(f"[PASS] recorded {directive_id}; state_revision={state['state_revision']}")
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
        report = validate_state(PROJECT_ROOT, strict=args.strict, task=args.task)
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
        validate_server_paths(PROJECT_ROOT, report, task=task)
        report.emit()
        return 0 if report.ok else 1
    raise ValueError(f"unknown paths command: {args.paths_command}")


def command_result(args: argparse.Namespace) -> int:
    result = import_result_bundle(PROJECT_ROOT, Path(args.bundle).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_agent(args: argparse.Namespace) -> int:
    report = validate_state(PROJECT_ROOT, strict=args.strict, task=args.task)
    report.emit()
    if not report.ok:
        print("[BLOCKED] Resolve critical state conflicts before continuing.")
        return 1
    print("[PASS] Agent start gate passed. Read CURRENT_STATE.md before acting.")
    return 0


def command_job(args: argparse.Namespace) -> int:
    if args.job_command == "dispatch":
        parameters = parse_key_values(args.param)
        registry = read_json(PROJECT_ROOT / "experiments" / "experiment_registry.json")
        experiment = next((item for item in registry["experiments"] if item["id"] == args.experiment_id), None)
        is_mpp_training = args.command_id == "standard_training" and experiment and (
            str(experiment.get("family", "")).startswith("mpp") or "mpp" in str(experiment.get("script", "")).lower()
        )
        report = validate_state(PROJECT_ROOT, strict=True, task="training" if is_mpp_training else "general")
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
        is_mpp_training = manifest["command_id"] == "standard_training" and (
            str(experiment.get("family", "")).startswith("mpp") or "mpp" in str(experiment.get("script", "")).lower()
        )
        report = validate_state(work_root, strict=True, task="training" if is_mpp_training else "server")
        report.emit()
        if not report.ok:
            print("[BLOCKED] Server job preflight failed")
            return 1
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
        print("[INFO] launching allowlisted standard training entry")
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
    sync = state_sub.add_parser("sync")
    sync.add_argument("--force-revision", action="store_true")
    state_sub.add_parser("migrate", help="one-time initial migration of provenance and document lifecycle")
    validate = state_sub.add_parser("validate")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--task", choices=["general", "server", "training"], default="general")

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

    result = sub.add_parser("result")
    result_sub = result.add_subparsers(dest="result_command", required=True)
    result_import = result_sub.add_parser("import")
    result_import.add_argument("--bundle", required=True)

    agent = sub.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    start = agent_sub.add_parser("start-check")
    start.add_argument("--strict", action="store_true")
    start.add_argument("--task", choices=["general", "server", "training"], default="general")

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    dispatch = job_sub.add_parser("dispatch")
    dispatch.add_argument("--job-id", required=True)
    dispatch.add_argument("--experiment-id", required=True)
    dispatch.add_argument("--phase", choices=["preflight", "smoke", "formal"], required=True)
    dispatch.add_argument("--command-id", choices=["state_preflight", "standard_training"], required=True)
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
        if args.command == "agent":
            return command_agent(args)
        if args.command == "job":
            return command_job(args)
        raise ValueError(f"unknown command: {args.command}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
