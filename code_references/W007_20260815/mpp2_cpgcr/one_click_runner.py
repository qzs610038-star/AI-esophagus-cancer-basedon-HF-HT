"""W007 一键测试、批准门禁训练与仅回传重试状态机。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "protocol_v001.yaml"
SERVER_CONFIG_PATH = PACKAGE_ROOT / "configs" / "server_v001.yaml"
SERVER_PATHS_PATH = PROJECT_ROOT / "configs" / "server_paths.yaml"
TESTS_PATH = PACKAGE_ROOT / "tests"
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mpp2_cpgcr.result_contract import (
    build_failure_payload,
    inventory_files,
    publish_gitee_return,
    recover_failure_gitee_return,
    recover_lfs_rejected_arm_gitee_return,
    retry_gitee_return,
    validate_arm_artifacts,
    write_json,
    write_return_manifest,
)


ARM_ORDER = ("FBR", "RCC", "HCR", "CPGCR")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@dataclass(frozen=True)
class StageResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


class StageFailure(RuntimeError):
    def __init__(self, message: str, *, exit_code: int):
        super().__init__(message)
        self.exit_code = int(exit_code)


class RunContext:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        mode: str,
        source_commit: str,
        resolved_config: Mapping[str, Any],
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.mode = mode
        self.source_commit = source_commit
        self.stdout_path = self.run_dir / "stdout.log"
        self.stderr_path = self.run_dir / "stderr.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.resolved_config_path = self.run_dir / "resolved_config.json"
        self.active_arm: str | None = None
        self.active_job: dict[str, Any] | None = None
        self.failure_stage: str | None = None
        self.failure_path: Path | None = None
        self.return_pending = False
        self.stdout_path.write_bytes(b"")
        self.stderr_path.write_bytes(b"")
        self.events_path.write_bytes(b"")
        write_json(self.resolved_config_path, resolved_config)

    def event(self, event_type: str, *, stage: str, **values: Any) -> None:
        event = {
            "event_type": event_type,
            "stage": stage,
            "recorded_at": utc_now(),
            "source_commit": self.source_commit,
            **values,
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def append_stdout(self, text: str) -> None:
        if not text:
            return
        with self.stdout_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    def append_stderr(self, text: str) -> None:
        if not text:
            return
        with self.stderr_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(text)
        sys.stderr.write(text)
        sys.stderr.flush()

    def write_failure(
        self,
        *,
        stage: str,
        exit_code: int,
        job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> Path:
        active_job = self.active_job or {}
        payload = build_failure_payload(
            stage=stage,
            exit_code=exit_code,
            source_commit=self.source_commit,
            stdout_path=self.stdout_path.name,
            stderr_path=self.stderr_path.name,
            events_path=self.events_path.name,
            resolved_config_path=self.resolved_config_path.name,
            job_id=job_id or active_job.get("job_id"),
            attempt_id=attempt_id or active_job.get("attempt_id"),
        )
        return write_json(self.run_dir / "failure.json", payload)

    def bind_job(self, arm: str, record: Mapping[str, Any]) -> None:
        self.active_arm = arm
        self.active_job = dict(record)


class StageMachine:
    """固定顺序、非零立即停止的最小状态机。"""

    def __init__(self, context: RunContext):
        self.context = context
        self.executed: list[str] = []

    def run(self, stages: Sequence[tuple[str, Callable[[], StageResult | None]]]) -> int:
        for stage_name, action in stages:
            self.context.event("stage_started", stage=stage_name)
            self.executed.append(stage_name)
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                    result = action() or StageResult()
                exit_code = int(result.exit_code)
                stage_stdout = captured_stdout.getvalue() + result.stdout
                stage_stderr = captured_stderr.getvalue() + result.stderr
            except StageFailure as exc:
                exit_code = exc.exit_code
                stage_stdout = captured_stdout.getvalue()
                stage_stderr = captured_stderr.getvalue() + f"{exc}\n"
            except Exception:
                exit_code = 1
                stage_stdout = captured_stdout.getvalue()
                stage_stderr = captured_stderr.getvalue() + traceback.format_exc()
            self.context.append_stdout(stage_stdout)
            self.context.append_stderr(stage_stderr)
            if exit_code != 0:
                failure_path = self.context.write_failure(stage=stage_name, exit_code=exit_code)
                self.context.failure_stage = stage_name
                self.context.failure_path = failure_path
                self.context.event(
                    "stage_failed",
                    stage=stage_name,
                    exit_code=exit_code,
                    failure_path=failure_path.name,
                )
                self.context.append_stderr(
                    f"[HARD] stage={stage_name} exit_code={exit_code}; "
                    f"failure={failure_path}\n"
                )
                return exit_code
            self.context.event("stage_completed", stage=stage_name, exit_code=0)
        self.context.event("run_completed", stage="complete", exit_code=0)
        return 0


def load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    server_config = yaml.safe_load(SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
    server_paths = yaml.safe_load(SERVER_PATHS_PATH.read_text(encoding="utf-8"))
    return protocol, server_config, server_paths


def default_run_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return RUNTIME_ROOT / f"{stamp}-{mode}"


def default_training_run_dir(
    server_config: Mapping[str, Any],
    server_paths: Mapping[str, Any],
    *,
    stamp: str | None = None,
) -> Path:
    path_id = str(server_config.get("run_root_path_id", ""))
    entry = server_paths.get("paths", {}).get(path_id, {})
    root = Path(str(entry.get("path", "")))
    if not path_id or not root.is_absolute():
        raise StageFailure("registered experiment run root is invalid", exit_code=10)
    workspace_id = str(server_config.get("workspace_id", ""))
    if workspace_id != "W007":
        raise StageFailure("training run root workspace identity mismatch", exit_code=10)
    run_stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root / workspace_id / f"{run_stamp}-formal-batch"


def environment_stage() -> StageResult:
    import pytest
    import torch

    return StageResult(stdout=(
        f"python={sys.executable}\n"
        f"python_version={platform.python_version()}\n"
        f"torch={torch.__version__}\n"
        f"cuda_available={torch.cuda.is_available()}\n"
        f"pytest={pytest.__version__}\n"
        f"yaml={yaml.__version__}\n"
        f"git={shutil.which('git')}\n"
    ))


def config_stage(protocol: dict[str, Any], server_config: dict[str, Any]) -> StageResult:
    from mpp2_cpgcr.train_eval import fairness_signature, validate_protocol

    validate_protocol(protocol)
    if server_config.get("workspace_id") != "W007":
        raise StageFailure("server config workspace identity mismatch", exit_code=11)
    if server_config.get("transport") != "gitee_only" or server_config.get("remote") != "gitee":
        raise StageFailure("server transport is not fixed to Gitee", exit_code=12)
    if server_config.get("force_push_allowed") is not False:
        raise StageFailure("force push must remain disabled", exit_code=13)
    external = protocol.get("external_policy", {})
    if (
        external.get("timing") != "once_after_each_internal_checkpoint_freeze"
        or external.get("fixed_order") != list(ARM_ORDER)
        or external.get("freeze_four_arm_configuration_before_first_external") is not True
    ):
        raise StageFailure("external XZY timing/order/freeze contract has drifted", exit_code=14)
    return StageResult(stdout=json.dumps(fairness_signature(protocol), ensure_ascii=False) + "\n")


def subprocess_stage(command: Sequence[str], *, cwd: str | Path) -> StageResult:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return StageResult(completed.returncode, completed.stdout, completed.stderr)


def pytest_stage() -> StageResult:
    basetemp_root = Path(os.environ.get("PFMVAL_PYTEST_BASETEMP", "D:/codex_pytest_tmp"))
    basetemp = basetemp_root / f"W007-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    return subprocess_stage(
        [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_PATH),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(basetemp),
        ],
        cwd=PROJECT_ROOT,
    )


def diagnostic_gate_stage() -> StageResult:
    return subprocess_stage(
        [
            sys.executable,
            str(PROJECT_ROOT / "deploy" / "pfmval_ops.py"),
            "agent",
            "start-check",
            "--task",
            "diagnostic",
        ],
        cwd=PROJECT_ROOT,
    )


def training_approval_path(
    protocol: Mapping[str, Any],
    *,
    server_config: Mapping[str, Any] | None = None,
    server_paths: Mapping[str, Any] | None = None,
) -> Path:
    relative_path = Path(str(protocol["approval"]["fixed_path"]))
    local_path = PROJECT_ROOT / relative_path
    if server_config is None or server_paths is None:
        return local_path
    path_id = str(server_config.get("governance_checkout_path_id", ""))
    entry = server_paths.get("paths", {}).get(path_id, {})
    governance_root = Path(str(entry.get("path", "")))
    if not path_id or not governance_root.is_absolute():
        raise StageFailure("registered governance checkout path is invalid", exit_code=20)
    return governance_root / relative_path


def load_training_approval(
    protocol: Mapping[str, Any],
    source_commit: str,
    *,
    server_config: Mapping[str, Any] | None = None,
    server_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    approval_path = training_approval_path(
        protocol, server_config=server_config, server_paths=server_paths
    )
    if not approval_path.is_file():
        raise StageFailure(
            f"正式训练未获批准：缺少固定批准文件 {approval_path}",
            exit_code=20,
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    required_identity = {
        "status": "approved",
        "workspace_id": "W007",
        "experiment_id": protocol["experiment_id"],
        "protocol_revision": protocol["protocol_revision"],
        "source_commit": source_commit,
        "authorization_directive": protocol["authorization_directive"],
        "critical_contract_sha256": protocol["critical_contract_sha256"],
        "data_manifest_id": protocol["data_manifest_id"],
    }
    for key, expected in required_identity.items():
        if approval.get(key) != expected:
            raise StageFailure(f"training approval {key} mismatch", exit_code=21)
    if approval.get("arms") != list(ARM_ORDER):
        raise StageFailure("training approval must freeze the four-arm order", exit_code=22)
    if approval.get("run_units") != 3 or approval.get("run_limit", 0) < 3:
        raise StageFailure("training approval lacks the three-attempt budget", exit_code=23)
    jobs = approval.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(ARM_ORDER):
        raise StageFailure("training approval must bind all four job records", exit_code=24)
    job_ids: list[str] = []
    attempt_ids: list[str] = []
    return_branches: list[str] = []
    staging_roots: list[str] = []
    for arm in ARM_ORDER:
        record = jobs[arm]
        if not record.get("job_id") or not record.get("return_branch") or not record.get("return_staging_root"):
            raise StageFailure(f"training approval job record incomplete: {arm}", exit_code=25)
        if arm != "FBR" and not record.get("attempt_id"):
            raise StageFailure(f"training approval attempt missing: {arm}", exit_code=26)
        return_branch = str(record["return_branch"])
        if not return_branch.startswith("automation/server/W007/"):
            raise StageFailure(f"training approval return branch is outside W007: {arm}", exit_code=28)
        staging_root = Path(str(record["return_staging_root"]))
        if not staging_root.is_absolute():
            raise StageFailure(f"training approval staging root is not absolute: {arm}", exit_code=29)
        job_ids.append(str(record["job_id"]))
        return_branches.append(return_branch)
        staging_roots.append(str(staging_root))
        if record.get("attempt_id"):
            attempt_ids.append(str(record["attempt_id"]))
    for label, values in (
        ("job_id", job_ids),
        ("attempt_id", attempt_ids),
        ("return_branch", return_branches),
        ("return_staging_root", staging_roots),
    ):
        if len(values) != len(set(values)):
            raise StageFailure(f"training approval contains duplicate {label}", exit_code=29)
    if approval.get("xzy_policy_ack") != "frozen_once_per_arm_no_selection_no_rerun":
        raise StageFailure("training approval does not acknowledge the XZY boundary", exit_code=27)
    return approval


def approval_stage(
    protocol: Mapping[str, Any],
    source_commit: str,
    holder: dict[str, Any],
    server_config: Mapping[str, Any] | None = None,
    server_paths: Mapping[str, Any] | None = None,
    context: RunContext | None = None,
) -> StageResult:
    holder["approval"] = load_training_approval(
        protocol,
        source_commit,
        server_config=server_config,
        server_paths=server_paths,
    )
    if context is not None:
        context.bind_job("FBR", holder["approval"]["jobs"]["FBR"])
    return StageResult(stdout="training approval gate passed before GPU access\n")


def source_identity_stage(source_commit: str) -> StageResult:
    actual_root = Path(git_output("rev-parse", "--show-toplevel")).resolve()
    if actual_root != PROJECT_ROOT.resolve():
        raise StageFailure("Git worktree root does not match the W007 source root", exit_code=31)
    if git_output("rev-parse", "HEAD") != source_commit:
        raise StageFailure("source HEAD changed after source_commit resolution", exit_code=32)
    dirty = git_output("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise StageFailure("source worktree is dirty; formal execution requires a frozen tree", exit_code=33)
    return StageResult(stdout=f"source_commit={source_commit}\nsource_tree=clean\n")


def _write_four_arm_freeze(
    run_dir: Path,
    *,
    protocol: Mapping[str, Any],
    approval: Mapping[str, Any],
    source_commit: str,
) -> Path:
    return write_json(run_dir / "four_arm_freeze.json", {
        "schema_version": "1.0",
        "workspace_id": "W007",
        "experiment_id": protocol["experiment_id"],
        "protocol_revision": protocol["protocol_revision"],
        "source_commit": source_commit,
        "arms": list(ARM_ORDER),
        "jobs": copy.deepcopy(approval["jobs"]),
        "hyperparameters": {
            "training": copy.deepcopy(protocol["training"]),
            "loss": copy.deepcopy(protocol["loss"]),
            "model": copy.deepcopy(protocol["model"]),
        },
        "xzy_policy": copy.deepcopy(protocol["external_policy"]),
        "frozen_before_first_xzy": True,
    })


def write_attempt_started(
    arm_dir: Path,
    *,
    arm: str,
    protocol: Mapping[str, Any],
    approval: Mapping[str, Any],
    job_record: Mapping[str, Any],
    source_commit: str,
) -> Path:
    """在真实拟合调用前记录一次预算已消耗的服务器原始事件。"""
    if arm == "FBR" or not job_record.get("attempt_id"):
        raise ValueError("EXPERIMENT_STARTED only applies to trainable attempts")
    return write_json(arm_dir / "attempt_started.json", {
        "event_type": "EXPERIMENT_STARTED",
        "event_id": f"started-{job_record['job_id']}",
        "recorded_at": utc_now(),
        "experiment_id": protocol["experiment_id"],
        "workspace_id": "W007",
        "attempt_id": job_record["attempt_id"],
        "job_id": job_record["job_id"],
        "approval_id": approval["approval_id"],
        "run_units": 1,
        "protocol_revision": protocol["protocol_revision"],
        "phase": "formal",
        "source_commit": source_commit,
        "critical_contract_sha256": approval["critical_contract_sha256"],
    })


def validate_resume(run_dir: str | Path) -> dict[str, Any]:
    """确认既有批次只完成了连续前缀；续接不得重跑已完成臂。"""

    run_root = Path(run_dir).resolve()
    freeze_path = run_root / "four_arm_freeze.json"
    for required in (freeze_path,):
        if not required.is_file():
            raise StageFailure(f"resume evidence missing: {required}", exit_code=41)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("workspace_id") != "W007"
        or freeze.get("arms") != list(ARM_ORDER)
        or freeze.get("frozen_before_first_xzy") is not True
    ):
        raise StageFailure("resume freeze contract mismatch", exit_code=41)
    present = [arm for arm in ARM_ORDER if (run_root / arm).exists()]
    provenance_path = run_root / "resume_provenance.json"
    prior_arms: list[str] = []
    prior_return_commits: dict[str, str] = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        prior_arms = list(provenance.get("completed_arms", ()))
        prior_return_commits = dict(provenance.get("completed_return_commits", {}))
        if not prior_arms and provenance.get("fbr_return_commit"):
            prior_arms = ["FBR"]
            prior_return_commits = {"FBR": str(provenance["fbr_return_commit"])}
    combined = [*prior_arms, *present]
    if not present or combined != list(ARM_ORDER[:len(combined)]) or len(combined) >= len(ARM_ORDER):
        raise StageFailure("resume arms are not one incomplete four-arm prefix", exit_code=41)
    return_commits: dict[str, str] = dict(prior_return_commits)
    for arm in present:
        manifest_path = run_root / arm / "return_manifest.json"
        published_path = run_root / arm / "return_published.json"
        for required in (manifest_path, published_path):
            if not required.is_file():
                raise StageFailure(f"resume evidence missing: {required}", exit_code=41)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = json.loads(published_path.read_text(encoding="utf-8"))
        if (
            manifest.get("workspace_id") != "W007"
            or manifest.get("source_commit") != freeze.get("source_commit")
            or manifest.get("retry_only") is not False
        ):
            raise StageFailure(f"{arm} resume manifest mismatch", exit_code=41)
        if published.get("return_commit") != published.get("remote_commit"):
            raise StageFailure(f"{arm} resume return was not published exactly", exit_code=41)
        return_commits[arm] = str(published["return_commit"])
    remaining = ARM_ORDER[len(combined):]
    return {
        "prior_run_dir": str(run_root),
        "prior_source_commit": str(freeze["source_commit"]),
        "completed_arms": tuple(combined),
        "completed_return_commits": return_commits,
        "remaining_arms": remaining,
    }


def validate_fbr_resume(run_dir: str | Path) -> dict[str, Any]:
    """兼容既有调用：确认仅 FBR 完成并返回原字段。"""

    resume = validate_resume(run_dir)
    if resume["completed_arms"] != ("FBR",):
        raise StageFailure("FBR resume requires exactly one completed arm", exit_code=41)
    return {
        **resume,
        "fbr_return_commit": resume["completed_return_commits"]["FBR"],
    }


def training_batch_stage(
    context: RunContext,
    *,
    protocol: dict[str, Any],
    server_paths: dict[str, Any],
    approval: dict[str, Any],
    resume_from: Path | None = None,
) -> StageResult:
    import torch

    from mpp2_cpgcr.train_eval import (
        external_benchmark_once,
        prepare_training_context,
        replay_fbr,
        train_arm,
    )

    if not torch.cuda.is_available():
        raise StageFailure("formal W007 batch requires CUDA after approval", exit_code=30)
    resume = validate_resume(resume_from) if resume_from is not None else None
    arms = resume["remaining_arms"] if resume is not None else ARM_ORDER
    context.bind_job(arms[0], approval["jobs"][arms[0]])
    freeze_path = _write_four_arm_freeze(
        context.run_dir,
        protocol=protocol,
        approval=approval,
        source_commit=context.source_commit,
    )
    training_context = prepare_training_context(
        protocol,
        server_paths,
        batch_plan_output=context.run_dir / "batch_plan.csv",
    )
    frozen_config = json.loads(freeze_path.read_text(encoding="utf-8"))
    if resume is not None:
        write_json(context.run_dir / "resume_provenance.json", resume)
    device = torch.device("cuda")
    for arm in arms:
        context.bind_job(arm, approval["jobs"][arm])
        arm_dir = context.run_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=False)
        if arm == "FBR":
            model, artifacts = replay_fbr(
                training_context,
                device=device,
                output_dir=arm_dir,
            )
        else:
            started_path = write_attempt_started(
                arm_dir,
                arm=arm,
                protocol=protocol,
                approval=approval,
                job_record=approval["jobs"][arm],
                source_commit=context.source_commit,
            )
            model, artifacts = train_arm(
                training_context,
                arm=arm,
                device=device,
                output_dir=arm_dir,
            )
            artifacts["attempt_started"] = started_path
        external = external_benchmark_once(
            training_context,
            model=model,
            device=device,
            output_dir=arm_dir,
            frozen_config=frozen_config,
        )
        artifacts.update(external)
        metrics_path = write_json(arm_dir / "metrics.json", {
            "internal": json.loads(Path(artifacts["internal_metrics"]).read_text(encoding="utf-8")),
            "external": json.loads(Path(artifacts["external_metrics"]).read_text(encoding="utf-8")),
            "performance_gate": "WARN_only",
        })
        artifacts["metrics"] = metrics_path
        validate_arm_artifacts(arm=arm, successful=True, artifacts=artifacts)
        job_record = approval["jobs"][arm]
        artifact_paths = list(dict.fromkeys(Path(value) for value in artifacts.values()))
        return_manifest = write_return_manifest(
            arm_dir / "return_manifest.json",
            source_commit=context.source_commit,
            return_branch=job_record["return_branch"],
            evidence_class="pending_review",
            artifacts=inventory_files(artifact_paths, root=context.run_dir),
            retry_only=False,
        )
        artifact_paths.append(return_manifest)
        staging_root = (
            Path(job_record["return_staging_root"])
            / "git-return"
        )
        try:
            published = publish_gitee_return(
                project_root=PROJECT_ROOT,
                source_commit=context.source_commit,
                return_branch=job_record["return_branch"],
                run_dir=context.run_dir,
                artifact_paths=artifact_paths,
                staging_root=staging_root,
                commit_message=f"return(w007): {arm} {job_record['job_id']}",
            )
        except Exception as exc:
            pending = {
                "schema_version": "1.0",
                "retry_mode": "return_only",
                "training_retry_allowed": False,
                "arm": arm,
                "job_id": job_record["job_id"],
                "return_branch": job_record["return_branch"],
                "staging_worktree": str(staging_root),
                "return_commit": git_output("-C", str(staging_root), "rev-parse", "HEAD")
                if (staging_root / ".git").exists()
                else None,
                "error": str(exc),
            }
            pending_path = write_json(RUNTIME_ROOT / "pending_return.json", pending)
            context.return_pending = True
            raise StageFailure(
                f"Gitee回传失败；不得重跑训练，仅可执行 {pending_path}",
                exit_code=40,
            ) from exc
        write_json(arm_dir / "return_published.json", published)
    return StageResult(stdout="four-arm fixed batch completed; no automatic rerun\n")


def publish_failure_return(context: RunContext) -> dict[str, str] | None:
    if context.failure_path is None or context.active_job is None or context.return_pending:
        return None
    artifacts = {
        "stdout": context.stdout_path,
        "stderr": context.stderr_path,
        "events": context.events_path,
        "resolved_config": context.resolved_config_path,
        "failure": context.failure_path,
    }
    if context.active_arm:
        started_path = context.run_dir / context.active_arm / "attempt_started.json"
        if started_path.is_file():
            artifacts["attempt_started"] = started_path
    validate_arm_artifacts(
        arm=context.active_arm or "BATCH",
        successful=False,
        artifacts=artifacts,
    )
    job_record = context.active_job
    artifact_paths = list(artifacts.values())
    return_manifest = write_return_manifest(
        context.run_dir / "failure_return_manifest.json",
        source_commit=context.source_commit,
        return_branch=job_record["return_branch"],
        evidence_class="non_evidence_failure",
        artifacts=inventory_files(artifact_paths, root=context.run_dir),
        retry_only=False,
    )
    artifact_paths.append(return_manifest)
    staging_root = Path(job_record["return_staging_root"]) / "git-return"
    try:
        published = publish_gitee_return(
            project_root=PROJECT_ROOT,
            source_commit=context.source_commit,
            return_branch=job_record["return_branch"],
            run_dir=context.run_dir,
            artifact_paths=artifact_paths,
            staging_root=staging_root,
            commit_message=(
                f"return(w007): failure {context.active_arm or 'batch'} "
                f"{job_record['job_id']}"
            ),
        )
    except Exception as exc:
        pending = {
            "schema_version": "1.0",
            "retry_mode": "return_only",
            "training_retry_allowed": False,
            "arm": context.active_arm,
            "job_id": job_record["job_id"],
            "return_branch": job_record["return_branch"],
            "staging_worktree": str(staging_root),
            "return_commit": git_output("-C", str(staging_root), "rev-parse", "HEAD")
            if (staging_root / ".git").exists()
            else None,
            "error": str(exc),
        }
        pending_path = write_json(RUNTIME_ROOT / "pending_return.json", pending)
        context.return_pending = True
        context.append_stderr(
            f"[HARD] failure return could not reach Gitee; retry-return only: {pending_path}\n"
        )
        return None
    write_json(context.run_dir / "failure_return_published.json", published)
    context.event(
        "failure_return_published",
        stage=context.failure_stage or "unknown",
        return_commit=published["return_commit"],
    )
    return published


def run_test_mode(
    *,
    run_dir: Path,
    protocol: dict[str, Any],
    server_config: dict[str, Any],
    server_paths: dict[str, Any],
    source_commit: str,
) -> int:
    context = RunContext(
        run_dir,
        mode="test",
        source_commit=source_commit,
        resolved_config={
            "mode": "test",
            "source_commit": source_commit,
            "protocol": protocol,
            "server": server_config,
            "server_paths_source": str(SERVER_PATHS_PATH),
        },
    )
    machine = StageMachine(context)
    return machine.run([
        ("source_identity", lambda: source_identity_stage(source_commit)),
        ("environment_check", environment_stage),
        ("config_validate", lambda: config_stage(protocol, server_config)),
        ("minimal_pytest", pytest_stage),
        ("pre_sync_diagnostic", diagnostic_gate_stage),
    ])


def run_training_mode(
    *,
    run_dir: Path,
    protocol: dict[str, Any],
    server_config: dict[str, Any],
    server_paths: dict[str, Any],
    source_commit: str,
    resume_from: Path | None = None,
) -> int:
    context = RunContext(
        run_dir,
        mode="train",
        source_commit=source_commit,
        resolved_config={
            "mode": "train",
            "source_commit": source_commit,
            "protocol": protocol,
            "server": server_config,
            "approval_path": protocol["approval"]["fixed_path"],
            "resume_from": str(resume_from) if resume_from is not None else None,
        },
    )
    holder: dict[str, Any] = {}
    machine = StageMachine(context)
    stages: list[tuple[str, Callable[[], StageResult | None]]] = [
        (
            "approval_gate",
            lambda: approval_stage(
                protocol,
                source_commit,
                holder,
                server_config=server_config,
                server_paths=server_paths,
                context=context,
            ),
        ),
        ("source_identity", lambda: source_identity_stage(source_commit)),
        ("config_validate", lambda: config_stage(protocol, server_config)),
    ]
    if resume_from is not None:
        stages.append((
            "validate_completed_prefix",
            lambda: StageResult(stdout=json.dumps(
                validate_resume(resume_from), ensure_ascii=False
            ) + "\n"),
        ))
    stages.append((
            "fixed_four_arm_batch",
            lambda: training_batch_stage(
                context,
                protocol=protocol,
                server_paths=server_paths,
                approval=holder["approval"],
                resume_from=resume_from,
            ),
        ))
    exit_code = machine.run(stages)
    if exit_code != 0 and "approval" in holder:
        publish_failure_return(context)
    return exit_code


def run_retry_mode(run_root: Path | None = None) -> int:
    pending_path = RUNTIME_ROOT / "pending_return.json"
    if not pending_path.is_file():
        print(f"没有待重试的 Gitee 回传：{pending_path}", file=sys.stderr)
        return 50
    result = (
        recover_failure_gitee_return(pending_path, run_root)
        if run_root is not None
        else retry_gitee_return(pending_path)
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_lfs_recovery_mode(run_root: Path) -> int:
    pending_path = RUNTIME_ROOT / "pending_return.json"
    if not pending_path.is_file():
        print(f"没有待恢复的 Gitee 回传：{pending_path}", file=sys.stderr)
        return 50
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    rejected_staging = Path(str(payload["staging_worktree"])).resolve()
    recovery_staging = rejected_staging.with_name("git-return-ordinary-r001")
    result = recover_lfs_rejected_arm_gitee_return(
        pending_path,
        run_root,
        project_root=PROJECT_ROOT,
        recovery_staging=recovery_staging,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W007 fixed one-click controller")
    parser.add_argument(
        "mode",
        choices=["test", "train", "resume", "retry-return", "recover-lfs-return"],
    )
    parser.add_argument("--run-root", type=Path, help="测试夹具使用；双击入口不传参数")
    return parser


def configuration_load_failure(
    *,
    run_dir: Path,
    mode: str,
    source_commit: str,
    error: Exception,
) -> int:
    context = RunContext(
        run_dir,
        mode=mode,
        source_commit=source_commit,
        resolved_config={
            "mode": mode,
            "source_commit": source_commit,
            "configuration_status": "load_failed",
        },
    )

    def fail() -> StageResult:
        raise StageFailure(f"configuration load failed: {error}", exit_code=10)

    return StageMachine(context).run([("configuration_load", fail)])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "retry-return":
        return run_retry_mode(args.run_root)
    if args.mode == "recover-lfs-return":
        if args.run_root is None:
            print("recover-lfs-return mode requires --run-root", file=sys.stderr)
            return 51
        return run_lfs_recovery_mode(args.run_root)
    if args.mode == "resume" and args.run_root is None:
        print("resume mode requires --run-root for the preserved failed batch", file=sys.stderr)
        return 51
    source_commit = git_output("rev-parse", "HEAD")
    resume_from = args.run_root if args.mode == "resume" else None
    fallback_run_dir = (
        default_run_dir(args.mode)
        if resume_from is not None
        else args.run_root or default_run_dir(args.mode)
    )
    try:
        protocol, server_config, server_paths = load_configs()
    except Exception as exc:
        return configuration_load_failure(
            run_dir=fallback_run_dir,
            mode=args.mode,
            source_commit=source_commit,
            error=exc,
        )
    if args.mode == "test":
        return run_test_mode(
            run_dir=fallback_run_dir,
            protocol=protocol,
            server_config=server_config,
            server_paths=server_paths,
            source_commit=source_commit,
        )
    training_run_dir = (
        default_training_run_dir(server_config, server_paths)
        if resume_from is not None
        else args.run_root or default_training_run_dir(server_config, server_paths)
    )
    return run_training_mode(
        run_dir=training_run_dir,
        protocol=protocol,
        server_config=server_config,
        server_paths=server_paths,
        source_commit=source_commit,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    raise SystemExit(main())
