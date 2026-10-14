"""四臂预测、失败记录与 Gitee 回传清单合同。"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE_ID = "W007"
RETURN_BRANCH_PREFIX = "automation/server/W007/"
RETURN_TREE_PREFIX = "automation/returns/W007/"
RETURN_ATTRIBUTES = "*.pth -filter -diff -merge -text\n"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def write_prediction_csv(
    path: str | Path,
    *,
    patients: Sequence[str],
    spot_ids: Sequence[str],
    split: str,
    pathway_names: Sequence[str],
    truth_z: np.ndarray,
    prediction_z: np.ndarray,
    truth_raw: np.ndarray,
    prediction_raw: np.ndarray,
) -> Path:
    """保存逐样本 z/raw truth/prediction，不以文件较大为由省略。"""

    expected_shape = (len(patients), len(pathway_names))
    arrays = (truth_z, prediction_z, truth_raw, prediction_raw)
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("prediction arrays do not match patient/pathway dimensions")
    if len(spot_ids) != len(patients):
        raise ValueError("spot_ids must align with patients")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient", "spot_id", "split"]
    for prefix in ("truth_z", "prediction_z", "truth_raw", "prediction_raw"):
        fieldnames.extend(f"{prefix}__{name}" for name in pathway_names)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (patient, spot_id) in enumerate(zip(patients, spot_ids)):
            row: dict[str, Any] = {
                "patient": patient,
                "spot_id": spot_id,
                "split": split,
            }
            for prefix, array in zip(
                ("truth_z", "prediction_z", "truth_raw", "prediction_raw"),
                arrays,
            ):
                row.update({
                    f"{prefix}__{name}": float(array[row_index, column_index])
                    for column_index, name in enumerate(pathway_names)
                })
            writer.writerow(row)
    return destination


def inventory_files(paths: Iterable[str | Path], *, root: str | Path) -> list[dict[str, Any]]:
    """普通日志/预测仅登记路径与大小，不扩展 SHA-256 治理。"""

    root_path = Path(root).resolve()
    inventory = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        inventory.append({
            "path": path.relative_to(root_path).as_posix(),
            "size_bytes": path.stat().st_size,
        })
    return inventory


def build_failure_payload(
    *,
    stage: str,
    exit_code: int,
    source_commit: str,
    stdout_path: str,
    stderr_path: str,
    events_path: str,
    resolved_config_path: str,
    job_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "workspace_id": WORKSPACE_ID,
        "stage": stage,
        "exit_code": int(exit_code),
        "source_commit": source_commit,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "logs": {
            "stdout": stdout_path,
            "stderr": stderr_path,
            "events": events_path,
            "resolved_config": resolved_config_path,
        },
        "training_retry_allowed": False,
        "return_retry_only": True,
    }


def validate_arm_artifacts(
    *,
    arm: str,
    successful: bool,
    artifacts: Mapping[str, str | Path],
) -> None:
    if successful:
        required = {"internal_predictions", "external_predictions", "metrics", "checkpoint"}
        if arm != "FBR":
            required.add("training_history")
    else:
        required = {"stdout", "stderr", "events", "resolved_config", "failure"}
    missing = sorted(key for key in required if key not in artifacts or not Path(artifacts[key]).is_file())
    if missing:
        raise ValueError(f"{arm} artifact contract is incomplete: {missing}")


def write_return_manifest(
    path: str | Path,
    *,
    source_commit: str,
    return_branch: str,
    evidence_class: str,
    artifacts: Sequence[Mapping[str, Any]],
    retry_only: bool,
) -> Path:
    if not return_branch.startswith(RETURN_BRANCH_PREFIX):
        raise ValueError("return branch is outside the W007 server-owned prefix")
    return write_json(path, {
        "schema_version": "1.0",
        "workspace_id": WORKSPACE_ID,
        "source_commit": source_commit,
        "return_branch": return_branch,
        "evidence_class": evidence_class,
        "verification": "inventory_size_and_git_tree_closure",
        "artifacts": list(artifacts),
        "retry_only": bool(retry_only),
    })


def publish_gitee_return(
    *,
    project_root: str | Path,
    source_commit: str,
    return_branch: str,
    run_dir: str | Path,
    artifact_paths: Sequence[str | Path],
    staging_root: str | Path,
    commit_message: str,
) -> dict[str, str]:
    """固定 Git 工作流发布日志；无训练重试、无 force push、无任意 shell。"""

    if not return_branch.startswith(RETURN_BRANCH_PREFIX) or not re.fullmatch(
        r"automation/server/W007/[A-Za-z0-9._/-]+", return_branch
    ):
        raise ValueError("unsafe W007 return branch")
    root = Path(project_root).resolve()
    run_root = Path(run_dir).resolve()
    staging = Path(staging_root).resolve()
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise FileExistsError(f"return staging path already exists: {staging}")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(staging), source_commit],
        cwd=root,
        check=True,
    )
    bundle_root = staging / "automation" / "returns" / WORKSPACE_ID / run_root.name
    bundle_root.mkdir(parents=True, exist_ok=False)
    attributes_path = bundle_root / ".gitattributes"
    attributes_path.write_text(RETURN_ATTRIBUTES, encoding="utf-8", newline="\n")
    staged_relatives = [attributes_path.relative_to(staging).as_posix()]
    for artifact in artifact_paths:
        source = Path(artifact).resolve()
        relative = source.relative_to(run_root)
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged_relatives.append(destination.relative_to(staging).as_posix())
    subprocess.run(["git", "add", "--force", "--", *staged_relatives], cwd=staging, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=staging, check=True)
    return_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "gitee", f"HEAD:refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
    )
    remote_commit = subprocess.run(
        ["git", "ls-remote", "gitee", f"refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.split()[0]
    if remote_commit != return_commit:
        raise RuntimeError("Gitee return SHA does not match the local return commit")
    return {"return_commit": return_commit, "remote_commit": remote_commit}


def recover_lfs_rejected_arm_gitee_return(
    pending_path: str | Path,
    run_dir: str | Path,
    *,
    project_root: str | Path,
    recovery_staging: str | Path,
) -> dict[str, str]:
    """把已完成单臂从被拒绝的 LFS 指针提交重建为普通 Git blob；绝不训练。"""

    path = Path(pending_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("retry_mode") != "return_only" or payload.get("training_retry_allowed") is not False:
        raise ValueError("pending return record does not enforce return-only retry")
    arm = str(payload.get("arm", ""))
    if arm not in {"FBR", "RCC", "HCR", "CPGCR"}:
        raise ValueError("pending return arm is invalid")
    rejected_staging = Path(str(payload["staging_worktree"])).resolve()
    rejected_commit = _git_output(rejected_staging, "rev-parse", "HEAD")
    if rejected_commit != payload.get("return_commit"):
        raise RuntimeError("rejected LFS staging changed; refusing recovery")
    if _git_output(rejected_staging, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("rejected LFS staging is dirty")
    return_branch = str(payload["return_branch"])
    remote_line = _git_output(rejected_staging, "ls-remote", "gitee", f"refs/heads/{return_branch}")
    if remote_line:
        raise RuntimeError("return branch already exists remotely; refusing recovery rewrite")

    run_root = Path(run_dir).resolve()
    arm_root = run_root / arm
    manifest_path = arm_root / "return_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_commit = str(manifest.get("source_commit", ""))
    if (
        manifest.get("workspace_id") != WORKSPACE_ID
        or manifest.get("return_branch") != return_branch
        or manifest.get("retry_only") is not False
    ):
        raise ValueError("completed arm manifest does not match pending return")
    rejected_parent = _git_output(rejected_staging, "rev-parse", f"{rejected_commit}^")
    if rejected_parent != source_commit:
        raise RuntimeError("rejected return commit is not based on the frozen source commit")
    artifact_paths: list[Path] = []
    for entry in manifest.get("artifacts", []):
        artifact = (run_root / str(entry["path"])).resolve()
        if run_root not in artifact.parents or not artifact.is_file():
            raise FileNotFoundError(f"completed arm artifact is missing or outside run root: {artifact}")
        if artifact.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"completed arm artifact size changed: {artifact}")
        artifact_paths.append(artifact)
    if not artifact_paths:
        raise RuntimeError("completed arm manifest contains no artifacts")
    artifact_paths.append(manifest_path)

    result = publish_gitee_return(
        project_root=project_root,
        source_commit=source_commit,
        return_branch=return_branch,
        run_dir=run_root,
        artifact_paths=artifact_paths,
        staging_root=recovery_staging,
        commit_message=f"return(w007): recover ordinary Git {arm} {payload.get('job_id', 'unknown')}",
    )
    payload.update({
        "status": "published_after_lfs_rejection",
        "rejected_lfs_commit": rejected_commit,
        "rejected_lfs_staging_worktree": str(rejected_staging),
        "staging_worktree": str(Path(recovery_staging).resolve()),
        "return_commit": result["return_commit"],
        "remote_commit": result["remote_commit"],
        "recovered_run_dir": str(run_root),
    })
    write_json(path, payload)
    write_json(arm_root / "return_published.json", result)
    return result


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _return_commit_paths(staging: Path, commit: str) -> list[str]:
    return _git_output(
        staging,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    ).splitlines()


def _require_return_artifacts(staging: Path, commit: str) -> None:
    if not any(path.startswith(RETURN_TREE_PREFIX) for path in _return_commit_paths(staging, commit)):
        raise RuntimeError("staged return commit contains no W007 return artifacts")


def recover_completed_arm_gitee_return(
    pending_path: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """把已完成单臂的现有产物追加到既有 return ref；绝不重新评估或训练。"""

    path = Path(pending_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("retry_mode") != "return_only" or payload.get("training_retry_allowed") is not False:
        raise ValueError("pending return record does not enforce return-only retry")
    arm = str(payload.get("arm", ""))
    return_branch = str(payload.get("return_branch", ""))
    if arm not in {"FBR", "RCC", "HCR", "CPGCR"}:
        raise ValueError("pending return arm is invalid")
    if not return_branch.startswith(RETURN_BRANCH_PREFIX):
        raise ValueError("pending return branch is outside W007")

    run_root = Path(run_dir).resolve()
    arm_root = run_root / arm
    manifest_path = arm_root / "return_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed arm return manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workspace_id") != WORKSPACE_ID:
        raise ValueError("completed arm manifest belongs to another workspace")
    if manifest.get("return_branch") != return_branch or manifest.get("retry_only") is not False:
        raise ValueError("completed arm manifest does not match pending return")
    artifact_paths = []
    for entry in manifest.get("artifacts", []):
        artifact = (run_root / str(entry["path"])).resolve()
        if run_root not in artifact.parents or not artifact.is_file():
            raise FileNotFoundError(f"completed arm artifact is missing or outside run root: {artifact}")
        if artifact.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"completed arm artifact size changed: {artifact}")
        artifact_paths.append(artifact)
    if not artifact_paths:
        raise RuntimeError("completed arm manifest contains no artifacts")
    artifact_paths.append(manifest_path)

    staging = Path(payload["staging_worktree"]).resolve()
    current_commit = _git_output(staging, "rev-parse", "HEAD")
    if current_commit != payload.get("return_commit"):
        raise RuntimeError("completed arm recovery staging changed; refusing recovery")
    if _git_output(staging, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("completed arm recovery staging is dirty")
    remote_line = _git_output(staging, "ls-remote", "gitee", f"refs/heads/{return_branch}")
    if not remote_line or remote_line.split()[0] != current_commit:
        raise RuntimeError("remote return branch does not match recovery staging")

    bundle_root = staging / RETURN_TREE_PREFIX / run_root.name
    if payload.get("status") == "completed_arm_published":
        published_path = arm_root / "return_published.json"
        if not published_path.is_file() or not (bundle_root / arm / manifest_path.name).is_file():
            raise RuntimeError("completed arm published receipt or return bundle is missing")
        published = json.loads(published_path.read_text(encoding="utf-8"))
        expected = {"return_commit": current_commit, "remote_commit": current_commit}
        if published != expected or payload.get("remote_commit") != current_commit:
            raise RuntimeError("completed arm published receipt does not match return ref")
        return expected
    if bundle_root.exists():
        raise FileExistsError(f"completed arm return bundle already exists: {bundle_root}")
    bundle_root.mkdir(parents=True, exist_ok=False)
    attributes_path = bundle_root / ".gitattributes"
    attributes_path.write_text(RETURN_ATTRIBUTES, encoding="utf-8", newline="\n")
    staged_relatives = [attributes_path.relative_to(staging).as_posix()]
    for source in artifact_paths:
        relative = source.relative_to(run_root)
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged_relatives.append(destination.relative_to(staging).as_posix())
    subprocess.run(["git", "add", "--force", "--", *staged_relatives], cwd=staging, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"return(w007): recover completed {arm} {payload.get('job_id', 'unknown')}"],
        cwd=staging,
        check=True,
    )
    return_commit = _git_output(staging, "rev-parse", "HEAD")
    _require_return_artifacts(staging, return_commit)
    subprocess.run(
        ["git", "push", "gitee", f"HEAD:refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
    )
    remote_commit = _git_output(staging, "ls-remote", "gitee", f"refs/heads/{return_branch}").split()[0]
    if remote_commit != return_commit:
        raise RuntimeError("Gitee completed arm recovery SHA mismatch")
    result = {"return_commit": return_commit, "remote_commit": remote_commit}
    payload.update({
        "status": "completed_arm_published",
        "return_commit": return_commit,
        "remote_commit": remote_commit,
        "recovered_run_dir": str(run_root),
    })
    write_json(path, payload)
    write_json(arm_root / "return_published.json", result)
    return result


def retry_gitee_return(pending_path: str | Path) -> dict[str, str]:
    """只重试已完成 return commit 的 push；绝不重新运行训练。"""

    path = Path(pending_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("retry_mode") != "return_only" or payload.get("training_retry_allowed") is not False:
        raise ValueError("pending return record does not enforce return-only retry")
    return_branch = str(payload["return_branch"])
    if not return_branch.startswith(RETURN_BRANCH_PREFIX):
        raise ValueError("pending return branch is outside W007")
    staging = Path(payload["staging_worktree"])
    return_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if return_commit != payload["return_commit"]:
        raise RuntimeError("staged return commit changed; refusing retry")
    _require_return_artifacts(staging, return_commit)
    subprocess.run(
        ["git", "push", "gitee", f"HEAD:refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
    )
    remote_commit = subprocess.run(
        ["git", "ls-remote", "gitee", f"refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.split()[0]
    if remote_commit != return_commit:
        raise RuntimeError("Gitee return SHA mismatch after retry")
    payload["status"] = "published"
    payload["remote_commit"] = remote_commit
    write_json(path, payload)
    return {"return_commit": return_commit, "remote_commit": remote_commit}


def recover_failure_gitee_return(
    pending_path: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """把既有 HARD 失败日志补成 return commit；不执行训练或诊断。"""

    path = Path(pending_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("retry_mode") != "return_only" or payload.get("training_retry_allowed") is not False:
        raise ValueError("pending return record does not enforce return-only retry")
    return_branch = str(payload["return_branch"])
    if not return_branch.startswith(RETURN_BRANCH_PREFIX):
        raise ValueError("pending return branch is outside W007")
    staging = Path(payload["staging_worktree"]).resolve()
    run_root = Path(run_dir).resolve()
    required_names = (
        "stdout.log",
        "stderr.log",
        "events.jsonl",
        "resolved_config.json",
        "failure.json",
    )
    required_paths = [run_root / name for name in required_names]

    def validate_run(candidate: Path) -> tuple[str, list[Path]]:
        candidate_paths = [candidate / name for name in required_names]
        missing = [str(item) for item in candidate_paths if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"failure recovery artifacts are missing: {missing}")
        candidate_failure = json.loads(
            (candidate / "failure.json").read_text(encoding="utf-8")
        )
        if candidate_failure.get("workspace_id") != WORKSPACE_ID:
            raise ValueError("failure artifact belongs to another workspace")
        return str(candidate_failure["source_commit"]), candidate_paths

    source_commit, required_paths = validate_run(run_root)
    current_commit = _git_output(staging, "rev-parse", "HEAD")
    if current_commit != payload.get("return_commit") or current_commit != source_commit:
        raise RuntimeError("failure recovery staging no longer matches the failed source commit")
    remote_line = _git_output(staging, "ls-remote", "gitee", f"refs/heads/{return_branch}")
    if remote_line and remote_line.split()[0] != current_commit:
        raise RuntimeError("remote return branch advanced; refusing non-fast-forward recovery")

    def stage_run(candidate: Path, *, allow_existing: bool) -> None:
        candidate_source, candidate_paths = validate_run(candidate)
        if candidate_source != source_commit:
            raise RuntimeError("interrupted failure bundle source_commit mismatch")
        manifest_path = candidate / "failure_return_manifest.json"
        if not manifest_path.is_file():
            write_return_manifest(
                manifest_path,
                source_commit=source_commit,
                return_branch=return_branch,
                evidence_class="non_evidence_failure",
                artifacts=inventory_files(candidate_paths, root=candidate),
                retry_only=True,
            )
        artifact_paths = [*candidate_paths, manifest_path]
        bundle_root = staging / RETURN_TREE_PREFIX / candidate.name
        if bundle_root.exists() and not allow_existing:
            raise FileExistsError(f"failure return bundle already exists: {bundle_root}")
        bundle_root.mkdir(parents=True, exist_ok=allow_existing)
        staged_relatives = []
        for source in artifact_paths:
            destination = bundle_root / source.name
            shutil.copy2(source, destination)
            staged_relatives.append(destination.relative_to(staging).as_posix())
        subprocess.run(
            ["git", "add", "--force", "--", *staged_relatives],
            cwd=staging,
            check=True,
        )

    prior_return_commit: str | None = None
    dirty = _git_output(staging, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        if _git_output(staging, "diff", "--name-only"):
            raise RuntimeError("failure recovery staging has unstaged modifications")
        if _git_output(staging, "ls-files", "--others", "--exclude-standard"):
            raise RuntimeError("failure recovery staging has untracked files")
        cached_paths = _git_output(staging, "diff", "--cached", "--name-only").splitlines()
        prefix_parts = Path(RETURN_TREE_PREFIX).parts
        run_names = {
            Path(item).parts[len(prefix_parts)]
            for item in cached_paths
            if len(Path(item).parts) > len(prefix_parts)
            and Path(item).parts[:len(prefix_parts)] == prefix_parts
        }
        if len(run_names) != 1 or any(
            Path(item).parts[:len(prefix_parts)] != prefix_parts
            for item in cached_paths
        ):
            raise RuntimeError("dirty staging is not one interrupted W007 return bundle")
        interrupted_name = next(iter(run_names))
        if interrupted_name == run_root.name:
            interrupted_root = run_root
        else:
            interrupted_root = run_root.parent / interrupted_name
        stage_run(interrupted_root, allow_existing=True)
        subprocess.run(
            ["git", "commit", "-m", f"return(w007): recover interrupted failure {interrupted_name}"],
            cwd=staging,
            check=True,
        )
        prior_return_commit = _git_output(staging, "rev-parse", "HEAD")
        _require_return_artifacts(staging, prior_return_commit)

    stage_run(run_root, allow_existing=False)
    subprocess.run(
        ["git", "commit", "-m", f"return(w007): recover failure {payload.get('job_id', 'unknown')}"],
        cwd=staging,
        check=True,
    )
    return_commit = _git_output(staging, "rev-parse", "HEAD")
    _require_return_artifacts(staging, return_commit)
    subprocess.run(
        ["git", "push", "gitee", f"HEAD:refs/heads/{return_branch}"],
        cwd=staging,
        check=True,
    )
    remote_commit = _git_output(staging, "ls-remote", "gitee", f"refs/heads/{return_branch}").split()[0]
    if remote_commit != return_commit:
        raise RuntimeError("Gitee failure recovery SHA mismatch")
    payload.update({
        "status": "recovered_published",
        "return_commit": return_commit,
        "remote_commit": remote_commit,
        "recovered_run_dir": str(run_root),
    })
    if prior_return_commit is not None:
        payload["recovered_prior_commit"] = prior_return_commit
    write_json(path, payload)
    return {"return_commit": return_commit, "remote_commit": remote_commit}
