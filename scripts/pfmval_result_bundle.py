"""Authoritative result_bundle_v1 build and validation entry points."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping

from scripts.pfmval_state import (
    MAX_RESULT_FILE_BYTES,
    MAX_RESULT_TOTAL_BYTES,
    read_json,
    validate_against_schema_strict,
    write_json_atomic,
)


RESULT_V2_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "project_state"
    / "schemas"
    / "result_envelope_v2.schema.json"
)
LEGACY_PACKAGING_SIDECARS = {
    "artifacts.json",
    "large_artifacts.json",
    "metrics.json",
}
INTEGRITY_PATH = "artifacts/bundle_integrity.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _safe_relative_path(raw_path: str) -> bool:
    if not raw_path or "\\" in raw_path or "\x00" in raw_path:
        return False
    path = PurePosixPath(raw_path)
    return not path.is_absolute() and ".." not in path.parts


def _is_text(payload: bytes) -> bool:
    if b"\x00" in payload:
        return False
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _git_normalize(payload: bytes) -> bytes:
    if not _is_text(payload):
        return payload
    return payload.replace(b"\r\n", b"\n")


def _crlf_materialize(payload: bytes) -> bytes:
    normalized = payload.replace(b"\r\n", b"\n")
    return normalized.replace(b"\n", b"\r\n")


def _line_ending(payload: bytes) -> str:
    if not _is_text(payload):
        return "binary"
    crlf = payload.count(b"\r\n")
    lf = payload.count(b"\n") - crlf
    if crlf and lf:
        return "mixed"
    if crlf:
        return "crlf"
    if lf:
        return "lf"
    return "none"


def _require_array(result: Mapping[str, Any], field: str) -> list[Any]:
    value = result.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _check_finite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metric at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{location}[{index}]")


def validate_result_manifest_v1(
    project_root: Path,
    result: Mapping[str, Any],
) -> None:
    """Apply the complete v2 JSON Schema and semantic constraints."""
    _require_array(result, "artifacts")
    _require_array(result, "large_artifacts")
    validate_against_schema_strict(
        result,
        project_root / "project_state" / "schemas" / "result_envelope_v2.schema.json",
        "result envelope v2",
    )
    artifacts = list(result["artifacts"])
    paths: set[str] = set()
    artifact_ids: set[str] = set()
    by_id: Dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id", ""))
        if not artifact_id or artifact_id in artifact_ids:
            raise ValueError("result artifact_id must be non-empty and unique")
        artifact_ids.add(artifact_id)
        by_id[artifact_id] = artifact
        raw_path = str(artifact.get("path", ""))
        if not _safe_relative_path(raw_path):
            raise ValueError(f"result artifact path is unsafe: {raw_path}")
        if raw_path in paths:
            raise ValueError("artifact paths must be unique")
        paths.add(raw_path)
        if artifact.get("source_attempt_id") != result.get("attempt_id"):
            raise ValueError("result artifact attempt binding mismatch")
        if not str(artifact.get("retention", "")):
            raise ValueError("result artifact requires retention")
        digest = str(artifact.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            if artifact.get("evidence_role") == "critical":
                raise ValueError("critical artifact requires SHA-256")
            raise ValueError("result artifact requires SHA-256")
    for artifact_id in result["metric_artifact_ids"]:
        artifact = by_id.get(str(artifact_id))
        if artifact is None:
            raise ValueError("metric source artifact is missing")
        if artifact.get("evidence_role") == "diagnostic":
            raise ValueError("diagnostic artifact cannot source accepted metrics")
        if artifact.get("evidence_role") != "critical":
            raise ValueError("metric source artifact must be critical")
    if (
        result["phase"] == "formal"
        and result["status"] == "success"
        and not any(
            item.get("kind") == "selection_proof"
            and item.get("evidence_role") == "critical"
            for item in artifacts
        )
    ):
        raise ValueError("formal success requires critical selection proof")
    for artifact in result["large_artifacts"]:
        artifact_id = str(artifact.get("artifact_id", ""))
        if not artifact_id or artifact_id in artifact_ids:
            raise ValueError("large artifact_id must be non-empty and globally unique")
        artifact_ids.add(artifact_id)
        if not artifact.get("retention") and not artifact.get("recompute_policy"):
            raise ValueError("large artifact requires retention or recompute policy")
    _check_finite(result.get("metrics", {}), "metrics")


def _terminal_status(bundle_dir: Path, result: Mapping[str, Any]) -> str:
    terminal_artifact = next(
        (
            item
            for item in result["artifacts"]
            if item.get("artifact_id") == "attempt_terminal"
            or PurePosixPath(str(item.get("path", ""))).name
            == "attempt_terminal.json"
        ),
        None,
    )
    if terminal_artifact is None:
        raise ValueError("result bundle requires an attempt terminal artifact")
    terminal = read_json(bundle_dir / Path(*PurePosixPath(terminal_artifact["path"]).parts))
    if terminal.get("event_type") != "EXPERIMENT_TERMINAL":
        raise ValueError("terminal artifact has invalid event_type")
    if (
        terminal.get("attempt_id") != result.get("attempt_id")
        or terminal.get("job_id") != result.get("job_id")
    ):
        raise ValueError("terminal artifact identity mismatch")
    if terminal.get("status") == "completed" and int(terminal.get("returncode", -1)) == 0:
        return "success"
    if terminal.get("status") in {"completed", "failed"}:
        return "failed"
    return "incomplete"


def _bundle_sha256(bundle_dir: Path) -> str:
    records = []
    for path in sorted(item for item in bundle_dir.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(bundle_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return _sha256_bytes(_canonical_json_bytes(records))


def _source_integrity(
    *,
    artifact: Mapping[str, Any],
    payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    expected_sha = str(artifact.get("sha256", ""))
    expected_size = int(artifact.get("size_bytes", -1))
    actual_sha = _sha256_bytes(payload)
    normalized = _git_normalize(payload)
    if actual_sha == expected_sha and len(payload) == expected_size:
        source_match = "raw"
        raw_payload = payload
    else:
        materialized = _crlf_materialize(payload)
        if (
            _sha256_bytes(materialized) != expected_sha
            or len(materialized) != expected_size
        ):
            raise ValueError(
                f"artifact source bytes do not match raw or LF/CRLF-normalized identity: "
                f"{artifact.get('path')}"
            )
        source_match = "git_normalized_from_crlf"
        raw_payload = materialized
    record = {
        "artifact_id": artifact["artifact_id"],
        "path": artifact["path"],
        "line_ending_source_materialized": _line_ending(payload),
        "source_match": source_match,
        "size_bytes_raw": len(raw_payload),
        "sha256_raw": _sha256_bytes(raw_payload),
        "size_bytes_git_normalized": len(normalized),
        "sha256_git_normalized": _sha256_bytes(normalized),
    }
    return record, normalized


def _validate_legacy_sidecars(
    source_bundle: Path,
    source_result: Mapping[str, Any],
) -> None:
    metrics_path = source_bundle / "metrics.json"
    if metrics_path.exists() and read_json(metrics_path) != source_result.get("metrics"):
        raise ValueError("legacy metrics.json does not match result metrics")
    artifacts_path = source_bundle / "artifacts.json"
    if artifacts_path.exists() and not isinstance(read_json(artifacts_path), list):
        raise ValueError("legacy artifacts.json must be an array")
    large_path = source_bundle / "large_artifacts.json"
    if large_path.exists() and not isinstance(read_json(large_path), list):
        raise ValueError("legacy large_artifacts.json must be an array")


def build_result_bundle_v1(
    project_root: Path,
    source_bundle: Path,
    staging_dir: Path,
    *,
    result_id: str,
    artifact_retention: str,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Build an immutable packaging-only revision in an empty staging directory."""
    project_root = project_root.resolve()
    source_bundle = source_bundle.resolve()
    staging_dir = staging_dir.resolve()
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise ValueError("staging directory must be empty")
    source_result = read_json(source_bundle / "result.json")
    source_artifacts = _require_array(source_result, "artifacts")
    source_large = _require_array(source_result, "large_artifacts")
    if not artifact_retention:
        raise ValueError("artifact_retention must be non-empty")
    paths = [str(item.get("path", "")) for item in source_artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("artifact paths must be unique")
    source_expected = {
        "result.json",
        *LEGACY_PACKAGING_SIDECARS,
        *paths,
    }
    source_actual = {
        item.relative_to(source_bundle).as_posix()
        for item in source_bundle.rglob("*")
        if item.is_file()
    }
    unexpected = sorted(source_actual - source_expected)
    if unexpected:
        raise ValueError(f"source bundle contains unregistered sidecar: {unexpected}")
    _validate_legacy_sidecars(source_bundle, source_result)

    staging_dir.mkdir(parents=True, exist_ok=True)
    built_artifacts: list[dict[str, Any]] = []
    integrity_records: list[dict[str, Any]] = []
    for source_artifact in source_artifacts:
        artifact = dict(source_artifact)
        artifact_id = str(artifact.get("artifact_id", ""))
        if not artifact_id:
            raise ValueError("result artifact_id must be non-empty")
        raw_path = str(artifact.get("path", ""))
        if not _safe_relative_path(raw_path):
            raise ValueError(f"result artifact path is unsafe: {raw_path}")
        source_path = source_bundle / Path(*PurePosixPath(raw_path).parts)
        if not source_path.is_file():
            raise ValueError(f"artifact missing: {raw_path}")
        integrity, normalized = _source_integrity(
            artifact=artifact,
            payload=source_path.read_bytes(),
        )
        destination = staging_dir / Path(*PurePosixPath(raw_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(normalized)
        artifact["size_bytes"] = len(normalized)
        artifact["sha256"] = _sha256_bytes(normalized)
        artifact["source_attempt_id"] = source_result["attempt_id"]
        artifact["retention"] = str(
            artifact.get("retention") or artifact_retention
        )
        built_artifacts.append(artifact)
        integrity_records.append(integrity)

    integrity_payload = {
        "schema_version": "1.0",
        "source_result_id": source_result["result_id"],
        "target_result_id": result_id,
        "normalization": "UTF-8 text CRLF to LF; binary unchanged",
        "artifacts": integrity_records,
    }
    integrity_bytes = _canonical_json_bytes(integrity_payload)
    integrity_path = staging_dir / Path(*PurePosixPath(INTEGRITY_PATH).parts)
    integrity_path.parent.mkdir(parents=True, exist_ok=True)
    integrity_path.write_bytes(integrity_bytes)
    built_artifacts.append(
        {
            "artifact_id": "bundle_integrity",
            "path": INTEGRITY_PATH,
            "kind": "bundle_integrity",
            "evidence_role": "critical",
            "size_bytes": len(integrity_bytes),
            "sha256": _sha256_bytes(integrity_bytes),
            "retention": "retain_with_immutable_result_bundle",
            "source_attempt_id": source_result["attempt_id"],
        }
    )
    large_artifacts = []
    used_ids = {item["artifact_id"] for item in built_artifacts}
    for index, source_large_artifact in enumerate(source_large, start=1):
        large_artifact = dict(source_large_artifact)
        artifact_id = str(large_artifact.get("artifact_id", ""))
        if not artifact_id:
            name = Path(str(large_artifact.get("server_path", ""))).stem
            artifact_id = name or f"large_artifact_{index:03d}"
        if artifact_id in used_ids:
            artifact_id = f"large_{artifact_id}"
        large_artifact["artifact_id"] = artifact_id
        used_ids.add(artifact_id)
        large_artifacts.append(large_artifact)
    result = {
        **source_result,
        "result_id": result_id,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": built_artifacts,
        "large_artifacts": large_artifacts,
    }
    result["status"] = _terminal_status(staging_dir, result)
    write_json_atomic(staging_dir / "result.json", result)
    report = validate_result_bundle_v1(project_root, staging_dir)
    return {**report, "status": result["status"], "result_id": result_id}


def validate_result_bundle_v1(
    project_root: Path,
    bundle_dir: Path,
) -> Dict[str, Any]:
    """Validate Schema, semantics, closure, hashes, budgets and normalization."""
    project_root = project_root.resolve()
    bundle_dir = bundle_dir.resolve()
    result = read_json(bundle_dir / "result.json")
    validate_result_manifest_v1(project_root, result)
    paths = {str(item["path"]) for item in result["artifacts"]}
    expected_files = {"result.json", *paths}
    actual_files = {
        item.relative_to(bundle_dir).as_posix()
        for item in bundle_dir.rglob("*")
        if item.is_file()
    }
    unexpected = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if unexpected or missing:
        raise ValueError(
            f"artifact closure mismatch: missing={missing}, unexpected={unexpected}"
        )
    total = 0
    by_path: dict[str, Mapping[str, Any]] = {}
    for artifact in result["artifacts"]:
        raw_path = str(artifact["path"])
        path = bundle_dir / Path(*PurePosixPath(raw_path).parts)
        size = path.stat().st_size
        if size != int(artifact["size_bytes"]):
            raise ValueError(f"artifact size mismatch: {raw_path}")
        if size > MAX_RESULT_FILE_BYTES:
            raise ValueError(f"artifact exceeds size budget: {raw_path}")
        if _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {raw_path}")
        total += size
        by_path[raw_path] = artifact
    total += (bundle_dir / "result.json").stat().st_size
    if total > MAX_RESULT_TOTAL_BYTES:
        raise ValueError("result bundle exceeds total size budget")
    mapped_status = _terminal_status(bundle_dir, result)
    if result["status"] != mapped_status:
        raise ValueError(
            f"terminal status mapping mismatch: expected {mapped_status}"
        )
    integrity_artifact = by_path.get(INTEGRITY_PATH)
    if integrity_artifact is None:
        raise ValueError("result bundle has no registered integrity sidecar")
    integrity = read_json(bundle_dir / Path(*PurePosixPath(INTEGRITY_PATH).parts))
    records = integrity.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("integrity artifacts must be an array")
    expected_integrity_paths = paths - {INTEGRITY_PATH}
    record_paths = {str(item.get("path", "")) for item in records}
    if record_paths != expected_integrity_paths:
        raise ValueError("integrity sidecar does not close over result artifacts")
    for record in records:
        raw_path = str(record["path"])
        actual = (bundle_dir / Path(*PurePosixPath(raw_path).parts)).read_bytes()
        normalized_sha = str(record.get("sha256_git_normalized", ""))
        if _sha256_bytes(actual) != normalized_sha:
            raise ValueError(f"Git-normalized integrity mismatch: {raw_path}")
        if by_path[raw_path]["sha256"] != normalized_sha:
            raise ValueError(f"envelope normalized SHA mismatch: {raw_path}")
        source_match = record.get("source_match")
        if source_match not in {"raw", "git_normalized_from_crlf"}:
            raise ValueError(f"invalid source integrity mode: {raw_path}")
        if source_match == "git_normalized_from_crlf":
            if _sha256_bytes(_crlf_materialize(actual)) != record.get("sha256_raw"):
                raise ValueError(f"raw CRLF integrity mismatch: {raw_path}")
    return {
        "status": "valid",
        "result_id": result["result_id"],
        "artifact_count": len(result["artifacts"]),
        "bundle_sha256": _bundle_sha256(bundle_dir),
    }


def record_result_bundle_import_v1(
    project_root: Path,
    bundle_dir: Path,
    *,
    bundle_sha256: str,
) -> Dict[str, Any]:
    """Record an import-verification event only after complete bundle validation."""
    report = validate_result_bundle_v1(project_root, bundle_dir)
    if report["bundle_sha256"] != bundle_sha256:
        raise ValueError(
            "bundle_sha256 does not match the authoritative bundle validation"
        )
    from scripts.pfmval_governance import record_result_import_v2

    result = read_json(bundle_dir / "result.json")
    return record_result_import_v2(
        project_root,
        result,
        bundle_sha256=bundle_sha256,
    )


def _run_git(
    repo_root: Path,
    *args: str,
    env: Mapping[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
        env=dict(env) if env is not None else None,
    )
    return completed.stdout.strip()


def _remote_head(repo_root: Path, remote_name: str, ref: str) -> str | None:
    output = _run_git(
        repo_root,
        "ls-remote",
        "--heads",
        remote_name,
        f"refs/heads/{ref}",
    )
    if not output:
        return None
    fields = output.split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{ref}":
        raise ValueError(f"unexpected remote ref response for {ref}")
    return fields[0]


def _git_object_exists(repo_root: Path, object_spec: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", object_spec],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def _git_bundle_sha256(
    repo_root: Path,
    commit_sha: str,
    revision_path: str,
) -> str:
    names = _run_git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit_sha,
        "--",
        revision_path,
    ).splitlines()
    prefix = revision_path.rstrip("/") + "/"
    records = []
    for name in sorted(names):
        if not name.startswith(prefix):
            raise ValueError(f"unexpected Git tree path outside revision: {name}")
        completed = subprocess.run(
            ["git", "show", f"{commit_sha}:{name}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        payload = completed.stdout
        records.append(
            {
                "path": name[len(prefix) :],
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return _sha256_bytes(_canonical_json_bytes(records))


def publish_result_bundle_v1(
    repo_root: Path,
    bundle_dir: Path,
    *,
    remote_name: str,
    ref: str,
    revision_path: str,
    expected_parent: str,
    commit_message: str,
) -> Dict[str, Any]:
    """Commit one revision, fast-forward push it, and verify the remote SHA."""
    repo_root = repo_root.resolve()
    bundle_dir = bundle_dir.resolve()
    if remote_name != "gitee":
        raise ValueError("result bundle publish is restricted to the gitee remote")
    if not _safe_relative_path(revision_path):
        raise ValueError("revision_path must be a safe repository-relative path")
    report = validate_result_bundle_v1(
        Path(__file__).resolve().parents[1],
        bundle_dir,
    )
    remote_sha = _remote_head(repo_root, remote_name, ref)
    if remote_sha != expected_parent:
        raise ValueError(
            f"remote SHA mismatch: expected {expected_parent}, observed {remote_sha}"
        )
    _run_git(
        repo_root,
        "fetch",
        "--no-tags",
        remote_name,
        f"refs/heads/{ref}",
    )
    revision_spec = f"{remote_sha}:{revision_path}"
    if _git_object_exists(repo_root, revision_spec):
        remote_bundle_sha = _git_bundle_sha256(
            repo_root,
            remote_sha,
            revision_path,
        )
        if remote_bundle_sha != report["bundle_sha256"]:
            raise ValueError(
                "immutable revision already exists with a different bundle SHA"
            )
        parent_line = _run_git(
            repo_root,
            "show",
            "-s",
            "--format=%P",
            remote_sha,
        )
        return {
            "status": "already_published",
            "commit_sha": remote_sha,
            "parent_sha": parent_line.split()[0] if parent_line else None,
            "bundle_sha256": report["bundle_sha256"],
            "remote_verified": True,
        }

    with tempfile.TemporaryDirectory(prefix="pfmval-result-bundle-") as temp_name:
        temp_root = Path(temp_name).resolve()
        worktree = temp_root / "worktree"
        _run_git(repo_root, "worktree", "add", "--detach", str(worktree), remote_sha)
        try:
            destination = worktree / Path(*PurePosixPath(revision_path).parts)
            if destination.exists():
                raise ValueError("immutable revision path already exists")
            shutil.copytree(bundle_dir, destination)
            _run_git(worktree, "add", "--", revision_path)
            result = read_json(bundle_dir / "result.json")
            commit_env = dict(os.environ)
            commit_env.update(
                {
                    "GIT_AUTHOR_NAME": "PFMval result_bundle_v1",
                    "GIT_AUTHOR_EMAIL": "pfmval-result-bundle@example.invalid",
                    "GIT_COMMITTER_NAME": "PFMval result_bundle_v1",
                    "GIT_COMMITTER_EMAIL": "pfmval-result-bundle@example.invalid",
                    "GIT_AUTHOR_DATE": str(result["created_at"]),
                    "GIT_COMMITTER_DATE": str(result["created_at"]),
                }
            )
            _run_git(
                worktree,
                "commit",
                "-m",
                commit_message,
                env=commit_env,
            )
            commit_sha = _run_git(worktree, "rev-parse", "HEAD")
            parent_sha = _run_git(worktree, "rev-parse", "HEAD^")
            if parent_sha != remote_sha:
                raise ValueError(
                    f"publish commit parent mismatch: {parent_sha} != {remote_sha}"
                )
            observed_before_push = _remote_head(repo_root, remote_name, ref)
            if observed_before_push != remote_sha:
                raise ValueError(
                    "remote SHA changed before push; refusing non-fast-forward publish"
                )
            _run_git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                remote_sha,
                commit_sha,
            )
            _run_git(
                repo_root,
                "push",
                remote_name,
                f"{commit_sha}:refs/heads/{ref}",
            )
            observed_after_push = _remote_head(repo_root, remote_name, ref)
            if observed_after_push != commit_sha:
                raise ValueError(
                    f"remote commit verification failed: {observed_after_push}"
                )
            remote_bundle_sha = _git_bundle_sha256(
                repo_root,
                commit_sha,
                revision_path,
            )
            if remote_bundle_sha != report["bundle_sha256"]:
                raise ValueError("remote bundle SHA verification failed")
            return {
                "status": "published",
                "commit_sha": commit_sha,
                "parent_sha": parent_sha,
                "bundle_sha256": report["bundle_sha256"],
                "remote_verified": True,
            }
        finally:
            _run_git(
                repo_root,
                "worktree",
                "remove",
                "--force",
                str(worktree),
            )
