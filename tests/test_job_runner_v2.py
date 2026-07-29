import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _load_ops():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pfmval_ops_runner_v2", root / "deploy" / "pfmval_ops.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _commit_repo(path: Path, filename: str, content: str) -> str:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "PFMval test"], cwd=path, check=True)
    (path / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()


def _manifest(source_commit: str, governance_commit: str, entrypoint_sha256: str) -> dict:
    return {
        "schema_version": "2.0",
        "job_id": "W001-A001",
        "experiment_id": "fixture-exp",
        "workspace_id": "W001",
        "attempt_id": "A001",
        "approval_id": "APR-fixture",
        "critical_contract_sha256": "a" * 64,
        "source_commit": source_commit,
        "resolved_argv": ["python", "train.py", "--fixed"],
        "input_binding": {
            "artifacts": [
                {
                    "artifact_id": "mpp2_training_entrypoint",
                    "sha256": entrypoint_sha256,
                }
            ]
        },
        "run_units": 1,
        "execution_binding": {
            "mode": "source_plus_governance_bundle",
            "governance_commit": governance_commit,
        },
    }


def test_job_runner_v2_uses_workspace_and_external_attempt_root(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)

    plan = ops.prepare_job_v2_execution(
        manifest,
        governance_root=governance,
        source_worktree=source,
        run_root=tmp_path / "runs",
    )

    assert Path(plan["source_worktree"]) == source.resolve()
    assert Path(plan["run_directory"]) == (tmp_path / "runs" / "W001" / "A001").resolve()
    assert "state_revision" not in manifest
    event_path = ops.write_job_v2_started_event(plan, manifest)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["event_type"] == "EXPERIMENT_STARTED"
    assert event["source_commit"] == source_sha
    assert not (source / "W001-A001").exists()


def test_job_runner_v2_rejects_legacy_unbound_v2_manifest(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    manifest.pop("execution_binding")
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)

    try:
        ops.prepare_job_v2_execution(
            manifest,
            governance_root=governance,
            source_worktree=source,
            run_root=tmp_path / "runs",
        )
    except ValueError as exc:
        assert "execution_binding" in str(exc)
    else:
        raise AssertionError("legacy unbound v2 manifest must be rejected")


def test_job_runner_v2_rejects_source_entrypoint_hash_drift(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest = _manifest(source_sha, governance_sha, "b" * 64)
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)

    try:
        ops.prepare_job_v2_execution(
            manifest,
            governance_root=governance,
            source_worktree=source,
            run_root=tmp_path / "runs",
        )
    except ValueError as exc:
        assert "entrypoint hash" in str(exc)
    else:
        raise AssertionError("source entrypoint hash drift must be rejected")


def test_job_runner_v2_cli_dry_run_validates_without_creating_attempt_root(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(
        json.dumps(_manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())),
        encoding="utf-8",
    )
    monkeypatch.setattr(ops, "PROJECT_ROOT", governance)
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pfmval_ops.py",
            "job",
            "run-v2",
            "--manifest",
            str(manifest_path),
            "--source-worktree",
            str(source),
            "--run-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
    )

    assert ops.main() == 0
    assert not (tmp_path / "runs").exists()
