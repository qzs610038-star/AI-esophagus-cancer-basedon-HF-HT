import importlib.util
import hashlib
import io
import json
import subprocess
import sys
import tarfile
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


def _archive_commit(repo: Path, commit: str, destination: Path) -> None:
    payload = subprocess.run(
        ["git", "archive", "--format=tar", commit], cwd=repo, check=True, capture_output=True,
    ).stdout
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        archive.extractall(destination, filter="data")


def test_job_runner_v2_uses_workspace_and_external_attempt_root(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    validation = {}

    def capture_validation(*_args, **kwargs):
        validation.update(kwargs)

    monkeypatch.setattr(ops, "validate_job_manifest", capture_validation)

    plan = ops.prepare_job_v2_execution(
        manifest,
        governance_root=governance,
        source_worktree=source,
        run_root=tmp_path / "runs",
    )

    assert Path(plan["source_worktree"]) == source.resolve()
    assert Path(plan["run_directory"]) == (tmp_path / "runs" / "W001" / "A001").resolve()
    assert Path(validation["source_git_root"]) == source.resolve()
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
    validation = {}

    def capture_validation(*_args, **kwargs):
        validation.update(kwargs)

    monkeypatch.setattr(ops, "validate_job_manifest", capture_validation)

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


def test_job_runner_v2_writes_terminal_event(tmp_path, monkeypatch):
    ops = _load_ops()
    governance = tmp_path / "governance"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)
    plan = ops.prepare_job_v2_execution(
        manifest, governance_root=governance, source_worktree=source, run_root=tmp_path / "runs",
    )
    ops.write_job_v2_started_event(plan, manifest)
    terminal = ops.write_job_v2_terminal_event(plan, manifest, returncode=7, error="fixture failure")
    event = json.loads(terminal.read_text(encoding="utf-8"))
    assert event["event_type"] == "EXPERIMENT_TERMINAL"
    assert event["status"] == "failed"
    assert event["returncode"] == 7


def test_job_runner_v2_mpp_command_uses_external_attempt_root(tmp_path, monkeypatch):
    ops = _load_ops()
    source = tmp_path / "W001"
    entrypoint = source / "train_mpp_uni2h_mlp.py"
    entrypoint.parent.mkdir()
    entrypoint.write_text("print('fixture')\n", encoding="utf-8")
    split = source / "mpp_standard_splits" / "group_2" / "split_manifest.csv"
    split.parent.mkdir(parents=True)
    split.write_text("fixture\n", encoding="utf-8")
    manifest = {
        "input_binding": {
            "data_manifest_id": "repair:fixture",
            "artifacts": [{"artifact_id": "mpp2_split_manifest", "sha256": hashlib.sha256(split.read_bytes()).hexdigest()}],
        }
    }
    monkeypatch.setattr(ops, "active_mpp_repair", lambda _root: {"data_manifest_id": "repair:fixture", "server_stage_path": "D:/labels"})
    monkeypatch.setattr(ops, "get_registered_path", lambda path_id, **_kwargs: Path(f"D:/{path_id}"))
    command = ops._job_v2_runtime_command(
        manifest,
        governance_root=tmp_path / "governance",
        source_worktree=source,
        run_directory=tmp_path / "runs" / "W001" / "A003",
        entrypoint=entrypoint,
        argv=["python", "train_mpp_uni2h_mlp.py", "--loss", "mse"],
    )
    assert "--output_root" in command
    assert command[command.index("--output_root") + 1] == str(tmp_path / "runs" / "W001" / "A003")
    assert command[command.index("--splits_root") + 1] == str((source / "mpp_standard_splits").resolve())


def test_job_runner_v2_accepts_byte_verified_regular_governance_bundle(tmp_path, monkeypatch):
    ops = _load_ops()
    governance_repo = tmp_path / "governance-repo"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance_repo, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    bundle = tmp_path / "governance-bundle"
    _archive_commit(governance_repo, governance_sha, bundle)
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    validation = {}

    def capture_validation(*_args, **kwargs):
        validation.update(kwargs)

    monkeypatch.setattr(ops, "validate_job_manifest", capture_validation)

    plan = ops.prepare_job_v2_execution(
        manifest,
        governance_root=bundle,
        source_worktree=source,
        run_root=tmp_path / "runs",
        governance_repo=governance_repo,
        governance_commit=governance_sha,
    )

    assert Path(plan["governance_root"]) == bundle.resolve()
    assert Path(validation["source_git_root"]) == source.resolve()


def test_job_runner_v2_rejects_tampered_regular_governance_bundle(tmp_path, monkeypatch):
    ops = _load_ops()
    governance_repo = tmp_path / "governance-repo"
    source = tmp_path / "W001"
    governance_sha = _commit_repo(governance_repo, "README.md", "governance\n")
    source_sha = _commit_repo(source, "train.py", "print('fixture')\n")
    bundle = tmp_path / "governance-bundle"
    _archive_commit(governance_repo, governance_sha, bundle)
    (bundle / "README.md").write_text("tampered\n", encoding="utf-8")
    manifest = _manifest(source_sha, governance_sha, hashlib.sha256((source / "train.py").read_bytes()).hexdigest())
    monkeypatch.setattr(ops, "validate_job_manifest", lambda *_args, **_kwargs: None)

    try:
        ops.prepare_job_v2_execution(
            manifest,
            governance_root=bundle,
            source_worktree=source,
            run_root=tmp_path / "runs",
            governance_repo=governance_repo,
            governance_commit=governance_sha,
        )
    except ValueError as exc:
        assert "file hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered governance archive must be rejected")
