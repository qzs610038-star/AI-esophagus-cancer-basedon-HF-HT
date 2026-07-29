import json
import subprocess
from pathlib import Path

import pytest

from deploy import pfmval_ops
from scripts.pfmval_governance import (
    acquire_workspace_lease,
    allocate_workspace,
    approve_experiment_protocol,
    assert_workspace_locus,
    build_job_v2,
    build_result_v2,
    build_critical_contract,
    initialize_workspace_registry,
    prepare_attempt,
    record_attempt_event,
    record_e1_adaptation,
    record_result_import_v2,
    register_governed_experiment,
    release_workspace_lease,
    resolve_attempt_run_root,
    scan_workspace_registry,
    validate_job_v2,
    validate_result_v2,
)


def test_workspace_registry_allocates_monotonic_ids_and_never_reuses_tombstones(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    initialize_workspace_registry(root)

    first = allocate_workspace(
        root,
        experiment_id="exp-a",
        display_name="alpha",
        local_relative_path="workspaces/W001",
    )
    second = allocate_workspace(
        root,
        experiment_id="exp-b",
        display_name="beta",
        local_relative_path="workspaces/W002",
    )

    assert first["workspace_id"] == "W001"
    assert second["workspace_id"] == "W002"

    registry_path = root / "project_state" / "workspace_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["workspaces"][0]["status"] = "tombstoned"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    third = allocate_workspace(
        root,
        experiment_id="exp-c",
        display_name="gamma",
        local_relative_path="workspaces/W003",
    )
    assert third["workspace_id"] == "W003"

    with pytest.raises(ValueError, match="already has workspace"):
        allocate_workspace(
            root,
            experiment_id="exp-a",
            display_name="duplicate",
            local_relative_path="workspaces/W004",
        )


def test_workspace_shadow_scan_and_locus_guard_have_no_filesystem_side_effects(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    initialize_workspace_registry(root)
    workspace = allocate_workspace(
        root,
        experiment_id="exp-a",
        display_name="alpha",
        local_relative_path="workspaces/W001",
        branch="codex/exp-a",
        source_commit="a" * 40,
    )
    registry_path = root / "project_state" / "workspace_registry.json"
    before = registry_path.read_bytes()

    report = scan_workspace_registry(
        root,
        host_scope="local",
        observed_worktrees=[
            {
                "path": str(root / "workspaces" / "W001"),
                "branch": "codex/exp-a",
                "head": "a" * 40,
                "dirty": False,
            },
            {
                "path": str(root / "workspaces" / "unregistered"),
                "branch": "codex/other",
                "head": "b" * 40,
                "dirty": False,
            },
        ],
    )

    assert report["status"] == "WARN"
    assert report["workspaces"][0]["workspace_id"] == "W001"
    assert report["unregistered"][0]["severity"] == "WARN"
    assert registry_path.read_bytes() == before

    assert_workspace_locus(
        workspace,
        cwd=root / "workspaces" / "W001",
        branch="codex/exp-a",
        head="a" * 40,
        dirty=False,
    )
    with pytest.raises(ValueError, match="dirty"):
        assert_workspace_locus(
            workspace,
            cwd=root / "workspaces" / "W001",
            branch="codex/exp-a",
            head="a" * 40,
            dirty=True,
        )
    with pytest.raises(ValueError, match="outside registered root"):
        allocate_workspace(
            root,
            experiment_id="exp-escape",
            display_name="escape",
            local_relative_path="../outside/W004",
        )


def test_external_git_worktree_binding_requires_exact_path_branch_and_head(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=PFMval Test",
            "-c",
            "user.email=pfmval-test@example.invalid",
            "commit",
            "-m",
            "baseline",
        ],
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    external = tmp_path / "W001"
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "-b", "codex/exp-a", str(external)],
        check=True,
        capture_output=True,
    )

    workspace = allocate_workspace(
        root,
        experiment_id="exp-a",
        display_name="external",
        local_relative_path=str(external),
        branch="codex/exp-a",
        source_commit=head,
    )
    assert workspace["hosts"]["local"]["relative_path"] == str(external.resolve())
    assert workspace["hosts"]["local"]["branch"] == "codex/exp-a"
    assert_workspace_locus(
        workspace,
        cwd=external,
        branch="codex/exp-a",
        head=head,
        dirty=False,
    )
    with pytest.raises(ValueError, match="branch"):
        allocate_workspace(
            root,
            experiment_id="exp-b",
            display_name="wrong-branch",
            local_relative_path=str(external),
            branch="codex/not-exp-a",
            source_commit=head,
        )
def test_attempt_run_roots_are_external_and_workspace_lease_requires_explicit_release(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    initialize_workspace_registry(root)
    workspace = allocate_workspace(
        root,
        experiment_id="exp-a",
        display_name="alpha",
        local_relative_path="workspaces/W001",
        branch="codex/exp-a",
        source_commit="a" * 40,
    )
    run_root = tmp_path / "runs"
    first_run = resolve_attempt_run_root(
        run_root,
        workspace_id=workspace["workspace_id"],
        attempt_id="A001",
    )
    second_run = resolve_attempt_run_root(
        run_root,
        workspace_id=workspace["workspace_id"],
        attempt_id="A002",
    )

    assert first_run == run_root / "W001" / "A001"
    assert second_run == run_root / "W001" / "A002"
    assert not first_run.exists()
    assert not second_run.exists()

    lease = acquire_workspace_lease(
        root,
        workspace_id="W001",
        experiment_id="exp-a",
        attempt_id="A001",
        source_commit="a" * 40,
        critical_contract_sha256="b" * 64,
        holder="fixture",
        pid=1234,
        run_root=first_run,
    )
    assert lease["attempt_id"] == "A001"
    with pytest.raises(ValueError, match="active lease"):
        acquire_workspace_lease(
            root,
            workspace_id="W001",
            experiment_id="exp-a",
            attempt_id="A002",
            source_commit="a" * 40,
            critical_contract_sha256="b" * 64,
            holder="fixture",
            pid=5678,
            run_root=second_run,
        )

    release_workspace_lease(root, workspace_id="W001", attempt_id="A001")
    second_lease = acquire_workspace_lease(
        root,
        workspace_id="W001",
        experiment_id="exp-a",
        attempt_id="A002",
        source_commit="a" * 40,
        critical_contract_sha256="b" * 64,
        holder="fixture",
        pid=5678,
        run_root=second_run,
    )
    assert second_lease["attempt_id"] == "A002"


def test_critical_contract_digest_is_canonical_and_sensitive_to_scientific_choices():
    base = {
        "experiment_id": "exp-a",
        "protocol_revision": 1,
        "source_commit": "a" * 40,
        "critical_code": ["train.py", "model.py"],
        "resolved_hyperparameters": {"lr": 0.001, "seed": 42},
        "loss_contract": {"name": "mse"},
        "regularization_contract": {"weight_decay": 0.0001},
        "feature_flags": {"lora": False},
        "data_manifest_id": "manifest-v1",
        "data_manifest_sha256": "b" * 64,
        "input_artifacts": [
            {"artifact_id": "checkpoint", "sha256": "c" * 64}
        ],
        "selection_policy": {"metric": "val_loss", "direction": "min"},
        "external_eval_policy": {"dataset": "XZY", "selection_allowed": False},
    }
    reordered = {
        "selection_policy": {"direction": "min", "metric": "val_loss"},
        "input_artifacts": [
            {"sha256": "c" * 64, "artifact_id": "checkpoint"}
        ],
        **{
            key: value
            for key, value in base.items()
            if key not in {"selection_policy", "input_artifacts"}
        },
    }

    first = build_critical_contract(base)
    second = build_critical_contract(reordered)
    changed_lr = build_critical_contract(
        {
            **base,
            "resolved_hyperparameters": {"lr": 0.002, "seed": 42},
        }
    )
    changed_code_order = build_critical_contract(
        {**base, "critical_code": ["model.py", "train.py"]}
    )

    assert first["canonical_sha256"] == second["canonical_sha256"]
    assert first["canonical_sha256"] != changed_lr["canonical_sha256"]
    assert first["canonical_sha256"] != changed_code_order["canonical_sha256"]
    with pytest.raises(ValueError, match="full 40-character"):
        build_critical_contract({**base, "source_commit": "abc1234"})


def _write_minimal_experiment_registry(root: Path) -> None:
    path = root / "experiments" / "experiment_registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-07-27T00:00:00+00:00",
                "experiments": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_protocol_approval_and_started_event_enforce_append_only_run_budget(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write_minimal_experiment_registry(root)
    contract = build_critical_contract(
        {
            "experiment_id": "exp-a",
            "protocol_revision": 1,
            "source_commit": "a" * 40,
            "critical_code": ["train.py"],
            "resolved_hyperparameters": {"seed": 42},
            "loss_contract": {"name": "mse"},
            "regularization_contract": {},
            "feature_flags": {},
            "data_manifest_id": "manifest-v1",
            "data_manifest_sha256": "b" * 64,
            "input_artifacts": [],
            "selection_policy": {"metric": "val_loss"},
            "external_eval_policy": {"selection_allowed": False},
        }
    )
    with pytest.raises(ValueError, match="run_limit"):
        register_governed_experiment(
            root,
            experiment_id="exp-a",
            display_name="alpha",
            phase="smoke",
            run_limit=0,
            critical_contract=contract,
            local_relative_path="workspaces/W001",
        )

    experiment = register_governed_experiment(
        root,
        experiment_id="exp-a",
        display_name="alpha",
        phase="smoke",
        run_limit=2,
        critical_contract=contract,
        local_relative_path="workspaces/W001",
    )
    assert experiment["workspace_id"] == "W001"
    approval = approve_experiment_protocol(
        root,
        approval_id="APR-exp-a-r001",
        experiment_id="exp-a",
        protocol_revision=1,
        phase="smoke",
        run_limit=2,
        critical_contract_sha256=contract["canonical_sha256"],
        adaptation_policy="e1_allowlist_only",
        source="explicit_user_instruction",
    )
    assert approval["status"] == "active"

    first = prepare_attempt(
        root,
        experiment_id="exp-a",
        workspace_id="W001",
        approval_id=approval["approval_id"],
        run_units=1,
        source_commit="a" * 40,
        critical_contract_sha256=contract["canonical_sha256"],
    )
    record_attempt_event(
        root,
        workspace_id="W001",
        attempt_id=first["attempt_id"],
        event_id="evt-preflight-failed",
        event_type="PREFLIGHT_FAILED",
    )
    assert first["budget"]["consumed_after_event"] == 0
    started = record_attempt_event(
        root,
        workspace_id="W001",
        attempt_id=first["attempt_id"],
        event_id="evt-start-a001",
        event_type="EXPERIMENT_STARTED",
    )
    assert started["budget"]["consumed"] == 1
    duplicate = record_attempt_event(
        root,
        workspace_id="W001",
        attempt_id=first["attempt_id"],
        event_id="evt-start-a001",
        event_type="EXPERIMENT_STARTED",
    )
    assert duplicate["status"] == "already_recorded"
    assert duplicate["budget"]["consumed"] == 1

    second = prepare_attempt(
        root,
        experiment_id="exp-a",
        workspace_id="W001",
        approval_id=approval["approval_id"],
        run_units=1,
        source_commit="a" * 40,
        critical_contract_sha256=contract["canonical_sha256"],
    )
    record_attempt_event(
        root,
        workspace_id="W001",
        attempt_id=second["attempt_id"],
        event_id="evt-start-a002",
        event_type="EXPERIMENT_STARTED",
    )
    with pytest.raises(ValueError, match="run budget exhausted"):
        prepare_attempt(
            root,
            experiment_id="exp-a",
            workspace_id="W001",
            approval_id=approval["approval_id"],
            run_units=1,
            source_commit="a" * 40,
            critical_contract_sha256=contract["canonical_sha256"],
        )


def test_e1_adaptation_requires_allowlist_and_unchanged_critical_contract(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    allowed = record_e1_adaptation(
        root,
        adaptation_record_id="ADP-W001-A001-r002",
        workspace_id="W001",
        attempt_id="A001",
        previous_source_commit="a" * 40,
        new_source_commit="b" * 40,
        previous_dispatch_revision=1,
        new_dispatch_revision=2,
        critical_contract_before="c" * 64,
        critical_contract_after="c" * 64,
        changed_items=[
            {
                "path": "deploy/run_experiment.ps1",
                "change_class": "launcher_argument_compatibility",
            },
            {
                "path": "configs/server_paths.yaml",
                "change_class": "registered_path",
            },
        ],
        tests=["tests/test_launcher.py::test_argv"],
    )
    assert allowed["status"] == "recorded"

    with pytest.raises(ValueError, match="not E1-allowlisted"):
        record_e1_adaptation(
            root,
            adaptation_record_id="ADP-W001-A001-r003",
            workspace_id="W001",
            attempt_id="A001",
            previous_source_commit="b" * 40,
            new_source_commit="d" * 40,
            previous_dispatch_revision=2,
            new_dispatch_revision=3,
            critical_contract_before="c" * 64,
            critical_contract_after="c" * 64,
            changed_items=[
                {"path": "train.py", "change_class": "loss_contract"}
            ],
            tests=["tests/test_train.py::test_loss"],
        )
    with pytest.raises(ValueError, match="critical contract changed"):
        record_e1_adaptation(
            root,
            adaptation_record_id="ADP-W001-A001-r004",
            workspace_id="W001",
            attempt_id="A001",
            previous_source_commit="b" * 40,
            new_source_commit="e" * 40,
            previous_dispatch_revision=2,
            new_dispatch_revision=4,
            critical_contract_before="c" * 64,
            critical_contract_after="f" * 64,
            changed_items=[
                {
                    "path": "deploy/run_experiment.ps1",
                    "change_class": "logging",
                }
            ],
            tests=["tests/test_launcher.py::test_logging"],
        )


def test_server_job_v2_binds_workspace_attempt_approval_contract_and_run_units():
    prepared = {
        "experiment_id": "exp-a",
        "workspace_id": "W001",
        "attempt_id": "A001",
        "job_id": "W001-A001",
        "approval_id": "APR-exp-a-r001",
        "protocol_revision": 1,
        "phase": "smoke",
        "run_units": 3,
        "source_commit": "a" * 40,
        "critical_contract_sha256": "b" * 64,
        "workspace_branch": "automation/local/W001/A001",
    }
    job = build_job_v2(
        prepared,
        command_id="standard_training",
        path_ids=["mpp_data_root", "server_mpp_results"],
        resolved_argv=["python", "train.py", "--seeds", "1", "2", "3"],
        input_binding={
            "data_manifest_id": "manifest-v1",
            "data_manifest_sha256": "c" * 64,
            "artifacts": [{"artifact_id": "checkpoint", "sha256": "d" * 64}],
        },
    )
    validate_job_v2(job)

    assert job["schema_version"] == "2.0"
    assert job["run_units"] == 3
    assert job["result_branch"] == "automation/server/W001/A001"
    with pytest.raises(ValueError, match="run_units"):
        validate_job_v2({**job, "run_units": 0})
    with pytest.raises(ValueError, match="full 40-character"):
        validate_job_v2({**job, "source_commit": "abc1234"})


def test_result_v2_tiers_artifacts_and_import_is_idempotent_without_budget_change(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    job = {
        "schema_version": "2.0",
        "job_id": "W001-A001",
        "experiment_id": "exp-a",
        "workspace_id": "W001",
        "attempt_id": "A001",
        "protocol_revision": 1,
        "approval_id": "APR-exp-a-r001",
        "critical_contract_sha256": "b" * 64,
        "source_commit": "a" * 40,
        "phase": "formal",
        "run_units": 1,
    }
    result = build_result_v2(
        job,
        result_id="RES-W001-A001",
        status="success",
        artifacts=[
            {
                "artifact_id": "metrics",
                "path": "metrics.json",
                "kind": "metrics",
                "size_bytes": 100,
                "sha256": "c" * 64,
                "evidence_role": "critical",
                "retention": "retain",
            },
            {
                "artifact_id": "selection",
                "path": "best_epoch.json",
                "kind": "selection_proof",
                "size_bytes": 80,
                "sha256": "d" * 64,
                "evidence_role": "critical",
                "retention": "retain",
            },
                {
                    "artifact_id": "stdout",
                    "path": "stdout.log",
                    "kind": "log",
                    "size_bytes": 200,
                    "sha256": "e" * 64,
                    "evidence_role": "diagnostic",
                    "retention": "short",
                },
        ],
        metrics={"pcc": 0.5},
        metric_artifact_ids=["metrics"],
    )
    validate_result_v2(result)
    before_attempt_log = (
        root / "project_state" / "attempt_events.jsonl"
    ).read_bytes() if (root / "project_state" / "attempt_events.jsonl").exists() else b""

    first = record_result_import_v2(root, result, bundle_sha256="e" * 64)
    second = record_result_import_v2(root, result, bundle_sha256="e" * 64)

    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    after_attempt_log = (
        root / "project_state" / "attempt_events.jsonl"
    ).read_bytes() if (root / "project_state" / "attempt_events.jsonl").exists() else b""
    assert after_attempt_log == before_attempt_log
    with pytest.raises(ValueError, match="different bundle SHA"):
        record_result_import_v2(root, result, bundle_sha256="f" * 64)

    missing_sha = {
        **result,
        "artifacts": [
            {
                **result["artifacts"][0],
                "sha256": "",
            },
            *result["artifacts"][1:],
        ],
    }
    with pytest.raises(ValueError, match="critical artifact requires SHA-256"):
        validate_result_v2(missing_sha)
    with pytest.raises(ValueError, match="diagnostic artifact"):
        validate_result_v2(
            {
                **result,
                "metric_artifact_ids": ["stdout"],
            }
        )


def test_p0a_local_cli_happy_path_requires_no_ad_hoc_state_writes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "PFMval Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "config",
            "user.email",
            "pfmval@example.invalid",
        ],
        check=True,
    )
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    _write_minimal_experiment_registry(root)
    monkeypatch.setattr(pfmval_ops, "PROJECT_ROOT", root)

    def run_cli(*args: str) -> None:
        monkeypatch.setattr("sys.argv", ["pfmval_ops.py", *args])
        assert pfmval_ops.main() == 0

    contract_input = root / "contract_input.json"
    contract_output = root / "contract.json"
    contract_input.write_text(
        json.dumps(
            {
                "experiment_id": "exp-cli",
                "protocol_revision": 1,
                "source_commit": "a" * 40,
                "critical_code": ["train.py"],
                "resolved_hyperparameters": {"seed": 42},
                "loss_contract": {"name": "mse"},
                "regularization_contract": {},
                "feature_flags": {},
                "data_manifest_id": "manifest-v1",
                "data_manifest_sha256": "b" * 64,
                "input_artifacts": [],
                "selection_policy": {"metric": "val_loss"},
                "external_eval_policy": {"selection_allowed": False},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_cli(
        "governance",
        "contract",
        "--input",
        str(contract_input),
        "--output",
        str(contract_output),
    )
    contract = json.loads(contract_output.read_text(encoding="utf-8"))
    run_cli(
        "experiment",
        "register",
        "--experiment-id",
        "exp-cli",
        "--display-name",
        "CLI fixture",
        "--phase",
        "formal",
        "--run-limit",
        "1",
        "--critical-contract",
        str(contract_output),
        "--workspace-relative-path",
        "workspaces/W001",
    )
    run_cli(
        "experiment",
        "approve",
        "--approval-id",
        "APR-exp-cli-r001",
        "--experiment-id",
        "exp-cli",
        "--protocol-revision",
        "1",
        "--phase",
        "formal",
        "--run-limit",
        "1",
        "--critical-contract-sha256",
        contract["canonical_sha256"],
    )
    prepared_path = root / "prepared.json"
    run_cli(
        "attempt",
        "prepare",
        "--experiment-id",
        "exp-cli",
        "--workspace-id",
        "W001",
        "--approval-id",
        "APR-exp-cli-r001",
        "--run-units",
        "1",
        "--source-commit",
        "a" * 40,
        "--critical-contract-sha256",
        contract["canonical_sha256"],
        "--output",
        str(prepared_path),
    )
    run_cli(
        "attempt",
        "event",
        "--workspace-id",
        "W001",
        "--attempt-id",
        "A001",
        "--event-id",
        "evt-validated",
        "--event-type",
        "VALIDATED",
    )
    run_cli(
        "attempt",
        "event",
        "--workspace-id",
        "W001",
        "--attempt-id",
        "A001",
        "--event-id",
        "evt-started",
        "--event-type",
        "EXPERIMENT_STARTED",
    )

    input_binding = root / "input_binding.json"
    resolved_argv = root / "resolved_argv.json"
    input_binding.write_text(
        json.dumps(
            {
                "data_manifest_id": "manifest-v1",
                "data_manifest_sha256": "b" * 64,
                "artifacts": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_argv.write_text(
        json.dumps(["python", "train.py", "--seed", "42"], indent=2) + "\n",
        encoding="utf-8",
    )
    job_path = root / "job_v2.json"
    run_cli(
        "governance",
        "job-v2",
        "build",
        "--prepared",
        str(prepared_path),
        "--input-binding",
        str(input_binding),
        "--resolved-argv",
        str(resolved_argv),
        "--command-id",
        "standard_training",
        "--path-id",
        "mpp_data_root",
        "--output",
        str(job_path),
    )

    artifacts_path = root / "artifacts.json"
    metrics_path = root / "metrics.json"
    artifacts_path.write_text(
        json.dumps(
            [
                {
                    "artifact_id": "metrics",
                    "path": "metrics.json",
                    "kind": "metrics",
                    "size_bytes": 100,
                    "sha256": "c" * 64,
                    "evidence_role": "critical",
                    "retention": "retain",
                },
                {
                    "artifact_id": "selection",
                    "path": "best_epoch.json",
                    "kind": "selection_proof",
                    "size_bytes": 80,
                    "sha256": "d" * 64,
                    "evidence_role": "critical",
                    "retention": "retain",
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"pcc": 0.5}, indent=2) + "\n",
        encoding="utf-8",
    )
    result_path = root / "result_v2.json"
    run_cli(
        "governance",
        "result-v2",
        "build",
        "--job",
        str(job_path),
        "--result-id",
        "RES-W001-A001",
        "--status",
        "success",
        "--artifacts",
        str(artifacts_path),
        "--metrics",
        str(metrics_path),
        "--metric-artifact-id",
        "metrics",
        "--output",
        str(result_path),
    )
    run_cli(
        "governance",
        "result-v2",
        "validate",
        "--manifest",
        str(result_path),
    )
    run_cli(
        "governance",
        "result-v2",
        "record-import",
        "--manifest",
        str(result_path),
        "--bundle-sha256",
        "e" * 64,
    )

    assert (root / "project_state" / "workspace_registry.json").is_file()
    assert (root / "project_state" / "experiment_approvals.jsonl").is_file()
    assert (root / "project_state" / "attempt_events.jsonl").is_file()
    assert (root / "project_state" / "result_import_events.jsonl").is_file()
