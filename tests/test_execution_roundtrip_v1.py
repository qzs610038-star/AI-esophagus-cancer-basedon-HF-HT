from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "g12_server_roundtrip_blockers_v1.json"
JOB_PATH = ROOT / "automation" / "jobs" / "W001-A003" / "job.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_probe(*, gpu_free_bytes: int = 8_000_000_000) -> dict:
    return {
        "os": {"name": "Windows", "release": "fixture"},
        "powershell": {"path": "pwsh", "version": "7.5.0"},
        "git": {"path": "git", "version": "2.50.0"},
        "python_candidates": [
            {
                "path": "python",
                "version": "3.10.0",
                "jsonschema": True,
                "torch": True,
            }
        ],
        "schema_backends": ["python-jsonschema", "powershell-test-json"],
        "selected_schema_backend": "python-jsonschema",
        "pytorch": {"version": "2.7.0"},
        "cuda": {"available": True, "version": "12.8"},
        "gpu": {"count": 1, "free_bytes": gpu_free_bytes},
        "disk": {"free_bytes": 100_000_000_000},
        "path_ids": {"server_automation_worktrees": "D:/fixture/worktrees"},
        "git_line_endings": {"core.autocrlf": "true"},
    }


def _synthetic_governed_project(
    root: Path,
    *,
    command_id: str,
) -> Path:
    from scripts.pfmval_execution_roundtrip import RUNTIME_FILES

    experiment_id = f"fixture_{command_id}"
    job_id = f"W900-A001-{command_id}"
    approval_id = f"APR-{command_id}"
    contract_sha = hashlib.sha256(command_id.encode("utf-8")).hexdigest()
    source_commit = "1" * 40
    workspace_id = "W900"
    attempt_id = "A001"
    for relative in RUNTIME_FILES:
        source = ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "experiments" / "experiment_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiments": [
                    {
                        "id": experiment_id,
                        "workspace_id": workspace_id,
                        "critical_contract_sha256": contract_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    governance = root / "project_state" / "governance"
    governance.mkdir(parents=True, exist_ok=True)
    (governance / f"{command_id}_critical_contract.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "canonical_sha256": contract_sha,
                "data_manifest_id": "fixture-data",
                "input_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    (governance / f"{command_id}_attempt.json").write_text(
        json.dumps(
            {
                "event_type": "ATTEMPT_PREPARED",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "budget": {"remaining_after_prepare": 1},
            }
        ),
        encoding="utf-8",
    )
    (root / "project_state" / "experiment_approvals.jsonl").write_text(
        json.dumps(
            {
                "approval_id": approval_id,
                "experiment_id": experiment_id,
                "critical_contract_sha256": contract_sha,
                "status": "active",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "project_state" / "attempt_events.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    (root / "project_state" / "workspace_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "updated_at": "2026-07-29T00:00:00+00:00",
                "next_workspace_number": 901,
                "workspaces": [
                    {
                        "workspace_id": workspace_id,
                        "experiment_id": experiment_id,
                        "lease": None,
                        "hosts": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "server_paths.yaml").write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "paths:",
                "  server_automation_worktrees:",
                "    path: 'D:\\fixture\\worktrees'",
                "  server_repo_worktree:",
                "    path: 'D:\\fixture\\repo'",
                "  server_mpp_results:",
                "    path: 'D:\\fixture\\results'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    job_dir = root / "automation" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "job_id": job_id,
                "experiment_id": experiment_id,
                "workspace_id": workspace_id,
                "attempt_id": attempt_id,
                "approval_id": approval_id,
                "source_commit": source_commit,
                "critical_contract_sha256": contract_sha,
                "run_units": 1,
                "resolved_argv": ["python", "fixture.py"],
                "command_id": command_id,
                "path_ids": [],
                "input_binding": {"data_manifest_id": "fixture-data", "artifacts": []},
                "created_at": "2026-07-29T00:00:00+00:00",
                "dispatch_revision": 1,
                "result_branch": f"automation/server/{workspace_id}/{attempt_id}",
                "execution_binding": {
                    "mode": "execution_bundle_v1",
                    "governance_commit": "2" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    return job_path


def test_g12_review_is_frozen_as_fourteen_named_regression_fixtures():
    payload = _read_json(FIXTURE_PATH)
    blockers = payload["blockers"]

    assert payload["source_review_sha256"] == (
        "4921ccfc055ef24410dfadeeef1e6bffe6096547c0f0cbcb07e66294995cdd08"
    )
    assert [item["id"] for item in blockers] == [
        "G12-B01",
        "G12-B02",
        "G12-B03",
        "G12-B04",
        "G12-B05",
        "G12-B06",
        "G12-R1",
        "G12-R2",
        "G12-R3",
        "G12-R4",
        "G12-RR1",
        "G12-RR2",
        "G12-RR3",
        "G12-RR4",
    ]
    assert {item["stage"] for item in blockers} == {
        "execution_bundle",
        "preflight_v2",
        "result_bundle_v1",
        "server_fingerprint_v1",
        "state_machine",
    }


def test_execution_bundle_is_minimal_closed_content_addressed_and_reproducible(
    tmp_path,
):
    from scripts.pfmval_execution_roundtrip import build_execution_bundle_v1

    first = tmp_path / "bundle-a"
    second = tmp_path / "bundle-b"
    report_a = build_execution_bundle_v1(ROOT, JOB_PATH, first)
    report_b = build_execution_bundle_v1(ROOT, JOB_PATH, second)
    manifest = _read_json(first / "bundle_manifest.json")

    assert report_a["bundle_id"] == report_b["bundle_id"]
    assert report_a["bundle_sha256"] == report_b["bundle_sha256"]
    assert manifest["identity"] == {
        "job_id": "W001-A003",
        "source_commit": "a04319a1d6aeafcd35d6adc7e2893f1d7b683bda",
        "critical_contract_sha256": (
            "df0c5df369a3049125d06b985683c7bf7c4027664fc48d8b86f103e0ad8044e5"
        ),
    }
    paths = {item["path"] for item in manifest["files"]}
    assert {
        "job/job.json",
        "governance/experiment.json",
        "governance/approval.json",
        "governance/critical_contract.json",
        "governance/attempt.json",
        "governance/workspace.json",
        "governance/path_snapshot.json",
    }.issubset(paths)
    assert not any(path.startswith("automation/results/") for path in paths)
    assert "experiments/experiment_dashboard.md" not in paths
    assert not any(
        path.startswith(prefix)
        for path in paths
        for prefix in ("histogene/", "egnv1/", "egnv2/")
    )
    for record in manifest["files"]:
        payload = (first / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_execution_bundle_cli_generates_one_path_operation_card(tmp_path):
    bundle = tmp_path / "bundle"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy" / "pfmval_ops.py"),
            "governance",
            "execution-bundle-v1",
            "build",
            "--job",
            str(JOB_PATH),
            "--output",
            str(bundle),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    operation_card = (bundle / "run_server.ps1").read_text(encoding="utf-8")
    assert "'--bundle', $Bundle" in operation_card
    assert "source_commit" not in operation_card
    assert "critical_contract_sha256" not in operation_card
    assert "refs/heads/" not in operation_card
    assert "commit message" not in operation_card.lower()

    validated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy" / "pfmval_ops.py"),
            "governance",
            "execution-bundle-v1",
            "validate",
            "--bundle",
            str(bundle),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_two_distinct_experiment_types_build_and_verify_without_execution(
    tmp_path,
):
    from scripts.pfmval_execution_roundtrip import build_execution_bundle_v1

    reports = []
    for command_id in ("standard_training", "mpp_pathway_ridge_calibration"):
        project = tmp_path / command_id
        job = _synthetic_governed_project(project, command_id=command_id)
        reports.append(
            build_execution_bundle_v1(
                project,
                job,
                tmp_path / f"{command_id}-bundle",
            )
        )

    assert all(report["status"] == "valid" for report in reports)
    assert reports[0]["bundle_id"] != reports[1]["bundle_id"]
    assert not list(tmp_path.rglob("attempt_started.json"))


def test_server_fingerprint_is_reused_until_expiry_and_then_refreshed(tmp_path):
    from scripts.pfmval_execution_roundtrip import ensure_server_fingerprint_v1

    now = datetime(2026, 7, 29, 16, 0, tzinfo=timezone.utc)
    state_dir = tmp_path / "state"
    calls = []

    def probe() -> dict:
        calls.append(len(calls) + 1)
        return _fixture_probe(gpu_free_bytes=8_000_000_000 + len(calls))

    first = ensure_server_fingerprint_v1(
        state_dir,
        now=now,
        ttl_seconds=60,
        probe=probe,
    )
    reused = ensure_server_fingerprint_v1(
        state_dir,
        now=now + timedelta(seconds=59),
        ttl_seconds=60,
        probe=probe,
    )
    refreshed = ensure_server_fingerprint_v1(
        state_dir,
        now=now + timedelta(seconds=61),
        ttl_seconds=60,
        probe=probe,
    )

    assert first["reused"] is False
    assert reused["reused"] is True
    assert reused["fingerprint_sha256"] == first["fingerprint_sha256"]
    assert refreshed["reused"] is False
    assert refreshed["fingerprint_sha256"] != first["fingerprint_sha256"]
    assert calls == [1, 2]
    fingerprint = _read_json(state_dir / "server_fingerprint_v1.json")
    assert "credentials" not in fingerprint
    assert fingerprint["expires_at"] == "2026-07-29T16:02:01+00:00"


def test_preflight_v2_aggregates_all_independent_blockers_without_fail_fast(
    tmp_path,
):
    from scripts.pfmval_execution_roundtrip import aggregate_preflight_v2

    observations = {
        "bundle": {"valid": True},
        "source": {
            "head_matches": False,
            "clean": False,
            "object_available": False,
            "entrypoint_sha_matches": False,
        },
        "bindings": {
            "approval_matches": False,
            "contract_matches": False,
            "job_matches": False,
            "data_manifest_matches": False,
            "split_matches": False,
        },
        "attempt": {
            "root_absent": False,
            "lease_available": False,
            "budget_available": False,
        },
        "runtime": {
            "schema_backend_available": False,
            "gpu_sufficient": False,
            "disk_sufficient": False,
        },
        "publish": {
            "fast_forward": False,
            "remote_identity_matches": False,
        },
    }

    report = aggregate_preflight_v2(
        bundle_manifest={"bundle_id": "b" * 64},
        fingerprint=_fixture_probe(),
        observations=observations,
    )
    failure_ids = {item["check_id"] for item in report["failures"]}

    assert report["status"] == "blocked"
    assert report["safe_to_start"] is False
    assert len(failure_ids) == 17
    assert {
        "source_head",
        "source_clean",
        "source_object",
        "entrypoint_sha",
        "approval_binding",
        "contract_binding",
        "job_binding",
        "data_manifest",
        "split_manifest",
        "attempt_root",
        "lease",
        "budget",
        "schema_backend",
        "gpu_capacity",
        "disk_capacity",
        "publish_fast_forward",
        "remote_identity",
    } == failure_ids


def test_preflight_v2_reports_bundle_tamper_and_missing_host_assets_together(
    tmp_path,
):
    from scripts.pfmval_execution_roundtrip import (
        build_execution_bundle_v1,
        run_preflight_v2,
    )

    bundle = tmp_path / "bundle"
    build_execution_bundle_v1(ROOT, JOB_PATH, bundle)
    runner = bundle / "runtime" / "deploy" / "pfmval_ops.py"
    runner.write_bytes(runner.read_bytes() + b"\n# tampered\n")

    report = run_preflight_v2(
        bundle,
        state_root=tmp_path / "state",
        now=datetime(2026, 7, 29, 16, 0, tzinfo=timezone.utc),
        probe=_fixture_probe,
    )
    failure_ids = {item["check_id"] for item in report["failures"]}

    assert report["status"] == "blocked"
    assert "bundle_integrity" in failure_ids
    assert {
        "source_head",
        "source_clean",
        "source_object",
        "entrypoint_sha",
        "publish_fast_forward",
        "remote_identity",
    }.issubset(failure_ids)
    assert "budget" not in failure_ids


def test_state_machine_replay_and_terminal_recovery_never_rerun(tmp_path):
    from scripts.pfmval_execution_roundtrip import execute_roundtrip_v1

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": "a" * 64,
                "identity": {
                    "job_id": "W777-A001",
                    "source_commit": "1" * 40,
                    "critical_contract_sha256": "2" * 64,
                },
                "run_units": 1,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    calls = {"run": 0, "package": 0, "publish": 0}

    def run_once(_context):
        calls["run"] += 1
        return {"returncode": 0}

    def package_once(_context):
        calls["package"] += 1
        return {"bundle_sha256": "3" * 64, "result_id": "R001"}

    def publish_once(_context):
        calls["publish"] += 1
        return {"status": "published", "commit_sha": "4" * 40}

    first = execute_roundtrip_v1(
        bundle,
        state_root=state_root,
        preflight={"safe_to_start": True, "failures": [], "warnings": []},
        executor=run_once,
        result_packager=package_once,
        result_publisher=publish_once,
    )
    replay = execute_roundtrip_v1(
        bundle,
        state_root=state_root,
        preflight={"safe_to_start": True, "failures": [], "warnings": []},
        executor=run_once,
        result_packager=package_once,
        result_publisher=publish_once,
    )

    assert first["status"] == "PUBLISHED"
    assert replay["status"] == "ALREADY_PUBLISHED"
    assert calls == {"run": 1, "package": 1, "publish": 1}
    assert first["run_units_consumed"] == replay["run_units_consumed"] == 1

    recovery_bundle = tmp_path / "recovery-bundle"
    recovery_bundle.mkdir()
    (recovery_bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": "b" * 64,
                "identity": {
                    "job_id": "W777-A002",
                    "source_commit": "1" * 40,
                    "critical_contract_sha256": "2" * 64,
                },
                "run_units": 1,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    recovery_state = state_root / ("b" * 64)
    recovery_state.mkdir(parents=True)
    (recovery_state / "attempt_started.json").write_text(
        json.dumps({"event_type": "EXPERIMENT_STARTED", "run_units": 1}),
        encoding="utf-8",
    )
    (recovery_state / "attempt_terminal.json").write_text(
        json.dumps(
            {
                "event_type": "EXPERIMENT_TERMINAL",
                "status": "completed",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )

    recovered = execute_roundtrip_v1(
        recovery_bundle,
        state_root=state_root,
        preflight={"safe_to_start": True, "failures": [], "warnings": []},
        executor=lambda _context: pytest.fail("terminal recovery must not execute"),
        result_packager=package_once,
        result_publisher=publish_once,
    )
    assert recovered["status"] == "PUBLISHED"
    assert recovered["recovered_from_terminal"] is True
    assert calls["run"] == 1


def test_state_machine_rejects_same_job_identity_with_different_bundle_sha(
    tmp_path,
):
    from scripts.pfmval_execution_roundtrip import execute_roundtrip_v1

    state_root = tmp_path / "state"
    preflight = {"safe_to_start": False, "failures": [], "warnings": []}
    for index, bundle_id in enumerate(("a" * 64, "b" * 64)):
        bundle = tmp_path / f"bundle-{index}"
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "bundle_id": bundle_id,
                    "identity": {
                        "job_id": "W777-A003",
                        "source_commit": "1" * 40,
                        "critical_contract_sha256": "2" * 64,
                    },
                    "run_units": 1,
                    "files": [],
                }
            ),
            encoding="utf-8",
        )
        if index == 0:
            report = execute_roundtrip_v1(
                bundle,
                state_root=state_root,
                preflight=preflight,
            )
            assert report["status"] == "BLOCKED"
        else:
            with pytest.raises(ValueError, match="different bundle SHA"):
                execute_roundtrip_v1(
                    bundle,
                    state_root=state_root,
                    preflight=preflight,
                )


def test_governance_maintenance_workspace_can_advance_clean_branch_head(tmp_path):
    from scripts.pfmval_governance import assert_workspace_locus

    workspace = {
        "workspace_kind": "governance_maintenance",
        "hosts": {
            "local": {
                "relative_path": str(tmp_path),
                "branch": "codex/w002-server-execution-roundtrip-v1-20260729",
                "current_source_commit": "1" * 40,
            }
        },
    }

    assert_workspace_locus(
        workspace,
        cwd=tmp_path,
        branch="codex/w002-server-execution-roundtrip-v1-20260729",
        head="2" * 40,
        dirty=False,
    )
    with pytest.raises(ValueError, match="dirty"):
        assert_workspace_locus(
            workspace,
            cwd=tmp_path,
            branch="codex/w002-server-execution-roundtrip-v1-20260729",
            head="2" * 40,
            dirty=True,
        )


def test_prediction_column_contract_reports_all_missing_columns_in_preflight():
    from scripts.pfmval_execution_roundtrip import (
        aggregate_preflight_v2,
        preflight_prediction_artifact_columns,
    )

    contract = {
        "schema_version": "prediction_artifact_contract_v1",
        "required_columns": [
            "sample_id",
            "spatial_cluster_id",
            "pathway_id",
            "y_true",
            "y_pred",
        ],
    }
    column_report = preflight_prediction_artifact_columns(
        contract,
        ["pathway_id", "y_pred", "pathway_id"],
    )
    assert column_report["status"] == "blocked"
    assert column_report["missing_columns"] == [
        "sample_id",
        "spatial_cluster_id",
        "y_true",
    ]
    assert column_report["duplicate_columns"] == ["pathway_id"]

    report = aggregate_preflight_v2(
        bundle_manifest={"bundle_id": "b" * 64},
        fingerprint=_fixture_probe(),
        observations={
            "bundle": {"valid": True},
            "source": {
                "head_matches": True,
                "clean": True,
                "object_available": True,
                "entrypoint_sha_matches": True,
            },
            "bindings": {
                "approval_matches": True,
                "contract_matches": True,
                "job_matches": True,
                "data_manifest_matches": True,
                "split_matches": True,
            },
            "attempt": {
                "root_absent": True,
                "lease_available": True,
                "budget_available": True,
            },
            "runtime": {
                "schema_backend_available": True,
                "gpu_sufficient": True,
                "disk_sufficient": True,
            },
            "publish": {
                "fast_forward": True,
                "remote_identity_matches": True,
            },
            "prediction_contract": column_report,
        },
    )
    failure = next(item for item in report["failures"] if item["check_id"] == "prediction_columns")
    assert failure["missing_columns"] == column_report["missing_columns"]
