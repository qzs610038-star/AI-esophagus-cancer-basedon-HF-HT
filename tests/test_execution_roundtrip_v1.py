from __future__ import annotations

import hashlib
import json
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
