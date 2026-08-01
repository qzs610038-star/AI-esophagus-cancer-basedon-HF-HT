import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.pfmval_result_bundle import (
    build_result_bundle_v1,
    publish_result_bundle_v1,
    record_result_bundle_import_v1,
    validate_result_bundle_v1,
)
from scripts.pfmval_governance import validate_result_v2
from scripts.pfmval_state import validate_against_schema_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _historical_source_bundle(
    root: Path,
    *,
    attempt_id: str,
    result_id: str,
    include_large_artifact_id: bool,
    enforce_prediction_return: bool = False,
    include_prediction_artifact: bool = False,
) -> Path:
    bundle = root / result_id
    artifacts_dir = bundle / "artifacts"
    artifacts_dir.mkdir(parents=True)
    job = {
        "schema_version": "2.0",
        "job_id": f"W001-{attempt_id}",
        "experiment_id": "mpp2_huber_loss_paired_v001_20260728",
        "workspace_id": "W001",
        "attempt_id": attempt_id,
        "protocol_revision": 2,
        "approval_id": "APR-mpp2-huber-loss-paired-v001-20260728-r002",
        "critical_contract_sha256": "d" * 64,
        "source_commit": "a" * 40,
        "phase": "formal",
        "run_units": 1,
    }
    if enforce_prediction_return:
        job["command_id"] = "standard_training"
        job["artifact_policy"] = {
            "critical": "sha256_required",
            "supporting": "size_inventory_default",
            "diagnostic": "non_evidence",
            "diagnostic_logs": "include_only_when_needed_for_agent_analysis",
            "large_artifacts": "registered_path_size_sha256_only",
            "raw_predictions": {
                "requirement": "required_inline_for_every_evaluated_split",
                "artifact_kind": "raw_prediction_table",
                "large_artifact_exemption": False,
                "required_columns": [
                    "sample_id",
                    "spatial_cluster_id",
                    "pathway_id",
                    "y_true",
                    "y_pred",
                ],
            },
        }
    terminal = {
        "event_type": "EXPERIMENT_TERMINAL",
        "job_id": job["job_id"],
        "attempt_id": attempt_id,
        "status": "completed",
        "returncode": 0,
    }
    metrics = {"loss": {"type": "mse"}, "external_xzy": {"pcc": 0.6549}}
    summary_lf = b"best epoch: 15\nselection: internal validation only\n"
    summary_raw_crlf = summary_lf.replace(b"\n", b"\r\n")
    payloads = {
        "artifacts/job.json": (json.dumps(job, indent=2) + "\n").encode(),
        "artifacts/attempt_terminal.json": (
            json.dumps(terminal, indent=2) + "\n"
        ).encode(),
        "artifacts/metrics.json": (json.dumps(metrics, indent=2) + "\n").encode(),
        "artifacts/training_summary.txt": summary_lf,
    }
    if include_prediction_artifact:
        payloads["artifacts/predictions_internal_val.csv"] = (
            b"sample_id,spatial_cluster_id,pathway_id,y_true,y_pred\n"
            b"sample-1,cluster-1,pathway-1,0.25,0.30\n"
        )
    for relative, payload in payloads.items():
        _write(bundle / relative, payload)
    artifacts = [
        {
            "artifact_id": "job_manifest",
            "path": "artifacts/job.json",
            "kind": "job_manifest",
            "evidence_role": "critical",
            "size_bytes": len(payloads["artifacts/job.json"]),
            "sha256": _sha256(payloads["artifacts/job.json"]),
            "source_attempt_id": attempt_id,
        },
        {
            "artifact_id": "attempt_terminal",
            "path": "artifacts/attempt_terminal.json",
            "kind": "attempt_event",
            "evidence_role": "critical",
            "size_bytes": len(payloads["artifacts/attempt_terminal.json"]),
            "sha256": _sha256(payloads["artifacts/attempt_terminal.json"]),
            "source_attempt_id": attempt_id,
        },
        {
            "artifact_id": "metrics",
            "path": "artifacts/metrics.json",
            "kind": "metrics",
            "evidence_role": "critical",
            "size_bytes": len(payloads["artifacts/metrics.json"]),
            "sha256": _sha256(payloads["artifacts/metrics.json"]),
            "source_attempt_id": attempt_id,
        },
        {
            "artifact_id": "selection_proof",
            "path": "artifacts/training_summary.txt",
            "kind": "selection_proof",
            "evidence_role": "critical",
            "size_bytes": len(summary_raw_crlf),
            "sha256": _sha256(summary_raw_crlf),
            "source_attempt_id": attempt_id,
        },
    ]
    if include_prediction_artifact:
        prediction_path = "artifacts/predictions_internal_val.csv"
        artifacts.append(
            {
                "artifact_id": "predictions_internal_validation",
                "path": prediction_path,
                "kind": "raw_prediction_table",
                "evaluation_split": "internal_validation",
                "evidence_role": "critical",
                "size_bytes": len(payloads[prediction_path]),
                "sha256": _sha256(payloads[prediction_path]),
                "source_attempt_id": attempt_id,
            }
        )
    large = {
        "server_path": (
            f"D:\\AIPatho\\qzs\\pfmval_experiment_runs\\W001\\{attempt_id}"
            "\\best_checkpoint.pth"
        ),
        "size_bytes": 6420688,
        "sha256": "c" * 64,
        "retention": "retain_on_server_until_explicit_user_disposition",
    }
    if include_large_artifact_id:
        large["artifact_id"] = "best_checkpoint"
    result = {
        **job,
        "result_id": result_id,
        "status": "success",
        "created_at": "2026-07-29T05:23:40+00:00",
        "artifacts": artifacts,
        "metrics": metrics,
        "metric_artifact_ids": ["metrics"],
        "large_artifacts": [large],
    }
    result.pop("command_id", None)
    result.pop("artifact_policy", None)
    if enforce_prediction_return and include_prediction_artifact:
        result["prediction_return"] = {
            "applicability": "required",
            "evaluated_splits": [
                {
                    "split_id": "internal_validation",
                    "artifact_id": "predictions_internal_validation",
                }
            ],
        }
    # Historical packages contained these redundant packaging sidecars.
    _write(bundle / "result.json", (json.dumps(result, indent=2) + "\n").encode())
    _write(bundle / "artifacts.json", (json.dumps(artifacts, indent=2) + "\n").encode())
    _write(bundle / "large_artifacts.json", (json.dumps([large], indent=2) + "\n").encode())
    _write(bundle / "metrics.json", (json.dumps(metrics, indent=2) + "\n").encode())
    return bundle


@pytest.mark.parametrize(
    ("attempt_id", "old_result_id", "new_result_id", "has_large_id"),
    [
        ("A003", "W001-A003-result-R003", "W001-A003-result-R004", False),
        ("A004", "W001-A004-result-R001", "W001-A004-result-R002", True),
    ],
)
def test_historical_g12_only_packaging_revision_passes_full_validator(
    tmp_path,
    attempt_id,
    old_result_id,
    new_result_id,
    has_large_id,
):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id=attempt_id,
        result_id=old_result_id,
        include_large_artifact_id=has_large_id,
    )
    staging = tmp_path / "staging" / new_result_id

    report = build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        staging,
        result_id=new_result_id,
        artifact_retention="retain_in_immutable_result_bundle",
        created_at="2026-07-29T12:00:00+00:00",
    )

    validated = validate_result_bundle_v1(PROJECT_ROOT, staging)
    result = json.loads((staging / "result.json").read_text(encoding="utf-8"))
    assert report["bundle_sha256"] == validated["bundle_sha256"]
    assert result["result_id"] == new_result_id
    assert result["status"] == "success"


def test_strict_schema_validation_uses_powershell_when_jsonschema_is_unavailable(
    tmp_path,
    monkeypatch,
):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A004",
        result_id="W001-A004-result-R001",
        include_large_artifact_id=True,
    )
    staging = tmp_path / "staging"
    build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        staging,
        result_id="W001-A004-result-R002",
        artifact_retention="retain_in_immutable_result_bundle",
        created_at="2026-07-29T12:00:00+00:00",
    )
    result = json.loads((staging / "result.json").read_text(encoding="utf-8"))
    schema = PROJECT_ROOT / "project_state" / "schemas" / "result_envelope_v2.schema.json"
    monkeypatch.setenv("PFMVAL_FORCE_POWERSHELL_SCHEMA", "1")

    backend = validate_against_schema_strict(result, schema, "result envelope v2")
    assert backend == "powershell-test-json"

    invalid = copy.deepcopy(result)
    invalid.pop("approval_id")
    with pytest.raises(ValueError, match="schema violation"):
        validate_against_schema_strict(invalid, schema, "result envelope v2")
    assert isinstance(result["artifacts"], list)
    assert isinstance(result["large_artifacts"], list)
    assert all(item["artifact_id"] for item in result["artifacts"])
    assert all(item["retention"] for item in result["artifacts"])
    assert all(item["artifact_id"] for item in result["large_artifacts"])
    assert not (staging / "metrics.json").exists()
    integrity = json.loads(
        (staging / "artifacts" / "bundle_integrity.json").read_text(
            encoding="utf-8"
        )
    )
    summary = next(
        item
        for item in integrity["artifacts"]
        if item["path"] == "artifacts/training_summary.txt"
    )
    assert summary["source_match"] == "git_normalized_from_crlf"
    assert summary["sha256_raw"] != summary["sha256_git_normalized"]


def test_build_requires_empty_staging_and_array_protocol_types(tmp_path):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A003",
        result_id="W001-A003-result-R003",
        include_large_artifact_id=False,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "existing.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="staging directory must be empty"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            staging,
            result_id="W001-A003-result-R004",
            artifact_retention="retain",
        )


def test_policy_aware_training_result_requires_inline_raw_predictions(tmp_path):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A005",
        result_id="W001-A005-result-R001",
        include_large_artifact_id=True,
        enforce_prediction_return=True,
        include_prediction_artifact=False,
    )

    with pytest.raises(ValueError, match="raw prediction return declaration"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            tmp_path / "missing-predictions",
            result_id="W001-A005-result-R002",
            artifact_retention="retain",
        )


def test_policy_aware_training_result_accepts_every_declared_prediction_split(tmp_path):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A006",
        result_id="W001-A006-result-R001",
        include_large_artifact_id=True,
        enforce_prediction_return=True,
        include_prediction_artifact=True,
    )

    report = build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        tmp_path / "complete-predictions",
        result_id="W001-A006-result-R002",
        artifact_retention="retain",
    )

    assert report["status"] == "success"


def test_raw_predictions_cannot_use_large_artifact_exemption(tmp_path):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A007",
        result_id="W001-A007-result-R001",
        include_large_artifact_id=True,
        enforce_prediction_return=True,
        include_prediction_artifact=True,
    )
    result_path = source / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["large_artifacts"].append(
        {
            "artifact_id": "predictions_external_xzy",
            "kind": "raw_prediction_table",
            "server_path": "D:\\fixture\\predictions_external_xzy.csv",
            "size_bytes": 10_000_000,
            "sha256": "e" * 64,
            "retention": "retain_on_server",
        }
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (source / "large_artifacts.json").write_text(
        json.dumps(result["large_artifacts"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be registered as a large artifact"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            tmp_path / "invalid-large-predictions",
            result_id="W001-A007-result-R002",
            artifact_retention="retain",
        )

    result_path = source / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["large_artifacts"] = result["large_artifacts"][0]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="large_artifacts must be an array"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            tmp_path / "empty-staging",
            result_id="W001-A003-result-R004",
            artifact_retention="retain",
        )


def test_build_rejects_duplicate_paths_unregistered_sidecars_and_bad_terminal(
    tmp_path,
):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A004",
        result_id="W001-A004-result-R001",
        include_large_artifact_id=True,
    )
    result_path = source / "result.json"
    original = json.loads(result_path.read_text(encoding="utf-8"))

    duplicate = copy.deepcopy(original)
    duplicate["artifacts"][1]["path"] = "artifacts/job.json"
    result_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            tmp_path / "duplicate-staging",
            result_id="W001-A004-result-R002",
            artifact_retention="retain",
        )

    result_path.write_text(json.dumps(original), encoding="utf-8")
    (source / "unregistered.txt").write_text("sidecar", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered sidecar"):
        build_result_bundle_v1(
            PROJECT_ROOT,
            source,
            tmp_path / "sidecar-staging",
            result_id="W001-A004-result-R002",
            artifact_retention="retain",
        )
    (source / "unregistered.txt").unlink()

    terminal_path = source / "artifacts" / "attempt_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["returncode"] = 9
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    original["artifacts"][1]["size_bytes"] = terminal_path.stat().st_size
    original["artifacts"][1]["sha256"] = _sha256(terminal_path.read_bytes())
    result_path.write_text(json.dumps(original), encoding="utf-8")
    report = build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        tmp_path / "failed-staging",
        result_id="W001-A004-result-R002",
        artifact_retention="retain",
    )
    assert report["status"] == "failed"


def test_validator_rejects_tampering_and_full_schema_violations(tmp_path):
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A004",
        result_id="W001-A004-result-R001",
        include_large_artifact_id=True,
    )
    staging = tmp_path / "staging"
    build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        staging,
        result_id="W001-A004-result-R002",
        artifact_retention="retain",
    )

    result_path = staging / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["artifacts"][0]["retention"]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="result envelope v2 schema"):
        validate_result_bundle_v1(PROJECT_ROOT, staging)
    with pytest.raises(ValueError, match="result envelope v2 schema"):
        validate_result_v2(result)


def test_record_import_uses_complete_validator_and_is_side_effect_safe(tmp_path):
    project = tmp_path / "project"
    schema_dir = project / "project_state" / "schemas"
    schema_dir.mkdir(parents=True)
    schema_dir.joinpath("result_envelope_v2.schema.json").write_bytes(
        PROJECT_ROOT.joinpath(
            "project_state",
            "schemas",
            "result_envelope_v2.schema.json",
        ).read_bytes()
    )
    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A004",
        result_id="W001-A004-result-R001",
        include_large_artifact_id=True,
    )
    bundle = tmp_path / "bundle"
    report = build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        bundle,
        result_id="W001-A004-result-R002",
        artifact_retention="retain",
    )
    event_path = project / "project_state" / "result_import_events.jsonl"

    with pytest.raises(ValueError, match="bundle_sha256 does not match"):
        record_result_bundle_import_v1(
            project,
            bundle,
            bundle_sha256="f" * 64,
        )
    assert not event_path.exists()

    metrics_path = bundle / "artifacts" / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics_path.write_bytes(original_metrics + b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        record_result_bundle_import_v1(
            project,
            bundle,
            bundle_sha256=report["bundle_sha256"],
        )
    assert not event_path.exists()
    metrics_path.write_bytes(original_metrics)

    first = record_result_bundle_import_v1(
        project,
        bundle,
        bundle_sha256=report["bundle_sha256"],
    )
    second = record_result_bundle_import_v1(
        project,
        bundle,
        bundle_sha256=report["bundle_sha256"],
    )
    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 1


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_publish_is_fast_forward_idempotent_and_verifies_remote_sha(tmp_path):
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "PFMval Test")
    _git(repo, "config", "user.email", "pfmval@example.invalid")
    (repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "baseline.txt")
    _git(repo, "commit", "-m", "baseline")
    parent = _git(repo, "rev-parse", "HEAD")
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "gitee", str(remote))
    ref = "automation/server/W001/A003"
    _git(repo, "push", "gitee", f"{parent}:refs/heads/{ref}")

    source = _historical_source_bundle(
        tmp_path / "source",
        attempt_id="A003",
        result_id="W001-A003-result-R003",
        include_large_artifact_id=False,
    )
    bundle = tmp_path / "bundle"
    build_result_bundle_v1(
        PROJECT_ROOT,
        source,
        bundle,
        result_id="W001-A003-result-R004",
        artifact_retention="retain",
        created_at="2026-07-29T12:00:00+00:00",
    )

    first = publish_result_bundle_v1(
        repo,
        bundle,
        remote_name="gitee",
        ref=ref,
        revision_path="automation/returns/W001/A003/R004",
        expected_parent=parent,
        commit_message="G12-R: return W001 A003 packaging-only R004",
    )
    second = publish_result_bundle_v1(
        repo,
        bundle,
        remote_name="gitee",
        ref=ref,
        revision_path="automation/returns/W001/A003/R004",
        expected_parent=first["commit_sha"],
        commit_message="G12-R: return W001 A003 packaging-only R004",
    )
    remote_sha = _git(
        repo,
        "ls-remote",
        "--heads",
        "gitee",
        f"refs/heads/{ref}",
    ).split()[0]
    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert first["commit_sha"] == second["commit_sha"] == remote_sha
    assert first["parent_sha"] == parent

    with pytest.raises(ValueError, match="remote SHA mismatch"):
        publish_result_bundle_v1(
            repo,
            bundle,
            remote_name="gitee",
            ref=ref,
            revision_path="automation/returns/W001/A003/R005",
            expected_parent="f" * 40,
            commit_message="must not publish",
        )


def test_g12r_operation_card_uses_zero_install_schema_backend_and_fresh_root():
    card = (
        PROJECT_ROOT
        / "project_state"
        / "governance"
        / "G12_R_W001_result_bundle_server_operation_card_20260729.md"
    ).read_text(encoding="utf-8")

    assert "git -C $Repo cat-file -e \"${ImplSha}^{commit}\"" in card
    assert "git -C $Repo merge-base --is-ancestor $ImplSha $ImplTip" in card
    assert "(git -C $Repo rev-parse $ImplRef).Trim() -ne $ImplSha" not in card
    assert "50fb973e4ce3092a408398c94310214a87ecdba0" in card
    assert "Get-Command Test-Json -ErrorAction SilentlyContinue" in card
    assert "PowerShell 7 Test-Json is unavailable" in card
    assert "$PythonCandidates = @(" in card
    assert "'C:\\Program Files\\Python313\\python.exe'" in card
    assert "'C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe'" in card
    assert "'D:\\miniconda\\python.exe'" in card
    assert "foreach ($Candidate in $PythonCandidates)" in card
    assert (
        "& $Candidate -c 'import sys; assert sys.version_info >= (3, 9); "
        "print(sys.executable)'"
    ) in card
    assert "$Python = $Candidate" in card
    assert "import jsonschema" not in card
    assert "no compatible server Python candidate found" in card
    assert "python (Join-Path $Impl" not in card
    assert "$RunId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'" in card
    assert "$Root = Join-Path $RootBase $RunId" in card
    assert "*>&1 | Tee-Object -FilePath $Build3Log" in card
