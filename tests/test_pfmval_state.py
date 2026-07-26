import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import finalize_experiment as finalize_module
from scripts.finalize_experiment import build_dashboard
from scripts.pfmval_governance import build_job_v2, build_result_v2
from scripts.pfmval_state import (
    MPP_TRAINING_ALLOWED_PARAMETERS,
    append_directive,
    activate_mpp_repair_evidence,
    build_mpp_path_index,
    build_result_envelope,
    create_job_manifest,
    compute_source_hashes,
    create_diagnostic_request,
    create_exploration_session,
    directive_lifecycle_candidates,
    evaluate_mpp2_baseline_guard,
    fold_directives,
    exploration_cleanup_candidates,
    git_commit_exists,
    import_result_bundle,
    import_mpp_repair_evidence_from_git,
    list_exploration_sessions,
    normalize_rel,
    promote_exploration_script,
    read_json,
    read_directive_events,
    record_diagnostic_outputs,
    recover_incomplete_result_transactions,
    safe_job_parameters,
    scan_documents,
    sha256_file,
    sync_state,
    transition_directive,
    validate_diagnostic_state,
    validate_state,
    validate_result_envelope,
    validate_job_manifest,
    validate_job_semantics,
    validate_mpp_sequence_gate,
    update_mpp2_baseline_guard_from_result,
    ValidationReport,
    validate_server_paths,
    verify_mpp_repair_server_assets,
)
from path_registry import get_registered_path
from config_utils import load_config


def test_normalize_rel_preserves_hidden_directory_prefixes():
    assert normalize_rel(Path(".claude") / "next-steps.md") == ".claude/next-steps.md"
    assert normalize_rel(Path(".qoder") / "experience.md") == ".qoder/experience.md"


def test_git_commit_exists_uses_real_git_object_database(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "pfmval-test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "PFMval Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert git_commit_exists(repo, commit) is True
    assert git_commit_exists(repo, "0" * 40) is False


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_minimal_project(root: Path) -> Path:
    (root / "experiments").mkdir(parents=True)
    (root / "project_state" / "plans").mkdir(parents=True)
    (root / "project_state" / "schemas").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "mpp_standard_splits").mkdir(parents=True)
    (root / ".claude").mkdir(parents=True)
    (root / "project_state" / "plans" / "mpp_training.md").write_text("# active\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (root / "README.md").write_text("# test repo\n", encoding="utf-8")
    write_json(root / "project_state" / "schemas" / "server_job.schema.json", {"type": "object"})
    write_json(root / "project_state" / "schemas" / "mpp_repair_registry.schema.json", {"type": "object"})
    (root / "configs" / "server_paths.yaml").write_text(
        "schema_version: '1.0'\npaths:\n  mpp_standard_splits:\n    path: mpp_standard_splits\n    kind: repo_directory\n    status: active\n    required_on: both\n",
        encoding="utf-8",
    )
    write_json(root / "mpp_standard_splits" / "path_index.json", {"schema_version": "1.0", "labels_validated": True})
    directive = {
        "event_type": "directive",
        "directive_id": "DIR-20260711-001",
        "issued_at": "2026-07-11T00:00:00+08:00",
        "summary": "Use MPP2",
        "scope": "mpp_training",
        "topic": "active_route",
        "status": "active",
        "supersedes": [],
        "effective_from_revision": 1,
        "affected_files": ["project_state/plans/mpp_training.md"],
        "source": "explicit_user_instruction",
    }
    (root / "project_state" / "directives.jsonl").write_text(json.dumps(directive) + "\n", encoding="utf-8")
    registry = {
        "version": 1,
        "updated_at": "2026-07-11T00:00:00+00:00",
        "current_mpp_policy": {"selected_mpp": 2, "next_recommended_experiment": "smoke"},
        "experiments": [{
            "id": "exp1",
            "family": "mpp",
            "status": "planned",
            "script": "train.py",
            "evidence_status": "pending",
            "provenance_complete": None,
            "result_id": None,
        }],
    }
    write_json(root / "experiments" / "experiment_registry.json", registry)
    (root / "experiments" / "experiment_dashboard.md").write_text(build_dashboard(registry), encoding="utf-8")
    state = {
        "schema_version": "1.0",
        "state_revision": 1,
        "updated_at": "2026-07-11T00:00:00+00:00",
        "source_commit": "unknown",
        "active_directive_ids": ["DIR-20260711-001"],
        "active_plans": {"mpp_training": {"doc_id": "plan-mpp-training", "path": "project_state/plans/mpp_training.md", "status": "active", "approved_at": "2026-07-11T00:00:00+08:00"}},
        "server_transport": {"mode": "gitee_only", "remote_name": "gitee", "allowed_operations": ["git_fetch"], "forbidden_direct_connections": ["ssh", "scp", "http_remote_command", "remote_tunnel"]},
        "active_training_jobs": [],
        "pending_result_ids": [],
        "latest_accepted_result_ids": [],
        "blocked_actions": [],
        "superseded_conclusions": [],
        "source_hashes": {},
    }
    write_json(root / "project_state" / "current_state.json", state)
    docs = {
        "schema_version": "1.0",
        "updated_at": "2026-07-11T00:00:00+00:00",
        "state_revision": 1,
        "documents": [{
            "doc_id": "plan-mpp-training",
            "path": "project_state/plans/mpp_training.md",
            "category": "状态方案",
            "scope": "mpp_training",
            "authority": "normative",
            "lifecycle": "active",
            "verified_at": "2026-07-11T00:00:00+00:00",
            "state_revision": 1,
            "content_sha256": sha256_file(root / "project_state" / "plans" / "mpp_training.md"),
            "supersedes": [],
            "superseded_by": [],
            "truth_sources": ["project_state/current_state.json"],
            "connectivity_modes": [],
        }],
    }
    write_json(root / "project_state" / "document_registry.json", docs)
    sync_state(root, force_revision=True)
    return root


def bind_job(root: Path, result_manifest: dict) -> Path:
    phase = result_manifest["phase"]
    job_id = result_manifest["job_id"]
    source_commit = result_manifest["source_commit"]
    approval = None
    if phase == "formal":
        approval = {
            "approved": True,
            "source": "explicit_user_instruction",
            "job_id": job_id,
            "source_commit": source_commit,
        }
    job = {
        "schema_version": "1.0",
        "job_id": job_id,
        "experiment_id": result_manifest["experiment_id"],
        "source_commit": source_commit,
        "state_revision": read_json(root / "project_state" / "current_state.json")["state_revision"],
        "phase": phase,
        "command_id": "state_preflight" if phase == "preflight" else "standard_training",
        "path_ids": [],
        "parameters": {} if phase == "preflight" else {"num_epochs": 2 if phase == "smoke" else 10},
        "data_manifest_id": result_manifest.get("data_manifest_id"),
        "path_index_version": result_manifest.get("path_index_version"),
        "created_at": "2026-07-11T00:00:00+00:00",
        "dispatch_branch": f"automation/local/{job_id}",
        "result_branch": f"automation/server/{job_id}",
        "formal_training_approval": approval,
        "artifact_policy": {
            "max_file_bytes": 20 * 1024 * 1024,
            "max_total_bytes": 50 * 1024 * 1024,
            "large_artifacts": "server_path_size_sha256_only",
        },
    }
    output = root / "automation" / "jobs" / job_id / "job.json"
    write_json(output, job)
    return output


def test_directive_supersession_is_explicit_and_append_only(tmp_path):
    root = make_minimal_project(tmp_path)
    new_id = append_directive(
        root,
        summary="Use a new route",
        scope="mpp_training",
        topic="active_route",
        supersedes=["DIR-20260711-001"],
        affected_files=["project_state/plans/mpp_training.md"],
    )
    events = read_directive_events(root)
    folded = fold_directives(events)
    assert folded["DIR-20260711-001"]["status"] == "superseded"
    assert folded[new_id]["status"] == "active"
    assert [event["event_type"] for event in events[-2:]] == ["status_update", "directive"]


def test_state_validation_rejects_unresolved_same_topic_directives(tmp_path):
    root = make_minimal_project(tmp_path)
    append_directive(
        root,
        summary="Conflicting route",
        scope="mpp_training",
        topic="active_route",
        supersedes=[],
        affected_files=[],
    )
    report = validate_state(root)
    assert any("directive conflicts" in message for message in report.fail_items)


def test_diagnostic_gate_allows_document_freshness_drift_but_keeps_transport_hard(tmp_path):
    root = make_minimal_project(tmp_path)
    # The production diagnostic profile validates the same three state schemas.
    # This focused fixture only needs permissive schema shells; schema semantics
    # are covered by the production schema tests.
    for schema_name in ("current_state.schema.json", "document_registry.schema.json", "directives.schema.json"):
        write_json(root / "project_state" / "schemas" / schema_name, {"type": "object"})
    # Full validation treats this as a normative-document failure; a read-only
    # diagnostic must surface it as a warning instead of blocking triage.
    (root / "project_state" / "plans" / "mpp_training.md").write_text("# changed\n", encoding="utf-8")
    full_report = validate_state(root)
    assert any("normative document hash is stale" in message for message in full_report.fail_items)

    diagnostic_report = validate_diagnostic_state(root)
    assert diagnostic_report.ok

    state_path = root / "project_state" / "current_state.json"
    state = read_json(state_path)
    state["server_transport"]["mode"] = "ssh"
    write_json(state_path, state)
    blocked_report = validate_diagnostic_state(root)
    assert not blocked_report.ok
    assert "server transport is not gitee_only" in blocked_report.fail_items


def test_diagnostic_request_is_allowlisted_non_evidence_audit(tmp_path):
    root = tmp_path / "diagnostic-repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "pfmval-test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "PFMval Test"], cwd=root, check=True)
    make_minimal_project(root)
    for schema_name in ("current_state.schema.json", "document_registry.schema.json", "directives.schema.json"):
        write_json(root / "project_state" / "schemas" / schema_name, {"type": "object"})
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "diagnostic fixture"], cwd=root, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()

    request_path = create_diagnostic_request(
        root,
        diagnostic_id="diagnostic-20260714-cache-probe",
        source_commit=commit,
        command_id="cache_probe",
        source_branch="main",
        return_branch="automation/diagnostics",
    )
    request = read_json(request_path)
    assert request["command_id"] == "cache_probe"
    assert request["execution_contract"]["arbitrary_shell"] is False
    assert request["execution_contract"]["training"] is False
    returned_output = request_path.parent / "stdout.txt"
    returned_output.write_text("cache readable\n", encoding="utf-8")
    completion = record_diagnostic_outputs(
        root,
        diagnostic_id="diagnostic-20260714-cache-probe",
        outputs=["automation/diagnostics/diagnostic-20260714-cache-probe/stdout.txt"],
    )
    assert completion["server_write"] is False
    assert completion["outputs"][0]["sha256"] == sha256_file(returned_output)
    events = (root / "project_state" / "diagnostics.jsonl").read_text(encoding="utf-8")
    assert "diagnostic_request" in events
    assert "diagnostic_completed" in events
    with pytest.raises(ValueError, match="not allowlisted"):
        create_diagnostic_request(
            root,
            diagnostic_id="diagnostic-20260714-bad-command",
            source_commit=commit,
            command_id="arbitrary_shell",
            source_branch="main",
            return_branch="automation/diagnostics",
        )


def test_explore_session_is_local_only_and_candidates_never_delete(tmp_path):
    root = make_minimal_project(tmp_path)
    manifest_path = create_exploration_session(
        root, session_id="explore-20260714-cache-layout", purpose="inspect local cache layout",
    )
    manifest = read_json(manifest_path)
    assert manifest["restrictions"]["server_execution"] is False
    assert manifest["restrictions"]["training_data"] is False
    assert list_exploration_sessions(root)[0]["status"] == "active"
    assert exploration_cleanup_candidates(root, older_than_days=1) == []
    assert manifest_path.exists()


def test_explore_promotion_requires_marker_and_active_directive(tmp_path):
    root = make_minimal_project(tmp_path)
    source = root / "scripts" / "explorations" / "probe_explore.py"
    source.parent.mkdir(parents=True)
    source.write_text("# PFMVAL_EXPLORE\nprint('probe')\n", encoding="utf-8")
    target = promote_exploration_script(
        root,
        source="scripts/explorations/probe_explore.py",
        target="scripts/probe.py",
        directive_id="DIR-20260711-001",
    )
    assert target.read_text(encoding="utf-8") == "print('probe')\n"
    assert source.exists()


def test_directive_completion_is_append_only_and_lifecycle_is_review_only(tmp_path):
    root = make_minimal_project(tmp_path)
    event = transition_directive(
        root,
        directive_id="DIR-20260711-001",
        status="completed",
        reason="the route close-out was recorded",
        completion_evidence=["experiments/decision_log.md"],
    )
    assert event["status"] == "completed"
    assert fold_directives(read_directive_events(root))["DIR-20260711-001"]["status"] == "completed"
    assert directive_lifecycle_candidates(root, older_than_days=1) == []
    with pytest.raises(ValueError, match="only active"):
        transition_directive(
            root,
            directive_id="DIR-20260711-001",
            status="cancelled",
            reason="must not rewrite history",
        )


def test_directive_lifecycle_candidate_does_not_change_active_status(tmp_path):
    root = make_minimal_project(tmp_path)
    candidates = directive_lifecycle_candidates(root, older_than_days=1)
    assert candidates[0]["directive_id"] == "DIR-20260711-001"
    assert candidates[0]["action"] == "review_only_no_automatic_transition"
    assert fold_directives(read_directive_events(root))["DIR-20260711-001"]["status"] == "active"


def test_document_scan_preserves_hidden_paths_and_marks_only_missing_qoder(tmp_path):
    root = make_minimal_project(tmp_path)
    (root / "project_state" / "document_registry.json").unlink()
    (root / ".claude" / "next-steps.md").write_text("# generated\n", encoding="utf-8")
    write_json(root / ".claude" / "doc-registry.json", {
        "documents": [
            {"path": ".claude/next-steps.md", "category": "local", "purpose": "", "created": "", "tags": []},
            {"path": ".qoder/experience.md", "category": "local", "purpose": "", "created": "", "tags": []},
        ]
    })
    registry = scan_documents(root)
    entries = {item["path"]: item for item in registry["documents"]}
    assert entries[".claude/next-steps.md"]["lifecycle"] == "active"
    assert entries[".qoder/experience.md"]["lifecycle"] == "missing"
    assert "claude/next-steps.md" not in entries


def test_document_scan_is_byte_stable_when_inputs_are_unchanged(tmp_path, monkeypatch):
    root = make_minimal_project(tmp_path)
    registry_path = root / "project_state" / "document_registry.json"
    timestamps = iter([
        "2026-07-27T00:00:00+00:00",
        "2026-07-27T00:01:00+00:00",
    ])
    monkeypatch.setattr("scripts.pfmval_state.utc_now", lambda: next(timestamps))
    first = scan_documents(root)
    write_json(registry_path, first)

    second = scan_documents(root)

    assert second == first


def test_document_scan_preserves_manually_curated_lineage(tmp_path):
    root = make_minimal_project(tmp_path)
    registry_path = root / "project_state" / "document_registry.json"
    registry = scan_documents(root)
    entry = next(item for item in registry["documents"] if item["path"] == "AGENTS.md")
    entry["supersedes"] = ["doc-old-agents"]
    entry["superseded_by"] = ["doc-next-agents"]
    entry["truth_sources"] = ["project_state/directives.jsonl"]
    write_json(registry_path, registry)

    rescanned = scan_documents(root)
    rescanned_entry = next(
        item for item in rescanned["documents"] if item["path"] == "AGENTS.md"
    )

    assert rescanned_entry["supersedes"] == ["doc-old-agents"]
    assert rescanned_entry["superseded_by"] == ["doc-next-agents"]
    assert rescanned_entry["truth_sources"] == ["project_state/directives.jsonl"]


def test_r0_baseline_manifest_freezes_v1_compatibility_samples():
    root = Path(__file__).resolve().parents[1]
    manifest_path = (
        root
        / "tests"
        / "fixtures"
        / "workflow_governance_v3"
        / "r0_baseline_manifest.json"
    )
    manifest = read_json(manifest_path)
    samples = {item["class"]: item for item in manifest["samples"]}

    assert set(samples) == {
        "preflight",
        "smoke",
        "formal",
        "success",
        "failed",
        "incomplete",
        "legacy_accepted_without_envelope",
    }
    for sample in samples.values():
        source = root / sample["source_path"]
        assert source.is_file()
        assert sha256_file(source) == sample["source_sha256"]
        assert sample["schema_version"] == "1.0"
        assert sample["evidence_boundary"] in {
            "current_compatibility_fixture",
            "accepted_legacy_measurement_incomplete_provenance",
        }

    incomplete = samples["incomplete"]
    assert not (root / incomplete["expected_result_path"]).exists()

    legacy = samples["legacy_accepted_without_envelope"]
    registry = read_json(root / legacy["source_path"])
    experiment = next(
        item
        for item in registry["experiments"]
        if item["id"] == legacy["selector"]["experiment_id"]
    )
    assert experiment["evidence_status"] == "accepted"
    assert experiment["provenance_complete"] is False
    assert experiment["source_commit"] is None
    assert experiment["job_id"] is None
    assert experiment["result_manifest_sha256"] is None


def test_document_scan_registers_canonical_skills_adapters_and_review_lifecycle(tmp_path):
    root = make_minimal_project(tmp_path)
    state_path = root / "project_state" / "current_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["active_skills"] = {
        "train": {
            "canonical_path": ".agents/skills/train/SKILL.md",
            "adapter_paths": [".claude/skills/train/SKILL.md"],
            "status": "compatibility_router",
            "approved_at": "2026-07-26T00:00:00+08:00",
        }
    }
    state["pending_plan_reviews"] = {
        "workspace_protocol": {
            "path": "project_state/plans/workspace_protocol.md",
            "status": "approved_design",
            "approved_at": "2026-07-26T00:00:00+08:00",
        },
        "asset_refactor": {
            "path": "project_state/plans/asset_refactor.md",
            "status": "pending_review",
        },
    }
    write_json(state_path, state)
    canonical = root / ".agents" / "skills" / "train" / "SKILL.md"
    adapter = root / ".claude" / "skills" / "train" / "SKILL.md"
    approved_design = root / "project_state" / "plans" / "workspace_protocol.md"
    pending_review = root / "project_state" / "plans" / "asset_refactor.md"
    for path in (canonical, adapter, approved_design, pending_review):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    registry = scan_documents(root)
    entries = {item["path"]: item for item in registry["documents"]}

    canonical_entry = entries[".agents/skills/train/SKILL.md"]
    assert canonical_entry["scope"] == "skill:train"
    assert canonical_entry["authority"] == "normative"
    assert canonical_entry["lifecycle"] == "active"
    assert canonical_entry["availability"] == "tracked"
    adapter_entry = entries[".claude/skills/train/SKILL.md"]
    assert adapter_entry["scope"] == "skill_adapter:train"
    assert adapter_entry["authority"] == "reference"
    assert adapter_entry["lifecycle"] == "active"
    assert adapter_entry["availability"] == "tracked"
    assert entries["project_state/plans/workspace_protocol.md"]["lifecycle"] == "approved_design"
    assert entries["project_state/plans/asset_refactor.md"]["lifecycle"] == "pending_review"


def test_state_validation_rejects_missing_active_skill_canonical_path(tmp_path):
    root = make_minimal_project(tmp_path)
    state_path = root / "project_state" / "current_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["active_skills"] = {
        "train": {
            "canonical_path": ".agents/skills/train/SKILL.md",
            "adapter_paths": [".claude/skills/train/SKILL.md"],
            "status": "compatibility_router",
            "approved_at": "2026-07-26T00:00:00+08:00",
        }
    }
    write_json(state_path, state)
    adapter = root / ".claude" / "skills" / "train" / "SKILL.md"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# adapter\n", encoding="utf-8")
    write_json(root / "project_state" / "document_registry.json", scan_documents(root))

    report = validate_state(root)

    assert any("active skill train canonical path is absent" in item for item in report.fail_items)


def test_mpp_index_reports_conflicting_duplicate_barcodes(tmp_path):
    split_root = tmp_path / "mpp_standard_splits"
    label = split_root / "group_1" / "labels" / "train" / "P1" / "P1_ssGSEA_zscore.csv"
    label.parent.mkdir(parents=True)
    label.write_text("barcode,pathway\na,1\na,2\nb,3\n", encoding="utf-8")
    for group in (3, 5):
        audit = split_root / f"group_{group}" / "overlap_embargo_audit.csv"
        audit.parent.mkdir(parents=True)
        audit.write_text("neighbor_final_split,leakage_pairs\nembargo,0\n", encoding="utf-8")
    index = build_mpp_path_index(tmp_path)
    asset = next(item for item in index["assets"] if item["role"] == "standardized_label")
    assert asset["row_count"] == 3
    assert asset["unique_barcode_count"] == 2
    assert asset["conflicting_duplicate_count"] == 1
    assert index["labels_validated"] is False


def test_manual_dashboard_edit_is_detected(tmp_path):
    root = make_minimal_project(tmp_path)
    with (root / "experiments" / "experiment_dashboard.md").open("a", encoding="utf-8") as handle:
        handle.write("manual edit\n")
    report = validate_state(root)
    assert any("experiment_dashboard_sha256" in message for message in report.fail_items)


def test_job_parameters_are_argv_safe():
    assert safe_job_parameters({"batch_size": 32, "fold": 1}) == ["--batch_size", "32", "--fold", "1"]
    with pytest.raises(ValueError, match="unsafe job parameter value"):
        safe_job_parameters({"fold": "1;Remove-Item"})


def test_mpp_job_parameter_names_match_argparse_flags_exactly():
    project_root = Path(__file__).resolve().parent.parent
    script = (project_root / "train_mpp_uni2h_mlp.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    declared_flags = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    missing = sorted(key for key in MPP_TRAINING_ALLOWED_PARAMETERS if f"--{key}" not in declared_flags)
    assert missing == []


def test_powershell_launcher_keeps_event_identifiers_for_cleanup():
    project_root = Path(__file__).resolve().parent.parent
    launcher = (project_root / "deploy" / "run_experiment.ps1").read_text(encoding="utf-8")
    assert "-SourceIdentifier $outSourceIdentifier" in launcher
    assert "-SourceIdentifier $errSourceIdentifier" in launcher
    assert "Unregister-Event -SourceIdentifier $outSourceIdentifier" in launcher
    assert "Unregister-Event -SourceIdentifier $errSourceIdentifier" in launcher
    assert "$outEvent = Register-ObjectEvent" not in launcher
    assert "$errEvent = Register-ObjectEvent" not in launcher
    assert "taskkill /F /IM python.exe" not in launcher


def test_powershell_launcher_does_not_inject_num_threads_into_cache_parity():
    project_root = Path(__file__).resolve().parent.parent
    launcher = (project_root / "deploy" / "run_experiment.ps1").read_text(encoding="utf-8")
    assert '$Script -notlike "*check_mpp_online_cache_parity.py"' in launcher
    assert '$fullArgs = "-u `"$scriptPath`" $effectiveArguments"' in launcher
    assert '$fullArgs = "-u `"$scriptPath`" $Arguments --num_threads 8"' not in launcher


def test_path_registry_resolves_relative_path_and_rejects_escape(tmp_path):
    registry_path = tmp_path / "server_paths.yaml"
    registry_path.write_text(
        "schema_version: '1.0'\npaths:\n  good:\n    path: mpp_standard_splits\n  bad:\n    path: ../outside\n",
        encoding="utf-8",
    )
    assert get_registered_path("good", registry_path=registry_path, project_root=tmp_path) == tmp_path / "mpp_standard_splits"
    with pytest.raises(ValueError, match="escapes project root"):
        get_registered_path("bad", registry_path=registry_path, project_root=tmp_path)


def test_server_task_skips_missing_local_only_legacy_path(tmp_path):
    root = make_minimal_project(tmp_path)
    (root / "configs" / "server_paths.yaml").write_text(
        (root / "configs" / "server_paths.yaml").read_text(encoding="utf-8")
        + "\n  legacy_partner_labels_local:\n"
        + "    path: parter_ljk_MPP1&4_patch_split_zscore\n"
        + "    kind: repo_directory\n"
        + "    status: legacy\n"
        + "    required_on: local\n",
        encoding="utf-8",
    )

    server_report = ValidationReport()
    validate_server_paths(root, server_report, task="general", host_scope="server")
    assert not any("legacy_partner_labels_local" in item for item in server_report.fail_items)

    local_report = ValidationReport()
    validate_server_paths(root, local_report, task="general")
    assert any("legacy_partner_labels_local" in item for item in local_report.fail_items)


def test_server_scope_still_requires_missing_both_path(tmp_path):
    root = make_minimal_project(tmp_path)
    (root / "configs" / "server_paths.yaml").write_text(
        (root / "configs" / "server_paths.yaml").read_text(encoding="utf-8")
        + "\n  required_on_both_missing:\n"
        + "    path: server_required_asset\n"
        + "    kind: repo_directory\n"
        + "    status: active\n"
        + "    required_on: both\n",
        encoding="utf-8",
    )

    report = ValidationReport()
    validate_server_paths(root, report, task="training", host_scope="server")
    assert any("required_on_both_missing" in item for item in report.fail_items)


def test_repository_declares_lf_checkout_policy():
    project_root = Path(__file__).resolve().parent.parent
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes


def test_server_config_resolves_machine_paths_from_stable_ids():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "config.server.yaml")
    assert Path(config["paths"]["patch_base"]) == get_registered_path("phase2_patch_root")
    assert Path(config["paths"]["ssgsea_base"]) == get_registered_path("phase2_ssgsea_zscore_root")


def test_formal_job_requires_bound_explicit_approval(tmp_path):
    root = make_minimal_project(tmp_path)
    with pytest.raises(ValueError, match="approval file"):
        create_job_manifest(
            root,
            job_id="job-1",
            experiment_id="exp1",
            phase="formal",
            command_id="standard_training",
            path_ids=["mpp_standard_splits"],
            parameters={"fold": 1, "num_epochs": 10},
        )


def test_result_import_updates_registry_dashboard_and_state(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    artifact = bundle / "training_summary.txt"
    artifact.write_text("ok\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "result_id": "result-1",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "formal",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "data_manifest_id": "fixture-v1",
        "path_index_version": "1.0",
        "formal_training_approved": True,
        "artifacts": [{"path": artifact.name, "kind": "txt", "size_bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}],
        "metrics": {"best_epoch": 2, "best_val_pcc": 0.5, "best_val_loss": 0.2},
    }
    write_json(bundle / "result.json", manifest)
    bind_job(root, manifest)
    result = import_result_bundle(root, bundle)
    registry = json.loads((root / "experiments" / "experiment_registry.json").read_text(encoding="utf-8"))
    state = json.loads((root / "project_state" / "current_state.json").read_text(encoding="utf-8"))
    assert result["status"] == "imported"
    assert registry["experiments"][0]["result_id"] == "result-1"
    assert registry["experiments"][0]["evidence_status"] == "accepted"
    assert "exp1" in state["latest_accepted_result_ids"]
    assert "exp1" in (root / "experiments" / "experiment_dashboard.md").read_text(encoding="utf-8")
    assert "result-1" in (root / "experiments" / "experiment_progress.md").read_text(encoding="utf-8")
    assert not (root / "project_state" / ".transactions").exists()


def test_result_import_dual_reads_v2_and_rejects_same_id_with_different_sha(
    tmp_path,
):
    root = make_minimal_project(tmp_path)
    source_commit = "a" * 40
    contract_sha = "b" * 64
    data_sha = "c" * 64
    registry_path = root / "experiments" / "experiment_registry.json"
    registry = read_json(registry_path)
    registry["experiments"][0].update(
        {
            "workspace_id": "W001",
            "protocol_revision": 1,
            "critical_contract_sha256": contract_sha,
            "source_commit": source_commit,
        }
    )
    write_json(registry_path, registry)
    workspace_registry = {
        "schema_version": "1.0",
        "updated_at": "2026-07-27T00:00:00+00:00",
        "next_workspace_number": 2,
        "workspaces": [
            {
                "workspace_id": "W001",
                "display_name": "fixture",
                "experiment_id": "exp1",
                "protocol_revision": 1,
                "status": "registered",
                "active_attempt_id": None,
                "lease": None,
                "opened_at": "2026-07-27T00:00:00+00:00",
                "last_verified_at": None,
                "close_eligibility": "not_evaluated",
                "retention_policy": "retain_until_explicit_close_approval",
                "hosts": {
                    "local": {
                        "host_scope": "local",
                        "branch": "automation/local/W001/A001",
                        "path_id": "local_experiment_workspaces",
                        "relative_path": "workspaces/W001",
                        "current_source_commit": source_commit,
                    }
                },
            }
        ],
    }
    write_json(
        root / "project_state" / "workspace_registry.json",
        workspace_registry,
    )
    approval = {
        "event_type": "protocol_approval",
        "approval_id": "APR-exp1-r001",
        "experiment_id": "exp1",
        "protocol_revision": 1,
        "phase": "formal",
        "run_limit": 1,
        "critical_contract_sha256": contract_sha,
        "adaptation_policy": "e1_allowlist_only",
        "source": "explicit_user_instruction",
        "status": "active",
        "approved_at": "2026-07-27T00:00:00+00:00",
    }
    (root / "project_state" / "experiment_approvals.jsonl").write_text(
        json.dumps(approval) + "\n",
        encoding="utf-8",
    )
    prepared = {
        "event_type": "ATTEMPT_PREPARED",
        "event_id": "prepare-W001-A001",
        "recorded_at": "2026-07-27T00:00:00+00:00",
        "experiment_id": "exp1",
        "workspace_id": "W001",
        "attempt_id": "A001",
        "job_id": "W001-A001",
        "approval_id": "APR-exp1-r001",
        "protocol_revision": 1,
        "phase": "formal",
        "run_units": 1,
        "source_commit": source_commit,
        "critical_contract_sha256": contract_sha,
        "workspace_branch": "automation/local/W001/A001",
    }
    (root / "project_state" / "attempt_events.jsonl").write_text(
        json.dumps(prepared) + "\n",
        encoding="utf-8",
    )
    job = build_job_v2(
        prepared,
        command_id="standard_training",
        path_ids=["mpp_standard_splits"],
        resolved_argv=["python", "train.py"],
        input_binding={
            "data_manifest_id": "fixture-v2",
            "data_manifest_sha256": data_sha,
            "path_index_version": "2.0",
            "artifacts": [],
        },
    )
    job_path = root / "automation" / "jobs" / job["job_id"] / "job.json"
    write_json(job_path, job)
    project_root = Path(__file__).resolve().parent.parent
    (root / "project_state" / "schemas" / "server_job_v2.schema.json").write_text(
        (
            project_root
            / "project_state"
            / "schemas"
            / "server_job_v2.schema.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    bundle = root / "bundle-v2"
    bundle.mkdir()
    artifact = bundle / "metrics.json"
    artifact.write_text('{"best_epoch":2,"best_val_loss":0.2}\n', encoding="utf-8")
    result_manifest = build_result_v2(
        job,
        result_id="RES-W001-A001",
        status="success",
        artifacts=[
            {
                "artifact_id": "selection-proof",
                "path": artifact.name,
                "kind": "selection_proof",
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
                "evidence_role": "critical",
                "retention": "retain",
            }
        ],
        metrics={"best_epoch": 2, "best_val_loss": 0.2},
        metric_artifact_ids=["selection-proof"],
    )
    write_json(bundle / "result.json", result_manifest)

    imported = import_result_bundle(root, bundle)
    assert imported["status"] == "imported"
    imported_registry = read_json(registry_path)
    experiment = imported_registry["experiments"][0]
    assert experiment["result_id"] == "RES-W001-A001"
    assert experiment["data_manifest_id"] == "fixture-v2"
    assert experiment["path_index_version"] == "2.0"

    same = import_result_bundle(root, bundle)
    assert same["status"] == "already_imported"
    result_manifest["created_at"] = "2026-07-27T00:00:01+00:00"
    write_json(bundle / "result.json", result_manifest)
    with pytest.raises(ValueError, match="different bundle SHA"):
        import_result_bundle(root, bundle)


def test_invalid_result_hash_writes_nothing(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    artifact = bundle / "training_summary.txt"
    artifact.write_text("ok\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "result_id": "result-bad",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "smoke",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "data_manifest_id": "fixture-v1",
        "path_index_version": "1.0",
        "artifacts": [{"path": artifact.name, "kind": "txt", "size_bytes": artifact.stat().st_size, "sha256": "0" * 64}],
        "metrics": {"best_epoch": 1, "best_val_loss": 0.1},
    }
    write_json(bundle / "result.json", manifest)
    before = (root / "experiments" / "experiment_registry.json").read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        import_result_bundle(root, bundle)
    assert (root / "experiments" / "experiment_registry.json").read_bytes() == before


def test_non_finite_result_metric_writes_nothing(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "result-nan",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "smoke",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "data_manifest_id": "fixture-v1",
        "path_index_version": "1.0",
        "artifacts": [],
        "metrics": {"best_epoch": 1, "best_val_loss": float("nan")},
    }
    write_json(bundle / "result.json", manifest)
    before = (root / "experiments" / "experiment_registry.json").read_bytes()
    with pytest.raises(ValueError, match="non-finite metric"):
        import_result_bundle(root, bundle)
    assert (root / "experiments" / "experiment_registry.json").read_bytes() == before


def test_successful_training_result_requires_core_metrics(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "result-empty",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "smoke",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "data_manifest_id": "fixture-v1",
        "path_index_version": "1.0",
        "artifacts": [],
        "metrics": {},
    }
    with pytest.raises(ValueError, match="missing best_epoch"):
        validate_result_envelope(bundle, manifest)


def test_preflight_success_is_not_promoted_to_training_evidence(tmp_path):
    root = make_minimal_project(tmp_path)
    registry_path = root / "experiments" / "experiment_registry.json"
    state_path = root / "project_state" / "current_state.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["experiments"][0].update({
        "status": "done",
        "evidence_status": "accepted",
        "result_id": "accepted-result",
        "source_commit": "def5678",
    })
    write_json(registry_path, registry)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["latest_accepted_result_ids"] = ["exp1"]
    write_json(state_path, state)
    sync_state(root, force_revision=True)
    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "preflight-1",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "preflight",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "artifacts": [],
        "metrics": {},
    }
    write_json(bundle / "result.json", manifest)
    bind_job(root, manifest)
    import_result_bundle(root, bundle)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    experiment = registry["experiments"][0]
    assert experiment["evidence_status"] == "accepted"
    assert experiment["status"] == "done"
    assert experiment["result_id"] == "accepted-result"
    assert experiment["source_commit"] == "def5678"
    assert experiment["last_preflight"]["result_id"] == "preflight-1"
    assert "exp1" in state["latest_accepted_result_ids"]


def test_failed_new_result_removes_experiment_from_latest_accepted(tmp_path):
    root = make_minimal_project(tmp_path)
    registry_path = root / "experiments" / "experiment_registry.json"
    state_path = root / "project_state" / "current_state.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["experiments"][0].update({"status": "done", "evidence_status": "accepted", "result_id": "old-result"})
    write_json(registry_path, registry)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["latest_accepted_result_ids"] = ["exp1"]
    write_json(state_path, state)
    sync_state(root, force_revision=True)

    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "failed-result",
        "job_id": "job-2",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "formal",
        "status": "failed",
        "created_at": "2026-07-11T00:00:00+00:00",
        "formal_training_approved": True,
        "artifacts": [],
        "metrics": {},
    }
    write_json(bundle / "result.json", manifest)
    bind_job(root, manifest)
    import_result_bundle(root, bundle)
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    final_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "exp1" not in final_state["latest_accepted_result_ids"]
    assert final_registry["experiments"][0]["evidence_status"] == "rejected"


def test_result_pack_rejects_duplicate_artifact_basenames(tmp_path):
    source_a = tmp_path / "a" / "training_summary.txt"
    source_b = tmp_path / "b" / "training_summary.txt"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_text("a\n", encoding="utf-8")
    source_b.write_text("b\n", encoding="utf-8")
    job = {
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "preflight",
        "data_manifest_id": None,
        "path_index_version": "1.0",
        "formal_training_approval": None,
    }
    with pytest.raises(ValueError, match="duplicate artifact basename"):
        build_result_envelope(
            job=job,
            status="failed",
            output_dir=tmp_path / "output",
            artifact_paths=[source_a, source_b],
            metrics={},
        )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.building-*"))


def test_result_import_requires_dispatched_job_envelope(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "unbound-result",
        "job_id": "job-missing",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "preflight",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "artifacts": [],
        "metrics": {},
    }
    write_json(bundle / "result.json", manifest)
    before = (root / "experiments" / "experiment_registry.json").read_bytes()
    with pytest.raises(ValueError, match="no dispatched job envelope"):
        import_result_bundle(root, bundle)
    assert (root / "experiments" / "experiment_registry.json").read_bytes() == before


def test_result_import_rejects_job_phase_mismatch(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "mismatch-result",
        "job_id": "job-mismatch",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "preflight",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "artifacts": [],
        "metrics": {},
    }
    write_json(bundle / "result.json", manifest)
    job_path = bind_job(root, manifest)
    job = read_json(job_path)
    job["phase"] = "smoke"
    job["command_id"] = "standard_training"
    job["parameters"] = {"num_epochs": 2}
    write_json(job_path, job)
    with pytest.raises(ValueError, match="binding mismatch for phase"):
        import_result_bundle(root, bundle)


def test_result_envelope_rejects_windows_backslash_traversal(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "bad-path",
        "job_id": "job-1",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "preflight",
        "status": "failed",
        "created_at": "2026-07-11T00:00:00+00:00",
        "artifacts": [{"path": r"..\training_summary.txt", "kind": "txt", "size_bytes": 0, "sha256": "0" * 64}],
        "metrics": {},
    }
    with pytest.raises(ValueError, match="does not match|POSIX separators"):
        validate_result_envelope(bundle, manifest)


def test_smoke_job_epoch_budget_is_bounded():
    with pytest.raises(ValueError, match="explicit num_epochs"):
        validate_job_semantics("smoke", "standard_training", {})
    with pytest.raises(ValueError, match="between 1 and 3"):
        validate_job_semantics("smoke", "standard_training", {"num_epochs": 4})


def test_mpp2_repaired_baseline_guard_passes_only_within_all_thresholds():
    guard = {
        "reference_metrics": {
            "external_xzy_pcc": 0.6489,
            "external_xzy_mae_raw": 1209.9316,
            "external_xzy_r2_raw": -0.1554,
        },
        "thresholds": {
            "pcc_abs_drop": 0.05,
            "raw_mae_relative_increase": 0.10,
            "raw_r2_abs_drop": 0.10,
        },
    }

    passed = evaluate_mpp2_baseline_guard(guard, {
        "external_xzy_pcc": 0.61,
        "external_xzy_mae_raw": 1250.0,
        "external_xzy_r2_raw": -0.20,
    })
    assert passed["status"] == "passed_allow_lora"
    assert passed["triggered_metrics"] == []

    triggered = evaluate_mpp2_baseline_guard(guard, {
        "external_xzy_pcc": 0.5989,
        "external_xzy_mae_raw": 1209.9316,
        "external_xzy_r2_raw": -0.1554,
    })
    assert triggered["status"] == "triggered_rerun_mpp1_5"
    assert triggered["triggered_metrics"] == ["external_xzy_pcc"]


def test_mpp2_repaired_baseline_guard_blocks_on_missing_metric():
    guard = {
        "reference_metrics": {
            "external_xzy_pcc": 0.6489,
            "external_xzy_mae_raw": 1209.9316,
            "external_xzy_r2_raw": -0.1554,
        },
        "thresholds": {
            "pcc_abs_drop": 0.05,
            "raw_mae_relative_increase": 0.10,
            "raw_r2_abs_drop": 0.10,
        },
    }
    result = evaluate_mpp2_baseline_guard(guard, {"external_xzy_pcc": 0.64})
    assert result["status"] == "incomplete_block_lora"
    assert set(result["missing_metrics"]) == {"external_xzy_mae_raw", "external_xzy_r2_raw"}


def test_lora_sequence_gate_requires_repaired_mpp2_baseline_pass(tmp_path):
    root = make_minimal_project(tmp_path)
    state_path = root / "project_state" / "current_state.json"
    state = read_json(state_path)
    state["mpp_repair"] = {"baseline_guard": {"status": "pending_repaired_mpp2_baseline"}}
    write_json(state_path, state)
    experiment = {"repair_guard_role": "mpp2_lora_r8_smoke"}

    with pytest.raises(ValueError, match="LoRA is blocked"):
        validate_mpp_sequence_gate(root, experiment)

    state["mpp_repair"]["baseline_guard"]["status"] = "passed_allow_lora"
    write_json(state_path, state)
    validate_mpp_sequence_gate(root, experiment)


def test_repaired_mpp2_result_trigger_adds_rerun_gate():
    state = {
        "blocked_actions": ["mpp2_lora_until_repaired_baseline_passes_guard"],
        "mpp_repair": {"baseline_guard": {
            "status": "pending_repaired_mpp2_baseline",
            "reference_metrics": {
                "external_xzy_pcc": 0.6489,
                "external_xzy_mae_raw": 1209.9316,
                "external_xzy_r2_raw": -0.1554,
            },
            "thresholds": {
                "pcc_abs_drop": 0.05,
                "raw_mae_relative_increase": 0.10,
                "raw_r2_abs_drop": 0.10,
            },
        }},
    }
    update_mpp2_baseline_guard_from_result(
        state,
        {"repair_guard_role": "mpp2_repaired_frozen_baseline"},
        {
            "result_id": "repaired-mpp2",
            "status": "success",
            "phase": "formal",
            "metrics": {
                "external_xzy_pcc": 0.64,
                "external_xzy_mae_raw": 1400.0,
                "external_xzy_r2_raw": -0.16,
            },
        },
    )
    guard = state["mpp_repair"]["baseline_guard"]
    assert guard["status"] == "triggered_rerun_mpp1_5"
    assert guard["triggered_metrics"] == ["external_xzy_mae_raw"]
    assert "rerun_repaired_mpp1_5_before_lora" in state["blocked_actions"]


def test_mpp_job_requires_registered_paths_and_manifest_mode():
    experiment = {"script": "train_mpp_uni2h_mlp.py"}
    parameters = {"num_epochs": 2, "train_mpp_id": 2, "val_strategy": "manifest"}
    with pytest.raises(ValueError, match="missing required path ids"):
        validate_job_semantics("smoke", "standard_training", parameters, experiment=experiment)
    with pytest.raises(ValueError, match="server_mpp_partner_cache"):
        validate_job_semantics(
            "smoke",
            "standard_training",
            parameters,
            experiment=experiment,
            path_ids=["mpp_data_root", "mpp_standard_splits", "server_mpp_flat_cache", "server_mpp_results"],
        )
    with pytest.raises(ValueError, match="cannot be overridden"):
        validate_job_semantics(
            "smoke",
            "standard_training",
            {**parameters, "cache_root": "elsewhere"},
            experiment=experiment,
            path_ids=[
                "mpp_data_root",
                "mpp_standard_splits",
                "server_mpp_partner_cache",
                "server_mpp_flat_cache",
                "server_mpp_results",
            ],
        )


def test_mpp_cache_parity_job_is_preflight_only_and_registry_bound():
    experiment = {
        "family": "mpp_preflight",
        "script": "scripts/check_mpp_online_cache_parity.py",
    }
    path_ids = [
        "mpp_data_root",
        "mpp_standard_splits",
        "server_mpp_partner_cache",
        "server_mpp_flat_cache",
        "server_mpp_results",
    ]
    validate_job_semantics(
        "preflight",
        "cache_parity",
        {"samples_per_patient": 8, "device": "cuda"},
        experiment=experiment,
        path_ids=path_ids,
    )
    with pytest.raises(ValueError, match="only valid for phase=preflight"):
        validate_job_semantics(
            "smoke", "cache_parity", {"samples_per_patient": 8},
            experiment=experiment, path_ids=path_ids,
        )
    with pytest.raises(ValueError, match="cannot be overridden"):
        validate_job_semantics(
            "preflight", "cache_parity", {"mpp_root": "elsewhere"},
            experiment=experiment, path_ids=path_ids,
        )
    with pytest.raises(ValueError, match="between 1 and 64"):
        validate_job_semantics(
            "preflight", "cache_parity", {"samples_per_patient": 65},
            experiment=experiment, path_ids=path_ids,
        )


def test_mpp2_lora_smoke_requires_paired_seed_and_registered_paths():
    experiment = {
        "family": "mpp_uni2h_lora",
        "script": "train_mpp_uni2h_lora.py",
    }
    path_ids = [
        "mpp_data_root",
        "mpp_standard_splits",
        "server_mpp_results",
        "server_mpp2_frozen_baseline_checkpoint",
    ]
    parameters = {
        "mode": "lora",
        "dataset_name": "mpp2_s1_lora_smoke",
        "num_epochs": 3,
        "seed": 42,
        "head_checkpoint_sha256": "a" * 64,
        "lora_rank": 8,
        "lora_alpha": 16.0,
    }
    validate_job_semantics(
        "smoke", "standard_training", parameters,
        experiment=experiment, path_ids=path_ids,
    )
    with pytest.raises(ValueError, match="seed=42"):
        validate_job_semantics(
            "smoke", "standard_training", {**parameters, "seed": 43},
            experiment=experiment, path_ids=path_ids,
        )
    with pytest.raises(ValueError, match="phase=smoke"):
        validate_job_semantics(
            "formal", "standard_training", {**parameters, "num_epochs": 1},
            experiment=experiment, path_ids=path_ids,
        )


def test_smoke_result_is_accepted_but_not_latest_formal_result(tmp_path):
    root = make_minimal_project(tmp_path)
    bundle = root / "bundle"
    bundle.mkdir()
    manifest = {
        "schema_version": "1.0",
        "result_id": "smoke-result",
        "job_id": "job-smoke",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "phase": "smoke",
        "status": "success",
        "created_at": "2026-07-11T00:00:00+00:00",
        "data_manifest_id": "fixture-v1",
        "path_index_version": "1.0",
        "artifacts": [],
        "metrics": {"best_epoch": 2, "best_val_loss": 0.2},
    }
    write_json(bundle / "result.json", manifest)
    bind_job(root, manifest)
    import_result_bundle(root, bundle)
    registry = read_json(root / "experiments" / "experiment_registry.json")
    state = read_json(root / "project_state" / "current_state.json")
    assert registry["experiments"][0]["evidence_status"] == "accepted"
    assert "exp1" not in state["latest_accepted_result_ids"]


def test_interrupted_result_transaction_rolls_back_from_backups(tmp_path):
    root = make_minimal_project(tmp_path)
    registry_path = root / "experiments" / "experiment_registry.json"
    before = registry_path.read_bytes()
    transaction = root / "project_state" / ".transactions" / "txn-fixture"
    backup = transaction / "backup"
    backup.mkdir(parents=True)
    for source, name in (
        (registry_path, "experiment_registry.json"),
        (root / "experiments" / "experiment_dashboard.md", "experiment_dashboard.md"),
        (root / "project_state" / "current_state.json", "current_state.json"),
        (root / "CURRENT_STATE.md", "CURRENT_STATE.md"),
    ):
        (backup / name).write_bytes(source.read_bytes())
    write_json(transaction / "transaction.json", {"status": "committing"})
    # One staged file remains while another has already been moved: partial commit.
    (transaction / "experiment_dashboard.md").write_text("staged\n", encoding="utf-8")
    registry_path.write_text('{"corrupt": true}\n', encoding="utf-8")
    recovered = recover_incomplete_result_transactions(root)
    assert recovered == ["txn-fixture:rolled_back"]
    assert registry_path.read_bytes() == before
    assert not (root / "project_state" / ".transactions").exists()


def test_validate_job_rejects_unsafe_job_id_before_git_lookup(tmp_path):
    root = make_minimal_project(tmp_path)
    manifest = {
        "schema_version": "1.0",
        "job_id": "../escape",
        "experiment_id": "exp1",
        "source_commit": "abc1234",
        "state_revision": 1,
        "phase": "preflight",
        "command_id": "state_preflight",
        "path_ids": [],
        "parameters": {},
        "created_at": "2026-07-11T00:00:00+00:00",
        "artifact_policy": {},
    }
    with pytest.raises(ValueError, match="unsafe job_id"):
        validate_job_manifest(root, manifest)


def test_document_registry_semantic_hash_ignores_derived_view_bookkeeping(tmp_path):
    root = make_minimal_project(tmp_path)
    registry_path = root / "project_state" / "document_registry.json"
    registry = read_json(registry_path)
    derived_path = root / "CURRENT_STATE.md"
    registry["documents"].append({
        "doc_id": "derived-current-state",
        "path": "CURRENT_STATE.md",
        "authority": "derived",
        "lifecycle": "active",
        "verified_at": "2026-07-11T00:00:00+00:00",
        "state_revision": 1,
        "content_sha256": sha256_file(derived_path),
    })
    write_json(registry_path, registry)
    before = compute_source_hashes(root)["document_registry_sha256"]
    registry["updated_at"] = "2026-07-12T00:00:00+00:00"
    registry["state_revision"] = 999
    registry["documents"][-1]["verified_at"] = "2026-07-12T00:00:00+00:00"
    registry["documents"][-1]["state_revision"] = 999
    registry["documents"][-1]["content_sha256"] = "f" * 64
    write_json(registry_path, registry)
    assert compute_source_hashes(root)["document_registry_sha256"] == before
    registry["documents"][0]["lifecycle"] = "superseded"
    write_json(registry_path, registry)
    assert compute_source_hashes(root)["document_registry_sha256"] != before


def test_finalize_honors_custom_registry_and_keeps_unverified_result_pending(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_history.csv").write_text(
        "epoch,train_pcc,val_pcc,val_loss\n1,0.4,0.3,0.8\n2,0.5,0.35,0.7\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "experiment_registry.json"
    write_json(registry_path, {
        "version": 1,
        "updated_at": "2026-07-11T00:00:00+00:00",
        "experiments": [{"id": "exp1", "family": "test", "status": "planned", "script": "train.py"}],
    })
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_experiment.py",
            "--run-dir", str(run_dir),
            "--experiment-id", "exp1",
            "--registry", str(registry_path),
        ],
    )
    finalize_module.main()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    experiment = registry["experiments"][0]
    assert experiment["best_epoch"] == 2
    assert experiment["evidence_status"] == "pending"
    assert experiment["provenance_complete"] is False
    assert (tmp_path / "experiment_dashboard.md").exists()


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def _make_repair_evidence_repo(tmp_path: Path, *, tamper_summary_hash: bool = False):
    root = make_minimal_project(tmp_path / "repo")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "pfmval-test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "PFmval Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "core.longpaths", "true"], cwd=root, check=True)
    (root / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8")
    split_paths = []
    for group in range(1, 6):
        split = root / "mpp_standard_splits" / f"group_{group}" / "split_manifest.csv"
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text("patient,patch_stem,split\nP1,patch_x1_y1,train\n", encoding="utf-8")
        split_paths.append((group, split))
    source_commit = _commit_all(root, "source")
    source_assets = []
    for group, split in split_paths:
        relative = split.relative_to(root).as_posix()
        split_blob = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"], cwd=root)
        source_assets.append({
            "path": rf"D:\server\worktree\mpp_standard_splits\group_{group}\split_manifest.csv",
            "size_bytes": len(split_blob),
            "sha256": hashlib.sha256(split_blob).hexdigest(),
        })

    version = "barcode-repair-test-v001"
    evidence_rel = Path("automation") / "evidence" / version
    evidence_dir = root / evidence_rel
    evidence_dir.mkdir(parents=True)
    generated_assets = [
        {
            "path": f"group_{group}/labels/train/P1/P1_ssGSEA_zscore.csv",
            "role": "standardized_label",
            "size_bytes": 10,
            "sha256": f"{group}" * 64,
            "row_count": 1,
            "unique_barcode_count": 1,
            "duplicate_barcode_count": 0,
        }
        for group in range(1, 6)
    ]
    audit = {
        "schema_version": "1.0",
        "generated_at": "2026-07-11T00:00:00+00:00",
        "staging_version": version,
        "source_commit": source_commit,
        "server_transport": "gitee_only",
        "mpp_ids": [1, 2, 3, 4, 5],
        "source_assets": source_assets,
        "generated_assets": generated_assets,
        "validation": {
            "patient_barcode_one_to_one": True,
            "dataset_labels_unique": True,
            "existing_mpp_assets_modified": False,
            "published_from_versioned_staging": True,
        },
        "training_gate": {"status": "blocked_pending_evidence_import_and_explicit_gate_release"},
    }
    write_json(evidence_dir / "server_asset_audit_manifest.json", audit)
    summary = {
        "schema_version": "1.0",
        "status": "repair_staging_verified",
        "source_commit": source_commit,
        "staging_version": version,
        "server_stage_path": rf"D:\staging\{version}",
        "audit_sha256": "f" * 64,
        "generated_asset_count": len(generated_assets),
        "source_asset_count": len(source_assets),
        "patient_barcode_one_to_one": True,
        "dataset_labels_unique": True,
        "training_gate": "blocked_pending_evidence_import_and_explicit_gate_release",
    }
    write_json(evidence_dir / "server_verification.json", summary)
    _commit_all(root, "evidence")
    audit_blob = subprocess.check_output(
        ["git", "show", f"HEAD:{evidence_rel.as_posix()}/server_asset_audit_manifest.json"],
        cwd=root,
    )
    audit_sha = hashlib.sha256(audit_blob).hexdigest()
    summary["audit_sha256"] = "0" * 64 if tamper_summary_hash else audit_sha
    write_json(evidence_dir / "server_verification.json", summary)
    evidence_commit = _commit_all(root, "canonical hash")
    return root, evidence_commit, evidence_rel.as_posix(), version, audit_sha


def test_repair_evidence_import_records_verified_pending_state(tmp_path):
    root, evidence_commit, evidence_path, version, audit_sha = _make_repair_evidence_repo(tmp_path)

    result = import_mpp_repair_evidence_from_git(
        root,
        git_ref=evidence_commit,
        evidence_path=evidence_path,
        expected_audit_sha256=audit_sha,
    )

    assert result["status"] == "imported"
    registry = read_json(root / "project_state" / "mpp_repair_registry.json")
    record = registry["repairs"][0]
    assert record["evidence_id"] == version
    assert record["status"] == "verified_pending_gate_release"
    state = read_json(root / "project_state" / "current_state.json")
    assert state["mpp_repair"]["verified_evidence_id"] == version
    assert state["mpp_repair"]["active_data_manifest_id"] is None


def test_repair_evidence_import_rejects_tampered_summary_without_state_change(tmp_path):
    root, evidence_commit, evidence_path, _, audit_sha = _make_repair_evidence_repo(
        tmp_path, tamper_summary_hash=True
    )
    before = (root / "project_state" / "current_state.json").read_bytes()

    with pytest.raises(ValueError, match="audit SHA-256"):
        import_mpp_repair_evidence_from_git(
            root,
            git_ref=evidence_commit,
            evidence_path=evidence_path,
            expected_audit_sha256=audit_sha,
        )

    assert (root / "project_state" / "current_state.json").read_bytes() == before
    assert not (root / "project_state" / "mpp_repair_registry.json").exists()


def test_repair_activation_requires_explicit_release_directive_and_removes_only_barcode_gate(tmp_path):
    root, evidence_commit, evidence_path, version, audit_sha = _make_repair_evidence_repo(tmp_path)
    imported = import_mpp_repair_evidence_from_git(
        root,
        git_ref=evidence_commit,
        evidence_path=evidence_path,
        expected_audit_sha256=audit_sha,
    )
    state_path = root / "project_state" / "current_state.json"
    state = read_json(state_path)
    state["blocked_actions"] = [
        "new_mpp_training_until_conflicting_duplicate_barcodes_are_resolved",
        "formal_training_without_user_approval",
    ]
    write_json(state_path, state)

    with pytest.raises(ValueError, match="gate-release directive"):
        activate_mpp_repair_evidence(
            root,
            evidence_id=version,
            directive_id="DIR-missing",
        )

    directive_id = append_directive(
        root,
        summary="Explicitly release the barcode repair gate after verified evidence import",
        scope="mpp_data",
        topic="barcode_repair_gate_release",
        supersedes=[],
        affected_files=["project_state/mpp_repair_registry.json"],
    )
    result = activate_mpp_repair_evidence(
        root,
        evidence_id=version,
        directive_id=directive_id,
    )

    assert result["status"] == "active"
    final_state = read_json(state_path)
    assert final_state["mpp_repair"]["active_data_manifest_id"] == imported["data_manifest_id"]
    assert "new_mpp_training_until_conflicting_duplicate_barcodes_are_resolved" not in final_state["blocked_actions"]
    assert "formal_training_without_user_approval" in final_state["blocked_actions"]


def test_server_revalidates_repaired_staging_bytes_before_training(tmp_path):
    root = tmp_path / "repo"
    stage = tmp_path / "stage"
    generated = stage / "group_2" / "labels" / "train" / "P1" / "P1_ssGSEA_zscore.csv"
    generated.parent.mkdir(parents=True)
    generated.write_text("barcode,pathway\na,1\n", encoding="utf-8")
    audit = {
        "generated_assets": [{
            "path": "group_2/labels/train/P1/P1_ssGSEA_zscore.csv",
            "size_bytes": generated.stat().st_size,
            "sha256": sha256_file(generated),
        }],
    }
    canonical = root / "project_state" / "evidence" / "mpp" / "repair-v1" / "server_asset_audit_manifest.json"
    write_json(canonical, audit)
    stage_audit = stage / "server_asset_audit_manifest.json"
    stage_audit.write_text(json.dumps(audit, separators=(",", ":")), encoding="utf-8")
    assert sha256_file(stage_audit) != sha256_file(canonical)
    record = {
        "evidence_id": "repair-v1",
        "audit_sha256": sha256_file(canonical),
        "server_stage_path": str(stage),
    }

    result = verify_mpp_repair_server_assets(root, record)
    assert result["verified_generated_assets"] == 1

    generated.write_text("barcode,pathway\na,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_mpp_repair_server_assets(root, record)
