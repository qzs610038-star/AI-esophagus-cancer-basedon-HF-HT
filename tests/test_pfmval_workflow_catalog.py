import json
from pathlib import Path

import pytest

from deploy import pfmval_ops
from scripts.pfmval_workflows import (
    activate_workflow,
    apply_workflow_discovery,
    deprecate_workflow,
    discover_workflows,
    list_workflows,
    merge_workflows,
    register_workflow_manifest,
    review_workflow,
    revise_workflow,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _marker(purpose: str = "sync registry") -> str:
    descriptor = {
        "purpose": purpose,
        "trigger": "pfmval workflow sync",
        "inputs": ["registry"],
        "outputs": ["derived-view"],
        "risk": "low",
        "approval_requirement": "explicit-local-write",
        "schema_refs": ["schema.json"],
        "test_refs": ["tests/test_sync.py"],
        "example_refs": ["examples/sync.json"],
        "version": "1.0.0",
        "stable_version": True,
    }
    return f"<!-- pfmval-workflow: {json.dumps(descriptor)} -->\n"


def _discover_pair(root: Path, purpose: str = "sync registry", **overrides):
    paths = ["docs/a.md", "docs/b.md"]
    _write(root / paths[0], _marker(purpose))
    _write(root / paths[1], _marker(purpose))
    return discover_workflows(
        root,
        selected_paths=paths,
        tracked_paths=paths,
        **overrides,
    )


def test_r11_t01_without_user_discovery_has_zero_scan_and_zero_write(tmp_path):
    root = tmp_path / "repo"
    reads = []
    result = discover_workflows(
        root,
        selected_paths=[],
        tracked_paths=[],
        reader=lambda path: reads.append(path) or "",
    )
    assert result == {
        "status": "not_requested",
        "reads": 0,
        "writes": 0,
        "candidates": [],
    }
    assert reads == []
    assert not (root / "project_state" / "workflow_catalog.json").exists()


def test_r11_t02_duplicate_structures_only_create_candidates(tmp_path):
    root = tmp_path / "repo"
    discovery = _discover_pair(root)
    assert discovery["status"] == "candidate_review_required"
    assert len(discovery["candidates"]) == 1
    candidate = discovery["candidates"][0]
    assert candidate["lifecycle"] == "candidate"
    assert candidate["standardized"] is False
    assert candidate["implementation_locations"] == ["docs/a.md", "docs/b.md"]

    applied = apply_workflow_discovery(root, discovery)
    assert applied["writes"] == 1
    stored = list_workflows(root)
    assert stored[0]["lifecycle"] == "candidate"


def test_r11_t03_high_risk_requires_review_schema_tests_and_approval(tmp_path):
    root = tmp_path / "repo"
    discovery = _discover_pair(root)
    applied = apply_workflow_discovery(root, discovery)
    workflow_id = applied["candidate_ids"][0]

    with pytest.raises(ValueError, match="reviewed"):
        activate_workflow(root, workflow_id)
    review_workflow(root, workflow_id)

    catalog_path = root / "project_state" / "workflow_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = catalog["entries"][0]
    entry["risk"] = "high"
    entry["schema_refs"] = []
    entry["test_refs"] = []
    entry["approval_requirement"] = ""
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="high-risk"):
        activate_workflow(root, workflow_id, standardized=True)


def test_r11_t04_revision_merge_and_deprecation_preserve_lineage(tmp_path):
    root = tmp_path / "repo"
    discovery = _discover_pair(root)
    workflow_id = apply_workflow_discovery(root, discovery)["candidate_ids"][0]
    review_workflow(root, workflow_id)
    activate_workflow(root, workflow_id, standardized=True)

    revised = revise_workflow(
        root,
        workflow_id,
        new_workflow_id="WF-SYNC-V2",
        version="2.0.0",
    )
    assert revised["supersedes"] == [workflow_id]
    by_id = {item["workflow_id"]: item for item in list_workflows(root)}
    assert by_id[workflow_id]["lifecycle"] == "deprecated"
    assert by_id[workflow_id]["replacement_workflow_ids"] == ["WF-SYNC-V2"]

    second = _discover_pair(root, purpose="publish registry")
    second_id = apply_workflow_discovery(root, second)["candidate_ids"][0]
    review_workflow(root, second_id)
    activate_workflow(root, second_id)
    merged = merge_workflows(
        root,
        ["WF-SYNC-V2", second_id],
        new_workflow_id="WF-SYNC-MERGED",
        purpose="merged sync",
        version="3.0.0",
    )
    assert sorted(merged["supersedes"]) == sorted(["WF-SYNC-V2", second_id])
    deprecate_workflow(root, "WF-SYNC-MERGED", replacement_workflow_ids=[])
    final = {item["workflow_id"]: item for item in list_workflows(root)}
    assert final["WF-SYNC-MERGED"]["lifecycle"] == "deprecated"


def test_r11_t05_same_scan_is_noop_and_out_of_scope_is_never_read(tmp_path):
    root = tmp_path / "repo"
    paths = ["docs/a.md", "docs/b.md"]
    for path in paths:
        _write(root / path, _marker())
    _write(root / "secrets.env", "TOKEN=must-not-read")
    reads = []

    def reader(path: Path) -> str:
        reads.append(path.relative_to(root).as_posix())
        return path.read_text(encoding="utf-8")

    discovery = discover_workflows(
        root,
        selected_paths=paths,
        tracked_paths=paths + ["secrets.env"],
        reader=reader,
    )
    first = apply_workflow_discovery(root, discovery)
    before = (root / "project_state" / "workflow_catalog.json").read_bytes()
    second = apply_workflow_discovery(root, discovery)
    after = (root / "project_state" / "workflow_catalog.json").read_bytes()
    assert first["writes"] == 1
    assert second["writes"] == 0
    assert before == after
    assert reads == paths


def test_r11_t06_occurrence_frequency_cannot_relax_experiment_approval(tmp_path):
    root = tmp_path / "repo"
    experiment_path = root / "experiments" / "experiment_registry.json"
    _write(experiment_path, '{"approval_required": true}\n')
    before = experiment_path.read_bytes()
    discovery = _discover_pair(root)
    discovery["occurrences"] = 1000000
    workflow_id = apply_workflow_discovery(root, discovery)["candidate_ids"][0]
    with pytest.raises(ValueError, match="reviewed"):
        activate_workflow(root, workflow_id)
    assert experiment_path.read_bytes() == before


def test_r11_t07_catalog_is_offline_and_cli_is_explicitly_scoped(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("external catalog/network access is forbidden")

    monkeypatch.setattr("socket.create_connection", forbidden)
    root = tmp_path / "repo"
    discovery = _discover_pair(root)
    assert discovery["reads"] == 2
    parser = pfmval_ops.build_parser()
    args = parser.parse_args(
        [
            "workflow",
            "discover",
            "--scope-manifest",
            "scope.json",
        ]
    )
    assert args.command == "workflow"
    assert args.workflow_command == "discover"


def test_g08_register_manifest_creates_candidates_and_same_content_is_noop(tmp_path):
    root = tmp_path / "repo"
    manifest_path = "project_state/workflows/g08-approved.json"
    manifest = {
        "schema_version": "1.0",
        "authorization_ref": "DIR-G08",
        "workflows": [
            {
                "workflow_id": "WF-PROJECT-FACT",
                "purpose": "Maintain non-experimental project facts",
                "trigger": "user requests a project fact update",
                "inputs": ["project fact intake"],
                "outputs": ["confirmation preview"],
                "implementation_locations": [
                    ".agents/skills/pfmval-governance/SKILL.md",
                    "scripts/pfmval_knowledge.py",
                ],
                "risk": "medium",
                "approval_requirement": "each fact requires explicit confirmation",
                "version": "0.1.0",
                "user_entry": "project_fact",
                "technical_routes": ["project_fact"],
            }
        ],
    }
    _write(root / manifest_path, json.dumps(manifest))

    first = register_workflow_manifest(
        root,
        manifest_path=manifest_path,
        authorization_ref="DIR-G08",
        tracked_paths=[manifest_path],
        active_authorization_refs=["DIR-G08"],
    )
    before = (root / "project_state" / "workflow_catalog.json").read_bytes()
    second = register_workflow_manifest(
        root,
        manifest_path=manifest_path,
        authorization_ref="DIR-G08",
        tracked_paths=[manifest_path],
        active_authorization_refs=["DIR-G08"],
    )
    after = (root / "project_state" / "workflow_catalog.json").read_bytes()

    assert first == {
        "status": "registered",
        "writes": 1,
        "candidate_ids": ["WF-PROJECT-FACT"],
    }
    assert second == {
        "status": "noop",
        "writes": 0,
        "candidate_ids": ["WF-PROJECT-FACT"],
    }
    assert before == after
    stored = list_workflows(root)
    assert stored[0]["lifecycle"] == "candidate"
    assert stored[0]["standardized"] is False
    assert stored[0]["authorization_ref"] == "DIR-G08"


def test_g08_cli_exposes_manifest_and_authorization_bound_register():
    parser = pfmval_ops.build_parser()
    args = parser.parse_args(
        [
            "workflow",
            "register",
            "--manifest",
            "project_state/workflows/g08-approved.json",
            "--authorization-ref",
            "DIR-G08",
        ]
    )
    assert args.workflow_command == "register"
    assert args.manifest == "project_state/workflows/g08-approved.json"
    assert args.authorization_ref == "DIR-G08"


def test_g08_register_reuses_same_entry_when_manifest_adds_another_candidate(tmp_path):
    root = tmp_path / "repo"
    manifest_path = "project_state/workflows/g08-approved.json"
    first_entry = {
        "workflow_id": "WF-PROJECT-FACT",
        "purpose": "Maintain non-experimental project facts",
        "trigger": "user requests a project fact update",
        "inputs": ["project fact intake"],
        "outputs": ["confirmation preview"],
        "implementation_locations": ["scripts/pfmval_knowledge.py"],
        "risk": "medium",
        "approval_requirement": "each fact requires explicit confirmation",
        "version": "0.1.0",
        "user_entry": "project_fact",
        "technical_routes": ["project_fact"],
    }
    manifest = {
        "schema_version": "1.0",
        "authorization_ref": "DIR-G08",
        "workflows": [first_entry],
    }
    _write(root / manifest_path, json.dumps(manifest))
    common = {
        "manifest_path": manifest_path,
        "authorization_ref": "DIR-G08",
        "tracked_paths": [manifest_path],
        "active_authorization_refs": ["DIR-G08"],
    }
    register_workflow_manifest(root, **common)

    manifest["workflows"].append(
        {
            **first_entry,
            "workflow_id": "WF-LEARNING-GUIDE",
            "purpose": "Maintain learning guide interfaces",
            "user_entry": "learning_guide",
            "technical_routes": ["learning_guide"],
        }
    )
    _write(root / manifest_path, json.dumps(manifest))
    result = register_workflow_manifest(root, **common)

    assert result["status"] == "registered"
    assert result["writes"] == 1
    assert [item["workflow_id"] for item in list_workflows(root)] == [
        "WF-LEARNING-GUIDE",
        "WF-PROJECT-FACT",
    ]


def test_g08_register_conflict_fails_before_adding_any_manifest_entry(tmp_path):
    root = tmp_path / "repo"
    manifest_path = "project_state/workflows/g08-approved.json"
    entry = {
        "workflow_id": "WF-PROJECT-FACT",
        "purpose": "Maintain project facts",
        "trigger": "user requests a fact update",
        "inputs": ["fact intake"],
        "outputs": ["confirmation preview"],
        "implementation_locations": ["scripts/pfmval_knowledge.py"],
        "risk": "medium",
        "approval_requirement": "explicit confirmation",
        "version": "0.1.0",
        "user_entry": "project_fact",
        "technical_routes": ["project_fact"],
    }
    manifest = {
        "schema_version": "1.0",
        "authorization_ref": "DIR-G08",
        "workflows": [entry],
    }
    _write(root / manifest_path, json.dumps(manifest))
    common = {
        "manifest_path": manifest_path,
        "authorization_ref": "DIR-G08",
        "tracked_paths": [manifest_path],
        "active_authorization_refs": ["DIR-G08"],
    }
    register_workflow_manifest(root, **common)
    before = (root / "project_state" / "workflow_catalog.json").read_bytes()

    manifest["workflows"] = [
        {**entry, "purpose": "Conflicting replacement"},
        {
            **entry,
            "workflow_id": "WF-PAPER-OUTPUT",
            "purpose": "Reserve paper output interface",
            "user_entry": "paper_output",
            "technical_routes": ["paper_output"],
        },
    ]
    _write(root / manifest_path, json.dumps(manifest))
    with pytest.raises(ValueError, match="different content"):
        register_workflow_manifest(root, **common)

    assert (root / "project_state" / "workflow_catalog.json").read_bytes() == before
    assert [item["workflow_id"] for item in list_workflows(root)] == [
        "WF-PROJECT-FACT"
    ]


@pytest.mark.parametrize(
    ("tracked_paths", "active_refs", "message"),
    [
        ([], ["DIR-G08"], "not tracked"),
        (["project_state/workflows/g08-approved.json"], [], "not active"),
    ],
)
def test_g08_register_requires_tracked_manifest_and_active_authorization(
    tmp_path,
    tracked_paths,
    active_refs,
    message,
):
    root = tmp_path / "repo"
    manifest_path = "project_state/workflows/g08-approved.json"
    _write(
        root / manifest_path,
        json.dumps(
            {
                "schema_version": "1.0",
                "authorization_ref": "DIR-G08",
                "workflows": [{}],
            }
        ),
    )
    with pytest.raises(ValueError, match=message):
        register_workflow_manifest(
            root,
            manifest_path=manifest_path,
            authorization_ref="DIR-G08",
            tracked_paths=tracked_paths,
            active_authorization_refs=active_refs,
        )
    assert not (root / "project_state" / "workflow_catalog.json").exists()
