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
