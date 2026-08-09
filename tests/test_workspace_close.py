import json

from scripts.pfmval_assets import build_close_preview


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_close_preview_is_stable_side_effect_free_and_reports_all_blockers(tmp_path):
    root = tmp_path / "repo"
    _write_json(
        root / "project_state" / "workspace_registry.json",
        {
            "schema_version": "1.0",
            "updated_at": "2026-07-27T00:00:00+00:00",
            "next_workspace_number": 2,
            "workspaces": [
                {
                    "workspace_id": "W001",
                    "experiment_id": "exp-a",
                    "status": "workspace_open",
                    "lease": {"attempt_id": "A001"},
                    "hosts": {
                        "local": {
                            "relative_path": "workspaces/W001",
                            "branch": "codex/exp-a",
                            "current_source_commit": "a" * 40,
                        }
                    },
                }
            ],
        },
    )
    (root / "project_state" / "attempt_events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "ATTEMPT_PREPARED",
                "workspace_id": "W001",
                "attempt_id": "A001",
                "job_id": "W001-A001",
            }
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "RESULT_RETURNED",
                "workspace_id": "W001",
                "attempt_id": "A001",
                "job_id": "W001-A001",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "project_state" / "asset_registry.json",
        {
            "schema_version": "1.0",
            "updated_at": "2026-07-27T00:00:00+00:00",
            "assets": [],
        },
    )
    before = {
        path: path.read_bytes()
        for path in (root / "project_state").rglob("*")
        if path.is_file()
    }
    observed = [
        {
            "path": str(root / "workspaces" / "W001"),
            "branch": "codex/exp-a",
            "head": "a" * 40,
            "dirty": True,
            "unpushed_commits": 1,
        }
    ]

    first = build_close_preview(root, workspace_id="W001", observed_worktrees=observed)
    second = build_close_preview(root, workspace_id="W001", observed_worktrees=observed)

    assert first == second
    assert first["status"] == "blocked"
    assert set(first["blockers"]) >= {
        "active_lease",
        "dirty_worktree",
        "unpushed_commits",
        "unimported_result",
    }
    after = {
        path: path.read_bytes()
        for path in (root / "project_state").rglob("*")
        if path.is_file()
    }
    assert after == before


def test_close_preview_reports_unreturned_result_and_can_reach_close_ready(tmp_path):
    root = tmp_path / "repo"
    workspace_registry_path = root / "project_state" / "workspace_registry.json"
    _write_json(
        workspace_registry_path,
        {
            "schema_version": "1.0",
            "updated_at": "2026-07-27T00:00:00+00:00",
            "next_workspace_number": 2,
            "workspaces": [
                {
                    "workspace_id": "W001",
                    "experiment_id": "exp-a",
                    "status": "workspace_open",
                    "lease": None,
                    "hosts": {
                        "local": {
                            "relative_path": "workspaces/W001",
                            "branch": "codex/exp-a",
                            "current_source_commit": "a" * 40,
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        root / "project_state" / "asset_registry.json",
        {
            "schema_version": "1.0",
            "updated_at": "2026-07-27T00:00:00+00:00",
            "assets": [],
        },
    )
    events_path = root / "project_state" / "attempt_events.jsonl"
    started = [
        {
            "event_type": "ATTEMPT_PREPARED",
            "workspace_id": "W001",
            "attempt_id": "A001",
        },
        {
            "event_type": "EXPERIMENT_STARTED",
            "workspace_id": "W001",
            "attempt_id": "A001",
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(item) for item in started) + "\n",
        encoding="utf-8",
    )
    observed = [
        {
            "path": str(root / "workspaces" / "W001"),
            "branch": "codex/exp-a",
            "head": "a" * 40,
            "dirty": False,
            "unpushed_commits": 0,
        }
    ]
    blocked = build_close_preview(
        root,
        workspace_id="W001",
        observed_worktrees=observed,
    )
    assert "unreturned_result" in blocked["blockers"]

    terminal = [
        *started,
        {
            "event_type": "RESULT_RETURNED",
            "workspace_id": "W001",
            "attempt_id": "A001",
        },
        {
            "event_type": "IMPORTED",
            "workspace_id": "W001",
            "attempt_id": "A001",
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(item) for item in terminal) + "\n",
        encoding="utf-8",
    )
    ready = build_close_preview(
        root,
        workspace_id="W001",
        observed_worktrees=observed,
    )
    assert ready["status"] == "close_ready"
    assert ready["blockers"] == []
    assert ready["execute_authorized"] is False


def test_close_preview_accepts_exact_registered_external_workspace(tmp_path):
    root = tmp_path / "repo"
    workspace_path = tmp_path / "repo_governed_workspaces" / "W002"
    _write_json(
        root / "project_state" / "workspace_registry.json",
        {
            "schema_version": "1.0",
            "updated_at": "2026-08-09T00:00:00+00:00",
            "next_workspace_number": 3,
            "workspaces": [
                {
                    "workspace_id": "W002",
                    "experiment_id": "governance-g15",
                    "status": "shadow",
                    "lease": None,
                    "hosts": {
                        "local": {
                            "path_id": "local_experiment_workspaces",
                            "relative_path": str(workspace_path),
                            "branch": "codex/w002",
                            "current_source_commit": "b" * 40,
                        }
                    },
                }
            ],
        },
    )
    (root / "project_state" / "attempt_events.jsonl").write_text(
        "", encoding="utf-8"
    )
    _write_json(
        root / "project_state" / "asset_registry.json",
        {
            "schema_version": "1.0",
            "updated_at": "2026-08-09T00:00:00+00:00",
            "assets": [],
        },
    )
    preview = build_close_preview(
        root,
        workspace_id="W002",
        observed_worktrees=[
            {
                "path": str(workspace_path),
                "branch": "codex/w002",
                "head": "b" * 40,
                "dirty": False,
                "unpushed_commits": 0,
            }
        ],
    )
    assert preview["status"] == "close_ready"
