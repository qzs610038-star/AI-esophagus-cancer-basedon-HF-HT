import json
from pathlib import Path

from scripts.pfmval_views import (
    build_experiment_progress,
    refresh_experiment_views,
)


def _registry() -> dict:
    experiments = [
        {
            "id": "current-planned",
            "display_name": "当前计划",
            "workspace_id": "W001",
            "phase": "smoke",
            "status": "planned",
            "evidence_status": "pending",
            "next_action": "运行 smoke",
            "updated_at": "2026-07-27T00:00:00+00:00",
        },
        {
            "id": "accepted-formal",
            "display_name": "已接纳正式实验",
            "workspace_id": "W002",
            "phase": "formal",
            "status": "done",
            "evidence_status": "accepted",
            "best_val_pcc": 0.61,
            "conclusion": "采用",
            "next_action": "无",
            "updated_at": "2026-07-27T00:00:00+00:00",
        },
    ]
    experiments.extend(
        {
            "id": f"historical-{index:02d}",
            "display_name": f"历史实验 {index:02d}",
            "phase": "formal",
            "status": "failed",
            "evidence_status": "historical",
            "next_action": "无",
            "updated_at": "2026-07-01T00:00:00+00:00",
        }
        for index in range(32)
    )
    return {
        "version": 1,
        "state_revision": 9,
        "updated_at": "2026-07-27T00:00:00+00:00",
        "experiments": experiments,
    }


def test_progress_view_keeps_current_accepted_and_history_separate(tmp_path):
    registry = _registry()
    text = build_experiment_progress(registry)
    current = text.split("## 已接纳结果", 1)[0]
    accepted = text.split("## 已接纳结果", 1)[1].split("## 历史记录", 1)[0]
    history = text.split("## 历史记录", 1)[1]

    assert "current-planned" in current
    assert "accepted-formal" not in current
    assert "accepted-formal" in accepted
    assert "historical-00" not in current
    assert "historical-00" in history
    assert text.count("historical-") == 32

    rows = [line for line in text.splitlines() if line.startswith("| ")]
    assert rows
    assert len({line.count("|") for line in rows}) == 1


def test_refresh_experiment_views_is_byte_stable(tmp_path):
    root = tmp_path / "repo"
    registry_path = root / "experiments" / "experiment_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(_registry(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    refresh_experiment_views(root)
    first_progress = (root / "experiments" / "experiment_progress.md").read_bytes()
    first_dashboard = (root / "experiments" / "experiment_dashboard.md").read_bytes()
    refresh_experiment_views(root)

    assert (root / "experiments" / "experiment_progress.md").read_bytes() == first_progress
    assert (root / "experiments" / "experiment_dashboard.md").read_bytes() == first_dashboard
