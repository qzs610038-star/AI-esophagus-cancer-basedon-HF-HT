import json
import re
from urllib.parse import unquote
from pathlib import Path

from scripts.pfmval_views import build_project_guide


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_project_guide_separates_active_and_historical_documents(tmp_path):
    root = tmp_path / "repo"
    active = root / "01_指南与解读" / "active.md"
    historical = root / "archive" / "old.md"
    active.parent.mkdir(parents=True)
    historical.parent.mkdir(parents=True)
    active.write_text("# active\n", encoding="utf-8")
    historical.write_text("# old\n", encoding="utf-8")
    _write_json(
        root / "project_state" / "document_registry.json",
        {
            "state_revision": 7,
            "updated_at": "2026-07-27T00:00:00+00:00",
            "documents": [
                {
                    "doc_id": "active-guide",
                    "path": "01_指南与解读/active.md",
                    "lifecycle": "active",
                    "authority": "normative",
                },
                {
                    "doc_id": "old-guide",
                    "path": "archive/old.md",
                    "lifecycle": "historical",
                    "authority": "reference",
                },
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

    first = build_project_guide(root)
    second = build_project_guide(root)

    assert first == second
    current_section = first.split("## 历史与过时资料", 1)[0]
    assert "01_指南与解读/active.md" in current_section
    assert "archive/old.md" not in current_section
    assert "Agent 专用" in first
    assert "CLI-only" in first


def test_tracked_project_guide_links_exist_and_registry_marks_it_active():
    root = Path(__file__).resolve().parent.parent
    guide_path = root / "PROJECT_GUIDE.md"
    registry = json.loads(
        (root / "project_state" / "document_registry.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item
        for item in registry["documents"]
        if item["path"] == "PROJECT_GUIDE.md"
    )
    assert entry["lifecycle"] == "active"
    assert entry["authority"] == "derived"
    assert entry["availability"] == "tracked"
    missing_paths = {
        item["path"]
        for item in registry["documents"]
        if item.get("lifecycle") == "missing"
        or item.get("availability") == "missing"
    }
    for target in re.findall(
        r"\[[^\]]+\]\(([^)]+)\)",
        guide_path.read_text(encoding="utf-8"),
    ):
        if unquote(target) in missing_paths:
            continue
        assert (root / unquote(target)).exists(), target


def test_project_guide_surfaces_deployment_learning_links_and_review_due(tmp_path):
    root = tmp_path / "repo"
    plan = root / "01_指南与解读" / "部署方案" / "plan.md"
    guide = root / "01_指南与解读" / "学习指南" / "guide.md"
    plan.parent.mkdir(parents=True)
    guide.parent.mkdir(parents=True)
    plan.write_text("# changed deployment\n", encoding="utf-8")
    guide.write_text("# learning guide\n", encoding="utf-8")
    _write_json(
        root / "project_state" / "document_registry.json",
        {
            "state_revision": 8,
            "updated_at": "2026-08-07T00:00:00+00:00",
            "documents": [
                {
                    "doc_id": "plan-a",
                    "path": "01_指南与解读/部署方案/plan.md",
                    "lifecycle": "active",
                    "authority": "normative",
                    "doc_role": "deployment_plan",
                    "content_sha256": "0" * 64,
                    "related_docs": ["guide-a"],
                },
                {
                    "doc_id": "guide-a",
                    "path": "01_指南与解读/学习指南/guide.md",
                    "lifecycle": "active",
                    "authority": "reference",
                    "doc_role": "learning_guide",
                    "source_refs": ["plan-a"],
                    "dependency_fingerprints": {"plan-a": "0" * 64},
                    "freshness": "fresh",
                },
            ],
        },
    )
    _write_json(
        root / "project_state" / "asset_registry.json",
        {"schema_version": "1.0", "assets": []},
    )

    rendered = build_project_guide(root)

    assert "## 部署方案与学习指南" in rendered
    assert "plan-a" in rendered
    assert "guide-a" in rendered
    assert "review_due" in rendered
    assert "changed_dependencies=plan-a" in rendered
