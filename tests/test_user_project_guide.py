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
