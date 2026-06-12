#!/usr/bin/env python3
"""
update_doc_registry.py — Scan project docs and update .claude/doc-registry.json.

Scans 01_指南与解读/, 02_组会汇报/, README.md, CLAUDE.md, and other key md files.
Preserves existing entries' purpose/tags/created fields.
New entries infer category from directory, date from filename.
Top-level 'updated' is set to today's date.

Usage:
    python scripts/update_doc_registry.py
    python scripts/update_doc_registry.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / ".claude" / "doc-registry.json"

# Scan directories for md files
SCAN_DIRS = {
    "01_指南与解读": "01_指南与解读",
    "02_组会汇报": "02_组会汇报",
    ".claude": ".claude",
    ".qoder": ".qoder",
}

# Category inference from directory
DIR_CATEGORY_MAP = {
    "学习指南": "学习指南",
    "部署方案": "部署方案",
    "分析报告": "分析报告",
    "02_组会汇报": "组会汇报",
    ".claude": "配置参考",
    ".qoder": "技术参考",
}

# Files at project root (or key paths) to include
ROOT_MD_FILES = [
    "README.md",
    "CLAUDE.md",
    "模型性能排名_Model_Performance_Ranking.md",
]

# Files to exclude from scanning
EXCLUDE_PATHS = {
    ".claude/doc-registry.json",  # self-referential
}


def extract_date_from_filename(filename):
    """Extract YYYY-MM-DD or YYYYMMDD from filename. Returns string or empty."""
    # Pattern: YYYYMMDD (8 digits)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Pattern: YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def infer_category(rel_path):
    """Infer category from relative path."""
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        subdir = parts[0]  # e.g. "01_指南与解读"
        if len(parts) >= 3:
            leaf = parts[1]  # e.g. "学习指南"
            if leaf in DIR_CATEGORY_MAP:
                return DIR_CATEGORY_MAP[leaf]
        if subdir in DIR_CATEGORY_MAP:
            return DIR_CATEGORY_MAP[subdir]
    return "未分类"


def load_registry():
    """Load existing doc-registry.json."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "updated": ""}


def save_registry(registry, dry_run=False):
    """Save doc-registry.json."""
    registry["updated"] = date.today().isoformat()
    if dry_run:
        print(f"\n[DRY RUN] Would save {len(registry['documents'])} entries to {REGISTRY_PATH}")
        return
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[PASS] Saved {len(registry['documents'])} entries to {REGISTRY_PATH}")


def scan_md_files():
    """Scan project for all md files to register. Returns list of (name, rel_path)."""
    discovered = []

    # Scan configured directories recursively
    for dir_key, dir_path in SCAN_DIRS.items():
        full_dir = PROJECT_ROOT / dir_path
        if not full_dir.exists():
            print(f"[SKIP] Directory not found: {full_dir}")
            continue
        for md_file in sorted(full_dir.rglob("*.md")):
            rel = str(md_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if rel in EXCLUDE_PATHS:
                continue
            discovered.append((md_file.name, rel))

    # Root-level md files
    for filename in ROOT_MD_FILES:
        p = PROJECT_ROOT / filename
        if p.exists():
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            discovered.append((p.name, rel))

    return discovered


def main():
    parser = argparse.ArgumentParser(description="Update PFMval doc registry")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without saving")
    args = parser.parse_args()

    registry = load_registry()
    existing = {doc["path"]: doc for doc in registry["documents"]}

    print(f"[INFO] Existing entries: {len(existing)}")
    print(f"[INFO] Scanning for md files...")

    discovered = scan_md_files()
    discovered_paths = {rel_path for _, rel_path in discovered}
    print(f"[INFO] Discovered {len(discovered)} md files")

    new_entries = []
    updated_count = 0
    missing_count = 0

    for name, rel_path in discovered:
        if rel_path in existing:
            # Preserve existing entry; update category if empty
            entry = existing[rel_path]
            changed = False
            if not entry.get("category"):
                entry["category"] = infer_category(rel_path)
                changed = True
            if not entry.get("created"):
                extracted = extract_date_from_filename(name)
                if extracted:
                    entry["created"] = extracted
                    changed = True
            if changed:
                updated_count += 1
                print(f"  [UPDATE] {rel_path} (category={entry['category']}, created={entry['created']})")
            continue

        # New file
        category = infer_category(rel_path)
        created = extract_date_from_filename(name)
        entry = {
            "name": name,
            "path": rel_path,
            "category": category,
            "purpose": "",
            "created": created,
            "tags": [],
        }
        new_entries.append(entry)
        print(f"  [NEW] {rel_path} (category={category}, created={created or '(not in filename)'})")

    # Build new document list: keep existing + mark orphans
    new_docs = []
    for doc in registry["documents"]:
        path = doc.get("path", "")
        if path in discovered_paths:
            new_docs.append(doc)
        else:
            missing_count += 1
            print(f"  [ORPHAN] {path} — file no longer exists, keeping in registry")
            new_docs.append(doc)  # Keep orphaned entries

    new_docs.extend(new_entries)

    # Sort by category then path
    new_docs.sort(key=lambda d: (d.get("category", ""), d.get("path", "")))
    registry["documents"] = new_docs

    print(f"\n[INFO] Summary:")
    print(f"       New entries: {len(new_entries)}")
    print(f"       Updated:     {updated_count}")
    print(f"       Orphaned:    {missing_count}")
    print(f"       Total:       {len(new_docs)}")

    save_registry(registry, dry_run=args.dry_run)

    if len(new_entries) > 0 and not args.dry_run:
        print(f"\n[INFO] New entries added with empty 'purpose' and 'tags'.")
        print(f"[INFO] Consider manually filling these fields for:")
        for entry in new_entries:
            print(f"       {entry['path']}")


if __name__ == "__main__":
    main()
