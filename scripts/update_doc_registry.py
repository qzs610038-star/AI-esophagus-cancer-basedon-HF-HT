#!/usr/bin/env python3
"""Compatibility wrapper for the tracked PFMval document registry.

Use ``python deploy/pfmval_ops.py docs scan`` for the canonical interface.
This wrapper defaults to a dry run and writes only when ``--write`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pfmval_state import scan_documents, write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan PFMval managed Markdown documents")
    parser.add_argument("--write", action="store_true", help="write project_state/document_registry.json")
    parser.add_argument("--dry-run", action="store_true", help="compatibility alias; dry run is already the default")
    args = parser.parse_args()

    registry = scan_documents(PROJECT_ROOT)
    print(json.dumps(registry["summary"], ensure_ascii=False, indent=2))
    if args.write:
        output = PROJECT_ROOT / "project_state" / "document_registry.json"
        write_json_atomic(output, registry)
        print(f"[PASS] wrote {output}")
    else:
        print("[DRY RUN] no files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
