"""Export current pathway predictions through the stable table contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from targets.registry import load_target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    package_dir = SRC_DIR.parent
    spec = load_target(
        str((config.get("parameters") or {}).get("target_id", "pathway_ssgsea")),
        Path((config.get("inputs") or {}).get("splits_dir") or (package_dir / "assets" / "group_2"))
        / "zscore_manifest.json",
    ).spec()
    print(
        f"pathway export uses target_id={spec.target_id}, n_outputs={spec.n_outputs}. "
        "Run the package with run_kind=predict or train to write raw/predictions_*.csv.",
        flush=True,
    )
    (args.run_dir / "raw" / "pathway_export_ready.json").write_text(
        json.dumps({"ready": True, "target": spec.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
