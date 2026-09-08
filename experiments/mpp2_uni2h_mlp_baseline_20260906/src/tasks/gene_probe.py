"""Reserved gene-prediction probe. Fails until the gene target contract exists."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from targets.gene_expression import MISSING, reserved_spec
from targets.protocol import MissingTargetContract
from targets.registry import load_target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        load_target("gene_expression", Path("unused.json")).spec()
    except MissingTargetContract as exc:
        payload = reserved_spec().to_dict()
        payload["error"] = str(exc)
        payload["read_next"] = "队友同步说明.md"
        (args.run_dir / "raw" / "gene_probe_missing.json").write_text(
            __import__("json").dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("gene_expression is reserved. Missing: " + ", ".join(MISSING), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
