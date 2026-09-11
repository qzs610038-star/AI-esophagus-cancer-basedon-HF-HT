"""Command-line adapter for read-only XZY external inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.xzy_external import run_external_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing Phase2 v4 checkpoints on read-only XZY inputs."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--native-step", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package = json.loads((PACKAGE_ROOT / "package.json").read_text(encoding="utf-8-sig"))
    try:
        result = run_external_inference(
            args.run_dir,
            native_step=args.native_step,
            batch_size=args.batch_size,
            resume=bool(args.resume),
            plan_only=bool(args.plan_only),
            code_version=str(package["code_version"]),
        )
    except Exception as error:
        print(f"XZY external inference failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())

