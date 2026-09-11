"""Thin command-line adapter for the package's shared orchestrator.

The two PowerShell entrypoints differ only in the protocol, default seeds and
folds they pass here.  Training, resuming and PlanOnly behavior belong to
``src.orchestrator``; this file deliberately contains no second implementation.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared Phase2 v4 orchestrator.")
    parser.add_argument("--config", required=True, type=Path, help="run-specific config snapshot")
    parser.add_argument("--run-dir", required=True, type=Path, help="raw output directory")
    parser.add_argument("--weights-dir", required=True, type=Path, help="server-only checkpoint directory")
    parser.add_argument("--protocol", choices=("original", "lopo6"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true", help="resume only compatible artifacts")
    parser.add_argument("--plan-only", action="store_true", help="validate and print the plan without training")
    return parser


def _orchestrator_callable():
    """Resolve the planned shared implementation without implementing it here."""
    package_dir = Path(__file__).resolve().parent
    source_dir = str(package_dir / "src")
    if source_dir not in sys.path:
        # The package modules use package-local imports and must not depend on
        # the repository root or the run directory as the current directory.
        sys.path.insert(0, source_dir)
    module = importlib.import_module("src.orchestrator")
    function = getattr(module, "execute_plan", None)
    if function is None:
        function = getattr(module, "main", None)
    if function is None:
        raise AttributeError("src.orchestrator must expose execute_plan or main")
    return function


def _exit_code(result) -> int:
    """Normalize the small set of result forms permitted from the orchestrator."""
    if result is None:
        return 0
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    if isinstance(result, dict) and "exit_code" in result:
        return int(result["exit_code"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execute_plan = _orchestrator_callable()
    result = execute_plan(
        config_path=args.config.resolve(),
        run_dir=args.run_dir.resolve(),
        weights_dir=args.weights_dir.resolve(),
        protocol=args.protocol,
        seeds=tuple(args.seeds),
        folds=None if args.folds is None else tuple(args.folds),
        resume=bool(args.resume),
        plan_only=bool(args.plan_only),
    )
    return _exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
