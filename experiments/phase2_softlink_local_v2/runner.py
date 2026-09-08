"""Compatibility shim: the formal main entry now lives in src/run_experiments.py."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from run_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
