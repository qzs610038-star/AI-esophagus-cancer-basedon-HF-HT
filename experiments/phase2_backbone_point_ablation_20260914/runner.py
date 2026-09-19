"""Official batch entrypoint; all scientific values come from frozen config.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root / "src"))
    from config import load_config
    from dispatch import execute_batch

    parser = argparse.ArgumentParser(description="Phase2 UNI / Virchow2 point ablation")
    parser.add_argument("--config", type=Path, default=package_root / "config.json")
    parser.add_argument("--scope", choices=("seed42", "remaining"), default="seed42")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = execute_batch(load_config(args.config), scope=args.scope, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result["exit_code"])
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
