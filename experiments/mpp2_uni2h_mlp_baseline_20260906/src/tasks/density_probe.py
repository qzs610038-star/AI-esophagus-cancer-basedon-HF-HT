"""Reserved VHD density probe. The stride hook exists; the formal 3-arm study does not."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sampling.registry import load_sampler

MISSING = (
    "matched_random_control_rule",
    "dense_held_out_test_definition_beyond_current_xzy",
    "formal_three_arm_protocol",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    parameters = json.loads(args.config.read_text(encoding="utf-8-sig")).get("parameters") or {}
    sampler = load_sampler(str(parameters.get("sampling_id", "dense")), parameters)
    payload = {
        "sampling_id": sampler.sampling_id,
        "hook_ready": True,
        "formal_experiment_ready": False,
        "missing": list(MISSING),
        "rule": "Filter only after the full split is loaded, and only the train split.",
        "read_next": "队友同步说明.md",
    }
    (args.run_dir / "raw" / "density_probe_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"sampling hook is {sampler.sampling_id}. Formal 3-arm density experiment is not ready. "
        "Missing: " + ", ".join(MISSING),
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
