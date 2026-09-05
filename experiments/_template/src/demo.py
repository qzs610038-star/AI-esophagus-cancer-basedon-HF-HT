"""Zero-training example. Outputs are demonstrations, never research evidence."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fail", action="store_true", help="Exercise error logging without training")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    print("零训练演示：仅检查独立运行和日志，不产生科研结果。", flush=True)
    if args.fail:
        raise RuntimeError("零训练演示：主动报错，用于验证日志回传。")
    demo = {"demo_only": True, "message": "No model was trained", "parameters": config["parameters"]}
    (args.run_dir / "raw" / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_dir / "metrics.json").write_text(json.dumps({"demo_only": True, "metrics": {}}), encoding="utf-8")
    print("演示完成；metrics 为空，不代表训练性能。", flush=True)


if __name__ == "__main__":
    main()
