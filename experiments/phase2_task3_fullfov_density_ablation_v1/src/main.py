"""Command-line actions for the task-3 package and task-2 read-only label audit."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

from config import PACKAGE_ROOT, load_config
from data_runtime import check_server_inputs, load_runtime_data
from label_audit import run_label_audit
from metrics_export import summarize_runs
from sampling import assert_random_equal_contract
from train_density import check_frozen_reference_compatibility, evaluate_frozen_reference, train_arm


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _batch_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"


def _load_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_paired_initialization(weights_root: Path, seed: int, arms: list[str]) -> None:
    states = [_load_state(weights_root / f"{arm}__seed{seed}" / "initial_weights.pt") for arm in arms]
    names = list(states[0])
    if any(list(state) != names for state in states[1:]):
        raise RuntimeError(f"seed={seed} 三臂初始参数结构不一致")
    for name in names:
        if any(not torch.equal(states[0][name], state[name]) for state in states[1:]):
            raise RuntimeError(f"seed={seed} 三臂初始参数不一致: {name}")


def run_batch(config: dict, mode: str, output_root: Path, weights_root: Path, device: str | None) -> dict:
    runtime = load_runtime_data(config)
    assert_random_equal_contract(runtime.full, config, config["training"]["seeds"])
    output_root.mkdir(parents=True, exist_ok=False); weights_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "feature_caches.json", {
        "status": "reused_read_only",
        "source_experiment_id": "phase2_fullfov_hpo_v1",
        "cache_dir": runtime.feature_cache_path,
        "common_identity_manifest": runtime.common_manifest_path,
        "new_feature_cache_created": False,
    })
    run_summaries = []
    seeds = config["training"]["seeds"] if mode == "formal" else [config["training"]["seeds"][0]]
    if mode == "formal":
        evaluate_frozen_reference(config, runtime, output_root / "frozen_seed45_reference", requested_device=device)
    for seed in seeds:
        for arm in config["training"]["arms"]:
            name = f"{arm}__seed{seed}"
            summary = train_arm(config, runtime, arm, int(seed), output_root / "runs" / name, weights_root / name, mode=mode, requested_device=device)
            run_summaries.append(summary)
        _assert_paired_initialization(weights_root, int(seed), list(config["training"]["arms"]))
    if mode == "formal":
        batch_summary = summarize_runs(run_summaries, output_root / "analysis")
    else:
        batch_summary = {"status": "smoke_complete_non_evidence", "n_runs": len(run_summaries), "formal_evidence": False}
        _write(output_root / "analysis" / "batch_summary.json", batch_summary)
    _write(output_root / "model_weights.json", {
        "return_policy": "paths_only_weight_files_not_returned_by_default",
        "weights_root": str(weights_root),
        "runs": [{"arm": item["arm"], "seed": item["seed"], "checkpoints": item["checkpoints"]} for item in run_summaries],
    })
    _write(output_root / "batch_manifest.json", {
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "status": "formal_complete_unregistered" if mode == "formal" else "smoke_complete_non_evidence",
        "task2_training_called": False,
        "frozen_reference_used_for_initialization": False,
        "external_used_for_selection": False,
        "formal_training_count": 9 if mode == "formal" else 0,
        "summary": batch_summary,
    })
    return {"output_root": str(output_root), "weights_root": str(weights_root), "summary": batch_summary}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["check-inputs", "render-label-audit", "smoke", "formal"])
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.json"))
    parser.add_argument("--output-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv); config = load_config(args.config)
    if args.action == "check-inputs":
        report = check_server_inputs(config)
        report["server_checks"].append(check_frozen_reference_compatibility(config))
        report["server_ready"] = all(item["status"] == "PASS" for item in report["server_checks"])
        destination = Path(args.output_dir) if args.output_dir else PACKAGE_ROOT / "analysis" / "check_inputs_report.json"
        _write(destination, report); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report["local_contract_ready"] and report["server_ready"] else 2
    if args.action == "render-label-audit":
        result = run_label_audit(config, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    batch_id = _batch_id(args.action)
    output_root = Path(args.output_dir) if args.output_dir else Path(config["paths"]["runs_root"]) / config["experiment_id"] / batch_id
    weights_root = Path(args.weights_dir) if args.weights_dir else Path(config["paths"]["weights_root"]) / config["experiment_id"] / batch_id
    result = run_batch(config, args.action, output_root, weights_root, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
