"""Explicit experiment actions. Default is a read-only input check."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import (
    BASELINE_MODELS,
    GEOMETRY_PROTOCOL,
    MODEL_DIMS,
    SEEDS,
    TRAIN_MODELS,
    load_config,
    preprocess_profile,
)
from errors import ConfigError
from run_io import utc_now, write_json


ACTIONS = (
    "check-environment",
    "check-inputs",
    "prepare-features",
    "train-spatial",
    "external-eval",
    "analyze-local",
)


@dataclass(frozen=True)
class TaskSpec:
    stage: str
    model: str
    protocol: str
    arm: str
    seed: int
    recipe: str

    @property
    def task_id(self) -> str:
        return "__".join((self.stage, self.model, self.protocol, self.arm, str(self.seed), self.recipe))


def plan_train_tasks(models: Sequence[str] | None = None) -> list[TaskSpec]:
    selected = tuple(TRAIN_MODELS if models is None else models)
    if not selected or len(set(selected)) != len(selected) or any(model not in TRAIN_MODELS for model in selected):
        raise ConfigError("models 必须是无重复的 hoptimus0/hoptimus1/phikonv2 非空子集")
    return [
        TaskSpec("backbone-batch2", model, GEOMETRY_PROTOCOL, "spatial", seed, "frozen_spatial")
        for model in selected
        for seed in SEEDS
    ]


def planned_external_keys() -> list[str]:
    return [task.task_id for task in plan_train_tasks()]


def _status(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "exists": path.exists(), "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_inputs(config: dict) -> dict:
    packaged = {name: _status(path) for name, path in config["inputs"].items()}
    server = {name: _status(config["paths"][name]) for name in ("image_root", "labels_root", "feature_caches_root", "hf_home")}
    warnings: list[str] = []
    errors: list[str] = []
    if not all(item["exists"] for item in packaged.values()):
        errors.append("包内小型输入不完整")
    if not all(item["exists"] for item in server.values()):
        warnings.append("服务器图像、标签、缓存根或 HF_HOME 尚未现场核验")
    models: dict[str, dict] = {}
    manifest_path = Path(config["inputs"]["model_manifest"])
    if manifest_path.is_file():
        model_manifest = _read_json(manifest_path)
        raw_models = model_manifest.get("models") or {}
        if set(raw_models) != set(TRAIN_MODELS):
            errors.append("模型清单必须且只能登记 hoptimus0、hoptimus1、phikonv2")
        for name in TRAIN_MODELS:
            entry = raw_models.get(name) or {}
            snapshot = entry.get("snapshot_path")
            revision = entry.get("revision")
            root = Path(snapshot) if snapshot else None
            expected_files = []
            if root and entry.get("checkpoint_filename"):
                expected_files.append(root / entry["checkpoint_filename"])
            if root and entry.get("config_filename"):
                expected_files.append(root / entry["config_filename"])
            pinned = isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
            snapshot_valid = bool(
                root and pinned and root.name == revision and root.parent.name == "snapshots"
                and expected_files and all(p.is_file() and p.stat().st_size > 0 for p in expected_files)
            )
            models[name] = {
                "snapshot_path": snapshot,
                "snapshot_exists": bool(root and root.is_dir()),
                "revision": revision,
                "feature_dim": entry.get("feature_dim"),
                "expected_feature_dim": MODEL_DIMS[name],
                "loader_type": entry.get("loader_type"),
                "output_mode": entry.get("output_mode"),
                "normalization_profile": entry.get("normalization_profile"),
                "preprocess_profile": preprocess_profile(name),
                "required_files": [str(p) for p in expected_files],
                "status": "snapshot_files_present" if snapshot_valid and entry.get("feature_dim") == MODEL_DIMS[name] else "unverified",
                "strict_model_load_status": "not_run_by_input_check",
            }
        if any(item["status"] != "snapshot_files_present" for item in models.values()) or len(models) != 3:
            warnings.append("新编码器快照、版本或维度尚未现场核验；严格加载只在下载预检或特征准备时执行")
    geometry_path = Path(config["inputs"]["slide_geometry"])
    if geometry_path.is_file():
        with geometry_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if any(not str(row.get("patch_coverage_size") or "").strip() for row in rows):
            warnings.append("patch_coverage_size 缺失；official_expected_mpp=0.5 与项目物理尺度关系仍为 unverified")
    baseline_status = None
    try:
        from baseline_reference import load_baseline_reference
        baseline = load_baseline_reference(config["inputs"]["baseline_reference_manifest"], require_prediction_files=False)
        baseline_status = {
            "status": "ok",
            "n_tasks": baseline["n_tasks"],
            "accepted_result_id": baseline["accepted_result_id"],
            "do_not_retrain": True,
            "embedded_metrics_valid": baseline["embedded_metrics_valid"],
            "models": list(BASELINE_MODELS),
        }
        missing_source = []
        missing_pred = []
        for task in baseline["tasks"]:
            for name in ("task_record", "internal_metrics", "formal_checkpoint_metadata"):
                if not task["files"][name]["exists"]:
                    missing_source.append(f"{task['task_id']}:{name}")
            for name in ("internal_predictions", "external_predictions"):
                if not task["files"][name]["exists"]:
                    missing_pred.append(f"{task['task_id']}:{name}")
        if missing_source:
            warnings.append("基线原始证据路径只作来源追溯且在当前机器不可用；运行与分析使用包内已核验的 accepted_*_metrics，不重训基线")
            baseline_status["source_evidence_unavailable"] = missing_source
        if missing_pred:
            warnings.append("基线预测 NPZ 未出现在本机回传中；分析将使用已接纳分析 JSON 中的冻结指标，不重训基线")
            baseline_status["prediction_npz_missing"] = missing_pred
    except Exception as exc:
        errors.append(f"基线清单核验失败: {type(exc).__name__}: {exc}")
        baseline_status = {"status": "error", "error": str(exc)}
    planned = [task.task_id for task in plan_train_tasks()]
    if len(planned) != 9 or len(set(planned)) != 9:
        errors.append("新任务计划不是 3 模型 × 3 种子")
    status = "error" if errors else ("warn" if warnings else "pass")
    return {
        "status": status,
        "packaged_inputs": packaged,
        "server_assets": server,
        "models": models,
        "baseline": baseline_status,
        "planned_train_tasks": planned,
        "warnings": warnings,
        "errors": errors,
        "source_group_is_physical_slide": False,
        "external_labels_used_for_selection": False,
        "hashes_used": False,
    }


def _record_stage(run_dir: Path, action: str, result: dict) -> dict:
    payload = {"action": action, "recorded_at": utc_now(), **result}
    write_json(run_dir / "action_result.json", payload)
    return payload


def execute_action(
    config: dict,
    *,
    action: str,
    run_dir: Path,
    weights_dir: Path,
    device: str,
    models: Sequence[str] | None = None,
) -> dict:
    if action not in ACTIONS:
        raise ConfigError(f"未知 action={action}；本批没有 search/freeze")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    write_json(run_dir / "config_snapshot.json", {k: v for k, v in config.items() if not k.startswith("_")})
    if action == "check-environment":
        from environment_check import build_environment_report
        return _record_stage(run_dir, action, build_environment_report(config))
    if action == "check-inputs":
        return _record_stage(run_dir, action, check_inputs(config))
    if action == "prepare-features":
        from feature_extract import prepare_feature_caches
        registry = prepare_feature_caches(config, device=device, models=models)
        write_json(run_dir / "feature_caches.json", registry)
        return _record_stage(run_dir, action, {
            "status": registry["status"],
            "cache_count": len(registry["selected_entries"]),
            "overall_cache_status": registry["overall_status"],
            "selected_models": registry["selected_models"],
            "failures": [item for item in registry["selected_entries"] if item["status"] == "failed"],
            "registry_path": registry["registry_path"],
        })
    if action == "train-spatial":
        from stage_runtime import execute_training_batch
        return _record_stage(run_dir, action, execute_training_batch(config, plan_train_tasks(models), run_dir=run_dir, weights_dir=weights_dir, device=device))
    if action == "external-eval":
        from external_eval import evaluate_external
        return _record_stage(run_dir, action, evaluate_external(config, run_dir=run_dir, device=device, models=models))
    from analyze import analyze_local
    return _record_stage(run_dir, action, analyze_local(config, run_dir=run_dir))


def action_exit_code(action: str, status: str | None) -> int:
    if status in {"pass", "warn", "ok", "completed"}:
        return 0
    if action == "analyze-local" and status == "partial":
        return 0
    if status == "partial":
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase2 second-batch spatial backbone ablation")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--action", choices=ACTIONS, default="check-inputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=TRAIN_MODELS)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = execute_action(
            config,
            action=args.action,
            run_dir=args.run_dir.resolve(),
            weights_dir=args.weights_dir.resolve(),
            device=args.device,
            models=args.models,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return action_exit_code(args.action, result.get("status"))
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
