"""Explicit experiment actions. Import heavy encoders only for requested work."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from config import MODEL_DIMS, SUPPORTED_MODELS, load_config, trial_config
from errors import ConfigError
from run_io import utc_now, write_json


ACTIONS = (
    "check-inputs", "prepare-features", "paired-view", "paired-recipe",
    "search-point", "search-spatial", "extend-point", "extend-spatial",
    "freeze", "final-ablation", "external-eval", "analyze-local", "export-phase3",
)
FULL_PROTOCOL = "full_fov_224_bicubic_v1"
LEGACY_PROTOCOL = "legacy_crop_0875_v1"
HISTORICAL_RECIPE = {
    "training.optimizer": "AdamW", "training.learning_rate": 3e-4,
    "training.batch_size": 256, "training.weight_decay": 1e-4,
    "training.lr_schedule": "constant", "training.b_lr_multiplier": 1.0,
}
CANDIDATE_RECIPE = {
    "training.optimizer": "Adam", "training.learning_rate": 1e-4,
    "training.batch_size": 32, "training.weight_decay": 0.0,
    "training.lr_schedule": "constant", "training.b_lr_multiplier": 1.0,
}


@dataclass(frozen=True)
class TaskSpec:
    stage: str
    model: str
    protocol: str
    arm: str
    seed: int
    recipe: str
    candidate_id: str | None = None

    @property
    def task_id(self) -> str:
        fields = (self.stage, self.model, self.protocol, self.arm, str(self.seed), self.recipe)
        return "__".join(fields) + ("__" + self.candidate_id if self.candidate_id else "")


def plan_paired_tasks(stage: str) -> list[TaskSpec]:
    if stage not in ("paired-view", "paired-recipe"):
        raise ValueError(stage)
    recipe = "historical" if stage == "paired-view" else "candidate"
    return [
        TaskSpec(stage, "uni2h", protocol, arm, seed, recipe)
        for seed in (42, 43, 44)
        for protocol in (LEGACY_PROTOCOL, FULL_PROTOCOL)
        for arm in ("point", "spatial")
    ]


def plan_final_tasks() -> list[TaskSpec]:
    return [
        TaskSpec("final-ablation", model, FULL_PROTOCOL, arm, seed,
                 "frozen_point" if arm == "point" else "frozen_spatial")
        for model in SUPPORTED_MODELS
        for seed in (45, 46, 47)
        for arm in ("point", "spatial", "no_b")
    ]


def planned_external_keys() -> list[str]:
    return [task.task_id for stage in ("paired-view", "paired-recipe") for task in plan_paired_tasks(stage)] + [
        task.task_id for task in plan_final_tasks()
    ]


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _status(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "exists": path.exists(), "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing"}


def check_inputs(config: dict) -> dict:
    """Inspect packaged and server assets without importing or running a model."""
    packaged = {name: _status(path) for name, path in config["inputs"].items()}
    server = {name: _status(config["paths"][name]) for name in ("image_root", "labels_root", "feature_caches_root")}
    manifest_path = Path(config["inputs"]["model_manifest"])
    models: dict[str, dict] = {}
    if manifest_path.is_file():
        model_manifest = _read_json(manifest_path)
        for name in SUPPORTED_MODELS:
            entry = (model_manifest.get("models") or {}).get(name) or {}
            snapshot = entry.get("snapshot_path")
            revision = entry.get("revision")
            root = Path(snapshot) if snapshot else None
            expected_files = [root / entry["checkpoint_filename"]] if root and entry.get("checkpoint_filename") else []
            if root and entry.get("config_filename"):
                expected_files.append(root / entry["config_filename"])
            pinned = isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
            layout_valid = entry.get("cls_dim") == MODEL_DIMS[name]
            snapshot_valid = bool(root and pinned and root.name == revision and root.parent.name == "snapshots"
                                  and expected_files and all(p.is_file() and p.stat().st_size > 0 for p in expected_files))
            models[name] = {
                "snapshot_path": snapshot,
                "snapshot_exists": bool(root and root.is_dir()),
                "revision": revision,
                "cls_dim": entry.get("cls_dim"),
                "expected_cls_dim": MODEL_DIMS[name],
                "required_files": [str(p) for p in expected_files],
                "status": "snapshot_files_present" if snapshot_valid and layout_valid else "unverified",
                "strict_model_load_status": "not_run_by_input_check",
            }
    geometry_warning = None
    geometry_path = Path(config["inputs"]["slide_geometry"])
    if geometry_path.is_file():
        with geometry_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if any(not str(row.get("patch_coverage_size") or "").strip() for row in rows):
            geometry_warning = "patch_coverage_size 缺失；来源组和观测网格不证明原图物理支持域或划分无重叠"
    warnings = []
    if not all(item["exists"] for item in packaged.values()):
        warnings.append("包内小型输入不完整")
    if not all(item["exists"] for item in server.values()):
        warnings.append("服务器图像、标签或缓存根目录尚未现场核验")
    if not all(item["status"] == "snapshot_files_present" for item in models.values()) or len(models) != 3:
        warnings.append("模型快照、版本或维度尚未现场核验")
    else:
        warnings.append("模型文件路径已核对；完整权重严格加载在显式特征准备时核验")
    if geometry_warning:
        warnings.append(geometry_warning)
    return {
        "status": "pass" if not warnings else "warn", "packaged_inputs": packaged,
        "server_assets": server, "models": models, "warnings": warnings,
        "source_group_is_physical_slide": False,
        "external_labels_used_for_selection": False,
    }


def _batch_dir(run_dir: Path) -> Path:
    return run_dir.parent


def _task_record_path(batch_dir: Path, task_id: str) -> Path:
    return batch_dir / "tasks" / f"{task_id}.json"


def _all_task_records(batch_dir: Path) -> dict[str, dict]:
    task_root = batch_dir / "tasks"
    if not task_root.is_dir():
        return {}
    records = {}
    for path in sorted(task_root.glob("*.json")):
        record = _read_json(path)
        task_id = str(record.get("task_id") or path.stem)
        if task_id in records:
            raise ConfigError(f"重复任务记录 {task_id}")
        records[task_id] = record
    return records


def _record_stage(run_dir: Path, action: str, result: dict) -> dict:
    payload = {"action": action, "recorded_at": utc_now(), **result}
    write_json(run_dir / "action_result.json", payload)
    return payload


def execute_action(config: dict, *, action: str, run_dir: Path, weights_dir: Path, device: str) -> dict:
    if action not in ACTIONS:
        raise ConfigError(f"未知 action={action}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    write_json(run_dir / "config_snapshot.json", {k: v for k, v in config.items() if not k.startswith("_")})
    if action == "check-inputs":
        return _record_stage(run_dir, action, check_inputs(config))
    if action == "prepare-features":
        return _record_stage(run_dir, action, _prepare_features(config, run_dir=run_dir, device=device))
    if action in ("paired-view", "paired-recipe"):
        return _record_stage(run_dir, action, _execute_training_batch(config, plan_paired_tasks(action), run_dir=run_dir, weights_dir=weights_dir, device=device))
    if action in ("search-point", "search-spatial", "extend-point", "extend-spatial"):
        return _record_stage(run_dir, action, _execute_search(config, action=action, run_dir=run_dir, weights_dir=weights_dir, device=device))
    if action == "freeze":
        return _record_stage(run_dir, action, _freeze_search(config, run_dir=run_dir))
    if action == "final-ablation":
        return _record_stage(run_dir, action, _execute_training_batch(config, plan_final_tasks(), run_dir=run_dir, weights_dir=weights_dir, device=device))
    if action == "external-eval":
        return _record_stage(run_dir, action, _evaluate_external(config, run_dir=run_dir, device=device))
    if action == "analyze-local":
        from stage_runtime import analyze_local
        return _record_stage(run_dir, action, analyze_local(run_dir=run_dir))
    return _record_stage(run_dir, action, _export_phase3(config, run_dir=run_dir, device=device))


def _prepare_features(config: dict, *, run_dir: Path, device: str) -> dict:
    from feature_extract import prepare_feature_caches
    registry = prepare_feature_caches(config, device=device, combinations=[
        ("uni2h", LEGACY_PROTOCOL), ("uni2h", FULL_PROTOCOL),
        ("uni", FULL_PROTOCOL), ("virchow2", FULL_PROTOCOL),
    ])
    write_json(run_dir / "feature_caches.json", registry)
    return {"status": "completed", "cache_count": len(registry["entries"]), "registry_path": registry["registry_path"]}


def _execute_training_batch(config: dict, tasks: list[TaskSpec], *, run_dir: Path, weights_dir: Path, device: str) -> dict:
    from stage_runtime import execute_training_batch
    return execute_training_batch(config, tasks, run_dir=run_dir, weights_dir=weights_dir, device=device)


def _execute_search(config: dict, *, action: str, run_dir: Path, weights_dir: Path, device: str) -> dict:
    from stage_runtime import execute_search
    return execute_search(config, action=action, run_dir=run_dir, weights_dir=weights_dir, device=device)


def _freeze_search(config: dict, *, run_dir: Path) -> dict:
    from stage_runtime import freeze_search
    return freeze_search(config, run_dir=run_dir)


def _evaluate_external(config: dict, *, run_dir: Path, device: str) -> dict:
    from stage_runtime import evaluate_external
    return evaluate_external(config, run_dir=run_dir, device=device)


def _export_phase3(config: dict, *, run_dir: Path, device: str) -> dict:
    from stage_runtime import export_phase3
    return export_phase3(config, run_dir=run_dir, device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase2 full-view HPO experiment package")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--action", choices=ACTIONS, default="check-inputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = execute_action(config, action=args.action, run_dir=args.run_dir.resolve(),
                                weights_dir=args.weights_dir.resolve(), device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("pass", "warn", "completed") else 1
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
