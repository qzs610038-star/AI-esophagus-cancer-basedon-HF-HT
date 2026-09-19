"""Train the nine frozen spatial-11 heads. No search or freeze stage."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from config import SEEDS, TRAIN_MODELS, model_config
from errors import ConfigError, IdentityMismatchError
from run_io import utc_now, write_json


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _batch(run_dir: Path) -> Path:
    return run_dir.parent


def _task_path(batch: Path, task_id: str) -> Path:
    return batch / "tasks" / f"{task_id}.json"


def _latest_success(record: dict | None) -> dict | None:
    return next((a for a in reversed((record or {}).get("attempts", [])) if a.get("status") == "completed"), None)


def _cache_table(config: dict, *, model_name: str, split: str):
    from data import attach_labels, load_pathway_names, load_slide_mapping, point_table_from_cache_arrays
    from feature_cache import load_feature_cache
    from feature_extract import cache_contract, cache_path
    from input_data import common_manifest_path, identities, load_common_manifest, select_dataset_rows
    from model_adapters import load_model_specs
    from transforms import GEOMETRY_PROTOCOL

    manifest_path = common_manifest_path(config)
    rows_all = load_common_manifest(manifest_path, expected_counts=config["data"]["expected_counts"])
    rows = select_dataset_rows(rows_all, "all", protocol=GEOMETRY_PROTOCOL)
    spec = load_model_specs(config["inputs"]["model_manifest"], require_local_files=False)[model_name]
    contract = cache_contract(config, spec, dataset="all", manifest_path=manifest_path)
    cache = load_feature_cache(
        cache_path(config, model=model_name, dataset="all"),
        expected_identities=identities(rows),
        expected_dim=spec.feature_dim,
        expected_metadata=contract,
    )
    mapping, status = load_slide_mapping(config["inputs"]["slide_mapping"])
    if status != "verified":
        raise IdentityMismatchError("来源组映射未核实")
    for row in rows:
        if mapping.get(row.patient_id) != row.source_group:
            raise IdentityMismatchError(f"来源组映射不一致: {row.identity_key}")
    chosen = [(i, row) for i, row in enumerate(rows) if row.split == split]
    if not chosen:
        raise IdentityMismatchError(f"缓存中没有 {split} 数据")
    indices = [i for i, _ in chosen]
    chosen_rows = [row for _, row in chosen]
    expected_patients = ({str(config["data"]["external_patient"])} if split == "external_test"
                         else set(map(str, config["data"]["development_patients"])))
    actual_patients = {row.patient_id for row in chosen_rows}
    if actual_patients != expected_patients:
        raise IdentityMismatchError(f"{split} 患者集合不符合冻结划分: {sorted(actual_patients)}")
    names = load_pathway_names(config["inputs"]["zscore_manifest"])
    table = point_table_from_cache_arrays(
        features=np.asarray(cache["features"][indices], dtype=np.float32),
        patient_ids=[r.patient_id for r in chosen_rows],
        slide_ids=[r.source_group for r in chosen_rows],
        spot_ids=[r.spot_id for r in chosen_rows],
        splits=[r.split for r in chosen_rows],
        x=[r.x for r in chosen_rows], y=[r.y for r in chosen_rows],
        pathway_names=names, source=str(cache["cache_dir"]),
    )
    if split != "external_test":
        table = attach_labels(table, config["paths"]["labels_root"], names, train_mpp_id=config["data"]["mpp_id"])
    return table, cache["cache_dir"], names


def _development_table(config: dict, *, model_name: str):
    from data import make_point_table
    train, path_train, names = _cache_table(config, model_name=model_name, split="train")
    val, path_val, _ = _cache_table(config, model_name=model_name, split="internal_val")
    if path_train != path_val:
        raise IdentityMismatchError("训练与验证集使用不同特征缓存")
    parts = (train, val)
    return make_point_table(
        patient_ids=[i.patient_id for p in parts for i in p.identities],
        slide_ids=[i.slide_id for p in parts for i in p.identities],
        spot_ids=[i.spot_id for p in parts for i in p.identities],
        splits=[str(s) for p in parts for s in p.split],
        x=np.concatenate([p.x for p in parts]), y=np.concatenate([p.y for p in parts]),
        features=np.concatenate([p.features for p in parts]),
        labels_z=np.concatenate([p.labels_z for p in parts]),
        slide_status="verified", pathway_names=names, source=path_train,
    ), path_train, names


def _run_task(config: dict, task, *, run_dir: Path, weights_dir: Path, device: str, table_cache: dict) -> dict:
    from train import train_arm
    batch = _batch(run_dir)
    path = _task_path(batch, task.task_id)
    record = _read(path) if path.is_file() else {"task_id": task.task_id, "spec": vars(task), "attempts": []}
    success = _latest_success(record)
    if success is not None:
        return {"task_id": task.task_id, "status": "reused_complete", "attempt": success}
    if task.arm != "spatial" or task.recipe != "frozen_spatial":
        raise ConfigError(f"本批只训练冻结空间臂: {task.task_id}")
    cfg = model_config(config, task.model)
    cfg["task_id"] = task.task_id
    cfg["task_stage"] = task.stage
    cfg["task_recipe"] = task.recipe
    if task.model not in table_cache:
        table_cache[task.model] = _development_table(config, model_name=task.model)
    table, cache_dir, _ = table_cache[task.model]
    attempt_no = len(record["attempts"]) + 1
    attempt_id = f"{task.task_id}__attempt{attempt_no:02d}"
    task_run = run_dir / "raw" / attempt_id
    task_weights = weights_dir / attempt_id
    if task_run.exists() or task_weights.exists():
        raise ConfigError(f"尝试目录已存在，拒绝覆盖: {attempt_id}")
    task_run.mkdir(parents=True, exist_ok=False)
    write_json(task_run / "effective_config.json", {k: v for k, v in cfg.items() if not k.startswith("_")})
    attempt = {
        "attempt_id": attempt_id, "status": "running", "started_at": utc_now(),
        "run_dir": str(task_run.resolve()), "weights_dir": str(task_weights.resolve()),
        "cache_dir": cache_dir,
    }
    record["attempts"].append(attempt)
    write_json(path, record)
    try:
        result = train_arm(cfg, task.arm, task.seed, task_run, checkpoint_dir=task_weights, point_table=table, device=device)
        attempt.update(status="completed" if result["status"] == "completed" else "failed", result=result, ended_at=utc_now())
    except Exception as exc:
        attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), ended_at=utc_now())
    write_json(path, record)
    return {"task_id": task.task_id, "status": attempt["status"], "attempt": attempt}


def _registries(config: dict, run_dir: Path, task_results: list[dict]) -> None:
    write_json(run_dir / "feature_caches.json", {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "return_policy": "server_feature_caches_excluded_from_local_result_copy",
        "entries": [{"task_id": r["task_id"], "cache_dir": r["attempt"].get("cache_dir"), "status": r["status"]} for r in task_results],
    })
    weights = []
    for r in task_results:
        result = r["attempt"].get("result") or {}
        weights.append({
            "task_id": r["task_id"], "status": r["status"],
            "weight_dir": r["attempt"].get("weights_dir"),
            "warmup_checkpoint": result.get("warmup_checkpoint"),
            "warmup_status": "not_applicable_formal_start_epoch_1" if int((config.get("selection") or {}).get("formal_start_epoch", 1)) == 1 else ("available" if result.get("warmup_checkpoint") else "not_produced"),
            "formal_checkpoint": result.get("formal_checkpoint"),
            "formal_status": "available" if result.get("formal_checkpoint") else "not_produced",
            "last_checkpoint": result.get("last_checkpoint"),
            "last_status": "available" if result.get("last_checkpoint") else "not_produced",
        })
    write_json(run_dir / "model_weights.json", {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "return_policy": "server_weights_excluded_from_local_result_copy",
        "entries": weights,
    })


def _record_preflight_failure(task, *, run_dir: Path, weights_dir: Path, exc: Exception) -> dict:
    batch = _batch(run_dir)
    path = _task_path(batch, task.task_id)
    record = _read(path) if path.is_file() else {"task_id": task.task_id, "spec": vars(task), "attempts": []}
    attempt_no = len(record["attempts"]) + 1
    attempt_id = f"{task.task_id}__attempt{attempt_no:02d}"
    now = utc_now()
    attempt = {
        "attempt_id": attempt_id,
        "status": "failed",
        "started_at": now,
        "ended_at": now,
        "run_dir": str((run_dir / "raw" / attempt_id).resolve()),
        "weights_dir": str((weights_dir / attempt_id).resolve()),
        "cache_dir": None,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "failure_stage": "task_preflight_or_execution",
    }
    record["attempts"].append(attempt)
    write_json(path, record)
    return {"task_id": task.task_id, "status": "failed", "attempt": attempt}


def execute_training_batch(config: dict, tasks: list, *, run_dir: Path, weights_dir: Path, device: str) -> dict:
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ConfigError("train-spatial 任务必须非空且无重复")
    if any(t.arm != "spatial" or t.recipe != "frozen_spatial" or t.model not in set(TRAIN_MODELS) for t in tasks):
        raise ConfigError("train-spatial 只能训练三个新编码器的空间臂")
    selected_models = tuple(model for model in TRAIN_MODELS if any(t.model == model for t in tasks))
    actual_scope = {(t.model, int(t.seed)) for t in tasks}
    expected_scope = {(model, int(seed)) for model in selected_models for seed in SEEDS}
    if actual_scope != expected_scope or len(tasks) != len(expected_scope):
        raise ConfigError("train-spatial 的每个选定模型必须恰好覆盖种子 45/46/47")
    table_cache: dict = {}
    results = []
    for task in tasks:
        try:
            result = _run_task(config, task, run_dir=run_dir, weights_dir=weights_dir, device=device, table_cache=table_cache)
        except Exception as exc:
            result = _record_preflight_failure(task, run_dir=run_dir, weights_dir=weights_dir, exc=exc)
        results.append(result)
    _registries(config, run_dir, results)
    completed = sum(r["status"] in ("completed", "reused_complete") for r in results)
    failed = [r for r in results if r["status"] not in ("completed", "reused_complete")]
    status = "completed" if completed == len(tasks) else ("partial" if completed else "failed")
    return {
        "status": status,
        "planned": len(tasks),
        "planned_full_comparison": len(TRAIN_MODELS) * len(SEEDS),
        "selected_models": list(selected_models),
        "completed": completed,
        "failed": len(failed),
        "failed_task_ids": [r["task_id"] for r in failed],
        "task_records": [str(_task_path(_batch(run_dir), t.task_id)) for t in tasks],
        "selected_scope_complete": completed == len(tasks),
        "complete_comparison": len(tasks) == len(TRAIN_MODELS) * len(SEEDS) and completed == len(tasks),
        "note": "缺失或失败任务已显式列出，不得用其他种子补位。",
    }
