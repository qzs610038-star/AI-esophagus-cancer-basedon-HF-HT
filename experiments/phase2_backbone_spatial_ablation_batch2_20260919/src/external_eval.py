"""External XZY inference only after a formal checkpoint exists."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from config import model_config
from errors import ConfigError
from run_io import write_json
from stage_runtime import _batch, _cache_table, _latest_success, _read, _task_path


def _formal_attempt(batch: Path, task) -> tuple[dict, dict]:
    path = _task_path(batch, task.task_id)
    if not path.is_file():
        raise ConfigError(f"缺少训练任务记录，不能外部评估: {task.task_id}")
    attempt = _latest_success(_read(path))
    if attempt is None:
        raise ConfigError(f"任务没有成功的正式检查点: {task.task_id}")
    result = attempt["result"]
    formal = result.get("formal_checkpoint")
    if not formal or not Path(formal).is_file():
        raise ConfigError(f"正式检查点缺失，拒绝外部评估: {task.task_id}: {formal}")
    return attempt, result


def _external_targets(table, config: dict, names: list[str]):
    from data import attach_labels
    labeled = table.subset(range(len(table)))
    labeled.split[:] = "external"
    labeled = attach_labels(labeled, config["paths"]["labels_root"], names, train_mpp_id=config["data"]["mpp_id"])
    return labeled.labels_z


def evaluate_external(config: dict, *, run_dir: Path, device: str, models: Sequence[str] | None = None) -> dict:
    from data import load_normalization
    from dispatch import plan_train_tasks
    from predict import load_formal_model, predict_table, save_predictions

    batch = _batch(run_dir)
    tasks = plan_train_tasks(models)
    table_cache: dict = {}
    outputs = []
    failures = []
    for task in tasks:
        try:
            _, result = _formal_attempt(batch, task)
            cfg = model_config(config, task.model)
            if task.model not in table_cache:
                table_cache[task.model] = _cache_table(config, model_name=task.model, split="external_test")
            table, cache_dir, names = table_cache[task.model]
            model, metadata = load_formal_model(cfg, result["formal_checkpoint"], arm=task.arm, seed=task.seed, device=device)
            metadata.update({
                "task_id": task.task_id, "stage": task.stage, "model": task.model,
                "protocol": task.protocol, "recipe": task.recipe, "split": "external_test",
                "selection_used_external": False,
            })
            pred = predict_table(cfg, arm=task.arm, model=model, table=table, split="external_test", device=device)
            norm = load_normalization(config["inputs"]["normalization"], names)
            target = _external_targets(table, config, names)
            output = save_predictions(
                run_dir / "raw" / f"{task.task_id}.npz",
                pred_z=pred, target_z=target, mean=norm.mean, std=norm.std,
                pathway_names=names, table=table, checkpoint_metadata=metadata,
            )
            outputs.append({
                "task_id": task.task_id,
                "prediction": str(output.resolve()),
                "formal_checkpoint": result["formal_checkpoint"],
                "cache_dir": cache_dir,
                "n_points": len(table),
                "selection_used_external": False,
            })
        except Exception as exc:
            failures.append({"task_id": task.task_id, "model": task.model, "seed": task.seed, "error": f"{type(exc).__name__}: {exc}"})
    status = "completed" if len(outputs) == len(tasks) else ("partial" if outputs else "failed")
    write_json(run_dir / "external_predictions.json", {
        "status": status,
        "count": len(outputs),
        "planned": len(tasks),
        "entries": outputs,
        "failures": failures,
    })
    write_json(run_dir / "model_weights.json", {
        "schema_version": "1.0",
        "return_policy": "server_weights_excluded_from_local_result_copy",
        "entries": [{
            "task_id": e["task_id"],
            "formal_checkpoint": e["formal_checkpoint"],
            "weight_dir": str(Path(e["formal_checkpoint"]).parent),
            "formal_status": "available",
        } for e in outputs],
    })
    write_json(run_dir / "feature_caches.json", {
        "schema_version": "1.0",
        "return_policy": "server_feature_caches_excluded_from_local_result_copy",
        "entries": [{"task_id": e["task_id"], "cache_dir": e["cache_dir"]} for e in outputs],
    })
    return {
        "status": status,
        "formal_prediction_count": len(outputs),
        "planned": len(tasks),
        "failed": len(failures),
        "failed_task_ids": [item["task_id"] for item in failures],
        "selected_scope_complete": len(outputs) == len(tasks),
        "complete_comparison": len(tasks) == 9 and len(outputs) == 9,
        "prediction_manifest": str(run_dir / "external_predictions.json"),
        "selection_used_external": False,
    }
