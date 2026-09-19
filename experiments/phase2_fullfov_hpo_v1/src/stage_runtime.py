"""Persistent, explicit Phase2 stage execution. No action runs at import time."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from config import trial_config
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


def _cache_table(config: dict, *, model_name: str, protocol: str, split: str):
    """Load only a COMPLETE, identity and source-group aligned CLS cache."""
    from adapters import load_model_specs
    from data import attach_labels, load_pathway_names, load_slide_mapping, point_table_from_cache_arrays
    from feature_cache import load_feature_cache
    from feature_extract import cache_contract, cache_path
    from input_data import common_manifest_path, identities, load_common_manifest, select_dataset_rows

    manifest_path = common_manifest_path(config)
    rows_all = load_common_manifest(manifest_path, expected_counts=config["data"]["expected_counts"])
    rows = select_dataset_rows(rows_all, "all", protocol=protocol)
    specs = load_model_specs(config["inputs"]["model_manifest"], require_local_files=False)
    spec = specs[model_name]
    contract = cache_contract(config, spec, protocol=protocol, dataset="all", manifest_path=manifest_path)
    cache = load_feature_cache(
        cache_path(config, model=model_name, protocol=protocol, dataset="all"),
        expected_identities=identities(rows), expected_dim=spec.layout.feature_dim,
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
        features=np.asarray(cache["cls"][indices], dtype=np.float32),
        patient_ids=[r.patient_id for r in chosen_rows],
        slide_ids=[r.source_group for r in chosen_rows],
        spot_ids=[r.spot_id for r in chosen_rows],
        splits=[r.split for r in chosen_rows],
        x=[r.x for r in chosen_rows], y=[r.y for r in chosen_rows],
        pathway_names=names, source=str(cache["cache_dir"]),
    )
    if split == "development":
        # The common manifest labels rows train/internal_val, so this branch is unused.
        raise ConfigError("使用显式 train/internal_val 集合")
    if split != "external_test":
        table = attach_labels(table, config["paths"]["labels_root"], names, train_mpp_id=config["data"]["mpp_id"])
    return table, cache["cache_dir"], names


def _development_table(config: dict, *, model_name: str, protocol: str):
    from data import make_point_table
    train, path_train, names = _cache_table(config, model_name=model_name, protocol=protocol, split="train")
    val, path_val, _ = _cache_table(config, model_name=model_name, protocol=protocol, split="internal_val")
    if path_train != path_val:
        raise IdentityMismatchError("训练与验证集使用不同特征缓存")
    parts = (train, val)
    table = make_point_table(
        patient_ids=[i.patient_id for p in parts for i in p.identities],
        slide_ids=[i.slide_id for p in parts for i in p.identities],
        spot_ids=[i.spot_id for p in parts for i in p.identities],
        splits=[str(s) for p in parts for s in p.split],
        x=np.concatenate([p.x for p in parts]), y=np.concatenate([p.y for p in parts]),
        features=np.concatenate([p.features for p in parts]),
        labels_z=np.concatenate([p.labels_z for p in parts]),
        slide_status="verified", pathway_names=names, source=path_train,
    )
    return table, path_train, names


def _frozen(batch: Path) -> dict:
    path = batch / "frozen_selection.json"
    if not path.is_file():
        raise ConfigError("需先显式执行 freeze，才可运行最终消融、外部评估或导出")
    return _read(path)


def _task_parameters(task, batch: Path) -> tuple[dict, str]:
    from dispatch import CANDIDATE_RECIPE, HISTORICAL_RECIPE
    if task.recipe == "historical":
        return HISTORICAL_RECIPE, "paired"
    if task.recipe == "candidate":
        return CANDIDATE_RECIPE, "paired"
    if task.recipe.startswith("frozen_"):
        frozen = _frozen(batch)
        kind = "point" if task.arm == "point" else "spatial"
        return dict(frozen[kind]["parameters"]), "search"
    if task.candidate_id:
        stage = "point" if task.arm == "point" else "spatial"
        plan_path = _plan_path(batch, stage, extension=task.stage.startswith("extend-"))
        if not plan_path.is_file():
            raise ConfigError(f"搜索计划不存在: {plan_path}")
        plan = _read(plan_path)
        matches = [c for c in plan["candidates"] if c["candidate_id"] == task.candidate_id]
        if len(matches) != 1:
            raise ConfigError(f"计划中没有候选 {task.candidate_id}")
        return dict(matches[0]["parameters"]), "search"
    raise ConfigError(f"未知任务配方 {task.recipe}")


def _run_task(config: dict, task, *, run_dir: Path, weights_dir: Path, device: str, table_cache: dict) -> dict:
    from train import train_arm
    batch = _batch(run_dir)
    path = _task_path(batch, task.task_id)
    record = _read(path) if path.is_file() else {"task_id": task.task_id, "spec": vars(task), "attempts": []}
    success = _latest_success(record)
    if success is not None:
        return {"task_id": task.task_id, "status": "reused_complete", "attempt": success}
    params, selection_mode = _task_parameters(task, batch)
    cfg = trial_config(config, model_name=task.model, parameters=params, selection_mode=selection_mode)
    cfg["task_id"] = task.task_id
    cfg["task_stage"] = task.stage
    cfg["input_protocol"] = task.protocol
    cfg["task_recipe"] = task.recipe
    cache_key = (task.model, task.protocol)
    if cache_key not in table_cache:
        table_cache[cache_key] = _development_table(config, model_name=task.model, protocol=task.protocol)
    table, cache_dir, _ = table_cache[cache_key]
    attempt_no = len(record["attempts"]) + 1
    attempt_id = f"{task.task_id}__attempt{attempt_no:02d}"
    task_run = run_dir / "raw" / attempt_id
    task_weights = weights_dir / attempt_id
    if task_run.exists() or task_weights.exists():
        raise ConfigError(f"尝试目录已存在，拒绝覆盖: {attempt_id}")
    task_run.mkdir(parents=True, exist_ok=False)
    write_json(task_run / "effective_config.json", {k: v for k, v in cfg.items() if not k.startswith("_")})
    attempt = {"attempt_id": attempt_id, "status": "running", "started_at": utc_now(),
               "run_dir": str(task_run.resolve()), "weights_dir": str(task_weights.resolve()),
               "cache_dir": cache_dir, "selection_mode": selection_mode}
    record["attempts"].append(attempt)
    write_json(path, record)
    try:
        result = train_arm(cfg, task.arm, task.seed, task_run, checkpoint_dir=task_weights,
                           point_table=table, device=device)
        attempt.update(status="completed" if result["status"] == "completed" else "failed",
                       result=result, ended_at=utc_now())
    except Exception as exc:
        attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(), ended_at=utc_now())
    write_json(path, record)
    return {"task_id": task.task_id, "status": attempt["status"], "attempt": attempt}


def _registries(config: dict, run_dir: Path, task_results: list[dict]) -> None:
    cache_entries = [{"task_id": r["task_id"], "cache_dir": r["attempt"].get("cache_dir"),
                      "status": r["status"]} for r in task_results]
    write_json(run_dir / "feature_caches.json", {"entries": cache_entries,
        "common_identity_manifest": str(Path(config["paths"]["feature_caches_root"]) / config["experiment_id"] / "common_identity_v1" / "common_identity_manifest.csv")})
    weights = []
    for r in task_results:
        result = r["attempt"].get("result") or {}
        weights.append({"task_id": r["task_id"], "status": r["status"],
                        "weight_dir": r["attempt"].get("weights_dir"),
                        "warmup_checkpoint": result.get("warmup_checkpoint"),
                        "warmup_status": "available" if result.get("warmup_checkpoint") else "not_applicable_or_not_produced",
                        "formal_checkpoint": result.get("formal_checkpoint"),
                        "formal_status": "available" if result.get("formal_checkpoint") else "not_produced",
                        "last_checkpoint": result.get("last_checkpoint"),
                        "last_status": "available" if result.get("last_checkpoint") else "not_produced"})
    write_json(run_dir / "model_weights.json", {"entries": weights})


def execute_training_batch(config: dict, tasks: list, *, run_dir: Path, weights_dir: Path, device: str) -> dict:
    if tasks and tasks[0].stage == "paired-recipe":
        from dispatch import plan_paired_tasks
        previous = [_task_path(_batch(run_dir), t.task_id) for t in plan_paired_tasks("paired-view")]
        incomplete = [p for p in previous if not p.is_file() or _latest_success(_read(p)) is None]
        if incomplete:
            raise ConfigError(f"历史配方配对尚未完成: 缺少/失败 {len(incomplete)} 次")
    if any(t.stage == "final-ablation" for t in tasks):
        _frozen(_batch(run_dir))
    table_cache: dict = {}
    results = []
    for task in tasks:
        results.append(_run_task(config, task, run_dir=run_dir, weights_dir=weights_dir,
                                 device=device, table_cache=table_cache))
    _registries(config, run_dir, results)
    completed = sum(r["status"] in ("completed", "reused_complete") for r in results)
    return {"status": "completed" if completed == len(tasks) else "partial", "planned": len(tasks),
            "completed": completed, "failed": len(tasks) - completed,
            "task_records": [str(_task_path(_batch(run_dir), t.task_id)) for t in tasks]}


def _candidate_result(batch: Path, task_id: str) -> dict | None:
    path = _task_path(batch, task_id)
    if not path.is_file():
        return None
    success = _latest_success(_read(path))
    if success is None:
        return None
    endpoint = success["result"]["formal_endpoint"]
    if endpoint is None:
        return None
    return {"score": float(endpoint["score"]), "mse": float(endpoint["mse"]),
            "run_dir": success["run_dir"]}


def _plan_path(batch: Path, stage: str, *, extension: bool = False) -> Path:
    return batch / f"search_{stage}{'_extension' if extension else ''}.json"


def _load_plan(path: Path):
    from search import search_plan_from_dict
    return search_plan_from_dict(_read(path))


def _save_plan(path: Path, plan) -> None:
    write_json(path, plan.as_dict())


def _stage_completed(batch: Path, stage: str) -> bool:
    path = _plan_path(batch, stage)
    if not path.is_file():
        return False
    plan = _load_plan(path)
    from search import rank_candidates
    return (len(plan.candidates) == 24 and
            all(any(r["seed"] == 42 and r["status"] == "completed" for r in c.results) for c in plan.candidates) and
            len(rank_candidates(plan.candidates, seeds=(42, 43, 44))) >= 3)


def _point_top3(batch: Path):
    from search import rank_candidates
    if not _stage_completed(batch, "point"):
        raise ConfigError("空间搜索需要已完成单点24组及前三名复核")
    point = _load_plan(_plan_path(batch, "point"))
    candidates = list(point.candidates)
    extension = _plan_path(batch, "point", extension=True)
    if extension.is_file():
        extra = _load_plan(extension)
        if (not all(any(r["seed"] == 42 and r["status"] == "completed" for r in c.results) for c in extra.candidates)
                or len(rank_candidates(extra.candidates, seeds=(42, 43, 44))) < 3):
            raise ConfigError("已启动的单点扩展未完成")
        candidates.extend(extra.candidates)
    top = rank_candidates(candidates, seeds=(42, 43, 44))[:3]
    if len(top) < 3:
        raise ConfigError("单点搜索缺少三个完成双种子复核的配置")
    return top


def _stage_plan(config: dict, batch: Path, stage: str, *, extension: bool):
    from search import build_extension_plan, build_point_search_plan, build_spatial_search_plan
    path = _plan_path(batch, stage, extension=extension)
    if path.is_file():
        return _load_plan(path), path
    if extension:
        if not _stage_completed(batch, stage):
            raise ConfigError(f"{stage} 初搜及复核未完成，不能扩展")
        base = _load_plan(_plan_path(batch, stage))
        plan = build_extension_plan(config, stage, base, point_top3=_point_top3(batch) if stage == "spatial" else None)
    elif stage == "point":
        plan = build_point_search_plan(config)
    else:
        plan = build_spatial_search_plan(config, _point_top3(batch))
    _save_plan(path, plan)
    return plan, path


def _search_task(stage: str, candidate, seed: int, *, extension: bool):
    from dispatch import FULL_PROTOCOL, TaskSpec
    return TaskSpec(f"{'extend-' if extension else 'search-'}{stage}", "uni2h",
                    FULL_PROTOCOL, "point" if stage == "point" else "spatial",
                    seed, "search", candidate.candidate_id)


def _sync_candidate_from_record(candidate, task, batch: Path) -> bool:
    result = _candidate_result(batch, task.task_id)
    if result is None:
        return False
    candidate.results = [r for r in candidate.results if int(r["seed"]) != task.seed]
    candidate.results.append({"seed": task.seed, "status": "completed",
        "patient_macro_pathway_pcc": result["score"],
        "patient_macro_z_mse_selection": result["mse"],
        "run_dir": result["run_dir"], "error": None})
    candidate.status = "completed"
    return True


def execute_search(config: dict, *, action: str, run_dir: Path, weights_dir: Path, device: str) -> dict:
    from search import next_tpe_candidate, rank_candidates
    from dispatch import plan_paired_tasks
    stage = "point" if action.endswith("point") else "spatial"
    extension = action.startswith("extend-")
    batch = _batch(run_dir)
    if stage == "point" and not extension:
        paired = plan_paired_tasks("paired-view") + plan_paired_tasks("paired-recipe")
        if any(not _task_path(batch, t.task_id).is_file() or
               _latest_success(_read(_task_path(batch, t.task_id))) is None for t in paired):
            raise ConfigError("搜索前须完成两批各12次配对实验")
    plan, plan_path = _stage_plan(config, batch, stage, extension=extension)
    table_cache: dict = {}
    results: list[dict] = []
    # A stage keeps the same sampler method even if Optuna disappears later.
    for candidate in plan.candidates:
        task = _search_task(stage, candidate, 42, extension=extension)
        if _sync_candidate_from_record(candidate, task, batch):
            continue
        if candidate.bucket == "tpe" and candidate.status == "pending":
            next_tpe_candidate(plan, config)
            _save_plan(plan_path, plan)
        # _run_task resolves parameters from the persisted plan; extension uses its own file.
        result = _run_task(config, task, run_dir=run_dir, weights_dir=weights_dir,
                           device=device, table_cache=table_cache)
        results.append(result)
        if not _sync_candidate_from_record(candidate, task, batch):
            candidate.status = "failed"
        _save_plan(plan_path, plan)
    seed42_complete = all(any(r["seed"] == 42 and r["status"] == "completed" for r in c.results) for c in plan.candidates)
    if seed42_complete:
        top = rank_candidates(plan.candidates, seeds=(42,))[:3]
        for candidate in top:
            for seed in (43, 44):
                task = _search_task(stage, candidate, seed, extension=extension)
                if _sync_candidate_from_record(candidate, task, batch):
                    continue
                result = _run_task(config, task, run_dir=run_dir, weights_dir=weights_dir,
                                   device=device, table_cache=table_cache)
                results.append(result)
                if not _sync_candidate_from_record(candidate, task, batch):
                    candidate.status = "failed"
                _save_plan(plan_path, plan)
    new_attempts = len(results)
    registered = {r["task_id"] for r in results}
    for candidate in plan.candidates:
        for seed in (42, 43, 44):
            task = _search_task(stage, candidate, seed, extension=extension)
            if task.task_id in registered:
                continue
            path = _task_path(batch, task.task_id)
            success = _latest_success(_read(path)) if path.is_file() else None
            if success is not None:
                results.append({"task_id": task.task_id, "status": "reused_complete", "attempt": success})
                registered.add(task.task_id)
    _registries(config, run_dir, results)
    replicated = len(rank_candidates(plan.candidates, seeds=(42, 43, 44)))
    return {"status": "completed" if seed42_complete and replicated >= 3 else "partial",
        "stage": stage, "extension": extension, "sampler_method": plan.sampler_method,
        "initial_configs": len(plan.candidates), "seed42_complete": sum(any(r["seed"] == 42 and r["status"] == "completed" for r in c.results) for c in plan.candidates),
        "top3_replicated": replicated, "plan_path": str(plan_path), "new_attempts": new_attempts}


def freeze_search(config: dict, *, run_dir: Path) -> dict:
    from search import rank_candidates
    batch = _batch(run_dir)
    target = batch / "frozen_selection.json"
    if target.exists():
        raise ConfigError("冻结配置已存在；拒绝用后续结果静默覆盖")
    frozen = {"schema_version": "1.0", "selection_split": "internal_val",
              "external_test_used_for_selection": False, "frozen_at": utc_now()}
    for stage in ("point", "spatial"):
        if not _stage_completed(batch, stage):
            raise ConfigError(f"{stage} 搜索初搜及复核未完成")
        initial = _load_plan(_plan_path(batch, stage))
        candidates = list(initial.candidates)
        extension_path = _plan_path(batch, stage, extension=True)
        if extension_path.is_file():
            extra = _load_plan(extension_path)
            if (len(extra.candidates) != 12 or
                not all(any(r["seed"] == 42 and r["status"] == "completed" for r in c.results) for c in extra.candidates) or
                len(rank_candidates(extra.candidates, seeds=(42, 43, 44))) < 3):
                raise ConfigError(f"{stage} 扩展尚未完成")
            candidates += extra.candidates
        winners = rank_candidates(candidates, seeds=(42, 43, 44))
        if len(winners) < 3:
            raise ConfigError(f"{stage} 没有三个完成复核的候选")
        winner = winners[0]
        frozen[stage] = {"candidate_id": winner.candidate_id,
                         "parameters": winner.parameters,
                         "replication_results": winner.results,
                         "sampler_method": initial.sampler_method}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=2)
    write_json(run_dir / "frozen_selection.json", frozen)
    return {"status": "completed", "frozen_path": str(target),
            "point_candidate": frozen["point"]["candidate_id"],
            "spatial_candidate": frozen["spatial"]["candidate_id"]}


def _formal_attempt(batch: Path, task) -> tuple[dict, dict]:
    path = _task_path(batch, task.task_id)
    if not path.is_file():
        raise ConfigError(f"缺少训练任务记录: {task.task_id}")
    attempt = _latest_success(_read(path))
    if attempt is None:
        raise ConfigError(f"任务没有成功的正式检查点: {task.task_id}")
    result = attempt["result"]
    formal = result.get("formal_checkpoint")
    if not formal or not Path(formal).is_file():
        raise ConfigError(f"正式检查点缺失: {task.task_id}: {formal}")
    return attempt, result


def _external_targets(table, config: dict, names: list[str]) -> np.ndarray:
    from data import attach_labels
    labeled = table.subset(range(len(table)))
    labeled.split[:] = "external"
    labeled = attach_labels(labeled, config["paths"]["labels_root"], names,
                            train_mpp_id=config["data"]["mpp_id"])
    return labeled.labels_z


def evaluate_external(config: dict, *, run_dir: Path, device: str) -> dict:
    from dispatch import plan_final_tasks, plan_paired_tasks
    from data import load_normalization
    from predict import load_formal_model, predict_table, save_predictions

    batch = _batch(run_dir)
    _frozen(batch)
    tasks = plan_paired_tasks("paired-view") + plan_paired_tasks("paired-recipe") + plan_final_tasks()
    # All 51 formal models are required. No external label is loaded until this gate passes.
    attempts = {task.task_id: _formal_attempt(batch, task) for task in tasks}
    table_cache: dict = {}
    outputs = []
    for task in tasks:
        _, result = attempts[task.task_id]
        params, mode = _task_parameters(task, batch)
        cfg = trial_config(config, model_name=task.model, parameters=params, selection_mode=mode)
        key = (task.model, task.protocol)
        if key not in table_cache:
            table_cache[key] = _cache_table(config, model_name=task.model,
                                            protocol=task.protocol, split="external_test")
        table, cache_dir, names = table_cache[key]
        model, metadata = load_formal_model(cfg, result["formal_checkpoint"],
                                            arm=task.arm, seed=task.seed, device=device)
        metadata.update({"task_id": task.task_id, "stage": task.stage,
                         "model": task.model, "protocol": task.protocol,
                         "recipe": task.recipe, "candidate_id": task.candidate_id,
                         "split": "external_test"})
        pred = predict_table(cfg, arm=task.arm, model=model, table=table,
                             split="external_test", device=device)
        norm = load_normalization(config["inputs"]["normalization"], names)
        # Truth is joined by patient/spot only after formal predictions are fixed.
        target = _external_targets(table, config, names)
        output = save_predictions(run_dir / "raw" / f"{task.task_id}.npz",
            pred_z=pred, target_z=target, mean=norm.mean, std=norm.std,
            pathway_names=names, table=table, checkpoint_metadata=metadata)
        outputs.append({"task_id": task.task_id, "prediction": str(output.resolve()),
                        "formal_checkpoint": result["formal_checkpoint"],
                        "cache_dir": cache_dir, "n_points": len(table),
                        "selection_used_external": False})
    write_json(run_dir / "external_predictions.json", {"status": "completed",
        "count": len(outputs), "entries": outputs})
    write_json(run_dir / "model_weights.json", {"entries": [
        {"task_id": e["task_id"], "formal_checkpoint": e["formal_checkpoint"],
         "weight_dir": str(Path(e["formal_checkpoint"]).parent),
         "last_checkpoint": str(Path(e["formal_checkpoint"]).parent / "last.pt"),
         "warmup_checkpoint": str(Path(e["formal_checkpoint"]).parent / "warmup.pt") if (Path(e["formal_checkpoint"]).parent / "warmup.pt").is_file() else None,
         "warmup_status": "available" if (Path(e["formal_checkpoint"]).parent / "warmup.pt").is_file() else "not_applicable_or_not_produced",
         "formal_status": "available", "last_status": "available"} for e in outputs]})
    write_json(run_dir / "feature_caches.json", {"entries": [
        {"task_id": e["task_id"], "cache_dir": e["cache_dir"]} for e in outputs]})
    return {"status": "completed", "formal_prediction_count": len(outputs),
            "prediction_manifest": str(run_dir / "external_predictions.json")}


def export_phase3(config: dict, *, run_dir: Path, device: str) -> dict:
    from dispatch import plan_final_tasks
    from export_phase3 import export_phase3_bundle

    batch = _batch(run_dir)
    frozen = _frozen(batch)
    seed_sources = []
    attempt = result = None
    for task in plan_final_tasks():
        if task.model != "uni2h" or task.arm != "spatial":
            continue
        item_attempt, item_result = _formal_attempt(batch, task)
        seed_sources.append({"task_id": task.task_id, "seed": task.seed,
                             "formal_checkpoint": item_result["formal_checkpoint"],
                             "formal_endpoint": item_result["formal_endpoint"],
                             "selected_for_phase3": task.seed == 45})
        if task.seed == 45:
            attempt, result = item_attempt, item_result
    if len(seed_sources) != 3 or attempt is None or result is None:
        raise ConfigError("Phase3 导出需要 UNI2-h 空间臂种子45/46/47的正式来源")
    selected_task_id = next(item["task_id"] for item in seed_sources if item["seed"] == 45)
    cfg = trial_config(config, model_name="uni2h",
        parameters=frozen["spatial"]["parameters"], selection_mode="search")
    output = export_phase3_bundle(cfg, checkpoint_path=result["formal_checkpoint"],
        output_dir=run_dir / "phase3_export",
        frozen_selection_path=batch / "frozen_selection.json", seed_sources=seed_sources)
    write_json(run_dir / "model_weights.json", {"entries": [{"task_id": selected_task_id,
        "weight_dir": attempt["weights_dir"],
        "formal_checkpoint": result["formal_checkpoint"],
        "last_checkpoint": result.get("last_checkpoint"), "warmup_checkpoint": result.get("warmup_checkpoint"),
        "warmup_status": "available" if result.get("warmup_checkpoint") else "not_applicable_or_not_produced", "formal_status": "available",
        "last_status": "available" if result.get("last_checkpoint") else "not_produced"}]})
    write_json(run_dir / "feature_caches.json", {"entries": [{
        "task_id": selected_task_id, "cache_dir": attempt["cache_dir"]}]})
    return {"status": "completed", "export": str(output.resolve())}


def analyze_local(*, run_dir: Path) -> dict:
    from dispatch import planned_external_keys
    from local_report import write_local_report

    batch = _batch(run_dir)
    files = sorted(batch.glob("external-eval_*/raw/*.npz"))
    if not files:
        raise ConfigError("批次下没有外部预测 NPZ；回传后可用 local_report.py --prediction-dir 显式分析")
    ids = [p.stem for p in files]
    if len(ids) != len(set(ids)):
        raise ConfigError("发现重复任务预测，请先明确选择一次 external-eval 回传")
    expected = set(planned_external_keys())
    missing = sorted(expected - set(ids))
    unexpected = sorted(set(ids) - expected)
    output = run_dir / "analysis" / "local_metrics.json"
    write_local_report(files, output)
    return {"status": "completed" if not missing and not unexpected else "partial",
            "n_predictions": len(files), "expected_predictions": len(expected),
            "missing_task_ids": missing, "unexpected_task_ids": unexpected,
            "analysis_path": str(output.resolve()), "computed_locally": True}
