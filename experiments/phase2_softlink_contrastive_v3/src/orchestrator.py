"""Resumable protocol-local task plans (planning is safe for dry-runs)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

try:  # package import (runner uses src.orchestrator)
    from .protocols import ProtocolSplit, build_lopo6_splits, build_original_split
    from .training import MAIN_CELLS, stage2_task_matrix
except ImportError:  # direct script/import with src placed on sys.path
    from protocols import ProtocolSplit, build_lopo6_splits, build_original_split
    from training import MAIN_CELLS, stage2_task_matrix

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class RunKey:
    protocol: str; fold: str; seed: int; cell: str; endpoint: int
    def value(self) -> str: return f"{self.protocol}/{self.fold}/s{self.seed}/{self.cell}/e{self.endpoint}"

def protocol_splits(manifest, *, protocol: str = "original") -> list[ProtocolSplit]:
    if protocol == "original":
        return [build_original_split(manifest)]
    if protocol == "lopo6":
        return build_lopo6_splits(manifest)
    raise ValueError("protocol 必须为 original 或 lopo6")

def build_task_dag(manifest, *, protocol: str="original", seeds: Sequence[int]=(42,43,44), best_epochs=None) -> list[dict]:
    """Build one protocol's DAG; LOPO calls never mix in original tasks."""
    tasks=[]
    for split in protocol_splits(manifest,protocol=protocol):
        # only seed 42 has the three 5-epoch LR pilot runs
        for lr in (1e-5,3e-5,1e-4):
            key=RunKey(split.protocol,split.fold,42,"r8_regression_lr_pilot",5)
            tasks.append({"key":key.value()+f"/lr{lr:g}","run_key":asdict(key),"kind":"lr_pilot","lr":lr,"depends_on":[]})
        selection_key=f"{split.protocol}/{split.fold}/s42/r8_regression_lr_selection/e5"
        tasks.append({"key":selection_key,"kind":"lr_selection","depends_on":[f"{split.protocol}/{split.fold}/s42/r8_regression_lr_pilot/e5/lr1e-05",f"{split.protocol}/{split.fold}/s42/r8_regression_lr_pilot/e5/lr3e-05",f"{split.protocol}/{split.fold}/s42/r8_regression_lr_pilot/e5/lr0.0001"]})
        for seed in seeds:
            for cell in MAIN_CELLS:
                # Selected seed-42 pilot is the r8 regression formal run.
                if int(seed) == 42 and cell == "r8_regression":
                    continue
                key=RunKey(split.protocol,split.fold,int(seed),cell,5)
                tasks.append({"key":key.value(),"run_key":asdict(key),"kind":"online","lr_source_seed":42,"depends_on":[selection_key]})
            for task in stage2_task_matrix(best_epochs):
                key=RunKey(split.protocol,split.fold,int(seed),task.cell,task.endpoint)
                # A sensitivity endpoint is a checkpoint from the same five-epoch
                # online run, not a second online adaptation job.
                online_key = RunKey(split.protocol, split.fold, int(seed), task.cell, 5).value()
                dependency = selection_key if int(seed) == 42 and task.cell == "r8_regression" else online_key
                tasks.append({"key":key.value()+f"/{task.h_mode}/{task.head}","run_key":asdict(key),"kind":"stage2","head":task.head,"h_mode":task.h_mode,"depends_on":[dependency]})
    return tasks

def initialise_state(path: str|Path, tasks: Iterable[dict]) -> dict:
    path=Path(path)
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    state={"schema_version":"v4","tasks":{task["key"]:{"status":"pending",**task} for task in tasks}}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    return state

def mark_task(path: str|Path, key: str, status: str) -> dict:
    path=Path(path); state=json.loads(path.read_text(encoding="utf-8"))
    if key not in state["tasks"]: raise KeyError(key)
    if state["tasks"][key]["status"]=="completed" and status!="completed": raise ValueError("已完成任务不可回退")
    state["tasks"][key]["status"]=status; path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); return state


def _resolve_packaged_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _task_fold(task: dict) -> str:
    if "run_key" in task:
        return str(task["run_key"]["fold"])
    return str(task["key"]).split("/", 2)[1]


def _plan_payload(config_path: Path, run_dir: Path, weights_dir: Path, protocol: str, seeds: Sequence[int], folds: Sequence[str] | None, resume: bool) -> tuple[dict, list[dict]]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    manifest_value = (config.get("data") or {}).get("split_manifest_file")
    if not manifest_value:
        raise ValueError("config.data.split_manifest_file 未设置")
    manifest_path = _resolve_packaged_path(manifest_value)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"包内 split_manifest 不存在: {manifest_path}")
    if protocol not in ("original", "lopo6"):
        raise ValueError("protocol 必须为 original 或 lopo6")
    if not seeds:
        raise ValueError("seeds 不能为空")
    manifest = pd.read_csv(manifest_path)
    splits = protocol_splits(manifest, protocol=protocol)
    requested = None if folds is None else {str(value) for value in folds}
    if requested:
        unknown = requested - {split.fold for split in splits}
        if unknown:
            raise ValueError(f"请求的 fold 不属于 {protocol}: {sorted(unknown)}")
        splits = [split for split in splits if split.fold in requested]
    if not splits:
        raise ValueError("fold 过滤后没有任务")
    if protocol == "original":
        expected_train = (config.get("data") or {}).get("expected_original_train_points")
        expected_val = (config.get("data") or {}).get("expected_original_internal_val_points")
        split = splits[0]
        if expected_train is not None and len(split.train_rows) != int(expected_train):
            raise ValueError(f"original train 计数不符: {len(split.train_rows)} != {expected_train}")
        if expected_val is not None and len(split.internal_val_rows) != int(expected_val):
            raise ValueError(f"original internal_val 计数不符: {len(split.internal_val_rows)} != {expected_val}")
    all_tasks = build_task_dag(manifest, protocol=protocol, seeds=tuple(int(seed) for seed in seeds))
    selected_folds = {split.fold for split in splits}
    tasks = [task for task in all_tasks if _task_fold(task) in selected_folds]
    split_counts = [{"fold": split.fold, "train_points": len(split.train_rows), "internal_val_points": len(split.internal_val_rows), "held_out_points": len(split.held_out_rows)} for split in splits]
    payload = {
        "schema_version": "v4", "config_path": str(config_path.resolve()),
        "split_manifest": str(manifest_path.resolve()), "run_dir": str(run_dir.resolve()),
        "weights_dir": str(weights_dir.resolve()), "protocol": protocol,
        "seeds": [int(seed) for seed in seeds], "folds": [split.fold for split in splits],
        "split_counts": split_counts, "resume": bool(resume), "task_count": len(tasks),
        "online_count": sum(task["kind"] in {"lr_pilot", "online"} for task in tasks),
        "stage2_count": sum(task["kind"] == "stage2" for task in tasks),
        "plan_only_does_not_load_large_inputs": True,
    }
    return payload, tasks


def execute_plan(config_path, run_dir, weights_dir, protocol, seeds, folds=None, resume=False, plan_only=False) -> dict:
    """Runner-facing planning/execution boundary.

    Planning reads only JSON and the packaged split CSV.  The heavyweight
    engine is deliberately imported only after a valid plan is persisted.
    """
    try:
        config_path, run_dir, weights_dir = Path(config_path), Path(run_dir), Path(weights_dir)
        if not config_path.is_file():
            raise FileNotFoundError(f"config 不存在: {config_path}")
        if run_dir.resolve() == weights_dir.resolve():
            raise ValueError("runs 与 weights 目录必须分离")
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payload, tasks = _plan_payload(config_path, run_dir, weights_dir, str(protocol), tuple(seeds), folds, bool(resume))
        plan_path, state_path = raw_dir / "task_plan.json", raw_dir / "task_state.json"
        if state_path.exists() and not resume:
            raise FileExistsError("task_state 已存在；需要 --resume 才能继续")
        payload["plan_only"] = bool(plan_only)
        plan_path.write_text(json.dumps({**payload, "tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
        state = initialise_state(state_path, tasks)
        if plan_only:
            return {"exit_code": 0, "status": "planned", "plan_path": str(plan_path), "state_path": str(state_path), "task_count": len(state["tasks"])}
        try:
            from .engine import execute_experiment
        except ImportError:
            try:
                from engine import execute_experiment
            except ImportError as exc:
                return {"exit_code": 1, "status": "failed", "error": f"训练引擎不可用: {exc}", "plan_path": str(plan_path)}
        result = execute_experiment(config_path=config_path, run_dir=run_dir, weights_dir=weights_dir, protocol=protocol, seeds=tuple(seeds), folds=folds, resume=bool(resume), plan_only=False)
        if isinstance(result, dict):
            exit_code = int(result.get("exit_code", 0))
            return {**result, "exit_code": exit_code, "plan_path": str(plan_path)}
        if isinstance(result, int):
            return {"exit_code": int(result), "plan_path": str(plan_path)}
        if result is False:
            return {"exit_code": 1, "status": "failed", "error": "训练引擎返回 False", "plan_path": str(plan_path)}
        return {"exit_code": 0, "status": "executed", "plan_path": str(plan_path)}
    except Exception as exc:
        return {"exit_code": 1, "status": "failed", "error": str(exc)}
