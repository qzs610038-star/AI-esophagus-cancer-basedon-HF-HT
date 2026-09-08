"""Unified v2.1 dispatcher for precheck, four arms, seeds, and frozen external prediction."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from config import SUPPORTED_ARMS, load_config, package_root, validate_config
from errors import ConfigError
from predict import predict_endpoint
from precheck import run_precheck
from run_io import (
    append_event,
    capture_stdio,
    copy_package_snapshot,
    copy_provenance,
    create_source_snapshot,
    create_batch_dir,
    create_run_dir,
    default_runs_root,
    default_weights_root,
    initial_run_record,
    package_payload,
    read_json,
    utc_now,
    write_config_snapshot,
    write_json,
)
from selection import recommend_arm
from train import train_arm


@dataclass(frozen=True)
class TrainingTask:
    arm: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"train_{self.seed}_{self.arm}"


def plan_training_tasks(config: dict, *, scope: str, seed: int | None = None) -> list[TrainingTask]:
    arms = list(config["training"]["arms"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    if scope == "all":
        return [TrainingTask(arm, run_seed) for run_seed in seeds for arm in arms]
    if scope in SUPPORTED_ARMS:
        return [TrainingTask(scope, run_seed) for run_seed in seeds]
    if scope == "seed":
        if seed is None or int(seed) not in seeds:
            raise ConfigError(f"--scope seed 需要 --seed，且必须属于 {seeds}")
        return [TrainingTask(arm, int(seed)) for arm in arms]
    raise ConfigError(f"未知训练范围 {scope!r}")


def _task_rows(tasks: list[TrainingTask]) -> list[dict]:
    return [{**asdict(task), "run_id": task.run_id, "status": "pending", "error": None, "result": None} for task in tasks]


def _finish_record(record: dict, status: str, exit_code: int, error: str | None = None) -> dict:
    record.update(status=status, exit_code=int(exit_code), ended_at=utc_now())
    if error:
        record["error"] = error
    return record


def _write_batch_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    write_json(path, manifest)


def _run_precheck_task(
    config: dict,
    batch_dir: Path,
    package: dict,
    runner: Callable,
    artifact_kind: str,
    source_snapshot: Path | None = None,
) -> dict:
    run_dir = create_run_dir(batch_dir, "00_precheck")
    copy_provenance(run_dir, source_snapshot)
    record = initial_run_record(
        package=package,
        run_id="00_precheck",
        run_dir=run_dir,
        arm=None,
        seed=None,
        artifact_kind=artifact_kind,
        entrypoint="src/precheck.py",
    )
    write_json(run_dir / "run.json", record)
    try:
        record["status"] = "running"
        write_json(run_dir / "run.json", record)
        with capture_stdio(run_dir):
            report = runner(
                config,
                run_dir / "raw",
                artifact_kind="official_precheck" if artifact_kind == "official" else artifact_kind,
            )
        ok = report.get("status") != "error"
        _finish_record(record, "succeeded" if ok else "failed", 0 if ok else 2)
        write_json(run_dir / "run.json", record)
        return {"status": record["status"], "run_id": "00_precheck", "report": report}
    except Exception as exc:
        detail = traceback.format_exc()
        (run_dir / "logs" / "errors.log").write_text(detail, encoding="utf-8")
        _finish_record(record, "failed", 1, str(exc))
        write_json(run_dir / "run.json", record)
        return {"status": "failed", "run_id": "00_precheck", "error": str(exc)}


def execute_training_batch(
    config: dict,
    *,
    runs_root: str | Path | None = None,
    weights_root: str | Path | None = None,
    scope: str = "all",
    seed: int | None = None,
    artifact_kind: str = "official",
    precheck_runner: Callable = run_precheck,
    train_runner: Callable = train_arm,
) -> dict:
    """Execute sequentially from a deep-copied snapshot; first failure stops the batch."""
    validate_config(config)
    frozen = copy.deepcopy(config)
    tasks = plan_training_tasks(frozen, scope=scope, seed=seed)
    batch_dir = create_batch_dir(runs_root or default_runs_root(frozen))
    resolved_weights_root = Path(weights_root or default_weights_root(frozen)).resolve()
    weight_experiment_dir = resolved_weights_root / frozen["experiment_id"]
    weight_batch_dir = weight_experiment_dir / batch_dir.name
    package = package_payload()
    snapshot_path = write_config_snapshot(batch_dir, frozen)
    copy_package_snapshot(batch_dir)
    source_snapshot = create_source_snapshot(batch_dir)
    manifest = {
        "experiment_id": frozen["experiment_id"],
        "plan_version": frozen["plan_version"],
        "config_version": frozen["config_version"],
        "comparison_name": frozen["comparison_name"],
        "run_kind": frozen["run_kind"],
        "scope": scope,
        "seed": seed,
        "artifact_kind": artifact_kind,
        "created_at": utc_now(),
        "config_snapshot": snapshot_path.name,
        "status": "running",
        "precheck": {"status": "pending"},
        "tasks": _task_rows(tasks),
        "external_tasks": [],
        "model_weights": {
            "registry": "model_weights.json",
            "weights_root": str(resolved_weights_root),
            "weight_experiment_directory": str(weight_experiment_dir),
            "weight_batch_directory": str(weight_batch_dir),
            "return_policy": "server_weights_excluded_from_local_result_copy",
        },
        "hashes_used": False,
    }
    manifest_path = batch_dir / "batch_manifest.json"
    _write_batch_manifest(manifest_path, manifest)
    weight_registry = {
        "schema_version": "1.0",
        "experiment_id": frozen["experiment_id"],
        "batch_id": batch_dir.name,
        "code_version": package.get("code_version"),
        "plan_version": frozen["plan_version"],
        "config_version": frozen["config_version"],
        "comparison_name": frozen["comparison_name"],
        "run_kind": frozen["run_kind"],
        "created_at": utc_now(),
        "weights_root": str(resolved_weights_root),
        "weight_experiment_directory": str(weight_experiment_dir),
        "weight_batch_directory": str(weight_batch_dir),
        "return_policy": "server_weights_excluded_from_local_result_copy",
        "entries": [
            {
                "run_id": task.run_id,
                "arm": task.arm,
                "seed": int(task.seed),
                "status": "pending",
                "weight_directory": str(weight_batch_dir / task.run_id),
                "warmup_checkpoint": str(weight_batch_dir / task.run_id / "warmup_best.pt"),
                "formal_checkpoint": str(weight_batch_dir / task.run_id / "formal_best.pt"),
                "last_checkpoint": str(weight_batch_dir / task.run_id / "last.pt"),
            }
            for task in tasks
        ],
    }
    weight_registry_path = batch_dir / "model_weights.json"
    write_json(weight_registry_path, weight_registry)
    append_event(batch_dir / "events.jsonl", stage="batch_start", scope=scope, seed=seed)

    precheck_result = _run_precheck_task(
        frozen, batch_dir, package, precheck_runner, artifact_kind, source_snapshot=source_snapshot
    )
    manifest["precheck"] = precheck_result
    if precheck_result["status"] != "succeeded":
        for row, weight_entry in zip(manifest["tasks"], weight_registry["entries"]):
            row["status"] = "not_run"
            row["error"] = "precheck_failed"
            weight_entry["status"] = "not_run"
        write_json(weight_registry_path, weight_registry)
        manifest["status"] = "failed"
        _write_batch_manifest(manifest_path, manifest)
        append_event(batch_dir / "events.jsonl", stage="batch_end", status="failed", reason="precheck")
        return {"exit_code": 1, "batch_dir": str(batch_dir), "manifest": manifest}

    weight_batch_dir.mkdir(parents=True, exist_ok=False)
    failed = False
    for task, row, weight_entry in zip(tasks, manifest["tasks"], weight_registry["entries"]):
        if failed:
            row["status"] = "not_run"
            row["error"] = "earlier_task_failed"
            weight_entry["status"] = "not_run"
            continue
        run_dir = create_run_dir(batch_dir, task.run_id)
        checkpoint_dir = weight_batch_dir / task.run_id
        copy_provenance(run_dir, source_snapshot)
        record = initial_run_record(
            package=package,
            run_id=task.run_id,
            run_dir=run_dir,
            arm=task.arm,
            seed=task.seed,
            artifact_kind=artifact_kind,
            entrypoint="src/train.py",
            extra={
                "weight_directory": str(checkpoint_dir),
                "weight_registry": str(weight_registry_path.resolve()),
            },
        )
        write_json(run_dir / "run.json", record)
        row["status"] = "running"
        _write_batch_manifest(manifest_path, manifest)
        try:
            record["status"] = "running"
            write_json(run_dir / "run.json", record)
            with capture_stdio(run_dir):
                result = train_runner(
                    frozen,
                    task.arm,
                    task.seed,
                    run_dir,
                    checkpoint_dir=checkpoint_dir,
                )
            if result.get("status") != "completed" or not result.get("formal_endpoint"):
                raise RuntimeError(f"训练未产生正式端点: {result.get('status')}")
            row["status"] = "succeeded"
            row["result"] = result
            weight_entry.update(
                status="succeeded",
                warmup_checkpoint=result.get("warmup_checkpoint"),
                formal_checkpoint=result.get("formal_checkpoint"),
                last_checkpoint=result.get("last_checkpoint"),
                formal_selection=result.get("formal_endpoint"),
                files=[
                    {
                        "kind": kind,
                        "path": str(Path(path).resolve()),
                        "size_bytes": Path(path).stat().st_size,
                    }
                    for kind, path in (
                        ("warmup", result.get("warmup_checkpoint")),
                        ("formal", result.get("formal_checkpoint")),
                        ("last", result.get("last_checkpoint")),
                    )
                    if path is not None
                ],
            )
            _finish_record(record, "succeeded", 0)
            write_json(run_dir / "run.json", record)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            weight_entry["status"] = "failed"
            detail = traceback.format_exc()
            with (run_dir / "logs" / "errors.log").open("a", encoding="utf-8") as handle:
                handle.write(detail)
            _finish_record(record, "failed", 1, str(exc))
            write_json(run_dir / "run.json", record)
            failed = True
        _write_batch_manifest(manifest_path, manifest)
        write_json(weight_registry_path, weight_registry)

    manifest["status"] = "failed" if failed else "succeeded"
    if not failed:
        _write_internal_summary(batch_dir, manifest, frozen)
    write_json(weight_registry_path, weight_registry)
    _write_batch_manifest(manifest_path, manifest)
    append_event(batch_dir / "events.jsonl", stage="batch_end", status=manifest["status"])
    return {"exit_code": 1 if failed else 0, "batch_dir": str(batch_dir), "manifest": manifest}


def _write_internal_summary(batch_dir: Path, manifest: dict, config: dict) -> dict:
    per_arm_values: dict[str, list[dict]] = {arm: [] for arm in config["training"]["arms"]}
    endpoints = []
    for row in manifest["tasks"]:
        if row["status"] != "succeeded":
            continue
        result = row["result"]
        formal = result["formal_endpoint"]
        per_arm_values[row["arm"]].append(formal)
        endpoints.append(
            {
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "checkpoint_kind": "formal",
                "epoch": int(formal["epoch"]),
                "score": float(formal["score"]),
                "mse": float(formal["mse"]),
                "checkpoint": str(Path(result["formal_checkpoint"]).resolve()),
            }
        )
    per_arm = {}
    hidden = int(config["model"]["hidden_dim"])
    input_dim = int(config["data"]["input_dim"])
    output_dim = int(config["data"]["output_dim"])
    point_parameters = input_dim * hidden + hidden + hidden * output_dim + output_dim
    spatial_parameters = point_parameters + hidden * output_dim
    for arm, values in per_arm_values.items():
        if values:
            per_arm[arm] = {
                "n_seeds": len(values),
                "mean_patient_macro_pathway_pcc": sum(float(v["score"]) for v in values) / len(values),
                "mean_patient_macro_z_mse_selection": sum(float(v["mse"]) for v in values) / len(values),
                "inference_parameters": spatial_parameters if arm in ("spatial", "joint") else point_parameters,
            }
    summary = {"per_arm": per_arm, "recommendation": recommend_arm(per_arm) if per_arm else None}
    write_json(batch_dir / "internal_summary.json", summary)
    endpoint_manifest = {
        "plan_version": config["plan_version"],
        "config_version": config["config_version"],
        "comparison_name": config["comparison_name"],
        "run_kind": config["run_kind"],
        "split_id": config["data"]["split_id"],
        "selection_source": "internal_val_only",
        "external_used_for_selection": False,
        "endpoints": endpoints,
    }
    write_json(batch_dir / "formal_endpoints.json", endpoint_manifest)
    return endpoint_manifest


def validate_endpoint_manifest(manifest: dict, config: dict) -> list[dict]:
    expected_meta = {
        "plan_version": config["plan_version"],
        "config_version": config["config_version"],
        "comparison_name": config["comparison_name"],
        "run_kind": "formal",
        "split_id": config["data"]["split_id"],
    }
    for key, expected in expected_meta.items():
        if manifest.get(key) != expected:
            raise ConfigError(f"端点清单 {key} 不匹配: {manifest.get(key)!r} != {expected!r}")
    endpoints = list(manifest.get("endpoints") or [])
    expected = {(arm, int(seed)) for seed in config["training"]["seeds"] for arm in config["training"]["arms"]}
    actual = {(str(item.get("arm")), int(item.get("seed", -1))) for item in endpoints}
    if actual != expected or len(endpoints) != len(expected):
        raise ConfigError("端点清单必须恰含 4臂×3种子，无重复或缺失")
    for item in endpoints:
        if item.get("checkpoint_kind") != "formal":
            raise ConfigError("端点清单只允许 formal checkpoint")
        if not Path(item["checkpoint"]).is_file():
            raise ConfigError(f"正式 checkpoint 不存在: {item['checkpoint']}")
    return endpoints


def execute_external_batch(config: dict, endpoint_manifest: dict, batch_dir: str | Path, *, device=None) -> int:
    endpoints = validate_endpoint_manifest(endpoint_manifest, config)
    batch = Path(batch_dir)
    source_snapshot = batch / "source_snapshot"
    if not source_snapshot.is_dir():
        source_snapshot = create_source_snapshot(batch)
    manifest_path = batch / "batch_manifest.json"
    batch_manifest = read_json(manifest_path)
    rows = [
        {"arm": item["arm"], "seed": item["seed"], "run_id": f"external_{item['seed']}_{item['arm']}", "status": "pending", "error": None}
        for item in endpoints
    ]
    batch_manifest["external_tasks"] = rows
    _write_batch_manifest(manifest_path, batch_manifest)
    failed = False
    package = package_payload(source_snapshot)
    for endpoint, row in zip(endpoints, rows):
        if failed:
            row.update(status="not_run", error="earlier_external_task_failed")
            continue
        run_dir = create_run_dir(batch, row["run_id"])
        copy_provenance(run_dir, source_snapshot)
        record = initial_run_record(
            package=package,
            run_id=row["run_id"],
            run_dir=run_dir,
            arm=str(endpoint["arm"]),
            seed=int(endpoint["seed"]),
            artifact_kind="official_external_prediction",
            entrypoint="src/predict.py",
            extra={"selected_checkpoint": endpoint["checkpoint"], "checkpoint_kind": "formal"},
        )
        write_json(run_dir / "run.json", record)
        try:
            record["status"] = "running"
            write_json(run_dir / "run.json", record)
            with capture_stdio(run_dir):
                result = predict_endpoint(config, endpoint, run_dir, device=device)
            row.update(status="succeeded", result=result)
            _finish_record(record, "succeeded", 0)
            write_json(run_dir / "run.json", record)
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            (run_dir / "logs" / "errors.log").write_text(traceback.format_exc(), encoding="utf-8")
            _finish_record(record, "failed", 1, str(exc))
            write_json(run_dir / "run.json", record)
            failed = True
        _write_batch_manifest(manifest_path, batch_manifest)
    if failed:
        batch_manifest["status"] = "failed"
    _write_batch_manifest(manifest_path, batch_manifest)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 软连接 v2.1 统一入口")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scope", choices=["all", *SUPPORTED_ARMS, "seed", "precheck", "external"], default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--endpoint-manifest", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    root = package_root()
    config = load_config(args.config, package_dir=root)
    if args.scope == "external":
        if args.endpoint_manifest is None:
            raise ConfigError("external 需要 --endpoint-manifest")
        endpoint_manifest = read_json(args.endpoint_manifest)
        batch = create_batch_dir(args.runs_root or default_runs_root(config))
        write_config_snapshot(batch, config)
        copy_package_snapshot(batch)
        create_source_snapshot(batch)
        write_json(batch / "batch_manifest.json", {"status": "running", "tasks": [], "external_tasks": [], "created_at": utc_now()})
        return execute_external_batch(config, endpoint_manifest, batch, device=args.device)
    if args.scope == "precheck":
        batch = create_batch_dir(args.runs_root or default_runs_root(config))
        write_config_snapshot(batch, config)
        copy_package_snapshot(batch)
        source_snapshot = create_source_snapshot(batch)
        result = _run_precheck_task(
            config, batch, package_payload(), run_precheck, "official", source_snapshot=source_snapshot
        )
        write_json(batch / "batch_manifest.json", {"status": result["status"], "precheck": result, "tasks": [], "external_tasks": []})
        print(json.dumps({"batch_dir": str(batch), "status": result["status"]}, ensure_ascii=False))
        return 0 if result["status"] == "succeeded" else 1
    result = execute_training_batch(
        config,
        runs_root=args.runs_root,
        weights_root=args.weights_root,
        scope=args.scope,
        seed=args.seed,
        artifact_kind="official",
    )
    formal_full_grid = (
        config.get("run_kind") == "formal"
        and list(config["training"]["arms"]) == list(SUPPORTED_ARMS)
        and [int(value) for value in config["training"]["seeds"]] == [42, 43, 44]
    )
    if result["exit_code"] == 0 and args.scope == "all" and formal_full_grid:
        endpoints = read_json(Path(result["batch_dir"]) / "formal_endpoints.json")
        result["exit_code"] = execute_external_batch(config, endpoints, result["batch_dir"], device=args.device)
    print(json.dumps({"batch_dir": result["batch_dir"], "exit_code": result["exit_code"]}, ensure_ascii=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
