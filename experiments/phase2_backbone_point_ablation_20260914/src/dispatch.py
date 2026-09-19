"""Sequential two-stage orchestration with an explicit stop after seed 42."""

from __future__ import annotations

import copy
import json
import os
import shutil
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from baseline_import import (
    extract_point_output_state,
    import_reference_artifacts,
    load_reference_entry,
    validate_reference_identity_alignment,
    validate_reference_files,
)
from config import validate_config
from data import (
    load_common_manifest,
    load_labels_for_rows,
    load_normalization,
    load_pathway_names,
)
from errors import ConfigError, IdentityMismatchError
from feature_cache import load_feature_cache
from predict import predict_external_arrays
from run_io import append_event, utc_now, write_json
from training import train_point_arrays


@dataclass(frozen=True)
class WorkUnit:
    kind: str
    model: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.kind}_{self.seed}_{self.model}"


def plan_scope(scope: str) -> list[WorkUnit]:
    if scope == "seed42":
        seeds = (42,)
    elif scope == "remaining":
        seeds = (43, 44)
    else:
        raise ValueError("scope 只允许 seed42 或 remaining；默认入口不能合并全种子")
    units: list[WorkUnit] = []
    for seed in seeds:
        units.append(WorkUnit("reference", "uni2h", seed))
        for model in ("uni", "virchow2"):
            units.append(WorkUnit("train", model, seed))
            units.append(WorkUnit("external", model, seed))
    return units


def execute_plan(units: Sequence[WorkUnit], executor: Callable[[WorkUnit], dict]) -> list[dict]:
    """Execute sequentially and make the first failure visible in all later rows."""

    records: list[dict] = []
    failed = False
    for unit in units:
        record = {
            **asdict(unit),
            "run_id": unit.run_id,
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "result": None,
            "error": None,
        }
        if failed:
            record.update(status="not_run", error="earlier_unit_failed")
            records.append(record)
            continue
        record["started_at"] = utc_now()
        try:
            result = executor(unit)
            record.update(status="succeeded", result=result, exit_code=0)
        except Exception as exc:
            record.update(status="failed", error=str(exc), exit_code=1)
            failed = True
        record["ended_at"] = utc_now()
        records.append(record)
    return records


def _feature_locations(config: dict, model_name: str) -> tuple[Path, Path, Path]:
    root = Path(config["paths"]["feature_caches_root"]) / config["experiment_id"]
    common = root / "common_identity_v1" / "common_identity_manifest.csv"
    cache = root / model_name / config["feature_extraction"]["feature_version"]
    registry = root / "feature_caches.json"
    return common, cache, registry


def _validate_feature_registry_entry(
    registry: dict, model_name: str, expected_cache: str | Path
) -> dict:
    if registry.get("status") != "complete":
        raise IdentityMismatchError("候选特征缓存登记未完成")
    matches = [
        entry
        for entry in registry.get("entries", [])
        if entry.get("model") == model_name
    ]
    if len(matches) != 1:
        raise IdentityMismatchError(f"feature_caches.json 必须唯一登记 {model_name}")
    entry = matches[0]
    if entry.get("status") not in {"complete", "reused_complete_cache"}:
        raise IdentityMismatchError(f"{model_name} 特征登记状态不是完整缓存")
    cache = Path(expected_cache).resolve()
    registered_cache = Path(str(entry.get("cache_dir") or "")).resolve()
    registered_cls = Path(str((entry.get("cls") or {}).get("path") or "")).resolve()
    if registered_cache != cache or registered_cls != cache / "cls.npy":
        raise IdentityMismatchError(f"{model_name} 特征登记路径与冻结缓存目录不一致")
    if (entry.get("cls") or {}).get("role") != "primary_training_input":
        raise IdentityMismatchError(f"{model_name} 未登记 CLS 为唯一训练输入")
    return entry


def _load_candidate_data(config: dict, model_name: str) -> dict:
    common_path, cache_path, registry_path = _feature_locations(config, model_name)
    if not registry_path.is_file():
        raise IdentityMismatchError("缺少 feature_caches.json；必须先完成 prepare_features.ps1")
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    _validate_feature_registry_entry(registry, model_name, cache_path)
    rows = load_common_manifest(
        common_path,
        expected_counts=config["data"]["expected_counts"],
        require_images=False,
    )
    identities = [row.identity_key for row in rows]
    input_dim = {"uni": 1024, "virchow2": 1280}[model_name]
    cache = load_feature_cache(cache_path, expected_identities=identities, expected_dim=input_dim)
    features = cache["cls"]
    development_indices = [index for index, row in enumerate(rows) if row.split != "external_test"]
    development_rows = [rows[index] for index in development_indices]
    pathway_names = load_pathway_names(config["inputs"]["zscore_manifest"])
    development_labels = load_labels_for_rows(
        development_rows,
        config["paths"]["labels_root"],
        pathway_names,
    )
    train_positions = [index for index, row in enumerate(development_rows) if row.split == "train"]
    val_positions = [index for index, row in enumerate(development_rows) if row.split == "internal_val"]
    if len(train_positions) != int(config["data"]["expected_counts"]["train"]):
        raise IdentityMismatchError("训练身份计数与冻结合同不一致")
    if len(val_positions) != int(config["data"]["expected_counts"]["internal_val"]):
        raise IdentityMismatchError("内部验证身份计数与冻结合同不一致")
    val_patients = {development_rows[index].patient_id for index in val_positions}
    if val_patients != set(config["data"]["development_patients"]):
        raise IdentityMismatchError("内部验证没有恰好覆盖六名开发患者")
    external_indices = [index for index, row in enumerate(rows) if row.split == "external_test"]
    return {
        "rows": rows,
        "pathway_names": pathway_names,
        "normalization": load_normalization(config["inputs"]["normalization"], pathway_names),
        "train_rows": [development_rows[index] for index in train_positions],
        "val_rows": [development_rows[index] for index in val_positions],
        "external_rows": [rows[index] for index in external_indices],
        "train_features": np.asarray(
            features[[development_indices[index] for index in train_positions]], dtype=np.float32
        ),
        "val_features": np.asarray(
            features[[development_indices[index] for index in val_positions]], dtype=np.float32
        ),
        "external_features": np.asarray(features[external_indices], dtype=np.float32),
        "train_targets": np.asarray(development_labels[train_positions], dtype=np.float32),
        "val_targets": np.asarray(development_labels[val_positions], dtype=np.float32),
        "cache_dir": str(cache_path.resolve()),
        "cache_metadata": cache["metadata"],
        "feature_registry": str(registry_path.resolve()),
    }


def _new_batch_id() -> str:
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%d_%H%M%S}_{now.microsecond:06d}_{os.getpid()}"


def _snapshot_package(package_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ("src", "inputs", "docs", "tests"):
        source = package_root / directory
        if source.is_dir():
            shutil.copytree(
                source,
                destination / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for name in (
        "config.json",
        "package.json",
        "requirements.txt",
        "README.md",
        "ACCEPTANCE.md",
        "runner.py",
    ):
        source = package_root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for source in package_root.glob("*.ps1"):
        shutil.copy2(source, destination / source.name)
    pytest_config = package_root / "pytest.ini"
    if pytest_config.is_file():
        shutil.copy2(pytest_config, destination / pytest_config.name)


def execute_batch(config: dict, *, scope: str, device: str = "cuda") -> dict:
    """Run one official batch. Scientific parameters cannot be supplied on the CLI."""

    validate_config(config)
    units = plan_scope(scope)
    frozen = copy.deepcopy(config)
    package_root = Path(frozen["_package_root"]).resolve()
    _, _, feature_registry_path = _feature_locations(frozen, "uni")
    if not feature_registry_path.is_file():
        raise ConfigError("特征缓存登记不存在；先执行 prepare_features.ps1")
    batch_id = _new_batch_id()
    batch_dir = Path(frozen["paths"]["runs_root"]) / frozen["experiment_id"] / batch_id
    weight_batch = Path(frozen["paths"]["weights_root"]) / frozen["experiment_id"] / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    write_json(batch_dir / "config_snapshot.json", {key: value for key, value in frozen.items() if not key.startswith("_")})
    _snapshot_package(package_root, batch_dir / "provenance" / "package_snapshot")
    shutil.copy2(feature_registry_path, batch_dir / "feature_caches.json")
    package_metadata = json.loads((package_root / "package.json").read_text(encoding="utf-8-sig"))
    manifest = {
        "schema_version": "1.0",
        "experiment_id": frozen["experiment_id"],
        "protocol_version": frozen["protocol_version"],
        "code_version": package_metadata.get("code_version"),
        "batch_id": batch_id,
        "scope": scope,
        "created_at": utc_now(),
        "status": "running",
        "selection_source": "internal_val_only",
        "external_used_for_selection": False,
        "tasks": [],
    }
    weights_registry = {
        "schema_version": "1.0",
        "experiment_id": frozen["experiment_id"],
        "batch_id": batch_id,
        "weights_root": str(Path(frozen["paths"]["weights_root"]).resolve()),
        "return_policy": "server_weights_excluded_from_local_result_copy",
        "entries": [
            {
                "model": unit.model,
                "seed": unit.seed,
                "status": "pending",
                "weight_directory": str((weight_batch / f"{unit.model}_seed{unit.seed}").resolve()),
                "warmup_checkpoint": str((weight_batch / f"{unit.model}_seed{unit.seed}" / "warmup_best.pt").resolve()),
                "formal_checkpoint": str((weight_batch / f"{unit.model}_seed{unit.seed}" / "formal_best.pt").resolve()),
                "last_checkpoint": str((weight_batch / f"{unit.model}_seed{unit.seed}" / "last.pt").resolve()),
            }
            for unit in units
            if unit.kind == "train"
        ],
    }
    write_json(batch_dir / "batch_manifest.json", manifest)
    write_json(batch_dir / "model_weights.json", weights_registry)
    append_event(batch_dir / "events.jsonl", stage="batch_start", scope=scope)

    validated_references: dict[int, dict] = {}
    output_states: dict[int, dict] = {}
    candidate_data: dict[str, dict] = {}
    train_results: dict[tuple[int, str], dict] = {}

    def run_unit(unit: WorkUnit) -> dict:
        if unit.kind == "reference":
            entry = load_reference_entry(frozen["inputs"]["baseline_reference_manifest"], unit.seed)
            validated = validate_reference_files(
                entry,
                frozen["paths"]["historical_reference_root"],
                train_count=int(frozen["data"]["expected_counts"]["train"]),
                internal_count=int(frozen["data"]["expected_counts"]["internal_val"]),
                external_count=int(frozen["data"]["expected_counts"]["external_test"]),
                output_dim=int(frozen["data"]["output_dim"]),
            )
            validated_references[unit.seed] = validated
            common_path, _, _ = _feature_locations(frozen, "uni")
            common_rows = load_common_manifest(
                common_path,
                expected_counts=frozen["data"]["expected_counts"],
                require_images=False,
            )
            pathway_names = load_pathway_names(frozen["inputs"]["zscore_manifest"])
            validate_reference_identity_alignment(validated, common_rows, pathway_names)
            output_states[unit.seed] = extract_point_output_state(
                validated["resolved_paths"]["initial_weights"]
            )
            destination = batch_dir / unit.run_id
            result = import_reference_artifacts(validated, destination / "raw")
            write_json(
                destination / "run.json",
                {
                    "status": "succeeded",
                    "exit_code": 0,
                    "run_id": unit.run_id,
                    "model": "uni2h",
                    "seed": unit.seed,
                    "artifact_kind": "historical_matched_reference_reused",
                    "trained_in_this_batch": False,
                    "resource_usage": result["resource_usage"],
                },
            )
            return {
                "artifact_kind": "historical_matched_reference_reused",
                "run_relative": unit.run_id,
                "formal_epoch": int(validated["formal_epoch"]),
                "trained_in_this_batch": False,
            }

        if unit.model not in candidate_data:
            candidate_data[unit.model] = _load_candidate_data(frozen, unit.model)
        data = candidate_data[unit.model]
        if unit.kind == "train":
            if unit.seed not in output_states:
                raise ConfigError("必须先验证同 seed 的历史输出层初值")
            run_dir = batch_dir / unit.run_id
            checkpoint_dir = weight_batch / f"{unit.model}_seed{unit.seed}"
            result = train_point_arrays(
                frozen,
                model_name=unit.model,
                seed=unit.seed,
                train_features=data["train_features"],
                train_targets=data["train_targets"],
                train_rows=data["train_rows"],
                val_features=data["val_features"],
                val_targets=data["val_targets"],
                val_rows=data["val_rows"],
                pathway_names=data["pathway_names"],
                output_state=output_states[unit.seed],
                output_state_source={
                    "status": "historical_matched_reference_reused",
                    "source_experiment": validated_references[unit.seed]["source_experiment"],
                    "source_batch": validated_references[unit.seed]["source_batch"],
                    "seed": unit.seed,
                    "initial_weights": validated_references[unit.seed]["resolved_paths"]["initial_weights"],
                },
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            if result.get("status") != "completed" or not result.get("formal_checkpoint"):
                raise RuntimeError("候选训练没有产生 formal 正式端点")
            train_results[(unit.seed, unit.model)] = result
            write_json(
                run_dir / "run.json",
                {
                    **result,
                    "status": "succeeded",
                    "result_status": result.get("status"),
                    "exit_code": 0,
                    "run_id": unit.run_id,
                    "artifact_kind": "new_candidate_training",
                },
            )
            weight_entry = next(
                item
                for item in weights_registry["entries"]
                if item["model"] == unit.model and int(item["seed"]) == unit.seed
            )
            weight_entry.update(
                status="succeeded",
                weight_directory=result["weight_directory"],
                warmup_checkpoint=result["warmup_checkpoint"],
                formal_checkpoint=result["formal_checkpoint"],
                last_checkpoint=result["last_checkpoint"],
                formal_selection=result["formal_endpoint"],
                pretrained_feature_cache=data["cache_dir"],
                pretrained_encoder={
                    "repo_id": data["cache_metadata"].get("repo_id"),
                    "revision": data["cache_metadata"].get("revision"),
                    "snapshot_path": data["cache_metadata"].get("snapshot_path"),
                    "strict_load": True,
                    "frozen": True,
                },
            )
            write_json(batch_dir / "model_weights.json", weights_registry)
            return {**result, "run_relative": unit.run_id}

        if unit.kind == "external":
            train_result = train_results.get((unit.seed, unit.model))
            if not train_result or not train_result.get("formal_checkpoint"):
                raise ConfigError("XZY 预测必须等待同模型同 seed 的 formal checkpoint 锁定")
            run_dir = batch_dir / unit.run_id
            output = run_dir / "raw" / "external_predictions.npz"
            result = predict_external_arrays(
                data["external_features"],
                data["external_rows"],
                frozen,
                unit.model,
                unit.seed,
                train_result["formal_checkpoint"],
                output,
                data["normalization"],
                device,
            )
            result["run_relative"] = unit.run_id
            result["prediction_relative"] = str(output.relative_to(batch_dir))
            write_json(
                run_dir / "run.json",
                {
                    **result,
                    "status": "succeeded",
                    "result_status": result.get("status"),
                    "exit_code": 0,
                    "run_id": unit.run_id,
                    "artifact_kind": "formal_external_prediction_without_targets",
                },
            )
            return result
        raise ConfigError(f"未知工作单元: {unit}")

    failed = False
    for unit in units:
        row = {
            **asdict(unit),
            "run_id": unit.run_id,
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "result": None,
            "error": None,
        }
        if failed:
            row.update(status="not_run", error="earlier_unit_failed")
            if unit.kind == "train":
                next(
                    item
                    for item in weights_registry["entries"]
                    if item["model"] == unit.model and int(item["seed"]) == unit.seed
                )["status"] = "not_run"
            manifest["tasks"].append(row)
            write_json(batch_dir / "batch_manifest.json", manifest)
            write_json(batch_dir / "model_weights.json", weights_registry)
            continue
        try:
            row["started_at"] = utc_now()
            row["status"] = "running"
            manifest["tasks"].append(row)
            write_json(batch_dir / "batch_manifest.json", manifest)
            row["result"] = run_unit(unit)
            row["status"] = "succeeded"
            row["exit_code"] = 0
        except Exception as exc:
            row["status"] = "failed"
            row["exit_code"] = 1
            row["error"] = str(exc)
            error_dir = batch_dir / unit.run_id / "logs"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "errors.log").write_text(traceback.format_exc(), encoding="utf-8")
            if unit.kind == "train":
                next(
                    item
                    for item in weights_registry["entries"]
                    if item["model"] == unit.model and int(item["seed"]) == unit.seed
                ).update(status="failed", error=str(exc))
            failed = True
        row["ended_at"] = utc_now()
        write_json(batch_dir / "batch_manifest.json", manifest)
        write_json(batch_dir / "model_weights.json", weights_registry)

    manifest["status"] = "failed" if failed else "succeeded"
    manifest["exit_code"] = 1 if failed else 0
    manifest["ended_at"] = utc_now()
    write_json(batch_dir / "batch_manifest.json", manifest)
    write_json(batch_dir / "model_weights.json", weights_registry)
    endpoints = []
    for seed, reference in sorted(validated_references.items()):
        endpoints.append(
            {
                "model": "uni2h",
                "seed": seed,
                "arm": "point",
                "checkpoint_kind": "formal",
                "epoch": int(reference["formal_epoch"]),
                "checkpoint": reference["resolved_paths"]["formal_checkpoint_registered"],
                "status": "historical_matched_reference_reused",
            }
        )
    for (seed, model_name), result in sorted(train_results.items()):
        endpoints.append(
            {
                "model": model_name,
                "seed": seed,
                "arm": "point",
                "checkpoint_kind": "formal",
                "epoch": int(result["formal_endpoint"]["epoch"]),
                "checkpoint": result["formal_checkpoint"],
                "status": "new_candidate_training",
            }
        )
    write_json(
        batch_dir / "formal_endpoints.json",
        {
            "schema_version": "1.0",
            "protocol_version": frozen["protocol_version"],
            "selection_source": "internal_val_only",
            "external_used_for_selection": False,
            "endpoints": endpoints,
        },
    )
    append_event(batch_dir / "events.jsonl", stage="batch_end", status=manifest["status"])
    return {
        "exit_code": 1 if failed else 0,
        "status": manifest["status"],
        "batch_dir": str(batch_dir.resolve()),
        "batch_id": batch_id,
        "scope": scope,
    }
