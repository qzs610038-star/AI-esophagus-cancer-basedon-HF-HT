"""Read-only XZY external inference for an already completed v4 run.

This module never trains, selects a checkpoint, or mutates the original run
state.  Its only writable location is ``<run>/external_xzy``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import gc
import json
from pathlib import Path
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import backbone as backbone_api
from . import data, engine, metrics, models, protocols, spatial


_PATCH_PATTERN = re.compile(r"^patch_x(-?\d+)_y(-?\d+)$")


@dataclass(frozen=True)
class ExternalTask:
    kind: str
    cell: str
    endpoint: int
    checkpoint: str
    head: str | None = None
    h_mode: str | None = None
    source_result: str | None = None

    @property
    def key(self) -> str:
        suffix = "" if self.head is None else f"/{self.h_mode}/{self.head}"
        return f"{self.kind}/{self.cell}/e{self.endpoint}{suffix}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def external_output_directory(run_dir: str | Path) -> Path:
    """Return the sole output root permitted for the external-only entry."""
    return Path(run_dir).resolve() / "external_xzy"


def _coordinate(stem: str) -> tuple[int, int]:
    match = _PATCH_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"XZY patch identity lacks auditable coordinates: {stem}")
    return int(match.group(1)), int(match.group(2))


def build_xzy_records(
    data_config: Mapping[str, Any],
    *,
    pathway_names: Sequence[str],
    native_step: int,
) -> tuple[list[data.OnlineRecord], dict[str, Any]]:
    """Materialise XZY from read-only labels/images and record its exact identity."""
    label_path = Path(str(data_config.get("external_xzy_labels", "")))
    if not label_path.is_file():
        raise FileNotFoundError(f"XZY label file does not exist: {label_path}")
    patient = str(data_config.get("external_patient", "XZY"))
    if patient != "XZY":
        raise ValueError(f"external_patient must be XZY, got {patient}")
    mpp_id = int(data_config.get("external_mpp_id", 2))
    expected_points = int(data_config.get("expected_external_points", 0))
    if expected_points < 1:
        raise ValueError("data.expected_external_points must be positive")
    if int(native_step) <= 0:
        raise ValueError("XZY native_step must be positive")

    frame = pd.read_csv(label_path)
    if frame.empty:
        raise ValueError("XZY raw label file is empty")
    id_column = frame.columns[0]
    stems = [Path(str(value)).stem for value in frame[id_column]]
    if len(stems) != expected_points:
        raise ValueError(
            f"XZY point count mismatch: labels={len(stems)}, expected={expected_points}"
        )
    if len(set(stems)) != len(stems):
        raise ValueError("XZY raw labels contain duplicate point identities")
    labels, resolved_pathways = data.load_patient_raw_labels(
        label_path, pathway_names=tuple(pathway_names)
    )
    if tuple(resolved_pathways) != tuple(pathway_names):
        raise AssertionError("XZY pathway order differs from the training contract")

    image_template = str(data_config.get("image_template", ""))
    if "{patch_stem}" not in image_template:
        raise ValueError("data.image_template must contain {patch_stem}")
    image_paths = {
        stem: Path(image_template.format(
            group=mpp_id, mpp_id=mpp_id, patient=patient, patch_stem=stem
        ))
        for stem in stems
    }
    image_directories = {path.parent.resolve() for path in image_paths.values()}
    if len(image_directories) != 1:
        raise ValueError("XZY images must resolve to one read-only directory")
    image_directory = next(iter(image_directories))
    actual_image_stems = {
        path.stem for path in image_directory.glob("*.png") if path.is_file()
    }
    if set(stems) != actual_image_stems:
        missing = sorted(set(stems) - actual_image_stems)[:5]
        extra = sorted(actual_image_stems - set(stems))[:5]
        raise ValueError(
            "XZY image identities do not match raw labels: "
            f"missing={missing}, extra={extra}"
        )

    coordinates = [_coordinate(stem) for stem in stems]
    origin_x = min(x for x, _ in coordinates)
    origin_y = min(y for _, y in coordinates)
    off_grid = [
        stem for stem, (x, y) in zip(stems, coordinates)
        if (x - origin_x) % int(native_step) or (y - origin_y) % int(native_step)
    ]
    if off_grid:
        raise ValueError(
            f"XZY coordinates violate the configured native spatial grid: {off_grid[:5]}"
        )

    records = [
        data.OnlineRecord(
            mpp_id=mpp_id,
            patient=patient,
            patch_stem=stem,
            x=x,
            y=y,
            original_split="external_test",
            block_id="external_xzy_read_only",
            image_path=image_paths[stem],
            raw_target=np.asarray(labels[stem], dtype=np.float32),
        )
        for stem, (x, y) in zip(stems, coordinates)
    ]
    audit = {
        "input_mode": "read_only",
        "selection_used": False,
        "patient": patient,
        "mpp_id": mpp_id,
        "point_count": len(records),
        "expected_point_count": expected_points,
        "native_step": int(native_step),
        "grid_origin": {"x": origin_x, "y": origin_y},
        "identity_source": "raw_label_first_column",
        "coordinate_source": "patch_stem_regex",
        "label_path": str(label_path.resolve()),
        "image_directory": str(image_directory),
        "pathway_names": list(pathway_names),
    }
    return records, audit


def _require_checkpoint(path: str | Path, weight_root: Path) -> str:
    checkpoint = Path(path).resolve()
    try:
        checkpoint.relative_to(weight_root.resolve())
    except ValueError as error:
        raise ValueError(f"checkpoint escapes the recorded weight directory: {checkpoint}") from error
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recorded checkpoint is missing: {checkpoint}")
    return str(checkpoint)


def _best_stage1_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if len(history) != 5:
        raise ValueError("stage1 history must contain exactly five fixed epochs")
    row = min(
        history,
        key=lambda item: (
            -float(item["patient_macro_pathway_pcc"]),
            float(item["zMSE"]),
            int(item["epoch"]),
        ),
    )
    return int(row["epoch"])


def build_external_plan(
    run_dir: str | Path,
    weight_root: str | Path,
    *,
    seed: int = 42,
) -> list[ExternalTask]:
    """Build an evaluation-only plan from already selected run artifacts."""
    run_root = Path(run_dir).resolve()
    weights = Path(weight_root).resolve()
    seed_root = run_root / "raw" / "original" / "original" / f"seed_{int(seed)}"
    stage1_results: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in sorted((seed_root / "stage1").glob("*.json")):
        stage1_results[path.stem] = (path, _read_json(path))
    if not stage1_results:
        raise FileNotFoundError("completed run contains no stage1 result files")

    tasks: list[ExternalTask] = []
    for cell, (source, result) in stage1_results.items():
        checkpoint_directory = Path(result["checkpoint_directory"])
        fixed_checkpoint = _require_checkpoint(checkpoint_directory / "epoch_5.pt", weights)
        tasks.append(ExternalTask(
            "stage1_fixed_e5", cell, 5, fixed_checkpoint,
            source_result=str(source),
        ))
    for cell, (source, result) in stage1_results.items():
        endpoint = _best_stage1_epoch(result["history"])
        if endpoint != 5:
            checkpoint = _require_checkpoint(
                Path(result["checkpoint_directory"]) / f"epoch_{endpoint}.pt", weights
            )
            tasks.append(ExternalTask(
                "stage1_best_epoch_sensitivity", cell, endpoint, checkpoint,
                source_result=str(source),
            ))

    groups = (
        ("stage2_fixed_e5", seed_root / "fixed_e5" / "stage2_results.json"),
        (
            "stage2_best_epoch_sensitivity",
            seed_root / "best_epoch_sensitivity" / "stage2_results.json",
        ),
    )
    for kind, source in groups:
        for item in _read_json(source):
            task, result = item["task"], item["result"]
            tasks.append(ExternalTask(
                kind=kind,
                cell=str(task["cell"]),
                endpoint=int(task["endpoint"]),
                checkpoint=_require_checkpoint(result["checkpoint"], weights),
                head=str(task["head"]),
                h_mode=str(task["h_mode"]),
                source_result=str(source),
            ))
    return tasks


def serialise_plan(tasks: Sequence[ExternalTask]) -> list[dict[str, Any]]:
    return [asdict(task) | {"key": task.key} for task in tasks]


def evaluate_stage2_checkpoint(
    task: ExternalTask,
    *,
    features: np.ndarray,
    frozen_features: np.ndarray,
    records: Sequence[data.OnlineRecord],
    normalization: Mapping[str, Any],
    native_step: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Evaluate one already-selected stage-two checkpoint on XZY."""
    if task.head not in {"point", "spatial"}:
        raise ValueError("stage2 external task must specify point or spatial head")
    if not records or {record.patient for record in records} != {"XZY"}:
        raise ValueError("stage2 external evaluation accepts XZY records only")
    feature_array = np.asarray(features, dtype=np.float32)
    frozen_array = np.asarray(frozen_features, dtype=np.float32)
    if feature_array.shape != frozen_array.shape or feature_array.shape[0] != len(records):
        raise ValueError("XZY feature arrays must align with the external identity list")
    target_raw = np.stack([record.raw_target for record in records]).astype(np.float32)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if mean.shape != (30,) or std.shape != (30,) or np.any(std <= 0):
        raise ValueError("training-only normalization must contain 30 positive scales")
    target_z = (target_raw - mean) / std

    torch_device = torch.device(device)
    head = models.Stage2Head(
        models.SharedProjection(), mode=str(task.head), shared_mode="freeze"
    ).to(torch_device)
    state = torch.load(task.checkpoint, map_location=torch_device, weights_only=True)
    head.load_state_dict(state, strict=True)
    head.eval()
    feature_tensor = torch.as_tensor(feature_array, device=torch_device)
    graph = None
    with torch.no_grad():
        if task.head == "spatial":
            graph = spatial.build_spatial_graph(
                records,
                frozen_array,
                native_steps={"XZY": float(native_step)},
            )
            delta = graph.spatial_delta(head.shared(feature_tensor))
            prediction_z = head(feature_tensor, delta).float().cpu().numpy()
        else:
            prediction_z = head(feature_tensor).float().cpu().numpy()
    prediction_raw = prediction_z * std + mean
    patient_ids = [record.patient for record in records]
    report = metrics.regression_metrics(
        prediction_z,
        target_z,
        patient_ids,
        raw_pred=prediction_raw,
        raw_target=target_raw,
    )
    if graph is not None:
        report["spatial_residual_metrics"] = metrics.spatial_residual_metrics(
            prediction_z, target_z, graph
        )
    return {
        "selection_used": False,
        "split": "external_xzy",
        "task": asdict(task),
        "metrics": report,
        "prediction_z": prediction_z,
        "target_z": target_z,
        "prediction_raw": prediction_raw,
        "target_raw": target_raw,
    }


def _validate_completed_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _read_json(run_root / "run.json")
    if record.get("experiment_id") != "phase2_softlink_contrastive_v3":
        raise ValueError("RunDirectory is not a phase2_softlink_contrastive_v3 run")
    if record.get("protocol") != "original" or [int(x) for x in record.get("seeds", [])] != [42]:
        raise ValueError("XZY external entry currently accepts the completed original/seed-42 run only")
    state = _read_json(run_root / "raw" / "engine_state.json")
    incomplete = [key for key, value in state.get("tasks", {}).items() if value.get("status") != "completed"]
    if not state.get("tasks") or incomplete:
        raise ValueError(f"original run is not complete: {incomplete[:5]}")
    resume_path = run_root / "logs" / "resume_history.jsonl"
    resume_rows = [
        json.loads(line) for line in resume_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not resume_rows or resume_rows[-1].get("status") != "succeeded" or int(resume_rows[-1].get("exit_code", 1)) != 0:
        raise ValueError("latest original-run resume record is not successful")
    return record, state


def _pathway_names(config: Mapping[str, Any], package_root: Path) -> tuple[str, ...]:
    location = config.get("data", {}).get("zscore_manifest_file")
    if not location:
        raise ValueError("run config lacks data.zscore_manifest_file")
    path = Path(str(location))
    if not path.is_absolute():
        path = package_root / path
    manifest = _read_json(path)
    names = tuple(str(value) for value in manifest.get("pathway_names", ()))
    if len(names) != 30:
        raise ValueError("zscore manifest must record the exact 30-pathway order")
    return names


def _task_output_paths(output_root: Path, task: ExternalTask) -> tuple[Path, Path]:
    directory = output_root / task.kind
    if task.head is None:
        stem = f"{task.cell}_e{task.endpoint}"
    else:
        stem = f"{task.cell}_{task.h_mode}_{task.head}_e{task.endpoint}"
    return directory / f"{stem}.json", directory / f"{stem}.npz"


def _compact_result(result: Mapping[str, Any], prediction_path: Path) -> dict[str, Any]:
    return {
        "selection_used": False,
        "split": "external_xzy",
        "task": result["task"],
        "metrics": result["metrics"],
        "prediction_file": str(prediction_path.resolve()),
    }


def _save_prediction_arrays(
    path: Path,
    result: Mapping[str, Any],
    records: Sequence[data.OnlineRecord],
    *,
    features: np.ndarray | None = None,
) -> None:
    payload: dict[str, Any] = {
        "prediction_z": np.asarray(result["prediction_z"], dtype=np.float32),
        "target_z": np.asarray(result["target_z"], dtype=np.float32),
        "prediction_raw": np.asarray(result["prediction_raw"], dtype=np.float32),
        "target_raw": np.asarray(result["target_raw"], dtype=np.float32),
        "patient": np.asarray([record.patient for record in records], dtype=str),
        "patch_stem": np.asarray([record.patch_stem for record in records], dtype=str),
        "x": np.asarray([record.x for record in records], dtype=np.int64),
        "y": np.asarray([record.y for record in records], dtype=np.int64),
    }
    if features is not None:
        payload["cls"] = np.asarray(features, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_external_feature_cache(
    path: str | Path,
    records: Sequence[data.OnlineRecord],
) -> np.ndarray:
    """Load a resumable CLS cache only when its complete XZY identity matches."""
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as arrays:
        required = {"cls", "patient", "patch_stem", "x", "y"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"external feature cache lacks identity fields: {missing}")
        expected = {
            "patient": np.asarray([record.patient for record in records], dtype=str),
            "patch_stem": np.asarray([record.patch_stem for record in records], dtype=str),
            "x": np.asarray([record.x for record in records], dtype=np.int64),
            "y": np.asarray([record.y for record in records], dtype=np.int64),
        }
        for name, value in expected.items():
            if not np.array_equal(np.asarray(arrays[name]), value):
                raise AssertionError(
                    f"external feature cache identity mismatch for {name}: {cache_path}"
                )
        features = np.asarray(arrays["cls"], dtype=np.float32)
    if features.shape != (len(records), 1536) or not np.isfinite(features).all():
        raise ValueError(
            f"external CLS cache must be finite with shape ({len(records)}, 1536): "
            f"{cache_path} has {features.shape}"
        )
    return features


def _stage1_external_inference(
    task: ExternalTask,
    *,
    runtime: engine.TorchRuntime,
    context: Mapping[str, Any],
    records: Sequence[data.OnlineRecord],
    source_result: Mapping[str, Any],
    seed: int,
    capture_frozen: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    device = runtime._device(context)
    model = runtime._stage_model(
        context, seed, task.cell, {"checkpoint": source_result["warmup_checkpoint"]}
    )
    state = torch.load(task.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cls_rows: list[np.ndarray] = []
    frozen_rows: list[np.ndarray] = []
    batch_size = int(context["resource_short_test"]["batch_size"])
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        images, target, _ = runtime._batch(
            context, records, range(start, stop), seed=seed, augment=False
        )
        with torch.no_grad():
            with runtime._autocast(context, device):
                output = model(images)
                if capture_frozen:
                    with models.lora_disabled(model.backbone):
                        frozen = backbone_api.forward_cls(model.backbone, images)
        predictions.append(output["pred"].float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
        cls_rows.append(output["cls"].float().cpu().numpy())
        if capture_frozen:
            frozen_rows.append(frozen.float().cpu().numpy())
    prediction_z = np.concatenate(predictions)
    target_z = np.concatenate(targets)
    cls_values = np.concatenate(cls_rows)
    frozen_values = np.concatenate(frozen_rows) if capture_frozen else None
    normalization = runtime._normalization(context)
    prediction_raw = prediction_z * normalization["std"] + normalization["mean"]
    target_raw = np.stack([record.raw_target for record in records]).astype(np.float32)
    report = metrics.regression_metrics(
        prediction_z,
        target_z,
        [record.patient for record in records],
        raw_pred=prediction_raw,
        raw_target=target_raw,
    )
    result = {
        "selection_used": False,
        "split": "external_xzy",
        "task": asdict(task),
        "metrics": report,
        "prediction_z": prediction_z,
        "target_z": target_z,
        "prediction_raw": prediction_raw,
        "target_raw": target_raw,
    }
    del model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, cls_values, frozen_values


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    task = result["task"]
    metric = result["metrics"]
    return {
        "kind": task["kind"],
        "cell": task["cell"],
        "endpoint": task["endpoint"],
        "h_mode": task.get("h_mode"),
        "head": task.get("head"),
        "patient_macro_pathway_pcc": metric["patient_macro_pathway_pcc"],
        "pooled_pcc": metric["pooled_pcc"],
        "CCC": metric["CCC"],
        "zRMSE": metric["zRMSE"],
        "zMAE": metric["zMAE"],
        "rawMAE": metric["rawMAE"],
        "R2": metric["R2"],
        "bias_z": metric["bias_z"],
        "selection_used": False,
    }


def _write_input_manifest(path: Path, records: Sequence[data.OnlineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mpp_id", "patient", "patch_stem", "x", "y", "split", "image_path"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "mpp_id": record.mpp_id,
                "patient": record.patient,
                "patch_stem": record.patch_stem,
                "x": record.x,
                "y": record.y,
                "split": record.original_split,
                "image_path": str(record.image_path.resolve()),
            })


def run_external_inference(
    run_dir: str | Path,
    *,
    native_step: int = 224,
    batch_size: int | None = None,
    resume: bool = False,
    plan_only: bool = False,
    code_version: str = "v3.0.5",
) -> dict[str, Any]:
    """Run XZY inference only and keep every new file under external_xzy."""
    package_root = Path(__file__).resolve().parents[1]
    run_root = Path(run_dir).resolve()
    output_root = external_output_directory(run_root)
    record, _ = _validate_completed_run(run_root)
    weight_root = Path(record["weight_directory"]).resolve()
    config = _read_json(run_root / "config.json")
    pathway_names = _pathway_names(config, package_root)
    records, input_audit = build_xzy_records(
        config["data"], pathway_names=pathway_names, native_step=int(native_step)
    )
    plan = build_external_plan(run_root, weight_root, seed=42)
    if plan_only:
        return {
            "exit_code": 0,
            "status": "planned",
            "mode": "external_inference_only",
            "selection_used": False,
            "training_performed": False,
            "input_mode": "read_only",
            "run_directory": str(run_root),
            "output_directory": str(output_root),
            "input": input_audit,
            "task_count": len(plan),
            "plan": serialise_plan(plan),
        }
    if output_root.exists() and not resume:
        raise FileExistsError(
            f"external output already exists; use -Resume to continue: {output_root}"
        )
    output_root.mkdir(parents=False, exist_ok=True)
    state_path = output_root / "state.json"
    if state_path.is_file():
        state = _read_json(state_path)
    elif resume:
        raise FileNotFoundError("-Resume requires external_xzy/state.json")
    else:
        state = {"schema_version": "v1", "tasks": {}}
    run_record = {
        "schema_version": "v1",
        "experiment_id": record["experiment_id"],
        "parent_run_id": record["run_id"],
        "code_version": code_version,
        "mode": "external_inference_only",
        "selection_used": False,
        "training_performed": False,
        "input_mode": "read_only",
        "native_step": int(native_step),
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(output_root / "run.json", run_record)
    _write_json(output_root / "input_validation.json", input_audit)
    _write_json(output_root / "plan.json", serialise_plan(plan))
    _write_input_manifest(output_root / "xzy_manifest.csv", records)

    training_artifacts = _read_json(
        run_root / "raw" / "original" / "original" / "training_artifacts.json"
    )
    resource = _read_json(
        run_root / "raw" / "original" / "original" / "resource_short_test.json"
    )
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("BatchSize must be positive")
        resource = {**resource, "batch_size": int(batch_size)}
    split = protocols.ProtocolSplit(
        "original", "original",
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
    )
    context = {
        "config": config,
        "split": split,
        "partitions": {"xzy": records},
        "artifacts": training_artifacts,
        "resource_short_test": resource,
    }
    runtime = engine.TorchRuntime()
    features: dict[tuple[str, int], np.ndarray] = {}
    frozen_path = output_root / "features" / "frozen_uni_cls.npz"
    frozen_features = None
    if frozen_path.is_file():
        frozen_features = load_external_feature_cache(frozen_path, records)
    summary_rows: list[dict[str, Any]] = []

    try:
        for task in [item for item in plan if item.head is None]:
            result_path, prediction_path = _task_output_paths(output_root, task)
            complete = state["tasks"].get(task.key, {}).get("status") == "completed"
            if complete and result_path.is_file() and prediction_path.is_file():
                saved = _read_json(result_path)
                features[(task.cell, task.endpoint)] = load_external_feature_cache(
                    prediction_path, records
                )
                summary_rows.append(_summary_row(saved))
                continue
            state["tasks"][task.key] = {"status": "running"}
            _write_json(state_path, state)
            source_result = _read_json(Path(str(task.source_result)))
            result, cls_values, frozen_values = _stage1_external_inference(
                task,
                runtime=runtime,
                context=context,
                records=records,
                source_result=source_result,
                seed=42,
                capture_frozen=frozen_features is None,
            )
            features[(task.cell, task.endpoint)] = cls_values
            if frozen_features is None:
                assert frozen_values is not None
                frozen_features = frozen_values
                frozen_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    frozen_path,
                    cls=frozen_features,
                    patient=np.asarray([record.patient for record in records], dtype=str),
                    patch_stem=np.asarray([record.patch_stem for record in records], dtype=str),
                    x=np.asarray([record.x for record in records]),
                    y=np.asarray([record.y for record in records]),
                )
            _save_prediction_arrays(prediction_path, result, records, features=cls_values)
            compact = _compact_result(result, prediction_path)
            _write_json(result_path, compact)
            state["tasks"][task.key] = {
                "status": "completed",
                "result_path": str(result_path.resolve()),
                "prediction_path": str(prediction_path.resolve()),
            }
            _write_json(state_path, state)
            summary_rows.append(_summary_row(compact))

        if frozen_features is None:
            raise RuntimeError("external inference did not produce frozen UNI morphology features")
        normalization = runtime._normalization(context)
        for task in [item for item in plan if item.head is not None]:
            result_path, prediction_path = _task_output_paths(output_root, task)
            complete = state["tasks"].get(task.key, {}).get("status") == "completed"
            if complete and result_path.is_file() and prediction_path.is_file():
                saved = _read_json(result_path)
                summary_rows.append(_summary_row(saved))
                continue
            key = (task.cell, task.endpoint)
            if key not in features:
                raise FileNotFoundError(f"missing XZY stage1 feature cache for {key}")
            state["tasks"][task.key] = {"status": "running"}
            _write_json(state_path, state)
            result = evaluate_stage2_checkpoint(
                task,
                features=features[key],
                frozen_features=frozen_features,
                records=records,
                normalization=normalization,
                native_step=int(native_step),
                device=runtime._device(context),
            )
            _save_prediction_arrays(prediction_path, result, records)
            compact = _compact_result(result, prediction_path)
            _write_json(result_path, compact)
            state["tasks"][task.key] = {
                "status": "completed",
                "result_path": str(result_path.resolve()),
                "prediction_path": str(prediction_path.resolve()),
            }
            _write_json(state_path, state)
            summary_rows.append(_summary_row(compact))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        frame = pd.DataFrame(summary_rows).sort_values(
            ["kind", "cell", "endpoint", "h_mode", "head"], na_position="first"
        )
        frame.to_csv(output_root / "metrics_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "schema_version": "v1",
            "status": "completed",
            "selection_used": False,
            "training_performed": False,
            "input_mode": "read_only",
            "point_count": len(records),
            "native_step": int(native_step),
            "task_count": len(plan),
            "completed_task_count": sum(
                value.get("status") == "completed" for value in state["tasks"].values()
            ),
            "fixed_stage1_count": sum(task.kind == "stage1_fixed_e5" for task in plan),
            "sensitivity_stage1_count": sum(task.kind == "stage1_best_epoch_sensitivity" for task in plan),
            "fixed_stage2_count": sum(task.kind == "stage2_fixed_e5" for task in plan),
            "sensitivity_stage2_count": sum(task.kind == "stage2_best_epoch_sensitivity" for task in plan),
            "metrics_summary": str((output_root / "metrics_summary.csv").resolve()),
        }
        _write_json(output_root / "summary.json", summary)
        run_record.update({
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "exit_code": 0,
            "task_count": len(plan),
        })
        _write_json(output_root / "run.json", run_record)
        return {"exit_code": 0, **summary, "output_directory": str(output_root)}
    except Exception as error:
        run_record.update({
            "status": "failed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "exit_code": 1,
            "error": str(error),
        })
        _write_json(output_root / "run.json", run_record)
        _write_json(output_root / "error.json", {"error": str(error)})
        raise
