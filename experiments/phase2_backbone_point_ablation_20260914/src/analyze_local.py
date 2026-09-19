"""Local-only derivation of double-PCC and error reports from returned raw outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from config import load_config
from data import Normalization, load_normalization, load_pathway_names
from errors import IdentityMismatchError, NonFiniteDataError
from run_io import utc_now, write_json
from selection import patient_macro_metrics, pooled_pathway_pcc, pooled_z_mse


def _pearson_flat(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    pred_centered = pred - pred.mean()
    target_centered = truth - truth.mean()
    denominator = float(np.sqrt(np.sum(pred_centered**2) * np.sum(target_centered**2)))
    if denominator == 0.0:
        return None
    return _clamp_correlation(float(np.sum(pred_centered * target_centered) / denominator))


def _clamp_correlation(value: float) -> float:
    """Remove harmless floating-point spill at the correlation boundaries."""

    if abs(value - 1.0) <= 1e-15:
        return 1.0
    if abs(value + 1.0) <= 1e-15:
        return -1.0
    return float(value)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        end = start + 1
        while end < flat.size and flat[order[end]] == flat[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman_flat(prediction: np.ndarray, target: np.ndarray) -> float | None:
    return _pearson_flat(_average_ranks(prediction), _average_ranks(target))


def _ccc_flat(prediction: np.ndarray, target: np.ndarray) -> float | None:
    """Lin's concordance correlation coefficient using population moments."""

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    pred_centered = pred - pred.mean()
    target_centered = truth - truth.mean()
    covariance = float(np.mean(pred_centered * target_centered))
    denominator = float(
        np.mean(pred_centered**2)
        + np.mean(target_centered**2)
        + (pred.mean() - truth.mean()) ** 2
    )
    if denominator == 0.0:
        return None
    return _clamp_correlation(2.0 * covariance / denominator)


def _r2(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    ss_residual = float(np.sum((pred - truth) ** 2))
    ss_total = float(np.sum((truth - truth.mean()) ** 2))
    return None if ss_total == 0.0 else float(1.0 - ss_residual / ss_total)


def compute_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    patient_ids: Sequence[str],
    normalization: Normalization,
    *,
    raw_target: np.ndarray | None = None,
) -> dict:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise IdentityMismatchError("预测与真值必须是同形二维矩阵")
    if pred.shape[1] != len(normalization.pathway_names):
        raise IdentityMismatchError("预测列与标准化通路顺序不一致")
    if len(patient_ids) != pred.shape[0]:
        raise IdentityMismatchError("患者身份数与预测行数不一致")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise NonFiniteDataError("分析输入含 NaN/Inf")
    macro = patient_macro_metrics(
        pred,
        target,
        patient_ids,
        pathway_names=normalization.pathway_names,
        constant_prediction_selection_penalty=-1.0,
    )
    pooled_pathways = pooled_pathway_pcc(pred, target, normalization.pathway_names)
    pred_raw = normalization.inverse_transform(pred)
    inverse_z_target_raw = normalization.inverse_transform(target)
    if raw_target is None:
        target_raw = inverse_z_target_raw
        raw_scope = "inverse_of_clipped_z_target_not_original_raw_truth"
        raw_truth_difference_mae = None
        raw_truth_difference_max = None
    else:
        target_raw = np.asarray(raw_target, dtype=np.float64)
        if target_raw.shape != pred.shape or not np.isfinite(target_raw).all():
            raise IdentityMismatchError("身份对齐的原始真值形状/数值非法")
        raw_scope = "identity_aligned_original_raw_truth"
        raw_truth_difference = target_raw - inverse_z_target_raw
        raw_truth_difference_mae = float(np.mean(np.abs(raw_truth_difference)))
        raw_truth_difference_max = float(np.max(np.abs(raw_truth_difference)))
    residual = pred_raw - target_raw
    z_mse = pooled_z_mse(pred, target)
    per_pathway_raw_rmse = np.sqrt(np.mean(residual**2, axis=0))
    return {
        "patient_macro_pathway_pcc_selection": macro.patient_macro_pathway_pcc,
        "patient_macro_pathway_pcc_measured": macro.patient_macro_pathway_pcc_measured,
        "patient_macro_z_mse_selection": macro.patient_macro_z_mse_selection,
        "pooled_flattened_pcc": _pearson_flat(pred, target),
        "pooled_pathway_pcc": pooled_pathways["macro_pathway_pcc"],
        "pooled_z_mse": z_mse,
        "pooled_z_rmse": float(np.sqrt(z_mse)),
        "pooled_z_mae": float(np.mean(np.abs(pred - target))),
        "pooled_spearman": _spearman_flat(pred, target),
        "pooled_ccc": _ccc_flat(pred, target),
        "raw_mae": float(np.mean(np.abs(residual))),
        "raw_r2": _r2(pred_raw, target_raw),
        "train_std_normalized_rmse_mean": float(
            np.mean(per_pathway_raw_rmse / normalization.std)
        ),
        "raw_truth_scope": raw_scope,
        "raw_target_vs_inverse_z_mae": raw_truth_difference_mae,
        "raw_target_vs_inverse_z_max_abs": raw_truth_difference_max,
        "n_points": int(pred.shape[0]),
        "n_pathways": int(pred.shape[1]),
        "n_patients": macro.n_patients,
        "n_valid_patient_pathway": macro.n_valid_patient_pathway,
        "n_excluded_patient_pathway": macro.n_excluded_patient_pathway,
        "n_constant_prediction_penalized": macro.n_constant_prediction_penalized,
    }


def _metric_details(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    patient_ids: Sequence[str],
    normalization: Normalization,
    *,
    raw_target: np.ndarray | None = None,
) -> dict:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    pred_raw = normalization.inverse_transform(pred)
    target_raw = (
        normalization.inverse_transform(target)
        if raw_target is None
        else np.asarray(raw_target, dtype=np.float64)
    )
    patients = np.asarray(list(patient_ids), dtype=str)
    macro = patient_macro_metrics(
        pred,
        target,
        patients,
        pathway_names=normalization.pathway_names,
        constant_prediction_selection_penalty=-1.0,
    )
    per_pathway: list[dict] = []
    for index, name in enumerate(normalization.pathway_names):
        z_residual = pred[:, index] - target[:, index]
        raw_residual = pred_raw[:, index] - target_raw[:, index]
        raw_rmse = float(np.sqrt(np.mean(raw_residual**2)))
        per_pathway.append(
            {
                "pathway": name,
                "n_points": int(pred.shape[0]),
                "pooled_pcc": _pearson_flat(pred[:, index], target[:, index]),
                "spearman": _spearman_flat(pred[:, index], target[:, index]),
                "ccc": _ccc_flat(pred[:, index], target[:, index]),
                "z_rmse": float(np.sqrt(np.mean(z_residual**2))),
                "z_mae": float(np.mean(np.abs(z_residual))),
                "raw_mae": float(np.mean(np.abs(raw_residual))),
                "raw_r2": _r2(pred_raw[:, index], target_raw[:, index]),
                "train_sd": float(normalization.std[index]),
                "train_std_normalized_rmse": raw_rmse / float(normalization.std[index]),
            }
        )
    per_patient: list[dict] = []
    for patient_id in sorted(set(patients.tolist())):
        mask = patients == patient_id
        z_residual = pred[mask] - target[mask]
        raw_residual = pred_raw[mask] - target_raw[mask]
        selection_record = macro.per_patient[patient_id]
        per_patient.append(
            {
                "patient_id": patient_id,
                "n_points": int(np.sum(mask)),
                "selection_pcc_mean": selection_record["selection_pcc_mean"],
                "measured_pcc_mean": selection_record["measured_pcc_mean"],
                "z_mse_mean": selection_record["z_mse_mean"],
                "pooled_flattened_pcc": _pearson_flat(pred[mask], target[mask]),
                "spearman": _spearman_flat(pred[mask], target[mask]),
                "ccc": _ccc_flat(pred[mask], target[mask]),
                "z_rmse": float(np.sqrt(np.mean(z_residual**2))),
                "z_mae": float(np.mean(np.abs(z_residual))),
                "raw_mae": float(np.mean(np.abs(raw_residual))),
                "raw_r2": _r2(pred_raw[mask], target_raw[mask]),
                "n_valid_pathways": selection_record["n_valid_pathways"],
            }
        )
    return {
        "per_pathway": per_pathway,
        "per_patient": per_patient,
        "patient_pathway_selection_details": macro.per_patient,
        "selection_exclusions": macro.exclusions,
    }


def _first_field(archive, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise IdentityMismatchError(f"预测附件缺字段，候选={list(names)}")


def _read_prediction(
    path: Path,
    *,
    require_target: bool,
    expected_model: str,
    expected_seed: int,
) -> dict:
    # The accepted historical archives use object string arrays. This local
    # analysis path only reads files explicitly selected by batch_manifest.json.
    with np.load(path, allow_pickle=True) as archive:
        endpoint_metadata = {}
        if "checkpoint_metadata" in archive.files:
            try:
                endpoint_metadata = dict(np.asarray(archive["checkpoint_metadata"]).item())
            except (TypeError, ValueError) as exc:
                raise IdentityMismatchError(f"预测附件 checkpoint_metadata 非法: {path}") from exc
        arm = (
            np.asarray(archive["arm"]).item()
            if "arm" in archive.files
            else endpoint_metadata.get("arm")
        )
        seed = (
            np.asarray(archive["seed"]).item()
            if "seed" in archive.files
            else endpoint_metadata.get("seed")
        )
        model = np.asarray(archive["model"]).item() if "model" in archive.files else None
        if arm != "point":
            raise IdentityMismatchError(f"预测附件不是 point 臂: {path}")
        if seed is None or int(seed) != int(expected_seed):
            raise IdentityMismatchError(
                f"预测附件 seed 不匹配: expected={expected_seed}, actual={seed}"
            )
        if expected_model != "uni2h" and model != expected_model:
            raise IdentityMismatchError(
                f"预测附件 model 不匹配: expected={expected_model}, actual={model}"
            )
        if not require_target:
            checkpoint_kind = (
                np.asarray(archive["checkpoint_kind"]).item()
                if "checkpoint_kind" in archive.files
                else endpoint_metadata.get("checkpoint_kind")
            )
            if checkpoint_kind != "formal":
                raise IdentityMismatchError(f"外部预测不是 formal 正式端点: {path}")
        result = {
            "pred_z": np.asarray(archive["pred_z"], dtype=np.float64),
            "patients": _first_field(archive, ("patients", "patient_id")).astype(str),
            "source_groups": _first_field(archive, ("source_groups", "slide_id")).astype(str),
            "spots": _first_field(archive, ("spots", "spot_id")).astype(str),
            "pathways": _first_field(archive, ("pathways", "pathway_names")).astype(str),
        }
        if require_target:
            if "target_z" not in archive.files:
                raise IdentityMismatchError(f"内部预测缺 target_z: {path}")
            result["target_z"] = np.asarray(archive["target_z"], dtype=np.float64)
        elif "target_z" in archive.files:
            raise IdentityMismatchError(f"服务器 XZY 预测不应携带 target_z: {path}")
        elif "pred_raw" not in archive.files:
            raise IdentityMismatchError(f"服务器 XZY 预测缺少一次逆变换的 pred_raw: {path}")
        else:
            result["pred_raw"] = np.asarray(archive["pred_raw"], dtype=np.float64)
    if result["pred_z"].ndim != 2 or not np.isfinite(result["pred_z"]).all():
        raise IdentityMismatchError(f"预测附件 pred_z 形状或数值非法: {path}")
    if not require_target and (
        result["pred_raw"].shape != result["pred_z"].shape
        or not np.isfinite(result["pred_raw"]).all()
    ):
        raise IdentityMismatchError(f"预测附件 pred_raw 形状或数值非法: {path}")
    if len(set(zip(result["patients"], result["source_groups"], result["spots"]))) != len(result["patients"]):
        raise IdentityMismatchError(f"预测附件身份重复: {path}")
    return result


def _align_external_target(prediction: dict, target_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(target_path, allow_pickle=False) as archive:
        target = {
            "target_z": np.asarray(archive["target_z"], dtype=np.float64),
            "patients": _first_field(archive, ("patients", "patient_id")).astype(str),
            "source_groups": _first_field(archive, ("source_groups", "slide_id")).astype(str),
            "spots": _first_field(archive, ("spots", "spot_id")).astype(str),
            "pathways": _first_field(archive, ("pathways", "pathway_names")).astype(str),
            "target_raw": np.asarray(archive["target_raw"], dtype=np.float64) if "target_raw" in archive.files else None,
        }
    if target["pathways"].tolist() != prediction["pathways"].tolist():
        raise IdentityMismatchError("XZY 真值与预测通路顺序不一致")
    target_index = {
        (patient, source, spot): index
        for index, (patient, source, spot) in enumerate(
            zip(target["patients"], target["source_groups"], target["spots"])
        )
    }
    prediction_keys = list(zip(prediction["patients"], prediction["source_groups"], prediction["spots"]))
    if len(target_index) != len(target["patients"]) or set(prediction_keys) != set(target_index):
        raise IdentityMismatchError("XZY 真值与预测身份集合不一致")
    order = [target_index[key] for key in prediction_keys]
    target_z = target["target_z"][order]
    target_raw = None if target["target_raw"] is None else target["target_raw"][order]
    return target_z, target_raw


def _aggregate(rows: list[dict], *, split: str = "internal_val") -> dict:
    metric_names = (
        "patient_macro_pathway_pcc_measured",
        "pooled_flattened_pcc",
        "pooled_pathway_pcc",
        "pooled_z_mse",
        "pooled_z_rmse",
        "pooled_z_mae",
        "pooled_spearman",
        "pooled_ccc",
        "train_std_normalized_rmse_mean",
        "raw_mae",
        "raw_r2",
    )
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == split:
            by_model.setdefault(row["model"], []).append(row)
    output: dict[str, dict] = {}
    for model, values in by_model.items():
        record = {
            "n_seeds": len(values),
            "seeds": sorted(int(value["seed"]) for value in values),
            "three_seed_scope_complete": sorted(int(value["seed"]) for value in values)
            == [42, 43, 44],
            "mean": {},
        }
        for metric in metric_names:
            finite = [
                float(value[metric])
                for value in values
                if value.get(metric) is not None and np.isfinite(float(value[metric]))
            ]
            record["mean"][metric] = float(np.mean(finite)) if finite else None
        if sorted(int(value["seed"]) for value in values) == [42, 43, 44]:
            record["sample_sd"] = {}
            for metric in metric_names:
                finite = [
                    float(value[metric])
                    for value in values
                    if value.get(metric) is not None and np.isfinite(float(value[metric]))
                ]
                record["sample_sd"][metric] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else None
        output[model] = record
    return output


def _head_parameter_count(model_name: str) -> int:
    input_dim = {"uni2h": 1536, "uni": 1024, "virchow2": 1280}[model_name]
    return int(input_dim * 256 + 256 + 256 * 30 + 30)


def _load_feature_runtime(batch: Path) -> dict[str, dict]:
    registry_path = batch / "feature_caches.json"
    if not registry_path.is_file():
        return {}
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    output: dict[str, dict] = {}
    for entry in registry.get("entries", []):
        model = str(entry.get("model") or "")
        runtime = entry.get("runtime")
        if model and isinstance(runtime, dict):
            output[model] = runtime
    return output


def _resource_fields(task: dict, feature_runtime: dict[str, dict]) -> dict:
    model = str(task["model"])
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    training_runtime = (
        result.get("resource_usage")
        if isinstance(result.get("resource_usage"), dict)
        else {}
    )
    extraction_runtime = feature_runtime.get(model, {})
    formal = result.get("formal_endpoint") if isinstance(result.get("formal_endpoint"), dict) else {}
    best_epoch = result.get("formal_epoch") if task.get("kind") == "reference" else formal.get("epoch")
    return {
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "head_parameter_count": _head_parameter_count(model),
        "training_elapsed_seconds": training_runtime.get("elapsed_seconds"),
        "training_peak_gpu_memory_bytes": training_runtime.get("peak_gpu_memory_bytes"),
        "feature_extraction_elapsed_seconds": extraction_runtime.get("elapsed_seconds"),
        "feature_extraction_peak_gpu_memory_bytes": extraction_runtime.get(
            "peak_gpu_memory_bytes"
        ),
    }


def _write_detail_csvs(output: Path, detail_rows: list[dict]) -> None:
    pathway_rows: list[dict] = []
    patient_rows: list[dict] = []
    for item in detail_rows:
        prefix = {
            key: item[key]
            for key in ("batch_id", "model", "seed", "split")
            if key in item
        }
        pathway_rows.extend({**prefix, **row} for row in item["per_pathway"])
        patient_rows.extend({**prefix, **row} for row in item["per_patient"])
    for filename, records in (
        ("per_pathway_metrics.csv", pathway_rows),
        ("per_patient_metrics.csv", patient_rows),
    ):
        if not records:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _read_batch_metrics(
    batch: Path,
    normalization: Normalization,
    external_truth_path: Path | None,
    expected_counts: dict[str, int] | None,
) -> tuple[list[dict], list[dict], dict]:
    manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8-sig"))
    batch_id = str(manifest.get("batch_id") or batch.name)
    rows: list[dict] = []
    detail_rows: list[dict] = []
    feature_runtime = _load_feature_runtime(batch)
    successful = [task for task in manifest.get("tasks", []) if task.get("status") == "succeeded"]
    for task in successful:
        if task.get("kind") not in {"reference", "train"}:
            continue
        expected_run_id = f"{task['kind']}_{int(task['seed'])}_{task['model']}"
        if task.get("run_id") != expected_run_id:
            raise IdentityMismatchError(f"批次任务 run_id 非法: {task.get('run_id')!r}")
        path = batch / str(task["run_id"]) / "raw" / "internal_best.npz"
        data = _read_prediction(
            path,
            require_target=True,
            expected_model=str(task["model"]),
            expected_seed=int(task["seed"]),
        )
        if expected_counts is not None and data["pred_z"].shape[0] != int(
            expected_counts["internal_val"]
        ):
            raise IdentityMismatchError(
                f"internal_val 预测行数应为 {expected_counts['internal_val']}，"
                f"实际={data['pred_z'].shape[0]}"
            )
        if data["pathways"].tolist() != list(normalization.pathway_names):
            raise IdentityMismatchError(f"内部预测通路顺序不匹配: {path}")
        metrics = compute_metrics(
            data["pred_z"], data["target_z"], data["patients"], normalization
        )
        details = _metric_details(
            data["pred_z"], data["target_z"], data["patients"], normalization
        )
        rows.append(
            {
                "batch_id": batch_id,
                "batch_dir": str(batch),
                "model": str(task["model"]),
                "seed": int(task["seed"]),
                "split": "internal_val",
                "prediction_relative": str(path.relative_to(batch)),
                **_resource_fields(task, feature_runtime),
                **metrics,
            }
        )
        detail_rows.append(
            {
                "batch_id": batch_id,
                "model": str(task["model"]),
                "seed": int(task["seed"]),
                "split": "internal_val",
                **details,
            }
        )

    for task in successful if external_truth_path is not None else []:
        expected_run_id = f"{task.get('kind')}_{int(task['seed'])}_{task['model']}"
        if task.get("run_id") != expected_run_id:
            raise IdentityMismatchError(f"批次任务 run_id 非法: {task.get('run_id')!r}")
        if task.get("kind") == "reference":
            path = batch / str(task["run_id"]) / "raw" / "external_predictions.npz"
            model = "uni2h"
        elif task.get("kind") == "external":
            path = batch / str(task["run_id"]) / "raw" / "external_predictions.npz"
            model = str(task["model"])
        else:
            continue
        prediction = _read_prediction(
            path,
            require_target=False,
            expected_model=model,
            expected_seed=int(task["seed"]),
        )
        if expected_counts is not None and prediction["pred_z"].shape[0] != int(
            expected_counts["external_test"]
        ):
            raise IdentityMismatchError(
                f"external_test 预测行数应为 {expected_counts['external_test']}，"
                f"实际={prediction['pred_z'].shape[0]}"
            )
        expected_pred_raw = normalization.inverse_transform(prediction["pred_z"])
        if not np.allclose(prediction["pred_raw"], expected_pred_raw, rtol=0.0, atol=1e-10):
            raise IdentityMismatchError("XZY pred_raw 不是 pred_z 按冻结训练参数的一次逆变换")
        target_z, target_raw = _align_external_target(prediction, external_truth_path)
        metrics = compute_metrics(
            prediction["pred_z"],
            target_z,
            prediction["patients"],
            normalization,
            raw_target=target_raw,
        )
        details = _metric_details(
            prediction["pred_z"],
            target_z,
            prediction["patients"],
            normalization,
            raw_target=target_raw,
        )
        rows.append(
            {
                "batch_id": batch_id,
                "batch_dir": str(batch),
                "model": model,
                "seed": int(task["seed"]),
                "split": "external_test",
                "prediction_relative": str(path.relative_to(batch)),
                **_resource_fields(task, feature_runtime),
                **metrics,
            }
        )
        detail_rows.append(
            {
                "batch_id": batch_id,
                "model": model,
                "seed": int(task["seed"]),
                "split": "external_test",
                **details,
            }
        )
    return rows, detail_rows, manifest


def analyze_batches(
    batch_dirs: Sequence[str | Path],
    output_dir: str | Path,
    normalization: Normalization,
    *,
    external_targets: str | Path | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    batches = [Path(value).resolve() for value in batch_dirs]
    if not batches or len(set(batches)) != len(batches):
        raise IdentityMismatchError("至少提供一个且不能重复提供批次目录")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"分析目录已存在，拒绝覆盖: {output}")
    external_truth_path = Path(external_targets).resolve() if external_targets else None
    rows: list[dict] = []
    detail_rows: list[dict] = []
    manifests: list[dict] = []
    for batch in batches:
        batch_rows, batch_details, manifest = _read_batch_metrics(
            batch, normalization, external_truth_path, expected_counts
        )
        rows.extend(batch_rows)
        detail_rows.extend(batch_details)
        manifests.append(manifest)
    if not any(row["split"] == "internal_val" for row in rows):
        raise IdentityMismatchError("所选批次没有成功的内部验证预测")
    row_keys = [(row["model"], int(row["seed"]), row["split"]) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise IdentityMismatchError("多个批次含重复的模型—seed—集合结果，请只保留一个正式来源")
    protocol_versions = {
        str(manifest["protocol_version"])
        for manifest in manifests
        if manifest.get("protocol_version") is not None
    }
    if protocol_versions and protocol_versions != {"v1.1-20260915"}:
        raise IdentityMismatchError(
            f"所选批次 protocol_version 不属于本包冻结协议: {sorted(protocol_versions)}"
        )
    experiment_ids = {
        str(manifest["experiment_id"])
        for manifest in manifests
        if manifest.get("experiment_id") is not None
    }
    if experiment_ids and experiment_ids != {"phase2_backbone_point_ablation_20260914"}:
        raise IdentityMismatchError(
            f"所选批次 experiment_id 不属于本实验: {sorted(experiment_ids)}"
        )

    output.mkdir(parents=True, exist_ok=False)
    aggregate = _aggregate(rows, split="internal_val")
    aggregate_by_split = {
        split_name: _aggregate(rows, split=split_name)
        for split_name in ("internal_val", "external_test")
        if any(row["split"] == split_name for row in rows)
    }
    seeds = sorted({int(row["seed"]) for row in rows if row["split"] == "internal_val"})
    per_model_seeds = {
        model: {
            int(row["seed"])
            for row in rows
            if row["split"] == "internal_val" and row["model"] == model
        }
        for model in ("uni2h", "uni", "virchow2")
    }
    if seeds == [42]:
        scope = "single_seed_42"
    elif all(values == {42, 43, 44} for values in per_model_seeds.values()):
        scope = "three_seed_complete"
    else:
        scope = "partial_multi_seed"
    external_row_count = sum(row["split"] == "external_test" for row in rows)
    report = {
        "schema_version": "1.1",
        "analyzed_at": utc_now(),
        "batch_dir": str(batches[0]) if len(batches) == 1 else None,
        "batch_dirs": [str(batch) for batch in batches],
        "scope": scope,
        "external_targets": str(external_truth_path) if external_truth_path else None,
        "expected_counts_enforced": expected_counts,
        "external_metrics_status": (
            "computed"
            if external_truth_path and external_row_count
            else "not_computed_no_successful_predictions"
            if external_truth_path
            else "not_computed_target_not_supplied"
        ),
        "external_used_for_selection": False,
        "fisher_z_aggregation": {
            "status": "not_applied",
            "reason": "protocol_requires_arithmetic_mean_after_per_seed_evaluation",
        },
        "spatial_metrics": {
            "status": "not_computed_not_predefined",
            "reason": "no_frozen_spatial_metric_definition_in_deployment_protocol",
        },
        "top_k_overlap": {
            "status": "not_computed_not_predefined",
            "reason": "no_frozen_top_k_or_overlap_definition_in_deployment_protocol",
        },
        "rows": rows,
        "aggregate": aggregate,
        "aggregate_by_split": aggregate_by_split,
    }
    write_json(output / "metrics_by_model_seed.json", report)
    write_json(
        output / "metric_details.json",
        {
            "schema_version": "1.0",
            "batch_dirs": [str(batch) for batch in batches],
            "rows": detail_rows,
        },
    )
    _write_detail_csvs(output, detail_rows)
    if rows:
        with (output / "model_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report_name = {
        "single_seed_42": "首批种子42审核报告.md",
        "three_seed_complete": "三种子统一分析报告.md",
        "partial_multi_seed": "部分种子分析报告.md",
    }[scope]
    scope_text = {
        "single_seed_42": "仅种子 42；未计算三种子标准差",
        "three_seed_complete": "三个模型均含种子 42、43、44；逐种子评价后汇总",
        "partial_multi_seed": "仅部分种子或模型齐备；不生成三种子标准差",
    }[scope]
    markdown = [
        "# UNI / Virchow2 基础模型替换消融分析",
        "",
        f"- 批次：{', '.join(f'`{batch.name}`' for batch in batches)}",
        f"- 范围：{scope_text}",
        f"- XZY：{'已按身份对齐真值并评价' if external_truth_path else '未提供本地真值，未计算指标'}",
        "- 选模来源：仅内部验证；XZY 未参与训练、超参数或检查点选择。",
        "",
        "| 模型 | seed | 集合 | 患者—通路等权 PCC（测量） | pooled 展平 PCC | z-RMSE | Spearman | CCC | raw-MAE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['seed']} | {row['split']} | "
            f"{fmt(row['patient_macro_pathway_pcc_measured'])} | {fmt(row['pooled_flattened_pcc'])} | "
            f"{fmt(row['pooled_z_rmse'])} | {fmt(row['pooled_spearman'])} | "
            f"{fmt(row['pooled_ccc'])} | {fmt(row['raw_mae'])} |"
        )
    if scope == "three_seed_complete":
        markdown.extend(
            [
                "",
                "## 三种子均值 ± 样本标准差",
                "",
                "| 模型 | 集合 | 患者—通路等权 PCC | pooled 展平 PCC | z-RMSE | raw-MAE |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )

        def mean_sd(record: dict, metric: str) -> str:
            mean_value = record["mean"].get(metric)
            sd_value = (record.get("sample_sd") or {}).get(metric)
            if mean_value is None or sd_value is None:
                return "NA"
            return f"{float(mean_value):.6f} ± {float(sd_value):.6f}"

        for split_name, model_records in aggregate_by_split.items():
            for model_name in ("uni2h", "uni", "virchow2"):
                record = model_records.get(model_name)
                if not record or not record.get("three_seed_scope_complete"):
                    continue
                markdown.append(
                    f"| {model_name} | {split_name} | "
                    f"{mean_sd(record, 'patient_macro_pathway_pcc_measured')} | "
                    f"{mean_sd(record, 'pooled_flattened_pcc')} | "
                    f"{mean_sd(record, 'pooled_z_rmse')} | "
                    f"{mean_sd(record, 'raw_mae')} |"
                )
    markdown.extend(
        [
            "",
            "| 模型 | seed | 最佳轮次 | 预测头参数量 | 训练耗时（秒） | 特征提取耗时（秒） | 训练显存峰值（bytes） | 提取显存峰值（bytes） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in (item for item in rows if item["split"] == "internal_val"):
        integer = lambda value: "NA" if value is None else str(int(value))
        markdown.append(
            f"| {row['model']} | {row['seed']} | {integer(row['best_epoch'])} | "
            f"{integer(row['head_parameter_count'])} | {fmt(row['training_elapsed_seconds'])} | "
            f"{fmt(row['feature_extraction_elapsed_seconds'])} | "
            f"{integer(row['training_peak_gpu_memory_bytes'])} | "
            f"{integer(row['feature_extraction_peak_gpu_memory_bytes'])} |"
        )
    markdown.extend(
        [
            "",
            "raw 指标若没有单独提供身份对齐的原始真值，仅对应训练 z-score 参数逆变换后的裁剪真值，不冒充原始未裁剪 ssGSEA。",
            "",
            "空间指标与 Top-k overlap 未计算：部署方案没有冻结其定义，报告中已显式标记为 `not_computed_not_predefined`。",
        ]
    )
    (output / report_name).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def analyze_batch(
    batch_dir: str | Path,
    output_dir: str | Path,
    normalization: Normalization,
    *,
    external_targets: str | Path | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Backward-compatible one-batch wrapper used for the seed-42 review."""

    return analyze_batches(
        [batch_dir],
        output_dir,
        normalization,
        external_targets=external_targets,
        expected_counts=expected_counts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="从服务器回传原始结果生成本地双 PCC 报告")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-targets", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    names = load_pathway_names(config["inputs"]["zscore_manifest"])
    normalization = load_normalization(config["inputs"]["normalization"], names)
    report = analyze_batches(
        args.batch_dir,
        args.output_dir,
        normalization,
        external_targets=args.external_targets,
        expected_counts=config["data"]["expected_counts"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
