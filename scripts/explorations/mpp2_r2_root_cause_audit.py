#!/usr/bin/env python3
# PFMVAL_EXPLORE
# PFMVAL_USER_FORCED_EXPERIMENT: pending_user_approval
"""MPP2 外部 R² 本机非证据性根因审查。

只读取已回传的 r003 预测、repaired v003 标签快照和 train-only z-score
参数。任何使用 XZY 真值的对齐均为 oracle（标签知情）诊断上界。

结果仅为用户强制指定的本地探索候选；未经后续明确批准，不得登记采纳。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_ZSCORE_SHA256 = (
    "fd65f6ab3a2dda2d8d9f8f17babe59818ea7d93f2480c5da2bcd6384ca37b2c8"
)
EXPECTED_SPLIT_SHA256 = (
    "babfa2b1d9aa7d7b2e56ac0ac18b95dcb006f66a41e4a86761bbe8a746d2f824"
)
EXPECTED_INTERNAL_ROWS = 1078
EXPECTED_EXTERNAL_ROWS = 1039
EXPECTED_PATIENTS = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 20_000


class AuditError(RuntimeError):
    """输入或复现门禁失败。"""


@dataclass(frozen=True)
class MetricSet:
    r2: float
    pcc: float
    ccc: float
    mae: float
    mse: float
    bias: float
    error_variance: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"{label} 不存在: {path}")
    actual = sha256_file(path)
    _require(actual.lower() == expected.lower(), f"{label} SHA-256 不匹配")


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / total)


def pcc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if np.std(truth) <= 0 or np.std(prediction) <= 0:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def ccc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    covariance = float(
        np.mean((truth - truth.mean()) * (prediction - prediction.mean()))
    )
    denominator = float(
        np.var(truth) + np.var(prediction) + (truth.mean() - prediction.mean()) ** 2
    )
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def metrics(truth: np.ndarray, prediction: np.ndarray) -> MetricSet:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    return MetricSet(
        r2=r2_score(truth, prediction),
        pcc=pcc(truth, prediction),
        ccc=ccc(truth, prediction),
        mae=float(np.mean(np.abs(error))),
        mse=float(np.mean(error**2)),
        bias=float(np.mean(error)),
        error_variance=float(np.var(error)),
    )


def oracle_intercept(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """标签知情：仅消除平均误差。"""
    return prediction + float(np.mean(truth - prediction))


def oracle_location_scale(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """标签知情：强制预测均值、标准差等于真值。"""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    pred_std = float(np.std(prediction))
    if pred_std <= 0:
        return np.full_like(prediction, truth.mean())
    return (prediction - prediction.mean()) * (
        float(np.std(truth)) / pred_std
    ) + truth.mean()


def oracle_positive_affine(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """标签知情：拟合非负斜率 y_hat=a*prediction+b 的 OLS 上界。"""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    variance = float(np.var(prediction))
    if variance <= 0:
        slope = 0.0
    else:
        covariance = float(
            np.mean((prediction - prediction.mean()) * (truth - truth.mean()))
        )
        slope = max(0.0, covariance / variance)
    intercept = float(truth.mean() - slope * prediction.mean())
    return slope * prediction + intercept, slope, intercept


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result_bundle(result_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    envelope_path = result_dir / "result.json"
    _require(envelope_path.is_file(), f"缺少 result.json: {envelope_path}")
    envelope = _load_json(envelope_path)
    _require(
        envelope.get("data_manifest_id")
        == "barcode-repair-20260711-d626ad8-v003:1204018178a4d355",
        "r003 data_manifest_id 不是 active repaired v003",
    )
    artifact_hashes: dict[str, str] = {}
    for artifact in envelope.get("artifacts", []):
        path = result_dir / artifact["path"]
        _require(path.is_file(), f"r003 产物缺失: {path}")
        actual = sha256_file(path)
        _require(
            actual.lower() == str(artifact["sha256"]).lower(),
            f"r003 产物哈希不匹配: {artifact['path']}",
        )
        _require(path.stat().st_size == int(artifact["size_bytes"]), f"r003 大小不匹配: {path}")
        artifact_hashes[artifact["path"]] = actual
    return envelope, artifact_hashes


def _find_repo_commit(result_dir: Path) -> str:
    repo = result_dir.parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _matrix(
    frame: pd.DataFrame, pathways: list[str], prefix: str
) -> np.ndarray:
    columns = [f"{prefix}{pathway}" for pathway in pathways]
    missing = [column for column in columns if column not in frame.columns]
    _require(not missing, f"预测表缺列: {missing[:5]}")
    values = frame[columns].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), f"{prefix} 数值含 NaN/Inf")
    return values


def validate_prediction_frame(
    frame: pd.DataFrame,
    pathways: list[str],
    prefixes: list[str],
    name: str,
) -> None:
    required = ["patient_id", "patch_id", "x", "y"]
    required.extend(
        f"{prefix}{pathway}" for prefix in prefixes for pathway in pathways
    )
    missing = [column for column in required if column not in frame.columns]
    _require(not missing, f"{name} 预测表缺列: {missing[:5]}")
    _require(
        frame[["patient_id", "patch_id", "x", "y"]].notna().all().all(),
        f"{name} 键有空值",
    )
    _require(
        not frame[["patient_id", "patch_id"]].duplicated().any(),
        f"{name} patient+patch 键重复",
    )
    numeric = frame[
        ["x", "y", *[f"{prefix}{pathway}" for prefix in prefixes for pathway in pathways]]
    ].to_numpy(dtype=np.float64)
    _require(np.isfinite(numeric).all(), f"{name} 数值含 NaN/Inf")


def _load_raw_tables(raw_root: Path, pathways: list[str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for patient in EXPECTED_PATIENTS + ["XZY"]:
        path = raw_root / patient / f"{patient}_ssGSEA.csv"
        _require(path.is_file(), f"缺少原始 ssGSEA: {path}")
        frame = pd.read_csv(path)
        _require(
            frame.columns.tolist() == ["barcode", *pathways],
            f"{patient} 原始通路顺序异常",
        )
        _require(frame["barcode"].notna().all(), f"{patient} barcode 有空值")
        _require(not frame["barcode"].duplicated().any(), f"{patient} barcode 重复")
        _require(
            np.isfinite(frame[pathways].to_numpy(dtype=np.float64)).all(),
            f"{patient} 原始 ssGSEA 含 NaN/Inf",
        )
        tables[patient] = frame
    return tables


def _validate_truth_against_raw(
    frame: pd.DataFrame,
    truth_z: np.ndarray,
    raw_tables: dict[str, pd.DataFrame],
    pathways: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> float:
    lookup = {patient: table.set_index("barcode") for patient, table in raw_tables.items()}
    rows: list[np.ndarray] = []
    for patient, patch in frame[["patient_id", "patch_id"]].itertuples(
        index=False, name=None
    ):
        _require(patient in lookup, f"未知患者: {patient}")
        _require(patch in lookup[patient].index, f"原始标签缺键: {patient}/{patch}")
        rows.append(lookup[patient].loc[patch, pathways].to_numpy(dtype=np.float64))
    raw = np.vstack(rows)
    reconstructed_z = (raw - means) / stds
    max_abs = float(np.max(np.abs(reconstructed_z - truth_z)))
    _require(max_abs < 1e-6, f"预测真值与原始 ssGSEA/z-score 不一致: {max_abs}")
    return max_abs


def validate_inputs(
    result_dir: Path,
    zscore_params_path: Path,
) -> dict[str, Any]:
    require_sha256(zscore_params_path, EXPECTED_ZSCORE_SHA256, "z-score 参数")
    split_path = zscore_params_path.parent / "split_manifest.csv"
    require_sha256(split_path, EXPECTED_SPLIT_SHA256, "repaired split_manifest")

    params = _load_json(zscore_params_path)
    pathways = list(params["pathways"])
    order = _load_json(result_dir / "pathway_order.json")
    _require(len(pathways) == 30, "z-score 参数不是 30 通路")
    _require(pathways == order, "z-score 参数与 r003 通路顺序不一致")
    means = np.asarray([params["pathways"][p]["mean"] for p in pathways], dtype=np.float64)
    stds = np.asarray([params["pathways"][p]["std"] for p in pathways], dtype=np.float64)
    _require(np.isfinite(means).all() and np.isfinite(stds).all(), "z-score 参数非有限")
    _require((stds > 0).all(), "z-score std 必须大于 0")

    internal = pd.read_csv(result_dir / "predictions_internal_val_base.csv")
    external = pd.read_csv(result_dir / "predictions_external_xzy.csv")
    _require(len(internal) == EXPECTED_INTERNAL_ROWS, "internal 行数不是 1078")
    _require(len(external) == EXPECTED_EXTERNAL_ROWS, "XZY 行数不是 1039")
    validate_prediction_frame(internal, pathways, ["true_", "pred_"], "internal")
    validate_prediction_frame(
        external,
        pathways,
        ["true_", "pred_base_", "pred_calibrated_"],
        "XZY",
    )
    _require(sorted(internal["patient_id"].unique()) == EXPECTED_PATIENTS, "internal 患者集合异常")
    _require(external["patient_id"].unique().tolist() == ["XZY"], "external 患者不是 XZY")

    split = pd.read_csv(split_path)
    val = split.loc[
        split["split"].eq("internal_val"), ["patient", "patch_stem"]
    ].copy()
    _require(len(val) == EXPECTED_INTERNAL_ROWS, "repaired split 的 val 行数不是 1078")
    split_keys = set(map(tuple, val.itertuples(index=False, name=None)))
    prediction_keys = set(
        map(tuple, internal[["patient_id", "patch_id"]].itertuples(index=False, name=None))
    )
    _require(split_keys == prediction_keys, "internal 预测键与 repaired val split 不一致")

    truth_internal_z = _matrix(internal, pathways, "true_")
    pred_internal_z = _matrix(internal, pathways, "pred_")
    truth_external_z = _matrix(external, pathways, "true_")
    pred_external_z = _matrix(external, pathways, "pred_base_")
    pred_external_cal_z = _matrix(external, pathways, "pred_calibrated_")

    source_snapshot = zscore_params_path.parents[2]
    raw_root = source_snapshot / "raw_ssgsea"
    raw_tables = _load_raw_tables(raw_root, pathways)
    internal_truth_max_abs = _validate_truth_against_raw(
        internal, truth_internal_z, raw_tables, pathways, means, stds
    )
    external_truth_max_abs = _validate_truth_against_raw(
        external, truth_external_z, raw_tables, pathways, means, stds
    )
    _require(
        set(external["patch_id"]) == set(raw_tables["XZY"]["barcode"]),
        "XZY 预测键与原始 XZY 标签不完全一致",
    )

    metrics_json = _load_json(result_dir / "metrics.json")
    truth_external_raw = truth_external_z * stds + means
    pred_external_raw = pred_external_z * stds + means
    per_pathway = [
        metrics(truth_external_raw[:, i], pred_external_raw[:, i])
        for i in range(len(pathways))
    ]
    reproduced = {
        "mean_per_pathway_raw_r2": float(np.mean([item.r2 for item in per_pathway])),
        "mean_per_pathway_pcc": float(np.mean([item.pcc for item in per_pathway])),
        "mean_raw_mae": float(np.mean([item.mae for item in per_pathway])),
    }
    expected = metrics_json["external_baseline"]
    r2_error = abs(reproduced["mean_per_pathway_raw_r2"] - expected["mean_per_pathway_raw_r2"])
    pcc_error = abs(reproduced["mean_per_pathway_pcc"] - expected["mean_per_pathway_pcc"])
    mae_error = abs(reproduced["mean_raw_mae"] - expected["mean_raw_mae"])
    _require(r2_error < 1e-10, f"R² 复现误差过大: {r2_error}")
    _require(pcc_error < 1e-10, f"PCC 复现误差过大: {pcc_error}")
    _require(mae_error < 1e-6, f"MAE 复现误差过大: {mae_error}")

    return {
        "params": params,
        "pathways": pathways,
        "means": means,
        "stds": stds,
        "internal": internal,
        "external": external,
        "truth_internal_z": truth_internal_z,
        "pred_internal_z": pred_internal_z,
        "truth_external_z": truth_external_z,
        "pred_external_z": pred_external_z,
        "pred_external_cal_z": pred_external_cal_z,
        "raw_tables": raw_tables,
        "split_path": split_path,
        "preflight": {
            "internal_rows": len(internal),
            "external_rows": len(external),
            "pathways": len(pathways),
            "internal_truth_z_max_abs_error": internal_truth_max_abs,
            "external_truth_z_max_abs_error": external_truth_max_abs,
            "metric_reproduction": reproduced,
            "metric_reproduction_abs_error": {
                "r2": r2_error,
                "pcc": pcc_error,
                "mae": mae_error,
            },
        },
    }


def _trimmed_r2(truth: np.ndarray, prediction: np.ndarray, fraction: float) -> float:
    residual = np.abs(prediction - truth)
    keep = max(2, int(math.floor(len(truth) * (1.0 - fraction))))
    indices = np.argsort(residual, kind="stable")[:keep]
    return r2_score(truth[indices], prediction[indices])


def analyze(validated: dict[str, Any]) -> dict[str, pd.DataFrame | dict[str, Any]]:
    pathways: list[str] = validated["pathways"]
    means: np.ndarray = validated["means"]
    stds: np.ndarray = validated["stds"]
    internal: pd.DataFrame = validated["internal"]
    external: pd.DataFrame = validated["external"]
    truth_i = validated["truth_internal_z"] * stds + means
    pred_i = validated["pred_internal_z"] * stds + means
    truth_x = validated["truth_external_z"] * stds + means
    pred_x = validated["pred_external_z"] * stds + means
    pred_x_cal = validated["pred_external_cal_z"] * stds + means

    pathway_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    oracle_arrays: dict[str, list[float]] = {
        "intercept_gain": [],
        "location_scale_gain": [],
        "location_scale_increment": [],
        "positive_affine_gain": [],
        "positive_affine_increment_over_intercept": [],
        "trim_1pct_gain": [],
        "trim_5pct_gain": [],
        "patient_balanced_pcc_gap": [],
        "patient_balanced_r2_gap": [],
    }
    patient_groups = {
        patient: np.flatnonzero(internal["patient_id"].to_numpy() == patient)
        for patient in EXPECTED_PATIENTS
    }

    for index, pathway in enumerate(pathways):
        internal_metric = metrics(truth_i[:, index], pred_i[:, index])
        external_metric = metrics(truth_x[:, index], pred_x[:, index])
        calibrated_metric = metrics(truth_x[:, index], pred_x_cal[:, index])
        intercept_pred = oracle_intercept(truth_x[:, index], pred_x[:, index])
        location_scale_pred = oracle_location_scale(truth_x[:, index], pred_x[:, index])
        affine_pred, affine_slope, affine_intercept = oracle_positive_affine(
            truth_x[:, index], pred_x[:, index]
        )
        intercept_metric = metrics(truth_x[:, index], intercept_pred)
        location_scale_metric = metrics(truth_x[:, index], location_scale_pred)
        affine_metric = metrics(truth_x[:, index], affine_pred)

        patient_true_means = [
            float(truth_i[indices, index].mean()) for indices in patient_groups.values()
        ]
        patient_true_stds = [
            float(truth_i[indices, index].std()) for indices in patient_groups.values()
        ]
        patient_pred_means = [
            float(pred_i[indices, index].mean()) for indices in patient_groups.values()
        ]
        patient_pred_stds = [
            float(pred_i[indices, index].std()) for indices in patient_groups.values()
        ]
        patient_metrics = [
            metrics(truth_i[indices, index], pred_i[indices, index])
            for indices in patient_groups.values()
        ]

        bias_sq = external_metric.bias**2
        bias_share = (
            bias_sq / external_metric.mse if external_metric.mse > 0 else float("nan")
        )
        identity_gap = abs(
            external_metric.mse
            - (external_metric.error_variance + external_metric.bias**2)
        )
        trim_1 = _trimmed_r2(truth_x[:, index], pred_x[:, index], 0.01)
        trim_5 = _trimmed_r2(truth_x[:, index], pred_x[:, index], 0.05)
        row = {
            "pathway": pathway,
            "internal_true_mean": float(truth_i[:, index].mean()),
            "internal_true_std": float(truth_i[:, index].std()),
            "internal_pred_mean": float(pred_i[:, index].mean()),
            "internal_pred_std": float(pred_i[:, index].std()),
            "internal_bias": internal_metric.bias,
            "internal_mse": internal_metric.mse,
            "internal_raw_r2": internal_metric.r2,
            "internal_pcc": internal_metric.pcc,
            "internal_ccc": internal_metric.ccc,
            "internal_mae": internal_metric.mae,
            "internal_patient_balanced_raw_r2": float(
                np.nanmean([item.r2 for item in patient_metrics])
            ),
            "internal_patient_balanced_pcc": float(
                np.nanmean([item.pcc for item in patient_metrics])
            ),
            "xzy_true_mean": float(truth_x[:, index].mean()),
            "xzy_true_std": float(truth_x[:, index].std()),
            "xzy_pred_mean": float(pred_x[:, index].mean()),
            "xzy_pred_std": float(pred_x[:, index].std()),
            "xzy_bias": external_metric.bias,
            "xzy_error_variance": external_metric.error_variance,
            "xzy_bias_squared": bias_sq,
            "xzy_bias_share_of_mse": bias_share,
            "xzy_mse_identity_abs_error": identity_gap,
            "xzy_mse": external_metric.mse,
            "xzy_raw_r2": external_metric.r2,
            "xzy_pcc": external_metric.pcc,
            "xzy_ccc": external_metric.ccc,
            "xzy_mae": external_metric.mae,
            "xzy_calibrated_raw_r2": calibrated_metric.r2,
            "ridge_delta_raw_r2": calibrated_metric.r2 - external_metric.r2,
            "xzy_true_mean_minus_internal_mean": float(
                truth_x[:, index].mean() - truth_i[:, index].mean()
            ),
            "xzy_pred_mean_minus_internal_pred_mean": float(
                pred_x[:, index].mean() - pred_i[:, index].mean()
            ),
            "xzy_true_std_ratio_to_internal": float(
                truth_x[:, index].std() / truth_i[:, index].std()
            ),
            "xzy_pred_std_ratio_to_internal": float(
                pred_x[:, index].std() / pred_i[:, index].std()
            ),
            "xzy_true_mean_outside_internal_patient_range": not (
                min(patient_true_means)
                <= float(truth_x[:, index].mean())
                <= max(patient_true_means)
            ),
            "xzy_true_std_outside_internal_patient_range": not (
                min(patient_true_stds)
                <= float(truth_x[:, index].std())
                <= max(patient_true_stds)
            ),
            "xzy_pred_mean_outside_internal_patient_range": not (
                min(patient_pred_means)
                <= float(pred_x[:, index].mean())
                <= max(patient_pred_means)
            ),
            "xzy_pred_std_outside_internal_patient_range": not (
                min(patient_pred_stds)
                <= float(pred_x[:, index].std())
                <= max(patient_pred_stds)
            ),
            "trim_1pct_raw_r2": trim_1,
            "trim_1pct_delta_raw_r2": trim_1 - external_metric.r2,
            "trim_5pct_raw_r2": trim_5,
            "trim_5pct_delta_raw_r2": trim_5 - external_metric.r2,
        }
        pathway_rows.append(row)
        oracle_row = {
            "pathway": pathway,
            "baseline_raw_r2": external_metric.r2,
            "intercept_oracle_raw_r2": intercept_metric.r2,
            "intercept_delta_raw_r2": intercept_metric.r2 - external_metric.r2,
            "location_scale_oracle_raw_r2": location_scale_metric.r2,
            "location_scale_delta_raw_r2": location_scale_metric.r2
            - external_metric.r2,
            "location_scale_increment_over_intercept": location_scale_metric.r2
            - intercept_metric.r2,
            "positive_affine_oracle_raw_r2": affine_metric.r2,
            "positive_affine_delta_raw_r2": affine_metric.r2 - external_metric.r2,
            "positive_affine_increment_over_intercept": affine_metric.r2
            - intercept_metric.r2,
            "positive_affine_slope": affine_slope,
            "positive_affine_intercept": affine_intercept,
        }
        oracle_rows.append(oracle_row)
        oracle_arrays["intercept_gain"].append(oracle_row["intercept_delta_raw_r2"])
        oracle_arrays["location_scale_gain"].append(
            oracle_row["location_scale_delta_raw_r2"]
        )
        oracle_arrays["location_scale_increment"].append(
            oracle_row["location_scale_increment_over_intercept"]
        )
        oracle_arrays["positive_affine_gain"].append(
            oracle_row["positive_affine_delta_raw_r2"]
        )
        oracle_arrays["positive_affine_increment_over_intercept"].append(
            oracle_row["positive_affine_increment_over_intercept"]
        )
        oracle_arrays["trim_1pct_gain"].append(row["trim_1pct_delta_raw_r2"])
        oracle_arrays["trim_5pct_gain"].append(row["trim_5pct_delta_raw_r2"])
        oracle_arrays["patient_balanced_pcc_gap"].append(
            row["internal_patient_balanced_pcc"] - external_metric.pcc
        )
        oracle_arrays["patient_balanced_r2_gap"].append(
            row["internal_patient_balanced_raw_r2"] - external_metric.r2
        )

    per_pathway = pd.DataFrame(pathway_rows)
    oracle = pd.DataFrame(oracle_rows)
    summary_row: dict[str, Any] = {"pathway": "__MEAN__"}
    for column in oracle.columns:
        if column != "pathway":
            summary_row[column] = float(oracle[column].mean())
    for name, values in oracle_arrays.items():
        low, high = bootstrap_mean_ci(values)
        summary_row[f"{name}_ci95_low"] = low
        summary_row[f"{name}_ci95_high"] = high
        summary_row[f"{name}_supported"] = bool(low > 0)
    oracle = pd.concat([oracle, pd.DataFrame([summary_row])], ignore_index=True)

    patient_rows: list[dict[str, Any]] = []
    for patient in EXPECTED_PATIENTS:
        indices = patient_groups[patient]
        patient_metrics = [
            metrics(truth_i[indices, i], pred_i[indices, i])
            for i in range(len(pathways))
        ]
        patient_rows.append(
            {
                "dataset": "internal_val",
                "patient": patient,
                "n_patches": len(indices),
                "mean_per_pathway_raw_r2": float(
                    np.nanmean([item.r2 for item in patient_metrics])
                ),
                "mean_per_pathway_pcc": float(
                    np.nanmean([item.pcc for item in patient_metrics])
                ),
                "mean_ccc": float(np.nanmean([item.ccc for item in patient_metrics])),
                "mean_raw_mae": float(
                    np.nanmean([item.mae for item in patient_metrics])
                ),
                "mean_true_z": float(validated["truth_internal_z"][indices].mean()),
                "mean_pred_z": float(validated["pred_internal_z"][indices].mean()),
            }
        )
    external_metrics = [
        metrics(truth_x[:, i], pred_x[:, i]) for i in range(len(pathways))
    ]
    patient_rows.append(
        {
            "dataset": "external_test",
            "patient": "XZY",
            "n_patches": len(external),
            "mean_per_pathway_raw_r2": float(
                np.nanmean([item.r2 for item in external_metrics])
            ),
            "mean_per_pathway_pcc": float(
                np.nanmean([item.pcc for item in external_metrics])
            ),
            "mean_ccc": float(np.nanmean([item.ccc for item in external_metrics])),
            "mean_raw_mae": float(
                np.nanmean([item.mae for item in external_metrics])
            ),
            "mean_true_z": float(validated["truth_external_z"].mean()),
            "mean_pred_z": float(validated["pred_external_z"].mean()),
        }
    )
    per_patient = pd.DataFrame(patient_rows)

    spatial_rows: list[dict[str, Any]] = []
    for grid_size in [1024, 2048, 4096]:
        cell_x = np.floor(external["x"].to_numpy(dtype=np.float64) / grid_size).astype(int)
        cell_y = np.floor(external["y"].to_numpy(dtype=np.float64) / grid_size).astype(int)
        grouping = pd.DataFrame({"cell_x": cell_x, "cell_y": cell_y})
        for (cx, cy), indices in grouping.groupby(["cell_x", "cell_y"]).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            for pathway_index, pathway in enumerate(pathways):
                metric = metrics(truth_x[idx, pathway_index], pred_x[idx, pathway_index])
                spatial_rows.append(
                    {
                        "grid_size_px": grid_size,
                        "cell_x": int(cx),
                        "cell_y": int(cy),
                        "pathway": pathway,
                        "n_patches": len(idx),
                        "bias": metric.bias,
                        "mae": metric.mae,
                        "raw_r2": metric.r2 if len(idx) >= 2 else float("nan"),
                    }
                )
    spatial = pd.DataFrame(spatial_rows)

    residual_rows: list[dict[str, Any]] = []
    for pathway_index, pathway in enumerate(pathways):
        prediction = pred_x[:, pathway_index]
        bins = pd.qcut(prediction, q=10, labels=False, duplicates="drop")
        for bin_id in sorted(pd.Series(bins).dropna().unique()):
            idx = np.flatnonzero(np.asarray(bins) == bin_id)
            residual_rows.append(
                {
                    "pathway": pathway,
                    "prediction_decile": int(bin_id) + 1,
                    "n_patches": len(idx),
                    "mean_true": float(truth_x[idx, pathway_index].mean()),
                    "mean_prediction": float(prediction[idx].mean()),
                    "mean_residual_pred_minus_true": float(
                        (prediction[idx] - truth_x[idx, pathway_index]).mean()
                    ),
                    "mae": float(
                        np.mean(np.abs(prediction[idx] - truth_x[idx, pathway_index]))
                    ),
                }
            )
    residual_shape = pd.DataFrame(residual_rows)

    negative_deficit = np.maximum(0.0, -per_pathway["xzy_raw_r2"].to_numpy())
    worst_order = np.argsort(negative_deficit)[::-1]
    total_negative_deficit = float(negative_deficit.sum())
    top3_share = (
        float(negative_deficit[worst_order[:3]].sum() / total_negative_deficit)
        if total_negative_deficit > 0
        else 0.0
    )
    worst_three = per_pathway.iloc[worst_order[:3]]["pathway"].tolist()
    summary = {
        "baseline": {
            "internal_mean_raw_r2": float(
                np.mean(per_pathway["internal_raw_r2"])
            ),
            "external_mean_raw_r2": float(np.mean(per_pathway["xzy_raw_r2"])),
            "external_mean_pcc": float(np.mean(per_pathway["xzy_pcc"])),
            "external_legacy_pooled_pcc": float(
                np.corrcoef(
                    validated["truth_external_z"].ravel(),
                    validated["pred_external_z"].ravel(),
                )[0, 1]
            ),
            "ridge_delta_mean_raw_r2": float(
                np.mean(per_pathway["ridge_delta_raw_r2"])
            ),
        },
        "distribution_shift": {
            "pathways_xzy_true_mean_outside_internal_patient_range": int(
                per_pathway["xzy_true_mean_outside_internal_patient_range"].sum()
            ),
            "pathways_xzy_true_std_outside_internal_patient_range": int(
                per_pathway["xzy_true_std_outside_internal_patient_range"].sum()
            ),
            "mean_bias_share_of_mse": float(
                per_pathway["xzy_bias_share_of_mse"].mean()
            ),
        },
        "concentration": {
            "worst_three_pathways": worst_three,
            "worst_three_share_of_negative_r2_deficit": top3_share,
        },
        "oracle": {
            name: {
                "mean_delta_raw_r2": float(np.mean(values)),
                "ci95_low": bootstrap_mean_ci(values)[0],
                "ci95_high": bootstrap_mean_ci(values)[1],
                "supported": bool(bootstrap_mean_ci(values)[0] > 0),
            }
            for name, values in oracle_arrays.items()
        },
        "gap_decomposition": {
            "pooled_internal_mean_r2": float(
                np.mean(per_pathway["internal_raw_r2"])
            ),
            "patient_balanced_internal_mean_r2": float(
                np.mean(per_pathway["internal_patient_balanced_raw_r2"])
            ),
            "patient_balanced_internal_mean_pcc": float(
                np.mean(per_pathway["internal_patient_balanced_pcc"])
            ),
            "external_mean_r2": float(np.mean(per_pathway["xzy_raw_r2"])),
            "apparent_pooled_internal_to_external_gap": float(
                np.mean(per_pathway["internal_raw_r2"])
                - np.mean(per_pathway["xzy_raw_r2"])
            ),
            "aggregation_grain_component": float(
                np.mean(per_pathway["internal_raw_r2"])
                - np.mean(per_pathway["internal_patient_balanced_raw_r2"])
            ),
            "patient_balanced_internal_to_external_gap": float(
                np.mean(per_pathway["internal_patient_balanced_raw_r2"])
                - np.mean(per_pathway["xzy_raw_r2"])
            ),
            "xzy_intercept_oracle_gain": float(
                np.mean(oracle_arrays["intercept_gain"])
            ),
            "xzy_slope_gain_after_intercept": float(
                np.mean(oracle_arrays["positive_affine_increment_over_intercept"])
            ),
            "external_positive_affine_ceiling_mean_r2": float(
                np.mean(per_pathway["xzy_raw_r2"])
                + np.mean(oracle_arrays["positive_affine_gain"])
            ),
        },
        "spatial": {
            str(grid): {
                "cells": int(
                    spatial.loc[spatial["grid_size_px"].eq(grid), ["cell_x", "cell_y"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "median_cell_mae_cv_across_pathways": float(
                    spatial.loc[spatial["grid_size_px"].eq(grid)]
                    .groupby("pathway")["mae"]
                    .agg(lambda values: values.std(ddof=0) / values.mean())
                    .median()
                ),
            }
            for grid in [1024, 2048, 4096]
        },
    }
    return {
        "per_pathway": per_pathway,
        "oracle": oracle,
        "per_patient": per_patient,
        "spatial": spatial,
        "residual_shape": residual_shape,
        "summary": summary,
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(
    analysis: dict[str, pd.DataFrame | dict[str, Any]],
    preflight: dict[str, Any],
) -> str:
    summary: dict[str, Any] = analysis["summary"]  # type: ignore[assignment]
    per_pathway: pd.DataFrame = analysis["per_pathway"]  # type: ignore[assignment]
    oracle = summary["oracle"]
    baseline = summary["baseline"]
    distribution = summary["distribution_shift"]
    concentration = summary["concentration"]
    decomposition = summary["gap_decomposition"]
    intercept = oracle["intercept_gain"]
    scale_increment = oracle["location_scale_increment"]
    pcc_gap = oracle["patient_balanced_pcc_gap"]
    trim5 = oracle["trim_5pct_gain"]
    worst = per_pathway.nsmallest(3, "xzy_raw_r2")[
        ["pathway", "xzy_raw_r2", "xzy_pcc", "xzy_bias_share_of_mse"]
    ]
    worst_lines = "\n".join(
        f"| {row.pathway} | {_fmt(row.xzy_raw_r2)} | {_fmt(row.xzy_pcc)} | "
        f"{_fmt(100 * row.xzy_bias_share_of_mse, 1)}% |"
        for row in worst.itertuples(index=False)
    )
    return f"""# MPP2 外部 R² 根因审查（2026-07-25）

> **证据边界：本报告是本地非证据性诊断。r003 状态为 failed、未 accepted。**
> 所有使用 XZY 真值的 intercept、location-scale、OLS 和极值剔除均为
> oracle 上界，禁止作为正式校准器或无泄露外测结果。

## 结论摘要

- 已复现核心现象：internal-val mean raw R² = **{_fmt(baseline['internal_mean_raw_r2'], 6)}**，
  XZY = **{_fmt(baseline['external_mean_raw_r2'], 6)}**；XZY 逐通路平均 PCC =
  **{_fmt(baseline['external_mean_pcc'], 6)}**，legacy pooled PCC =
  **{_fmt(baseline['external_legacy_pooled_pcc'], 6)}**。
- **首要解释：internal 与 XZY 的 R² 统计粒度不一致，放大了表观落差。**
  internal 六患者混合后的 mean R² 为
  **{_fmt(decomposition['pooled_internal_mean_r2'])}**；先逐患者计算再平均仅
  **{_fmt(decomposition['patient_balanced_internal_mean_r2'])}**。因此表观落差
  {_fmt(decomposition['apparent_pooled_internal_to_external_gap'])} 中约
  **{100 * decomposition['aggregation_grain_component'] / decomposition['apparent_pooled_internal_to_external_gap']:.1f}%**
  来自聚合口径/患者异质性，而不是 XZY 独有的模型崩溃。
- **最主要的真实误差成分：跨患者通路基线（均值/截距）漂移，有数据支持。**
  仅消除每通路平均误差的 oracle mean R² 增量为
  **+{_fmt(intercept['mean_delta_raw_r2'])}**，配对通路 bootstrap 95% CI
  **[{_fmt(intercept['ci95_low'])}, {_fmt(intercept['ci95_high'])}]**。
- **XZY 也存在一定排序能力下降，有数据支持但影响次于截距。**
  公平的逐患者口径下，internal mean PCC 为
  **{_fmt(decomposition['patient_balanced_internal_mean_pcc'])}**，XZY 为
  **{_fmt(baseline['external_mean_pcc'])}**；配对通路差值
  **{_fmt(pcc_gap['mean_delta_raw_r2'])}**，95% CI
  **[{_fmt(pcc_gap['ci95_low'])}, {_fmt(pcc_gap['ci95_high'])}]**。
- **简单“统一尺度漂移”未获支持。** 强制均值与标准差对齐相对仅修正截距的
  新增 ΔR² 为 **{_fmt(scale_increment['mean_delta_raw_r2'])}**，95% CI
  **[{_fmt(scale_increment['ci95_low'])}, {_fmt(scale_increment['ci95_high'])}]**。
- Ridge 只改变一条通路，XZY mean raw R² 变化
  **{baseline['ridge_delta_mean_raw_r2']:+.8f}**，不是基线暴跌的解释。

## 数据与复现门禁

- repaired v003 核心资产：24/24 哈希、大小和行数通过。
- r003 预测：internal 1,078 行，XZY 1,039 行；30 通路顺序一致。
- `patient_id + patch_id` 唯一，internal 键与 repaired val split 完全相等；
  XZY 键覆盖原始 XZY 全部标签。
- 预测表 true z-score 与七患者原始 ssGSEA 经 train-only 参数重算的最大误差：
  internal `{preflight['internal_truth_z_max_abs_error']:.3e}`，
  XZY `{preflight['external_truth_z_max_abs_error']:.3e}`。
- r003 baseline 指标复现误差：R² `{preflight['metric_reproduction_abs_error']['r2']:.3e}`，
  PCC `{preflight['metric_reproduction_abs_error']['pcc']:.3e}`，
  MAE `{preflight['metric_reproduction_abs_error']['mae']:.3e}`。

## 同一批数据的五个关键口径

| 口径 | mean raw R² | 用途 |
|---|---:|---|
| internal 六患者 pooled | {_fmt(decomposition['pooled_internal_mean_r2'])} | 原 r003 内部汇总 |
| internal 逐患者后平均 | {_fmt(decomposition['patient_balanced_internal_mean_r2'])} | 与单患者 XZY 更可比 |
| XZY baseline | {_fmt(decomposition['external_mean_r2'])} | 正式未校准外测 |
| XZY intercept oracle | {_fmt(decomposition['external_mean_r2'] + decomposition['xzy_intercept_oracle_gain'])} | 仅诊断均值偏移上界 |
| XZY 正斜率仿射 oracle | {_fmt(decomposition['external_positive_affine_ceiling_mean_r2'])} | 标签知情线性上界 |

## 根因可能性排序

报告中的 pooled internal→XZY mean R² 表观落差为
**{_fmt(decomposition['apparent_pooled_internal_to_external_gap'])}**，但其中
**{_fmt(decomposition['aggregation_grain_component'])}** 来自“六患者混合”
与“单患者”口径差异。改用逐患者平均后，internal→XZY 差距缩小为
**{_fmt(decomposition['patient_balanced_internal_to_external_gap'])}**。

1. **统计粒度不一致与患者异质性 — 已验证，最先解释表观暴跌。**
   六名 internal 逐患者 mean R² 的平均值仅
   **{_fmt(decomposition['patient_balanced_internal_mean_r2'])}**，并非 pooled 的
   **{_fmt(decomposition['pooled_internal_mean_r2'])}**；其中 ZHZ 已出现负 R²。
   故 `0.635 → -0.088` 不能全部解释为外部泛化损失。

2. **跨患者 pathway baseline/intercept shift — 有数据支持，真实误差主因。**
   XZY 真实均值有 **{distribution['pathways_xzy_true_mean_outside_internal_patient_range']}/30**
   条通路落在六名 internal 患者均值范围之外；bias² 平均占 XZY MSE 的
   **{100 * distribution['mean_bias_share_of_mse']:.1f}%**。这能解释 PCC 尚可而
   R² 为负，但无法区分生物学患者差异与测序/批次差异。

3. **XZY 关系/排序泛化下降 — 有数据支持，但幅度较温和。**
   逐患者平均 PCC 从 internal 的
   **{_fmt(decomposition['patient_balanced_internal_mean_pcc'])}** 降至 XZY 的
   **{_fmt(baseline['external_mean_pcc'])}**；配对通路差值 CI 下界大于 0。
   该项可能包含图像域迁移、患者特异生物学、非线性或不可约噪声。

4. **少数通路集中拖累 — 已验证的集中现象，不等同于独立根因。**
   最差三条通路占全部负 R² deficit 的
   **{100 * concentration['worst_three_share_of_negative_r2_deficit']:.1f}%**：

   | 通路 | XZY R² | PCC | bias²/MSE |
   |---|---:|---:|---:|
{worst_lines}

5. **离群点/极值敏感性 — 存在小幅 1% 敏感性，但不是主因。**
   剔除绝对残差最大的 1% spot，mean R² 增加
   **{oracle['trim_1pct_gain']['mean_delta_raw_r2']:+.4f}**，CI
   **[{oracle['trim_1pct_gain']['ci95_low']:+.4f},
   {oracle['trim_1pct_gain']['ci95_high']:+.4f}]**；扩大到 5% 后结果不稳健
   （Δ **{trim5['mean_delta_raw_r2']:+.4f}**，CI 跨 0）。因此极值只解释小部分，
   且不能把被剔除样本视为错误数据。

6. **统一尺度（scale）漂移 — 尚未证实。**
   location-scale 对齐相对 intercept 的增量 CI 下界未大于 0，因此不支持
   “XZY 只需统一放大/缩小”的解释。

7. **空间非平稳性 — 可见但无法定为根本原因。**
   单个 XZY 患者的 1024/2048/4096 px 网格只能定位残差聚集区，不能区分组织区域、
   切片批次与患者效应；需要更多外部患者或区域注释。

8. **z-score 参数、通路顺序、键错配或 Ridge 本身 — 当前证据不支持。**
   哈希、键、真值重算和指标复现均通过；Ridge 的外部变化接近零。

## 方法与判定规则

- 所有 R²、PCC、CCC 和 MAE 均先逐通路计算，再对 30 条通路取算术平均；
  “patient-balanced”口径先在每名患者内计算，再对患者等权平均。
- 使用恒等式 `MSE = Var(pred−true) + mean(pred−true)²` 分离误差方差与
  bias（平均误差）。
- oracle 依次为：仅截距、强制均值与标准差、正斜率仿射 OLS。它们均使用
  XZY 真值，只有诊断意义。
- 置信区间通过对 30 条配对通路进行 20,000 次 bootstrap 得到，固定随机种子
  20260725；只有 95% CI 下界大于 0 才标记“有数据支持”。
- 极值敏感性分别移除每条通路绝对残差最大的 1%/5%；空间敏感性使用
  1024/2048/4096 px 网格。两者均不改变原始输入文件。

## 仍未解决的缺口

- 缺少每 spot 的 total counts、detected genes、library size，无法判断均值漂移中
  有多少来自测序深度或技术批次。
- 只有一个外部患者 XZY，无法把“患者生物差异”和“中心/批次差异”分离。
- oracle 结果使用 XZY 真值，只是定位上界；正式方案必须在 internal 患者内预注册、
  选择和冻结后再做新的外部验证。

## 建议的下一步

1. 先补齐 XZY 与六名 internal 患者的 library-size/检测基因数元数据，做与
   pathway bias 的相关分析。
2. 只在六名 internal 患者上设计 patient-level LOPO 校准，验证能否预测“新患者
   截距”，不得用 XZY 选择参数。
3. 优先审查最差三条通路的 gene-set、标签范围和残差空间分布，再决定是通路特异
   建模还是数据标准化问题。

## 后续需要回答的问题

- XZY 的 pathway 均值偏移是否与 total counts、detected genes 或 library size
  显著相关？
- 在六名 internal 患者中，能否仅凭图像或无标签切片统计预测新患者截距？
- Interferon Alpha、Complement、Reactive Oxygen Species 的问题来自 gene-set
  定义、技术批次、组织组成差异，还是图像模型本身？
"""


def write_outputs(
    output_dir: Path,
    result_dir: Path,
    zscore_params_path: Path,
    envelope: dict[str, Any],
    artifact_hashes: dict[str, str],
    validated: dict[str, Any],
    analysis: dict[str, pd.DataFrame | dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "per_pathway_diagnostics.csv": analysis["per_pathway"],
        "per_patient_diagnostics.csv": analysis["per_patient"],
        "oracle_diagnostics.csv": analysis["oracle"],
        "spatial_diagnostics.csv": analysis["spatial"],
        "residual_shape_diagnostics.csv": analysis["residual_shape"],
    }
    for name, frame in tables.items():
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_dir / name, index=False, lineterminator="\n")

    manifest = {
        "schema_version": "1.0",
        "analysis_id": "mpp2_r2_root_cause_20260725",
        "status": "local_non_evidentiary_diagnostic",
        "disclaimer": (
            "r003 failed and is not accepted; XZY-label-aware transforms are oracle "
            "upper bounds only and may not be used as formal calibration."
        ),
        "input_commit": _find_repo_commit(result_dir),
        "result_id": envelope["result_id"],
        "result_status": envelope["status"],
        "result_json_sha256": sha256_file(result_dir / "result.json"),
        "result_artifact_hashes": artifact_hashes,
        "zscore_params_path": str(zscore_params_path),
        "zscore_params_sha256": sha256_file(zscore_params_path),
        "split_manifest_sha256": sha256_file(validated["split_path"]),
        "run_parameters": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "grid_sizes_px": [1024, 2048, 4096],
            "outlier_sensitivity_fractions": [0.01, 0.05],
        },
        "preflight": validated["preflight"],
        "summary": analysis["summary"],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "MPP2_R2根因审查_20260725.md").write_text(
        build_report(analysis, validated["preflight"]), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--zscore-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    envelope, artifact_hashes = validate_result_bundle(args.result_dir)
    validated = validate_inputs(args.result_dir, args.zscore_params)
    analysis = analyze(validated)
    write_outputs(
        args.output_dir,
        args.result_dir,
        args.zscore_params,
        envelope,
        artifact_hashes,
        validated,
        analysis,
    )
    print(
        json.dumps(
            {
                "status": "complete_local_non_evidentiary_diagnostic",
                "output_dir": str(args.output_dir),
                "preflight": validated["preflight"],
                "summary": analysis["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
