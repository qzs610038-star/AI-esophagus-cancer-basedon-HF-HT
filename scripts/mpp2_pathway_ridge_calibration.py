"""Deterministic, train-only fitting primitives for MPP2 pathway calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np


IDENTITY_EPS = 1e-6


@dataclass(frozen=True)
class AffineFit:
    slope: float
    intercept: float
    fallback_reason: str | None = None


def patient_weights(patients: Sequence[str]) -> np.ndarray:
    values = np.asarray(patients, dtype=str)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("patients must be a non-empty one-dimensional sequence")
    counts = Counter(values.tolist())
    return np.asarray([1.0 / (len(counts) * counts[item]) for item in values], dtype=np.float64)


def fit_positive_affine(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
    ridge_lambda: float,
) -> AffineFit:
    """Fit k*x+b with a patient-balanced identity prior and a positive slope."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth and prediction must be equally shaped one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("calibration inputs must be finite")
    if ridge_lambda not in {0.1, 1.0, 10.0}:
        raise ValueError("ridge lambda must be one of the preregistered values")
    if float(np.var(prediction)) < IDENTITY_EPS:
        return AffineFit(1.0, 0.0, "pred_z_variance_below_1e-6")
    weights = patient_weights(patients)
    design = np.column_stack([prediction, np.ones_like(prediction)])
    lhs = design.T @ (weights[:, None] * design) + ridge_lambda * np.eye(2)
    rhs = design.T @ (weights * truth) + ridge_lambda * np.asarray([1.0, 0.0])
    try:
        slope, intercept = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return AffineFit(1.0, 0.0, "singular_ridge_system")
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        return AffineFit(1.0, 0.0, "nonpositive_or_nonfinite_slope")
    return AffineFit(float(slope), float(intercept))


def fit_all_pathways(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("pathway matrices must be equally shaped [N, P] arrays")
    fits = [fit_positive_affine(truth[:, index], prediction[:, index], patients, ridge_lambda)
            for index in range(truth.shape[1])]
    return (
        np.asarray([fit.slope for fit in fits], dtype=np.float64),
        np.asarray([fit.intercept for fit in fits], dtype=np.float64),
        [fit.fallback_reason for fit in fits],
    )


def patient_balanced_mse(truth: np.ndarray, prediction: np.ndarray, patients: Sequence[str]) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    values = np.asarray(patients, dtype=str)
    return float(np.mean([
        np.mean((truth[values == patient] - prediction[values == patient]) ** 2)
        for patient in sorted(set(values.tolist()))
    ]))


def choose_lambda_one_se(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
    lambda_grid: Iterable[float] = (0.1, 1.0, 10.0),
) -> tuple[float, Dict[str, Any]]:
    """Nested inner-LOPO selection; choose the stronger lambda within one SE."""
    values = np.asarray(patients, dtype=str)
    unique = sorted(set(values.tolist()))
    if len(unique) < 2:
        raise ValueError("lambda selection requires at least two patients")
    records: Dict[float, list[float]] = {}
    for ridge_lambda in lambda_grid:
        fold_scores: list[float] = []
        for held_out in unique:
            train_mask = values != held_out
            test_mask = ~train_mask
            slopes, intercepts, _ = fit_all_pathways(
                truth[train_mask], prediction[train_mask], values[train_mask], float(ridge_lambda),
            )
            calibrated = prediction[test_mask] * slopes + intercepts
            fold_scores.append(float(np.mean((truth[test_mask] - calibrated) ** 2)))
        records[float(ridge_lambda)] = fold_scores
    means = {value: float(np.mean(scores)) for value, scores in records.items()}
    best = min(means, key=means.get)
    best_se = float(np.std(records[best], ddof=1) / np.sqrt(len(records[best])))
    eligible = [value for value, mean in means.items() if mean <= means[best] + best_se + 1e-15]
    selected = max(eligible)
    return selected, {
        "fold_scores": {str(key): value for key, value in records.items()},
        "mean_scores": {str(key): value for key, value in means.items()},
        "best_lambda": best,
        "best_lambda_se": best_se,
        "selected_lambda": selected,
        "one_se_eligible": eligible,
    }


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    return float(1.0 - np.sum((truth - prediction) ** 2) / total) if total > 0 else float("nan")


def ccc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    denominator = float(np.var(truth) + np.var(prediction) + (truth.mean() - prediction.mean()) ** 2)
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def per_pathway_metrics(
    truth_z: np.ndarray,
    prediction_z: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> list[Dict[str, float]]:
    rows: list[Dict[str, float]] = []
    for index in range(truth_z.shape[1]):
        truth_raw = truth_z[:, index] * stds[index] + means[index]
        prediction_raw = prediction_z[:, index] * stds[index] + means[index]
        if np.std(truth_raw) > 0 and np.std(prediction_raw) > 0:
            pcc = float(np.corrcoef(truth_raw, prediction_raw)[0, 1])
        else:
            pcc = float("nan")
        rows.append({
            "pcc": pcc,
            "raw_r2": r2_score(truth_raw, prediction_raw),
            "raw_mae": float(np.mean(np.abs(truth_raw - prediction_raw))),
            "ccc": ccc(truth_raw, prediction_raw),
        })
    return rows


def nested_lopo(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
) -> Dict[str, Any]:
    values = np.asarray(patients, dtype=str)
    unique = sorted(set(values.tolist()))
    if len(unique) != 6:
        raise ValueError(f"nested LOPO requires exactly six patients, got {len(unique)}")
    oof = np.empty_like(prediction, dtype=np.float64)
    slopes = np.empty((len(unique), prediction.shape[1]), dtype=np.float64)
    intercepts = np.empty_like(slopes)
    records = []
    for fold_index, held_out in enumerate(unique):
        train_mask = values != held_out
        test_mask = ~train_mask
        selected, selection = choose_lambda_one_se(truth[train_mask], prediction[train_mask], values[train_mask])
        slope, intercept, reasons = fit_all_pathways(
            truth[train_mask], prediction[train_mask], values[train_mask], selected,
        )
        oof[test_mask] = prediction[test_mask] * slope + intercept
        slopes[fold_index] = slope
        intercepts[fold_index] = intercept
        records.append({
            "held_out_patient": held_out,
            "selected_lambda": selected,
            "selection": selection,
            "fallback_reasons": reasons,
        })
    return {"oof_prediction": oof, "slopes": slopes, "intercepts": intercepts, "folds": records}


def pathway_decisions(
    truth: np.ndarray,
    base_prediction: np.ndarray,
    calibrated_oof: np.ndarray,
    patients: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    slopes: np.ndarray,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    values = np.asarray(patients, dtype=str)
    unique = sorted(set(values.tolist()))
    baseline = per_pathway_metrics(truth, base_prediction, means, stds)
    calibrated = per_pathway_metrics(truth, calibrated_oof, means, stds)
    mask = np.zeros(truth.shape[1], dtype=bool)
    rows: list[Dict[str, Any]] = []
    for index in range(truth.shape[1]):
        reasons: list[str] = []
        per_patient_base = []
        per_patient_cal = []
        per_patient_mae_base = []
        per_patient_mae_cal = []
        for patient in unique:
            patient_mask = values == patient
            per_patient_base.append(float(np.mean((truth[patient_mask, index] - base_prediction[patient_mask, index]) ** 2)))
            per_patient_cal.append(float(np.mean((truth[patient_mask, index] - calibrated_oof[patient_mask, index]) ** 2)))
            scale = stds[index]
            per_patient_mae_base.append(float(np.mean(np.abs(truth[patient_mask, index] - base_prediction[patient_mask, index])) * scale))
            per_patient_mae_cal.append(float(np.mean(np.abs(truth[patient_mask, index] - calibrated_oof[patient_mask, index])) * scale))
        stable_ratio = float(np.std(slopes[:, index], ddof=0) / (abs(np.mean(slopes[:, index])) + 1e-8))
        if float(np.var(base_prediction[:, index])) < IDENTITY_EPS:
            reasons.append("pred_z_variance_below_1e-6")
        if np.any(slopes[:, index] <= 0):
            reasons.append("nonpositive_outer_slope")
        if sum(cal <= base + 1e-12 for base, cal in zip(per_patient_base, per_patient_cal)) < 4:
            reasons.append("fewer_than_4_of_6_patient_z_mse_nonworse")
        if float(np.median(per_patient_mae_cal)) >= float(np.median(per_patient_mae_base)) - 1e-12:
            reasons.append("patient_median_raw_mae_not_lower")
        if calibrated[index]["raw_r2"] <= baseline[index]["raw_r2"] + 1e-12:
            reasons.append("outer_oof_raw_r2_not_improved")
        if calibrated[index]["raw_mae"] > baseline[index]["raw_mae"] + 1e-12:
            reasons.append("outer_oof_raw_mae_worsened")
        if stable_ratio > 0.5:
            reasons.append("outer_slope_instability")
        enabled = not reasons
        mask[index] = enabled
        rows.append({
            "enabled": enabled,
            "fallback_reason": ";".join(reasons) if reasons else "",
            "baseline_raw_r2": baseline[index]["raw_r2"],
            "calibrated_raw_r2": calibrated[index]["raw_r2"],
            "delta_raw_r2": calibrated[index]["raw_r2"] - baseline[index]["raw_r2"],
            "baseline_raw_mae": baseline[index]["raw_mae"],
            "calibrated_raw_mae": calibrated[index]["raw_mae"],
            "delta_raw_mae": calibrated[index]["raw_mae"] - baseline[index]["raw_mae"],
            "median_patient_raw_mae_base": float(np.median(per_patient_mae_base)),
            "median_patient_raw_mae_calibrated": float(np.median(per_patient_mae_cal)),
            "nonworse_patient_z_mse_count": int(sum(cal <= base + 1e-12 for base, cal in zip(per_patient_base, per_patient_cal))),
            "outer_slope_mean": float(np.mean(slopes[:, index])),
            "outer_slope_std": float(np.std(slopes[:, index], ddof=0)),
            "outer_slope_stability_ratio": stable_ratio,
        })
    return mask, rows


def lambda_stability(folds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    selected = [float(fold["selected_lambda"]) for fold in folds]
    counts = Counter(selected)
    weak_count = counts.get(0.1, 0)
    failure_reasons = []
    if len(counts) == 3:
        failure_reasons.append("all_three_lambda_values_selected_across_outer_folds")
    if 0 < weak_count < 3:
        failure_reasons.append("weakest_lambda_0.1_lacks_cross_fold_consistency")
    return {
        "selected_lambdas": selected,
        "selection_counts": {str(key): value for key, value in sorted(counts.items())},
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


try:  # Keep fitting utilities importable in environments without PyTorch.
    import torch
    from torch import nn

    class CalibratedMPP2(nn.Module):
        """Frozen head with calibrated, final raw outputs and strict provenance binding."""

        def __init__(self, base_head: nn.Module, *, k: Sequence[float], b: Sequence[float],
                     mean: Sequence[float], std: Sequence[float], pathway_order: Sequence[str],
                     provenance: Dict[str, Any]) -> None:
            super().__init__()
            lengths = {len(k), len(b), len(mean), len(std), len(pathway_order)}
            if lengths != {30}:
                raise ValueError("CalibratedMPP2 requires exactly 30 aligned pathway values")
            required = {"base_checkpoint_sha256", "data_manifest_id", "zscore_params_sha256", "pathway_order_sha256"}
            missing = sorted(required - set(provenance))
            if missing:
                raise ValueError(f"calibrator provenance is incomplete: {missing}")
            self.base_head = base_head
            for parameter in self.base_head.parameters():
                parameter.requires_grad_(False)
            self.register_buffer("k", torch.as_tensor(k, dtype=torch.float32), persistent=True)
            self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32), persistent=True)
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32), persistent=True)
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32), persistent=True)
            self.pathway_order = tuple(pathway_order)
            self.provenance = dict(provenance)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            pred_z = self.base_head(features)
            if pred_z.shape[-1] != 30:
                raise RuntimeError(f"base head output dimension must be 30, got {pred_z.shape[-1]}")
            return (pred_z * self.k + self.b) * self.std + self.mean

        def assert_binding(self, actual: Dict[str, str]) -> None:
            for key in ("base_checkpoint_sha256", "data_manifest_id", "zscore_params_sha256", "pathway_order_sha256"):
                if actual.get(key) != self.provenance.get(key):
                    raise RuntimeError(f"CalibratedMPP2 provenance mismatch for {key}")
except ImportError:  # pragma: no cover
    CalibratedMPP2 = None  # type: ignore[assignment,misc]
