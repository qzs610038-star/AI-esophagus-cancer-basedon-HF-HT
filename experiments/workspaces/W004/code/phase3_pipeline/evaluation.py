"""Metrics, patient-cluster uncertainty, and output-contract checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


OOF_COLUMNS = ("case_id", "slide_id", "task", "y_true", "y_pred", "seed", "fold", "model")


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("binary metric inputs must be aligned finite vectors")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def aggregate_patient_predictions(oof: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    required = {"case_id", "y_true", "y_pred"}
    if missing := required - set(oof.columns):
        raise ValueError(f"OOF table missing columns: {sorted(missing)}")
    if method != "mean":
        raise ValueError("only frozen mean aggregation is supported in tabular evaluation")
    label_counts = oof.groupby("case_id")["y_true"].nunique()
    if (label_counts > 1).any():
        raise ValueError("inconsistent labels within patient")
    return oof.groupby("case_id", as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), n_slides=("slide_id", "nunique"))


def patient_cluster_bootstrap_auc(
    oof: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Resample patients, retaining all their slides in each draw."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    required = {"case_id", "y_true", "y_pred"}
    if missing := required - set(oof.columns):
        raise ValueError(f"OOF table missing columns: {sorted(missing)}")
    groups = {str(case): frame for case, frame in oof.groupby("case_id", sort=False)}
    patient_ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        frames = [groups[str(case)] for case in sampled]
        y = np.concatenate([frame["y_true"].to_numpy() for frame in frames])
        p = np.concatenate([frame["y_pred"].to_numpy() for frame in frames])
        if len(np.unique(y)) == 2:
            aucs.append(float(roc_auc_score(y, p)))
    if not aucs:
        raise ValueError("no bootstrap draw contained both classes")
    return {
        "valid_draws": len(aucs),
        "median_auc": float(np.median(aucs)),
        "ci95_low": float(np.quantile(aucs, 0.025)),
        "ci95_high": float(np.quantile(aucs, 0.975)),
    }


def validate_oof_table(oof: pd.DataFrame) -> pd.DataFrame:
    if missing := set(OOF_COLUMNS) - set(oof.columns):
        raise ValueError(f"OOF table missing columns: {sorted(missing)}")
    if oof.empty:
        raise ValueError("OOF table is empty")
    if oof[list(OOF_COLUMNS)].isna().any().any():
        raise ValueError("OOF contract columns contain missing values")
    if oof.duplicated(["task", "seed", "fold", "model", "slide_id"]).any():
        raise ValueError("OOF table contains duplicate model predictions")
    if not set(oof["y_true"].unique()).issubset({0, 1}):
        raise ValueError("OOF y_true must be binary")
    if ((oof["y_pred"] < 0) | (oof["y_pred"] > 1)).any():
        raise ValueError("OOF y_pred must be probabilities")
    return oof


def validate_partition_oof(oof: pd.DataFrame, expected_slide_ids: set[str]) -> pd.DataFrame:
    """Require each expected slide exactly once per task/seed/model partition."""
    validate_oof_table(oof)
    if "prediction_kind" not in oof or set(oof["prediction_kind"]) != {"partition_oof"}:
        raise ValueError("strict OOF rows must declare prediction_kind=partition_oof")
    for key, group in oof.groupby(["task", "seed", "model"], sort=False):
        actual = set(group["slide_id"].astype(str))
        if actual != {str(x) for x in expected_slide_ids} or group["slide_id"].duplicated().any():
            raise ValueError(f"OOF coverage is not exactly once for {key}")
    return oof


def validate_repeated_holdout_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    """Audit repeated test subsets without calling them partition OOF."""
    validate_oof_table(predictions)
    if "prediction_kind" not in predictions or set(predictions["prediction_kind"]) != {"repeated_holdout_test"}:
        raise ValueError("rows must declare prediction_kind=repeated_holdout_test")
    required_context = {"evidence_scope", "warnings"}
    if missing := required_context - set(predictions.columns):
        raise ValueError(f"repeated holdout rows missing evidence context: {sorted(missing)}")
    if predictions[list(required_context)].isna().any().any():
        raise ValueError("repeated holdout evidence context cannot be missing")
    counts = predictions.groupby(["task", "seed", "model", "slide_id"]).size()
    return {
        "prediction_kind": "repeated_holdout_test",
        "unique_slides": int(len(counts)),
        "rows": int(len(predictions)),
        "slides_repeated_across_folds": int((counts > 1).sum()),
        "max_repetitions": int(counts.max()),
        "evidence_scope": "repeated_holdout_not_partition_oof",
        "source_evidence_scopes": sorted(predictions["evidence_scope"].unique().tolist()),
        "warning_values": sorted(predictions["warnings"].unique().tolist()),
    }


def paired_fold_comparison(oof: pd.DataFrame, reference_model: str, candidate_model: str) -> pd.DataFrame:
    """Compare models on identical task/seed/fold/slide rows."""
    validate_oof_table(oof)
    keys = ["case_id", "slide_id", "task", "seed", "fold", "y_true"]
    left = oof[oof["model"] == reference_model][keys + ["y_pred"]].rename(columns={"y_pred": "reference_pred"})
    right = oof[oof["model"] == candidate_model][keys + ["y_pred"]].rename(columns={"y_pred": "candidate_pred"})
    paired = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError("models do not have identical paired OOF rows")
    rows = []
    for (task, seed, fold), group in paired.groupby(["task", "seed", "fold"], sort=True):
        y = group["y_true"].to_numpy()
        if len(np.unique(y)) != 2:
            reference_auc = candidate_auc = float("nan")
        else:
            reference_auc = float(roc_auc_score(y, group["reference_pred"]))
            candidate_auc = float(roc_auc_score(y, group["candidate_pred"]))
        rows.append({"task": task, "seed": seed, "fold": fold, "reference_auc": reference_auc,
                     "candidate_auc": candidate_auc, "delta_auc": candidate_auc - reference_auc})
    return pd.DataFrame(rows)


def leave_one_patient_out_auc(oof: pd.DataFrame) -> pd.DataFrame:
    """Measure whether removing one patient flips the aggregate AUC direction."""
    required = {"case_id", "y_true", "y_pred"}
    if missing := required - set(oof.columns):
        raise ValueError(f"OOF table missing columns: {sorted(missing)}")
    rows = []
    for patient in sorted(oof["case_id"].astype(str).unique()):
        subset = oof[oof["case_id"].astype(str) != patient]
        auc = float(roc_auc_score(subset["y_true"], subset["y_pred"])) if subset["y_true"].nunique() == 2 else float("nan")
        rows.append({"excluded_case_id": patient, "remaining_n": int(len(subset)), "auc": auc})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CalibrationModel:
    slope: float
    intercept: float


def fit_calibration(y_validation: np.ndarray, p_validation: np.ndarray) -> CalibrationModel:
    """Fit a logistic calibration map on validation predictions only."""

    y = np.asarray(y_validation, dtype=int)
    p = np.clip(np.asarray(p_validation, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) != 2:
        raise ValueError("calibration validation set must contain both classes")
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logits, y)
    return CalibrationModel(float(model.coef_[0, 0]), float(model.intercept_[0]))


def apply_calibration(probability: np.ndarray, model: CalibrationModel) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    z = model.intercept + model.slope * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z))


def conformal_risk_coverage(
    y_calibration: np.ndarray,
    p_calibration: np.ndarray,
    p_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Binary split-conformal prediction sets and their retained coverage."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y = np.asarray(y_calibration, dtype=int)
    p = np.asarray(p_calibration, dtype=float)
    if y.shape != p.shape or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("invalid calibration inputs")
    true_probability = np.where(y == 1, p, 1 - p)
    scores = 1 - true_probability
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    threshold = float(np.quantile(scores, level, method="higher"))
    test = np.asarray(p_test, dtype=float)
    include_zero = (1 - test) >= 1 - threshold
    include_one = test >= 1 - threshold
    set_size = include_zero.astype(int) + include_one.astype(int)
    return {
        "threshold": threshold,
        "include_zero": include_zero,
        "include_one": include_one,
        "set_size": set_size,
        "singleton_coverage": float(np.mean(set_size == 1)),
    }
