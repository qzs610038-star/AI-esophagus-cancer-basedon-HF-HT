"""Recompute exploratory paper-facing metrics from existing prediction CSVs only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ANALYSIS_ID = "phase2_phase3_reanalysis_v002"
ANALYSIS_CLASS = "exploratory_reanalysis"
EVIDENCE_STATUS = "pending_user_review"
W007_ARMS = ("FBR", "RCC", "HCR", "CPGCR")
W007_SPLITS = {
    "internal_val_predictions.csv": "internal_val",
    "XZY_predictions.csv": "XZY",
}
W004_BUNDLES = {
    "A001": {
        "arm": "ABS",
        "card_id": "S2-ABS-001",
        "relative_root": Path(
            "automation/returns/W004/A001/R002-repack-R001/artifacts/predictions"
        ),
    },
    "A002": {
        "arm": "ORD",
        "card_id": "S2-ORD-SLIDE-001",
        "relative_root": Path(
            "automation/returns/W004/A002/R001-repack/artifacts/predictions"
        ),
    },
    "A003": {
        "arm": "SPATIAL_IDENTITY",
        "card_id": "S2-SPATIAL-IDENTITY-001",
        "relative_root": Path(
            "automation/returns/W004/A003/R001-repack/artifacts/predictions"
        ),
    },
    "A004": {
        "arm": "SPATIAL_SHUFFLE",
        "card_id": "S2-SPATIAL-SHUFFLE-001",
        "relative_root": Path(
            "automation/returns/W004/A004/R001-repack/artifacts/predictions"
        ),
    },
}
COORDINATE_PATTERN = re.compile(r"patch_x(-?\d+)_y(-?\d+)")
W007_METRIC_COLUMNS = (
    "pcc",
    "ccc",
    "z_rmse",
    "raw_r2",
    "spearman",
    "pcc_minus_ccc",
    "pcc_squared_minus_raw_r2",
    "z_mse",
    "bias_squared",
    "scale_mismatch_squared",
    "decorrelation_component",
    "decomposition_sum",
    "decomposition_residual",
)


def _finite_pair(left: Iterable[float], right: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    keep = np.isfinite(left_array) & np.isfinite(right_array)
    return left_array[keep], right_array[keep]


def _pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    if truth.size < 2 or np.std(truth) == 0.0 or np.std(prediction) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def _concordance_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    mean_truth = float(np.mean(truth))
    mean_prediction = float(np.mean(prediction))
    variance_truth = float(np.var(truth))
    variance_prediction = float(np.var(prediction))
    covariance = float(np.mean((truth - mean_truth) * (prediction - mean_prediction)))
    denominator = variance_truth + variance_prediction + (mean_truth - mean_prediction) ** 2
    if denominator == 0.0:
        return float("nan")
    return float(2.0 * covariance / denominator)


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_rank = pd.Series(truth).rank(method="average").to_numpy(dtype=float)
    prediction_rank = pd.Series(prediction).rank(method="average").to_numpy(dtype=float)
    return _pearson(truth_rank, prediction_rank)


def _raw_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def _find_w007_prediction_files(project_root: Path) -> list[tuple[str, str, Path]]:
    quarantine_root = project_root / "project_state" / "inbox" / "W007" / "quarantine"
    located: list[tuple[str, str, Path]] = []
    for arm in W007_ARMS:
        arm_root = quarantine_root / arm
        for filename, split in W007_SPLITS.items():
            matches = sorted(arm_root.rglob(filename))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one W007 {arm}/{filename}, found {len(matches)}"
                )
            located.append((arm, split, matches[0]))
    return located


def _single_pathway_metrics(
    truth_z_values: Iterable[float],
    prediction_z_values: Iterable[float],
    truth_raw_values: Iterable[float],
    prediction_raw_values: Iterable[float],
) -> dict[str, float]:
    truth_z, prediction_z = _finite_pair(truth_z_values, prediction_z_values)
    truth_raw, prediction_raw = _finite_pair(truth_raw_values, prediction_raw_values)
    if truth_z.size != len(list(truth_z_values)) or truth_raw.size != len(
        list(truth_raw_values)
    ):
        raise RuntimeError("Non-finite W007 prediction values")

    pcc = _pearson(truth_z, prediction_z)
    ccc = _concordance_correlation(truth_z, prediction_z)
    raw_r2 = _raw_r2(truth_raw, prediction_raw)
    z_error = prediction_z - truth_z
    z_mse = float(np.mean(z_error**2))
    bias_squared = float((np.mean(prediction_z) - np.mean(truth_z)) ** 2)
    scale_mismatch_squared = float((np.std(prediction_z) - np.std(truth_z)) ** 2)
    decorrelation_component = float(
        2.0 * np.std(truth_z) * np.std(prediction_z) * (1.0 - pcc)
    )
    decomposition_sum = bias_squared + scale_mismatch_squared + decorrelation_component
    return {
        "pcc": pcc,
        "ccc": ccc,
        "z_rmse": float(np.sqrt(z_mse)),
        "raw_r2": raw_r2,
        "spearman": _spearman(truth_z, prediction_z),
        "pcc_minus_ccc": pcc - ccc,
        "pcc_squared_minus_raw_r2": pcc**2 - raw_r2,
        "z_mse": z_mse,
        "bias_squared": bias_squared,
        "scale_mismatch_squared": scale_mismatch_squared,
        "decorrelation_component": decorrelation_component,
        "decomposition_sum": decomposition_sum,
        "decomposition_residual": z_mse - decomposition_sum,
    }


def compute_patient_balanced_pathway_metrics(
    table: pd.DataFrame, arm: str, split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute each patient first, then take an equal-patient arithmetic mean."""

    pathways = sorted(
        column.removeprefix("truth_z__")
        for column in table.columns
        if column.startswith("truth_z__")
    )
    patient_rows: list[dict[str, object]] = []
    for patient, patient_table in table.groupby("patient", sort=True):
        for pathway in pathways:
            metrics = _single_pathway_metrics(
                patient_table[f"truth_z__{pathway}"],
                patient_table[f"prediction_z__{pathway}"],
                patient_table[f"truth_raw__{pathway}"],
                patient_table[f"prediction_raw__{pathway}"],
            )
            patient_rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_class": ANALYSIS_CLASS,
                    "evidence_status": EVIDENCE_STATUS,
                    "arm": arm,
                    "split": split,
                    "patient": patient,
                    "pathway": pathway,
                    "aggregation": "patient_level",
                    "n_spots": len(patient_table),
                    **metrics,
                }
            )
    patient_level = pd.DataFrame(patient_rows)

    balanced_rows: list[dict[str, object]] = []
    for pathway, pathway_table in patient_level.groupby("pathway", sort=True):
        metric_means = {
            metric: float(pathway_table[metric].mean())
            for metric in W007_METRIC_COLUMNS
        }
        balanced_rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "analysis_class": ANALYSIS_CLASS,
                "evidence_status": EVIDENCE_STATUS,
                "arm": arm,
                "split": split,
                "pathway": pathway,
                "aggregation": "patient_balanced_mean",
                "n_patients": pathway_table["patient"].nunique(),
                "n_spots": int(pathway_table["n_spots"].sum()),
                **metric_means,
            }
        )
    return pd.DataFrame(balanced_rows), patient_level


def _w007_pathway_metrics(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    balanced_tables: list[pd.DataFrame] = []
    patient_tables: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for arm, split, prediction_path in _find_w007_prediction_files(project_root):
        table = pd.read_csv(prediction_path)
        pathway_count = sum(column.startswith("truth_z__") for column in table.columns)
        if pathway_count != 30:
            raise RuntimeError(
                f"Expected 30 W007 pathways in {prediction_path}, found {pathway_count}"
            )
        balanced, patient_level = compute_patient_balanced_pathway_metrics(
            table, arm=arm, split=split
        )
        balanced_tables.append(balanced)
        patient_tables.append(patient_level)
        source_paths.append(prediction_path)
    return (
        pd.concat(balanced_tables, ignore_index=True),
        pd.concat(patient_tables, ignore_index=True),
        source_paths,
    )


def _positive_step(values: np.ndarray) -> int | None:
    differences = np.diff(np.unique(values))
    positive = differences[differences > 0]
    return int(np.min(positive)) if positive.size else None


def _rook_neighbors(coordinates: list[tuple[int, int]]) -> list[list[int]]:
    x_values = np.asarray([coordinate[0] for coordinate in coordinates], dtype=int)
    y_values = np.asarray([coordinate[1] for coordinate in coordinates], dtype=int)
    x_step = _positive_step(x_values)
    y_step = _positive_step(y_values)
    coordinate_to_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    neighbors: list[list[int]] = []
    for x_coordinate, y_coordinate in coordinates:
        candidates: list[tuple[int, int]] = []
        if x_step is not None:
            candidates.extend(
                [
                    (x_coordinate - x_step, y_coordinate),
                    (x_coordinate + x_step, y_coordinate),
                ]
            )
        if y_step is not None:
            candidates.extend(
                [
                    (x_coordinate, y_coordinate - y_step),
                    (x_coordinate, y_coordinate + y_step),
                ]
            )
        neighbors.append(
            [coordinate_to_index[candidate] for candidate in candidates if candidate in coordinate_to_index]
        )
    return neighbors


def _morans_i(values: np.ndarray, neighbors: list[list[int]]) -> float:
    centered = values - np.mean(values)
    denominator = float(np.sum(centered**2))
    weight_sum = sum(len(indices) for indices in neighbors)
    if denominator == 0.0 or weight_sum == 0:
        return float("nan")
    cross_product = sum(
        centered[index] * centered[neighbor]
        for index, indices in enumerate(neighbors)
        for neighbor in indices
    )
    return float((len(values) / weight_sum) * (cross_product / denominator))


def _hotspot_quadrant_counts(
    values: np.ndarray, neighbors: list[list[int]]
) -> dict[str, int]:
    centered = values - np.mean(values)
    counts = {"hh_count": 0, "ll_count": 0, "hl_count": 0, "lh_count": 0, "neutral_count": 0, "isolated_count": 0}
    for index, indices in enumerate(neighbors):
        if not indices:
            counts["isolated_count"] += 1
            continue
        local_value = centered[index]
        neighbor_mean = float(np.mean(centered[indices]))
        if local_value > 0.0 and neighbor_mean > 0.0:
            counts["hh_count"] += 1
        elif local_value < 0.0 and neighbor_mean < 0.0:
            counts["ll_count"] += 1
        elif local_value > 0.0 and neighbor_mean < 0.0:
            counts["hl_count"] += 1
        elif local_value < 0.0 and neighbor_mean > 0.0:
            counts["lh_count"] += 1
        else:
            counts["neutral_count"] += 1
    return counts


def _w007_spatial_outputs(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial_rows: list[dict[str, object]] = []
    hotspot_rows: list[dict[str, object]] = []
    for arm, split, prediction_path in _find_w007_prediction_files(project_root):
        table = pd.read_csv(prediction_path)
        parsed = table["spot_id"].astype(str).str.extract(COORDINATE_PATTERN)
        parsed.columns = ["x", "y"]
        coordinate_complete = not parsed.isna().any().any()
        if coordinate_complete:
            table = table.assign(
                _x=parsed["x"].astype(int).to_numpy(),
                _y=parsed["y"].astype(int).to_numpy(),
            )
        pathways = sorted(
            column.removeprefix("truth_z__")
            for column in table.columns
            if column.startswith("truth_z__")
        )
        for patient, patient_table in table.groupby("patient", sort=True):
            patient_table = patient_table.reset_index(drop=True)
            if not coordinate_complete:
                coordinate_reason = "spot_id_coordinate_parse_incomplete"
                coordinates: list[tuple[int, int]] = []
                neighbors: list[list[int]] = []
            else:
                coordinates = list(
                    zip(
                        patient_table["_x"].astype(int),
                        patient_table["_y"].astype(int),
                    )
                )
                coordinate_reason = ""
                if len(set(coordinates)) != len(coordinates):
                    coordinate_reason = "duplicate_coordinates_within_patient"
                    neighbors = []
                else:
                    neighbors = _rook_neighbors(coordinates)
                    if sum(len(indices) for indices in neighbors) == 0:
                        coordinate_reason = "no_rook_neighbors"

            for pathway in pathways:
                residual = (
                    patient_table[f"prediction_z__{pathway}"].to_numpy(dtype=float)
                    - patient_table[f"truth_z__{pathway}"].to_numpy(dtype=float)
                )
                moran_value = (
                    _morans_i(residual, neighbors) if not coordinate_reason else float("nan")
                )
                if not coordinate_reason and not np.isfinite(moran_value):
                    status = "not_computable"
                    reason = "constant_residual_or_zero_spatial_weight"
                elif coordinate_reason:
                    status = "not_computable"
                    reason = coordinate_reason
                else:
                    status = "computed"
                    reason = ""
                common = {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_class": ANALYSIS_CLASS,
                    "evidence_status": EVIDENCE_STATUS,
                    "arm": arm,
                    "split": split,
                    "patient": patient,
                    "pathway": pathway,
                    "n": len(patient_table),
                }
                spatial_rows.append(
                    {
                        **common,
                        "spatial_variable": "z_residual",
                        "neighbor_definition": "rook_minimum_positive_axis_step",
                        "morans_i": moran_value,
                        "computation_status": status,
                        "reason": reason,
                    }
                )
                if status == "computed":
                    hotspot_rows.append(
                        {
                            **common,
                            **_hotspot_quadrant_counts(residual, neighbors),
                            "hotspot_definition": "local_sign_quadrant_descriptive_no_significance_test",
                            "computation_status": "computed_descriptive",
                            "reason": "",
                        }
                    )
                else:
                    hotspot_rows.append(
                        {
                            **common,
                            "hh_count": 0,
                            "ll_count": 0,
                            "hl_count": 0,
                            "lh_count": 0,
                            "neutral_count": 0,
                            "isolated_count": 0,
                            "hotspot_definition": "local_sign_quadrant_descriptive_no_significance_test",
                            "computation_status": "not_computable",
                            "reason": reason,
                        }
                    )
    return pd.DataFrame(spatial_rows), pd.DataFrame(hotspot_rows)


def _load_w004_predictions(project_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    tables: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for attempt_id, bundle in W004_BUNDLES.items():
        prediction_root = project_root / bundle["relative_root"]
        paths = sorted(prediction_root.rglob("*.csv"))
        if len(paths) != 80:
            raise RuntimeError(
                f"Expected 80 W004 prediction CSVs for {attempt_id}, found {len(paths)}"
            )
        for path in paths:
            table = pd.read_csv(path)
            required_columns = {
                "case_id",
                "slide_id",
                "y_true",
                "y_pred",
                "task",
                "seed",
                "fold",
                "split",
            }
            missing = required_columns.difference(table.columns)
            if missing:
                raise RuntimeError(f"Missing W004 columns {sorted(missing)} in {path}")
            table = table.assign(
                attempt_id=attempt_id,
                arm=bundle["arm"],
                analysis_card_id=bundle["card_id"],
                source_file=str(path.relative_to(project_root)),
            )
            tables.append(table)
            source_paths.append(path)
    return pd.concat(tables, ignore_index=True), source_paths


def _calibration_fit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, str, str]:
    if np.unique(y_true).size != 2:
        return float("nan"), float("nan"), "not_computable", "single_class_outcome"
    clipped = np.clip(y_pred, 1e-6, 1.0 - 1e-6)
    logit_prediction = np.log(clipped / (1.0 - clipped))
    if float(np.std(logit_prediction)) == 0.0:
        return float("nan"), float("nan"), "not_computable", "constant_prediction_logit"
    design = np.column_stack([np.ones(len(y_true)), logit_prediction])
    prevalence = float(np.mean(y_true))
    coefficients = np.array([np.log(prevalence / (1.0 - prevalence)), 1.0])
    converged = False
    for _ in range(100):
        linear_predictor = np.clip(design @ coefficients, -35.0, 35.0)
        fitted_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        weights = fitted_probability * (1.0 - fitted_probability)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y_true - fitted_probability)
        try:
            update = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), "not_computable", "singular_calibration_information"
        coefficients = coefficients + update
        if not np.isfinite(coefficients).all() or np.max(np.abs(coefficients)) > 1e6:
            return float("nan"), float("nan"), "not_computable", "unstable_or_separated_calibration_fit"
        if float(np.max(np.abs(update))) < 1e-10:
            converged = True
            break
    if not converged:
        return float("nan"), float("nan"), "not_computable", "calibration_fit_did_not_converge"
    return float(coefficients[0]), float(coefficients[1]), "computed", ""


def _binary_metrics(table: pd.DataFrame) -> dict[str, object]:
    y_true = table["y_true"].to_numpy(dtype=float)
    y_pred = table["y_pred"].to_numpy(dtype=float)
    intercept, slope, calibration_status, calibration_reason = _calibration_fit(
        y_true, y_pred
    )
    return {
        "n": len(table),
        "n_positive": int(np.sum(y_true == 1.0)),
        "n_negative": int(np.sum(y_true == 0.0)),
        "auc": float(roc_auc_score(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_status": calibration_status,
        "calibration_reason": calibration_reason,
    }


def _tagged(row: dict[str, object]) -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_class": ANALYSIS_CLASS,
        "evidence_status": EVIDENCE_STATUS,
        **row,
    }


def _w004_metric_outputs(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_fit_rows: list[dict[str, object]] = []
    per_fit_grouping = [
        "attempt_id",
        "arm",
        "analysis_card_id",
        "task",
        "seed",
        "fold",
        "split",
    ]
    for keys, table in predictions.groupby(per_fit_grouping, sort=True):
        identity = dict(zip(per_fit_grouping, keys))
        per_fit_rows.append(_tagged({**identity, **_binary_metrics(table)}))
    per_fit = pd.DataFrame(per_fit_rows)

    pooled_rows: list[dict[str, object]] = []
    pooled_grouping = ["attempt_id", "arm", "analysis_card_id", "task", "split"]
    for keys, table in predictions.groupby(pooled_grouping, sort=True):
        identity = dict(zip(pooled_grouping, keys))
        pooled_rows.append(
            _tagged(
                {
                    **identity,
                    "pooling_scope": "all_seed_fold_prediction_rows_descriptive",
                    **_binary_metrics(table),
                }
            )
        )
    pooled = pd.DataFrame(pooled_rows)

    contrast_rows: list[dict[str, object]] = []
    contrast_definitions = [
        ("ORD_MINUS_ABS", "ORD", "ABS"),
        (
            "SPATIAL_IDENTITY_MINUS_SPATIAL_SHUFFLE",
            "SPATIAL_IDENTITY",
            "SPATIAL_SHUFFLE",
        ),
    ]
    test_per_fit = per_fit.loc[per_fit["split"].eq("test")]
    test_pooled = pooled.loc[pooled["split"].eq("test")]
    for contrast, candidate, reference in contrast_definitions:
        for task in ("pCR", "MPR"):
            candidate_fits = test_per_fit.loc[
                test_per_fit["arm"].eq(candidate) & test_per_fit["task"].eq(task),
                ["seed", "fold", "auc"],
            ]
            reference_fits = test_per_fit.loc[
                test_per_fit["arm"].eq(reference) & test_per_fit["task"].eq(task),
                ["seed", "fold", "auc"],
            ]
            paired = candidate_fits.merge(
                reference_fits,
                on=["seed", "fold"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            difference = paired["auc_candidate"] - paired["auc_reference"]
            candidate_pooled_auc = float(
                test_pooled.loc[
                    test_pooled["arm"].eq(candidate) & test_pooled["task"].eq(task),
                    "auc",
                ].iloc[0]
            )
            reference_pooled_auc = float(
                test_pooled.loc[
                    test_pooled["arm"].eq(reference) & test_pooled["task"].eq(task),
                    "auc",
                ].iloc[0]
            )
            contrast_rows.append(
                _tagged(
                    {
                        "contrast": contrast,
                        "task": task,
                        "split": "test",
                        "candidate_arm": candidate,
                        "reference_arm": reference,
                        "n_paired_fits": len(paired),
                        "mean_per_fit_auc_difference": float(difference.mean()),
                        "median_per_fit_auc_difference": float(difference.median()),
                        "min_per_fit_auc_difference": float(difference.min()),
                        "max_per_fit_auc_difference": float(difference.max()),
                        "pooled_auc_difference": candidate_pooled_auc
                        - reference_pooled_auc,
                    }
                )
            )
    contrasts = pd.DataFrame(contrast_rows)

    case_rows: list[dict[str, object]] = []
    test_predictions = predictions.loc[predictions["split"].eq("test")]
    for (attempt_id, arm, task, case_id), table in test_predictions.groupby(
        ["attempt_id", "arm", "task", "case_id"], sort=True
    ):
        labels = table["y_true"].drop_duplicates()
        prediction = table["y_pred"].to_numpy(dtype=float)
        truth = table["y_true"].to_numpy(dtype=float)
        case_rows.append(
            _tagged(
                {
                    "attempt_id": attempt_id,
                    "arm": arm,
                    "task": task,
                    "case_id": case_id,
                    "y_true": float(labels.iloc[0]) if len(labels) == 1 else float("nan"),
                    "n_distinct_y_true": len(labels),
                    "n_predictions": len(table),
                    "n_unique_slides": table["slide_id"].nunique(),
                    "n_unique_seeds": table["seed"].nunique(),
                    "n_unique_folds": table["fold"].nunique(),
                    "mean_y_pred": float(np.mean(prediction)),
                    "median_y_pred": float(np.median(prediction)),
                    "std_y_pred": float(np.std(prediction)),
                    "min_y_pred": float(np.min(prediction)),
                    "max_y_pred": float(np.max(prediction)),
                    "mean_absolute_error": float(np.mean(np.abs(prediction - truth))),
                    "mean_squared_error": float(np.mean((prediction - truth) ** 2)),
                    "interpretation": "post_hoc_descriptive",
                }
            )
        )
    case_posthoc = pd.DataFrame(case_rows)

    spatial_rows: list[dict[str, object]] = []
    for (attempt_id, arm, task), table in predictions.groupby(
        ["attempt_id", "arm", "task"], sort=True
    ):
        has_coordinate_columns = {"x", "y"}.issubset(table.columns)
        spatial_rows.append(
            _tagged(
                {
                    "attempt_id": attempt_id,
                    "arm": arm,
                    "task": task,
                    "computation_status": "computed"
                    if has_coordinate_columns
                    else "not_computable",
                    "reason": ""
                    if has_coordinate_columns
                    else "prediction_csv_has_no_coordinate_columns",
                }
            )
        )
    spatial_status = pd.DataFrame(spatial_rows)
    return per_fit, pooled, contrasts, case_posthoc, spatial_status


def _markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return "not_computable"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table.loc[:, columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _write_review_report(
    output_path: Path,
    w007_metrics: pd.DataFrame,
    w007_patient_metrics: pd.DataFrame,
    w007_spatial: pd.DataFrame,
    w007_hotspots: pd.DataFrame,
    w004_per_fit: pd.DataFrame,
    w004_pooled: pd.DataFrame,
    w004_contrasts: pd.DataFrame,
    w004_case_posthoc: pd.DataFrame,
    w004_spatial_status: pd.DataFrame,
) -> None:
    w007_summary_columns = [
        "pcc",
        "ccc",
        "z_rmse",
        "raw_r2",
        "spearman",
        "pcc_minus_ccc",
        "pcc_squared_minus_raw_r2",
    ]
    w007_summary = (
        w007_metrics.groupby(["arm", "split"], as_index=False)[w007_summary_columns]
        .mean()
        .sort_values(["arm", "split"])
    )
    pooled_columns = [
        "arm",
        "task",
        "split",
        "n",
        "auc",
        "auprc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "calibration_status",
    ]
    contrast_columns = [
        "contrast",
        "task",
        "n_paired_fits",
        "mean_per_fit_auc_difference",
        "pooled_auc_difference",
    ]
    not_computable_reasons = sorted(
        set(
            w004_spatial_status.loc[
                w004_spatial_status["computation_status"].eq("not_computable"),
                "reason",
            ]
        )
    )
    report = f"""# Phase 2 / Phase 3 本地探索性复算待审报告

- analysis_id: `{ANALYSIS_ID}`
- analysis_class: `{ANALYSIS_CLASS}`
- evidence_status: `{EVIDENCE_STATUS}`
- Registry: 未读取、未修改、未晋级。
- 输入边界：仅现有 W007 8 张 prediction CSV 与 W004 四个预测包的 320 张 prediction CSV。

## W007 pathway-wise 汇总

口径：每个 patient×pathway 先独立计算指标，再对 patient 做等权算术平均；internal_val 为 6 人等权，XZY 为 1 人。下表再对 30 个 pathway 做等权汇总；患者平衡的逐 pathway 数值见 `w007_pathway_metrics.csv`，逐患者明细见 `w007_patient_pathway_metrics.csv`（共 `{len(w007_patient_metrics)}` 行）。

{_markdown_table(w007_summary, ["arm", "split", *w007_summary_columns])}

W007 空间输出：Moran's I 可计算 `{int(w007_spatial["computation_status"].eq("computed").sum())}` 行；局部象限热点为无显著性检验的描述性计数，共 `{len(w007_hotspots)}` 行。

## W004 pooled 分类指标

{_markdown_table(w004_pooled.sort_values(["arm", "task", "split"]), pooled_columns)}

## W004 固定 AUC 差值

{_markdown_table(w004_contrasts.sort_values(["contrast", "task"]), contrast_columns)}

per-fit 共 `{len(w004_per_fit)}` 行；case_id post hoc 描述共 `{len(w004_case_posthoc)}` 行。case_id 结果仅用于事后描述，不作为预注册推断。

## 不可计算项

- W004 Moran's I / 热点：`not_computable`。
- 原因：`{', '.join(not_computable_reasons)}`。
- W007 热点象限仅为 `computed_descriptive`，不包含置换检验或多重比较校正。

## 审核边界

所有输出均为 `{ANALYSIS_CLASS}` / `{EVIDENCE_STATUS}`；不得直接写入论文结论或晋级为 accepted evidence。
"""
    output_path.write_text(report, encoding="utf-8")


def run_reanalysis(project_root: Path, output_dir: Path) -> dict[str, object]:
    """Run the prediction-only exploratory reanalysis and write review artifacts."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    w007_metrics, w007_patient_metrics, w007_sources = _w007_pathway_metrics(
        project_root
    )
    w007_spatial, w007_hotspots = _w007_spatial_outputs(project_root)
    w004_predictions, w004_sources = _load_w004_predictions(project_root)
    (
        w004_per_fit,
        w004_pooled,
        w004_contrasts,
        w004_case_posthoc,
        w004_spatial_status,
    ) = _w004_metric_outputs(w004_predictions)

    w007_metrics.to_csv(output_dir / "w007_pathway_metrics.csv", index=False)
    w007_patient_metrics.to_csv(
        output_dir / "w007_patient_pathway_metrics.csv", index=False
    )
    w007_spatial.to_csv(output_dir / "w007_spatial_metrics.csv", index=False)
    w007_hotspots.to_csv(output_dir / "w007_hotspot_summary.csv", index=False)
    w004_per_fit.to_csv(output_dir / "w004_per_fit_metrics.csv", index=False)
    w004_pooled.to_csv(output_dir / "w004_pooled_metrics.csv", index=False)
    w004_contrasts.to_csv(output_dir / "w004_fixed_auc_differences.csv", index=False)
    w004_case_posthoc.to_csv(output_dir / "w004_case_posthoc.csv", index=False)
    w004_spatial_status.to_csv(output_dir / "w004_spatial_status.csv", index=False)
    _write_review_report(
        output_dir / "review_report.md",
        w007_metrics,
        w007_patient_metrics,
        w007_spatial,
        w007_hotspots,
        w004_per_fit,
        w004_pooled,
        w004_contrasts,
        w004_case_posthoc,
        w004_spatial_status,
    )

    metadata: dict[str, object] = {
        "analysis_id": ANALYSIS_ID,
        "analysis_class": ANALYSIS_CLASS,
        "evidence_status": EVIDENCE_STATUS,
        "registry_mutated": False,
        "input_scope": "existing_W007_and_W004_prediction_csvs_only",
        "w007_prediction_file_count": len(w007_sources),
        "w007_aggregation": "patient_level_then_equal_patient_arithmetic_mean",
        "w007_internal_val_patient_count": 6,
        "w007_xzy_patient_count": 1,
        "w004_prediction_file_count": len(w004_sources),
        "inputs": [
            str(path.relative_to(project_root))
            for path in [*w007_sources, *w004_sources]
        ],
        "artifacts": [
            "w007_pathway_metrics.csv",
            "w007_patient_pathway_metrics.csv",
            "w007_spatial_metrics.csv",
            "w007_hotspot_summary.csv",
            "w004_per_fit_metrics.csv",
            "w004_pooled_metrics.csv",
            "w004_fixed_auc_differences.csv",
            "w004_case_posthoc.csv",
            "w004_spatial_status.csv",
            "review_report.md",
            "metadata.json",
        ],
        "limitations": [
            "Exploratory reanalysis only; not accepted evidence.",
            "No experiment Registry or current-state file was read or modified.",
            "W004 prediction CSVs contain no coordinate columns, so Moran's I and hotspot summaries are not computable.",
            "W007 hotspot quadrants are descriptive and include no significance test.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root
        / "experiments"
        / "explorations"
        / "phase2_phase3_reanalysis_v002",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    summary = run_reanalysis(arguments.project_root, arguments.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
