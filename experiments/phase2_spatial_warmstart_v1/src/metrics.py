"""Protocol metrics and deterministic internal model-selection ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _paired_inputs(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same [points, pathways] shape")
    if pred.shape[0] < 1 or pred.shape[1] < 1:
        raise ValueError("metric arrays cannot be empty")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("metric arrays contain non-finite values")
    return pred, truth


def _pcc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.size < 2 or x.size != y.size:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def patient_macro_pathway_pcc(
    prediction: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]
) -> float:
    """Equal-weight mean of every valid patient-pathway PCC."""

    return patient_macro_pathway_pcc_details(prediction, target, patient_ids)["value"]


def patient_macro_pathway_pcc_details(
    prediction: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]
) -> dict[str, float | int]:
    pred, truth = _paired_inputs(prediction, target)
    patients = np.asarray(patient_ids, dtype=str)
    if patients.ndim != 1 or patients.shape[0] != pred.shape[0]:
        raise ValueError("patient_ids length must equal the number of points")
    values: list[float] = []
    invalid = 0
    for patient in sorted(set(patients.tolist())):
        mask = patients == patient
        for pathway in range(pred.shape[1]):
            score = _pcc(pred[mask, pathway], truth[mask, pathway])
            if np.isfinite(score):
                values.append(score)
            else:
                invalid += 1
    return {
        "value": float(np.mean(values)) if values else float("nan"),
        "valid_count": len(values),
        "invalid_count": invalid,
    }


def flattened_pooled_pcc(prediction: np.ndarray, target: np.ndarray) -> float:
    """One PCC over all point-by-pathway z scores after flattening."""

    pred, truth = _paired_inputs(prediction, target)
    return _pcc(pred.ravel(), truth.ravel())


pooled_pcc = flattened_pooled_pcc


def z_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _paired_inputs(prediction, target)
    return float(np.mean((pred - truth) ** 2))


def compute_regression_metrics(
    prediction: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]
) -> dict[str, float | int]:
    """Return the two required PCCs, zMSE, and invalid macro item count."""

    macro = patient_macro_pathway_pcc_details(prediction, target, patient_ids)
    flattened = flattened_pooled_pcc(prediction, target)
    return {
        "patient_macro_pathway_pcc": macro["value"],
        "flattened_pooled_pcc": flattened,
        "pooled_pcc": flattened,
        "zMSE": z_mse(prediction, target),
        "valid_patient_pathway_count": macro["valid_count"],
        "invalid_patient_pathway_count": macro["invalid_count"],
    }


compute_metrics = compute_regression_metrics


@dataclass(frozen=True)
class SelectionCandidate:
    """Internally evaluated candidate; update zero is intentionally valid."""

    update: int
    patient_macro_pathway_pcc: float
    zMSE: float
    arm: str | None = None

    def __post_init__(self) -> None:
        if int(self.update) < 0:
            raise ValueError("candidate update must be non-negative")


_ARM_COMPLEXITY_ORDER = {
    "point_continue": 0,
    "spatial_residual_only": 1,
    "spatial_joint": 2,
}


def selection_key(
    candidate: SelectionCandidate, *, architecture_tiebreak: bool = False
) -> tuple[float | int, ...]:
    """Ascending key implementing the predeclared internal-only ordering."""

    pcc = float(candidate.patient_macro_pathway_pcc)
    mse = float(candidate.zMSE)
    pcc_invalid = not np.isfinite(pcc)
    mse_invalid = not np.isfinite(mse)
    prefix: tuple[float | int, ...] = (
        int(pcc_invalid),
        -pcc if not pcc_invalid else float("inf"),
        int(mse_invalid),
        mse if not mse_invalid else float("inf"),
    )
    if architecture_tiebreak:
        if candidate.arm not in _ARM_COMPLEXITY_ORDER:
            raise ValueError(f"unknown or missing architecture arm: {candidate.arm!r}")
        return prefix + (_ARM_COMPLEXITY_ORDER[candidate.arm],)
    return prefix + (int(candidate.update),)


def select_best_candidate(
    candidates: Sequence[SelectionCandidate], *, architecture_tiebreak: bool = False
) -> SelectionCandidate:
    """Select by macro PCC, zMSE, then update or declared architecture complexity."""

    if not candidates:
        raise ValueError("cannot select from an empty candidate sequence")
    return min(
        candidates,
        key=lambda candidate: selection_key(
            candidate, architecture_tiebreak=architecture_tiebreak
        ),
    )
