"""Float64 internal-validation metrics and the frozen checkpoint rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from errors import NonFiniteDataError


def _is_constant(values: np.ndarray) -> bool:
    return values.size == 0 or bool(np.max(values) == np.min(values))


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = float(
        np.sqrt(
            np.sum(pred_centered * pred_centered)
            * np.sum(target_centered * target_centered)
        )
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(pred_centered * target_centered) / denominator)


def pooled_pathway_pcc(
    pred_z: np.ndarray, target_z: np.ndarray, pathway_names: Sequence[str] | None = None
) -> dict:
    """Report pooled-by-spot PCC; never use it for checkpoint selection."""

    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pred_z/target_z 必须是形状相同的二维矩阵")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise NonFiniteDataError("pooled PCC 遇到非有限值")
    names = list(pathway_names or [f"p{i}" for i in range(pred.shape[1])])
    if len(names) != pred.shape[1]:
        raise ValueError("pathway_names 长度与列数不一致")
    per_pathway: dict[str, float | None] = {}
    valid: list[float] = []
    for index, name in enumerate(names):
        value = _pearson(pred[:, index], target[:, index])
        per_pathway[name] = value if np.isfinite(value) else None
        if np.isfinite(value):
            valid.append(value)
    return {
        "macro_pathway_pcc": float(np.mean(valid)) if valid else None,
        "per_pathway_pcc": per_pathway,
        "n_valid_pathways": len(valid),
    }


def pooled_z_mse(pred_z: np.ndarray, target_z: np.ndarray) -> float:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致: {pred.shape} vs {target.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise NonFiniteDataError("pooled z-MSE 遇到非有限值")
    return float(np.mean((pred - target) ** 2))


@dataclass
class SelectionMetrics:
    patient_macro_pathway_pcc: float
    patient_macro_pathway_pcc_measured: float
    patient_macro_z_mse_selection: float
    pooled_z_mse: float
    n_patients: int
    n_valid_patient_pathway: int
    n_excluded_patient_pathway: int
    n_constant_prediction_penalized: int
    exclusions: list[dict] = field(default_factory=list)
    per_patient: dict = field(default_factory=dict)
    computable: bool = True
    incomputable_reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def patient_macro_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    patient_ids: Sequence[str],
    *,
    pathway_names: Sequence[str] | None = None,
    constant_prediction_selection_penalty: float = -1.0,
    require_patient_count: int | None = None,
) -> SelectionMetrics:
    """Compute pathway PCC inside each patient, then weight patients equally."""

    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    patients = np.asarray(list(patient_ids), dtype=object)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pred_z/target_z 必须是形状相同的二维矩阵")
    if pred.shape[0] != patients.shape[0]:
        raise ValueError("身份数量与预测行数不一致")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise NonFiniteDataError("预测或目标含非有限值")

    names = list(pathway_names or [f"p{i}" for i in range(pred.shape[1])])
    if len(names) != pred.shape[1]:
        raise ValueError("pathway_names 长度与列数不一致")
    unique_patients = sorted({str(value) for value in patients.tolist()})
    if require_patient_count is not None and len(unique_patients) != int(require_patient_count):
        raise NonFiniteDataError(
            f"患者等权指标要求 {require_patient_count} 名患者，实际={len(unique_patients)}"
        )

    penalty = float(constant_prediction_selection_penalty)
    patient_selection: list[float] = []
    patient_measured: list[float] = []
    patient_mse: list[float] = []
    exclusions: list[dict] = []
    per_patient: dict[str, dict] = {}
    valid_count = 0
    excluded_count = 0
    penalized_count = 0

    for patient_id in unique_patients:
        row_mask = patients == patient_id
        selection_values: list[float] = []
        measured_values: list[float] = []
        mse_values: list[float] = []
        pathways: dict[str, dict] = {}
        for index, name in enumerate(names):
            target_column = target[row_mask, index]
            prediction_column = pred[row_mask, index]
            item: dict = {"n": int(target_column.size)}
            if target_column.size < 2 or _is_constant(target_column):
                reason = "target_n_lt_2" if target_column.size < 2 else "target_variance_0"
                item.update(excluded=True, reason=reason)
                exclusions.append(
                    {"patient_id": patient_id, "pathway": name, "reason": reason, "n": int(target_column.size)}
                )
                pathways[name] = item
                excluded_count += 1
                continue
            measured = _pearson(prediction_column, target_column)
            if _is_constant(prediction_column) or not np.isfinite(measured):
                selection_value = penalty
                measured_value = float("nan")
                penalized_count += 1
                constant_prediction = True
            else:
                selection_value = float(measured)
                measured_value = float(measured)
                constant_prediction = False
            mse = float(np.mean((prediction_column - target_column) ** 2))
            item.update(
                excluded=False,
                measured_pcc=measured_value,
                selection_pcc=selection_value,
                z_mse=mse,
                constant_prediction=constant_prediction,
            )
            pathways[name] = item
            selection_values.append(selection_value)
            measured_values.append(measured_value)
            mse_values.append(mse)
            valid_count += 1

        if not selection_values:
            return SelectionMetrics(
                float("nan"),
                float("nan"),
                float("nan"),
                pooled_z_mse(pred, target),
                len(unique_patients),
                valid_count,
                excluded_count,
                penalized_count,
                exclusions,
                per_patient,
                False,
                f"patient={patient_id} 没有有效目标，不能静默删除患者",
            )
        finite_measured = [value for value in measured_values if np.isfinite(value)]
        record = {
            "n_points": int(np.sum(row_mask)),
            "pathways": pathways,
            "n_valid_pathways": len(selection_values),
            "selection_pcc_mean": float(np.mean(selection_values)),
            "measured_pcc_mean": float(np.mean(finite_measured)) if finite_measured else float("nan"),
            "z_mse_mean": float(np.mean(mse_values)),
        }
        per_patient[patient_id] = record
        patient_selection.append(record["selection_pcc_mean"])
        patient_measured.append(record["measured_pcc_mean"])
        patient_mse.append(record["z_mse_mean"])

    finite_patient_measured = [value for value in patient_measured if np.isfinite(value)]
    return SelectionMetrics(
        float(np.mean(patient_selection)),
        float(np.mean(finite_patient_measured)) if finite_patient_measured else float("nan"),
        float(np.mean(patient_mse)),
        pooled_z_mse(pred, target),
        len(unique_patients),
        valid_count,
        excluded_count,
        penalized_count,
        exclusions,
        per_patient,
    )


@dataclass
class CheckpointChoice:
    epoch: int | None = None
    score: float | None = None
    mse: float | None = None
    reason: str | None = None
    kind: str = "formal"

    def as_dict(self) -> dict:
        return asdict(self)


def update_checkpoint_choice(
    current: CheckpointChoice,
    *,
    epoch: int,
    score: float,
    mse: float,
    tolerance: float = 1e-6,
    kind: str = "formal",
) -> CheckpointChoice:
    if not np.isfinite(score) or not np.isfinite(mse):
        return current
    if current.epoch is None:
        return CheckpointChoice(int(epoch), float(score), float(mse), "first_eligible", kind)
    delta = float(score) - float(current.score)
    if delta > float(tolerance):
        return CheckpointChoice(int(epoch), float(score), float(mse), "higher_pcc", kind)
    if abs(delta) <= float(tolerance) and float(mse) < float(current.mse):
        return CheckpointChoice(
            int(epoch), float(score), float(mse), "pcc_tie_lower_patient_macro_z_mse", kind
        )
    return current


@dataclass
class EarlyStopState:
    reference: float | None = None
    count: int = 0
    stopped: bool = False
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def update_early_stop(
    state: EarlyStopState,
    *,
    epoch: int,
    score: float,
    formal_start_epoch: int = 6,
    count_start_epoch: int = 16,
    min_delta: float = 1e-4,
    patience: int = 10,
) -> EarlyStopState:
    next_state = EarlyStopState(state.reference, int(state.count), False, state.reason)
    if not np.isfinite(score):
        next_state.reason = "score_not_finite"
        return next_state
    if int(epoch) < int(formal_start_epoch):
        next_state.reason = "warmup_not_counted"
        return next_state
    if int(epoch) < int(count_start_epoch):
        if next_state.reference is None or float(score) > float(next_state.reference):
            next_state.reference = float(score)
            next_state.reason = "reference_window_update"
        else:
            next_state.reason = "reference_window_hold"
        return next_state
    if next_state.reference is None:
        next_state.reason = "no_reference"
        return next_state
    if float(score) > float(next_state.reference) + float(min_delta):
        next_state.reference = float(score)
        next_state.count = 0
        next_state.reason = "improved_reset"
        return next_state
    next_state.count += 1
    next_state.stopped = next_state.count >= int(patience)
    next_state.reason = "patience_exhausted" if next_state.stopped else "no_improve_increment"
    return next_state
