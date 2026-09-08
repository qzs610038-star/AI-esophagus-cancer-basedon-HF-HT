"""Checkpoint selection, early stopping, lambda schedule, and patient-macro metrics.

All selection arithmetic is float64. Measured PCC never stores the -1 penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cmp_to_key
from typing import Sequence

import numpy as np

from errors import NonFiniteDataError

ARM_ORDER = ("point", "relation", "spatial", "joint")
INFERENCE_PARAMETER_COUNT = {
    "point": 401182,
    "relation": 401182,
    "spatial": 408862,
    "joint": 408862,
}


def relation_lambda(
    epoch: int,
    *,
    warmup_epochs: int = 5,
    ramp_epochs: int = 10,
    lambda_max: float = 0.05,
) -> float:
    epoch = int(epoch)
    if epoch < 1:
        raise ValueError("epoch 从 1 开始")
    if epoch <= int(warmup_epochs):
        return 0.0
    step = epoch - int(warmup_epochs)
    if step >= int(ramp_epochs):
        return float(lambda_max)
    return float(lambda_max) * float(step) / float(ramp_epochs)


def lambda_sequence(
    max_epoch: int,
    *,
    warmup_epochs: int = 5,
    ramp_epochs: int = 10,
    lambda_max: float = 0.05,
) -> list[float]:
    return [
        relation_lambda(
            epoch,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            lambda_max=lambda_max,
        )
        for epoch in range(1, int(max_epoch) + 1)
    ]


def _is_constant(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    return bool(np.max(values) == np.min(values))


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    pred_c = pred - pred.mean()
    tgt_c = target - target.mean()
    denom = float(np.sqrt(np.sum(pred_c * pred_c) * np.sum(tgt_c * tgt_c)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(pred_c * tgt_c) / denom)


def pooled_z_mse(pred_z: np.ndarray, target_z: np.ndarray) -> float:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致: {pred.shape} vs {target.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise NonFiniteDataError("pooled z-MSE 遇到非有限预测或目标")
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
        payload = asdict(self)
        return payload


def patient_macro_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    patient_ids: Sequence[str],
    *,
    pathway_names: Sequence[str] | None = None,
    constant_prediction_selection_penalty: float = -1.0,
    require_patient_count: int | None = None,
) -> SelectionMetrics:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    patients = np.asarray(list(patient_ids), dtype=object)
    if pred.shape != target.shape:
        raise ValueError("pred_z 与 target_z 形状必须一致")
    if pred.ndim != 2:
        raise ValueError("pred_z 必须是 [N, n_pathways]")
    if pred.shape[0] != patients.shape[0]:
        raise ValueError("身份数量必须与预测行数一致")
    if not np.isfinite(target).all():
        raise NonFiniteDataError("目标含非有限值")
    if not np.isfinite(pred).all():
        raise NonFiniteDataError("预测含非有限值，记为运行故障，不填假数")

    n_pathways = int(pred.shape[1])
    names = list(pathway_names) if pathway_names is not None else [f"p{i}" for i in range(n_pathways)]
    if len(names) != n_pathways:
        raise ValueError("pathway_names 长度必须等于通路数")

    unique_patients = sorted({str(p) for p in patients.tolist()})
    if require_patient_count is not None and len(unique_patients) != int(require_patient_count):
        raise NonFiniteDataError(
            f"选择分数需要 {require_patient_count} 名患者等权，当前={len(unique_patients)}"
        )

    penalty = float(constant_prediction_selection_penalty)
    exclusions: list[dict] = []
    per_patient: dict[str, dict] = {}
    patient_sel_pcc: list[float] = []
    patient_meas_pcc: list[float] = []
    patient_mse: list[float] = []
    n_valid_pp = 0
    n_excl = 0
    n_penalized = 0

    for pid in unique_patients:
        mask = patients == pid
        rec = {
            "n_points": int(np.sum(mask)),
            "pathways": {},
            "n_valid_pathways": 0,
            "selection_pcc_mean": float("nan"),
            "measured_pcc_mean": float("nan"),
            "z_mse_mean": float("nan"),
        }
        sel_vals: list[float] = []
        meas_vals: list[float] = []
        mse_vals: list[float] = []
        for p_i, name in enumerate(names):
            y = target[mask, p_i]
            yhat = pred[mask, p_i]
            item = {"pathway": name, "n": int(y.size)}
            if y.size < 2:
                item["excluded"] = True
                item["reason"] = "target_n_lt_2"
                rec["pathways"][name] = item
                exclusions.append({"patient_id": pid, "pathway": name, "reason": "target_n_lt_2", "n": int(y.size)})
                n_excl += 1
                continue
            if _is_constant(y):
                item["excluded"] = True
                item["reason"] = "target_variance_0"
                rec["pathways"][name] = item
                exclusions.append(
                    {"patient_id": pid, "pathway": name, "reason": "target_variance_0", "n": int(y.size)}
                )
                n_excl += 1
                continue
            measured = _pearson(yhat, y)
            if _is_constant(yhat) or not np.isfinite(measured):
                measured_out = float("nan")
                selected = penalty
                n_penalized += 1
                item["constant_prediction"] = True
            else:
                measured_out = float(measured)
                selected = float(measured)
                item["constant_prediction"] = False
            mse = float(np.mean((yhat - y) ** 2))
            item.update(
                {
                    "excluded": False,
                    "measured_pcc": measured_out,
                    "selection_pcc": selected,
                    "z_mse": mse,
                }
            )
            rec["pathways"][name] = item
            sel_vals.append(selected)
            meas_vals.append(measured_out)
            mse_vals.append(mse)
            n_valid_pp += 1
        rec["n_valid_pathways"] = len(sel_vals)
        if not sel_vals:
            rec["incomputable"] = True
            rec["incomputable_reason"] = "no_valid_patient_pathway"
            per_patient[pid] = rec
            return SelectionMetrics(
                patient_macro_pathway_pcc=float("nan"),
                patient_macro_pathway_pcc_measured=float("nan"),
                patient_macro_z_mse_selection=float("nan"),
                pooled_z_mse=pooled_z_mse(pred, target),
                n_patients=len(unique_patients),
                n_valid_patient_pathway=n_valid_pp,
                n_excluded_patient_pathway=n_excl,
                n_constant_prediction_penalized=n_penalized,
                exclusions=exclusions,
                per_patient={**{k: per_patient[k] for k in per_patient}, pid: rec},
                computable=False,
                incomputable_reason=f"patient={pid} 没有任何有效目标，不能静默删患者继续等权排名",
            )
        rec["selection_pcc_mean"] = float(np.mean(sel_vals))
        finite_meas = [v for v in meas_vals if np.isfinite(v)]
        rec["measured_pcc_mean"] = float(np.mean(finite_meas)) if finite_meas else float("nan")
        rec["z_mse_mean"] = float(np.mean(mse_vals))
        per_patient[pid] = rec
        patient_sel_pcc.append(rec["selection_pcc_mean"])
        patient_meas_pcc.append(rec["measured_pcc_mean"])
        patient_mse.append(rec["z_mse_mean"])

    return SelectionMetrics(
        patient_macro_pathway_pcc=float(np.mean(patient_sel_pcc)),
        patient_macro_pathway_pcc_measured=float(np.nanmean(patient_meas_pcc)),
        patient_macro_z_mse_selection=float(np.mean(patient_mse)),
        pooled_z_mse=pooled_z_mse(pred, target),
        n_patients=len(unique_patients),
        n_valid_patient_pathway=n_valid_pp,
        n_excluded_patient_pathway=n_excl,
        n_constant_prediction_penalized=n_penalized,
        exclusions=exclusions,
        per_patient=per_patient,
        computable=True,
        incomputable_reason=None,
    )


@dataclass
class CheckpointChoice:
    epoch: int | None = None
    score: float | None = None
    mse: float | None = None
    reason: str | None = None
    lambda_value: float | None = None
    kind: str = "formal"

    def as_dict(self) -> dict:
        return asdict(self)


def update_checkpoint_choice(
    current: CheckpointChoice,
    *,
    epoch: int,
    score: float,
    mse: float,
    lambda_value: float,
    tolerance: float = 1e-6,
    kind: str = "formal",
) -> CheckpointChoice:
    if not np.isfinite(score) or not np.isfinite(mse):
        return current
    if current.epoch is None:
        return CheckpointChoice(
            epoch=int(epoch),
            score=float(score),
            mse=float(mse),
            reason="first_eligible",
            lambda_value=float(lambda_value),
            kind=kind,
        )
    delta = float(score) - float(current.score)
    if delta > float(tolerance):
        return CheckpointChoice(
            epoch=int(epoch),
            score=float(score),
            mse=float(mse),
            reason="higher_pcc",
            lambda_value=float(lambda_value),
            kind=kind,
        )
    if abs(delta) <= float(tolerance):
        if float(mse) < float(current.mse):
            return CheckpointChoice(
                epoch=int(epoch),
                score=float(score),
                mse=float(mse),
                reason="pcc_tie_lower_patient_macro_z_mse",
                lambda_value=float(lambda_value),
                kind=kind,
            )
        return current
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
    """Reference is max selection PCC over epochs [formal_start, count_start). Counting starts at count_start."""
    next_state = EarlyStopState(
        reference=state.reference,
        count=int(state.count),
        stopped=False,
        reason=state.reason,
    )
    if not np.isfinite(score):
        next_state.reason = "score_not_finite"
        return next_state
    if epoch < int(formal_start_epoch):
        next_state.reason = "warmup_not_counted"
        return next_state
    if epoch < int(count_start_epoch):
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
    next_state.count = int(next_state.count) + 1
    if next_state.count >= int(patience):
        next_state.stopped = True
        next_state.reason = "patience_exhausted"
    else:
        next_state.reason = "no_improve_increment"
    return next_state


def history_epoch_status(recorded_epochs: Sequence[int], targets: Sequence[int] = (15, 25)) -> dict[int, dict]:
    recorded = {int(e) for e in recorded_epochs}
    out: dict[int, dict] = {}
    for epoch in targets:
        if int(epoch) in recorded:
            out[int(epoch)] = {"available": True, "reason": "recorded_in_history"}
        else:
            out[int(epoch)] = {
                "available": False,
                "reason": "unavailable_not_interpolated",
            }
    return out


def recommend_arm(per_arm: dict[str, dict], *, pcc_tolerance: float = 1e-6) -> dict:
    """Rank arms by 3-seed mean selection PCC, then mean patient-macro z-MSE, then fewer inference params, then ARM_ORDER."""
    ranked = []
    for arm in ARM_ORDER:
        if arm not in per_arm:
            continue
        stats = per_arm[arm]
        ranked.append(
            {
                "arm": arm,
                "mean_patient_macro_pathway_pcc": float(stats["mean_patient_macro_pathway_pcc"]),
                "mean_patient_macro_z_mse_selection": float(stats["mean_patient_macro_z_mse_selection"]),
                "inference_parameters": int(
                    stats.get("inference_parameters") or INFERENCE_PARAMETER_COUNT[arm]
                ),
                "order_index": ARM_ORDER.index(arm),
            }
        )
    if not ranked:
        raise ValueError("没有可排序的臂")

    def compare(left: dict, right: dict) -> int:
        pcc_delta = float(left["mean_patient_macro_pathway_pcc"]) - float(right["mean_patient_macro_pathway_pcc"])
        if abs(pcc_delta) > float(pcc_tolerance):
            return -1 if pcc_delta > 0 else 1
        mse_delta = float(left["mean_patient_macro_z_mse_selection"]) - float(right["mean_patient_macro_z_mse_selection"])
        if mse_delta != 0.0:
            return -1 if mse_delta < 0 else 1
        param_delta = int(left["inference_parameters"]) - int(right["inference_parameters"])
        if param_delta != 0:
            return -1 if param_delta < 0 else 1
        return int(left["order_index"]) - int(right["order_index"])

    ranked.sort(key=cmp_to_key(compare))
    return {
        "recommended_arm": ranked[0]["arm"],
        "ranking": ranked,
        "rule": "mean_patient_macro_pathway_pcc -> mean_patient_macro_z_mse_selection -> fewer_inference_params -> point/relation/spatial/joint",
        "used_for_accepted_conclusion": False,
    }
