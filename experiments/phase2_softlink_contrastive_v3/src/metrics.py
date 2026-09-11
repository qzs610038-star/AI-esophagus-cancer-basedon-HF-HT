"""Metrics required by the v4 Phase2 protocol (all calculations use float64)."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def _inputs(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, y = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2:
        raise ValueError("pred 和 target 必须是同形状 [points, pathways] 数组")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("指标输入含非有限数")
    return p, y


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    den = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    return float(np.sum(dx * dy) / den) if den > 0 else float("nan")


def concordance_correlation_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    x, y = np.asarray(pred, dtype=np.float64).ravel(), np.asarray(target, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    vx, vy = np.var(x), np.var(y)
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (vx + vy + (x.mean() - y.mean()) ** 2)) if vx + vy + (x.mean() - y.mean()) ** 2 else float("nan")


def patient_macro_pathway_pcc(pred: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]) -> float:
    p, y = _inputs(pred, target)
    ids = np.asarray(patient_ids, dtype=object)
    if ids.size != p.shape[0]:
        raise ValueError("patient_ids 长度不匹配")
    values = []
    for patient in sorted(set(ids.astype(str))):
        mask = ids.astype(str) == patient
        for pathway in range(p.shape[1]):
            score = _pcc(p[mask, pathway], y[mask, pathway])
            if np.isfinite(score):
                values.append(score)
    return float(np.mean(values)) if values else float("nan")


def pooled_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    p, y = _inputs(pred, target)
    return _pcc(p.ravel(), y.ravel())


def mse_decomposition(pred: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]) -> dict[str, float]:
    p, y = _inputs(pred, target)
    ids = np.asarray(patient_ids, dtype=object).astype(str)
    if len(ids) != len(p):
        raise ValueError("patient_ids 长度不匹配")
    centered, mean_bias, totals = [], [], []
    for patient in sorted(set(ids)):
        mask = ids == patient
        for pathway in range(p.shape[1]):
            error = p[mask, pathway] - y[mask, pathway]
            centered.append(float(np.mean((error - error.mean()) ** 2)))
            mean_bias.append(float(error.mean() ** 2))
            totals.append(float(np.mean(error ** 2)))
    total = float(np.mean(totals))
    return {"demeaned_mse": float(np.mean(centered)), "mean_bias_squared": float(np.mean(mean_bias)), "total_mse": total, "mse": total}


def per_patient_pathway_demeaned_metrics(pred: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]) -> list[dict]:
    p, y = _inputs(pred, target)
    ids = np.asarray(patient_ids, dtype=object).astype(str)
    rows = []
    for patient in sorted(set(ids)):
        mask = ids == patient
        for pathway in range(p.shape[1]):
            a, b = p[mask, pathway], y[mask, pathway]
            a, b = a-a.mean(), b-b.mean()
            rows.append({"patient": patient, "pathway": pathway, "demeaned_rmse": float(np.sqrt(np.mean((a-b)**2))), "demeaned_ccc": concordance_correlation_coefficient(a,b)})
    return rows


def per_patient_pathway_metrics(pred: np.ndarray, target: np.ndarray, patient_ids: Sequence[str]) -> list[dict]:
    """Unaggregated patient/pathway rows retained for later paired analyses."""
    p, y = _inputs(pred, target)
    ids = np.asarray(patient_ids, dtype=object).astype(str)
    rows = []
    for patient in sorted(set(ids)):
        mask = ids == patient
        for pathway in range(p.shape[1]):
            a, b = p[mask, pathway], y[mask, pathway]
            error = a - b
            rows.append({
                "patient": patient, "pathway": pathway, "n": int(mask.sum()),
                "pcc": _pcc(a, b), "ccc": concordance_correlation_coefficient(a, b),
                "zRMSE": float(np.sqrt(np.mean(error ** 2))),
                "zMAE": float(np.mean(np.abs(error))), "bias_z": float(error.mean()),
                "prediction_target_sd_ratio": float(np.std(a) / np.std(b)) if np.std(b) else float("nan"),
            })
    return rows


def spatial_residual_metrics(pred: np.ndarray, target: np.ndarray, graph) -> dict[str, float | int]:
    """Weighted one-hop residual agreement on an already fixed graph."""
    p,y=_inputs(pred,target); residual=p-y
    pairs_i=[]; pairs_j=[]; weights=[]
    for i in range(len(residual)):
        for slot in range(graph.neighbor_index.shape[1]):
            if bool(graph.neighbor_mask[i,slot]):
                pairs_i.append(i); pairs_j.append(int(graph.neighbor_index[i,slot])); weights.append(float(graph.neighbor_weight[i,slot]))
    if not weights:
        return {"edge_count":0,"weighted_neighbor_residual_correlation":float("nan"),"weighted_residual_semivariance":float("nan")}
    w=np.asarray(weights,dtype=np.float64); a=residual[np.asarray(pairs_i)].reshape(len(w),-1); b=residual[np.asarray(pairs_j)].reshape(len(w),-1)
    repeated=np.repeat(w,a.shape[1]); av=a.ravel(); bv=b.ravel(); total=float(repeated.sum())
    mean_a=float(np.sum(repeated*av)/total); mean_b=float(np.sum(repeated*bv)/total)
    covariance=float(np.sum(repeated*(av-mean_a)*(bv-mean_b))/total)
    variance_a=float(np.sum(repeated*(av-mean_a)**2)/total); variance_b=float(np.sum(repeated*(bv-mean_b)**2)/total)
    denominator=np.sqrt(variance_a*variance_b)
    return {"edge_count":len(w),"weighted_neighbor_residual_correlation":covariance/denominator if denominator else float("nan"),"weighted_residual_semivariance":float(np.sum(repeated*(av-bv)**2)/(2*total))}


def regression_metrics(pred: np.ndarray, target: np.ndarray, patient_ids: Sequence[str], *, raw_pred: np.ndarray | None = None, raw_target: np.ndarray | None = None) -> dict:
    p, y = _inputs(pred, target)
    err = p-y
    mse = float(np.mean(err**2))
    target_var = float(np.sum((y-y.mean())**2))
    result = {
        "patient_macro_pathway_pcc": patient_macro_pathway_pcc(p,y,patient_ids),
        "pooled_pcc": pooled_pcc(p,y), "zMSE": mse, "zRMSE": float(np.sqrt(mse)),
        "zMAE": float(np.mean(np.abs(err))), "rawMAE": None, "CCC": concordance_correlation_coefficient(p,y),
        "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err**2)/target_var) if target_var else float("nan"),
        "bias": float(err.mean()), "bias_z": float(err.mean()),
        "prediction_target_sd_ratio": float(np.std(p)/np.std(y)) if np.std(y) else float("nan"),
    }
    if (raw_pred is None) != (raw_target is None):
        raise ValueError("raw_pred 与 raw_target 必须同时提供")
    if raw_pred is not None:
        raw_p, raw_y = _inputs(raw_pred, raw_target)
        result["rawMAE"] = float(np.mean(np.abs(raw_p - raw_y)))
    result.update(mse_decomposition(p,y,patient_ids))
    result["per_patient_pathway"] = per_patient_pathway_metrics(p,y,patient_ids)
    demeaned = per_patient_pathway_demeaned_metrics(p,y,patient_ids)
    result["per_patient_pathway_demeaned"] = demeaned
    result["demeaned_zRMSE"] = float(np.mean([row["demeaned_rmse"] for row in demeaned])) if demeaned else float("nan")
    finite_ccc = [row["demeaned_ccc"] for row in demeaned if np.isfinite(row["demeaned_ccc"])]
    result["demeaned_CCC"] = float(np.mean(finite_ccc)) if finite_ccc else float("nan")
    return result


# Convenient aliases for callers that use the wording from the plan.
compute_metrics = regression_metrics
z_mse = lambda pred, target: float(np.mean((_inputs(pred,target)[0]-_inputs(pred,target)[1])**2))
