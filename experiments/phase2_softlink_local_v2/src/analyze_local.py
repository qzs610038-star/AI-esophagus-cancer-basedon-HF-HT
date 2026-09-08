"""Local-only derived metrics, epoch comparisons, fixed smoothing, and Fable checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from config import load_config, package_root
from data import load_normalization_from_config
from run_io import write_json


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    return None if denom == 0 else float(np.sum(xc * yc) / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _ccc(x: np.ndarray, y: np.ndarray):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    vx, vy = float(np.var(x)), float(np.var(y))
    denom = vx + vy + float((x.mean() - y.mean()) ** 2)
    if denom == 0:
        return 1.0 if np.array_equal(x, y) else None
    return float(2.0 * np.mean((x - x.mean()) * (y - y.mean())) / denom)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def compute_metric_bundle(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    patient_ids: Sequence[str],
    pathway_names: Sequence[str],
    top_k: int | None = None,
    residual_moran: dict | None = None,
) -> dict:
    pred = np.asarray(pred_z, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("pred_z/target_z 必须是同形状有限二维数组")
    mean_v, std_v = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    if mean_v.size != pred.shape[1] or std_v.size != pred.shape[1] or len(pathway_names) != pred.shape[1]:
        raise ValueError("标准化参数或通路名维数不一致")
    pred_raw = pred * std_v + mean_v
    target_raw = target * std_v + mean_v
    residual = pred - target
    pathway_pcc = [_pearson(pred[:, j], target[:, j]) for j in range(pred.shape[1])]
    valid_pcc = [value for value in pathway_pcc if value is not None]
    ss_res = float(np.sum((pred_raw - target_raw) ** 2))
    ss_tot = float(np.sum((target_raw - target_raw.mean()) ** 2))
    if top_k is None:
        top_k_result = {"status": "not_applicable", "value": None, "reason": "top_k_not_predefined"}
    else:
        k = int(top_k)
        overlaps = []
        for pred_row, target_row in zip(pred, target):
            p = set(np.argsort(pred_row)[-k:].tolist())
            t = set(np.argsort(target_row)[-k:].tolist())
            overlaps.append(len(p & t) / k)
        top_k_result = {"status": "computed", "k": k, "value": float(np.mean(overlaps))}
    per_patient = {}
    patient_array = np.asarray(patient_ids, dtype=object)
    for patient in sorted({str(value) for value in patient_array.tolist()}):
        mask = patient_array == patient
        patient_pcc = [_pearson(pred[mask, j], target[mask, j]) for j in range(pred.shape[1])]
        valid = [value for value in patient_pcc if value is not None]
        per_patient[patient] = {
            "n_points": int(mask.sum()),
            "valid_pathways": len(valid),
            "mean_pathway_pcc": float(np.mean(valid)) if valid else None,
            "z_rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
        }
    return {
        "pooled_pathway_pcc": {
            "mean": float(np.mean(valid_pcc)) if valid_pcc else None,
            "valid_pathways": len(valid_pcc),
            "per_pathway": dict(zip(pathway_names, pathway_pcc)),
        },
        "flattened_pcc": _pearson(pred.ravel(), target.ravel()),
        "z_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "z_mae": float(np.mean(np.abs(residual))),
        "z_mse": float(np.mean(residual ** 2)),
        "raw_mae": float(np.mean(np.abs(pred_raw - target_raw))),
        "raw_r2": None if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        "ccc": _ccc(pred.ravel(), target.ravel()),
        "spearman": _pearson(_rank(pred.ravel()), _rank(target.ravel())),
        "residual_morans_i": residual_moran or {"status": "not_applicable", "value": None, "reason": "graph_edges_not_supplied"},
        "masked_ssim": {"status": "not_applicable", "value": None, "reason": "window_and_mask_not_predefined"},
        "train_sd_nrmse": {
            "mean": float(np.mean(np.sqrt(np.mean((pred_raw - target_raw) ** 2, axis=0)) / std_v)),
            "per_pathway": dict(zip(pathway_names, (np.sqrt(np.mean((pred_raw - target_raw) ** 2, axis=0)) / std_v).tolist())),
        },
        "top_k_overlap": top_k_result,
        "absolute_z_error_distribution": _summary(np.abs(residual).ravel()),
        "per_patient": per_patient,
        "inverse_transform_count": 1,
    }


def fixed_graph_smoothing(
    pred_z: np.ndarray,
    identities: Sequence[tuple[str, str, str]],
    edge_rows: Sequence[dict],
    *,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if float(beta) != 1.0:
        raise ValueError("v2.1 固定平滑只允许 beta=1，不拟合")
    pred = np.asarray(pred_z, dtype=np.float64)
    lookup = {tuple(map(str, key)): i for i, key in enumerate(identities)}
    smooth = np.zeros_like(pred)
    degree = np.zeros(len(pred), dtype=np.int64)
    seen = set()
    for row in edge_rows:
        center_key = (str(row["patient_id"]), str(row["slide_id"]), str(row["spot_id"]))
        if center_key not in lookup:
            raise ValueError(f"图中心身份不在预测中: {center_key}")
        center = lookup[center_key]
        if center_key not in seen:
            smooth[center] = float(row["a_self"]) * pred[center]
            degree[center] = int(row["degree"])
            seen.add(center_key)
        neighbor_spot = str(row.get("neighbor_spot_id") or "")
        if neighbor_spot:
            neighbor_key = (
                str(row["neighbor_patient_id"]),
                str(row["neighbor_slide_id"]),
                neighbor_spot,
            )
            if neighbor_key not in lookup:
                raise ValueError(f"图邻居身份不在预测中: {neighbor_key}")
            smooth[center] += float(row["a"]) * pred[lookup[neighbor_key]]
    for key, index in lookup.items():
        if key not in seen:
            smooth[index] = pred[index]
    return smooth, degree


def residual_morans_i(residual: np.ndarray, identities, edge_rows) -> dict:
    residual = np.asarray(residual, dtype=np.float64)
    lookup = {tuple(map(str, key)): i for i, key in enumerate(identities)}
    weights = []
    for row in edge_rows:
        if not str(row.get("neighbor_spot_id") or ""):
            continue
        i = lookup[(str(row["patient_id"]), str(row["slide_id"]), str(row["spot_id"]))]
        j = lookup[(str(row["neighbor_patient_id"]), str(row["neighbor_slide_id"]), str(row["neighbor_spot_id"]))]
        weights.append((i, j, float(row["a"])))
    s0 = sum(w for _, _, w in weights)
    values = []
    if s0 > 0:
        for column in range(residual.shape[1]):
            centered = residual[:, column] - residual[:, column].mean()
            denom = float(np.sum(centered * centered))
            if denom > 0:
                numerator = sum(w * centered[i] * centered[j] for i, j, w in weights)
                values.append(len(residual) / s0 * numerator / denom)
    return {
        "status": "computed" if values else "not_applicable",
        "value": float(np.mean(values)) if values else None,
        "valid_pathways": len(values),
        "directed_weighted": True,
    }


def degree_strata_summary(data: dict, smooth: np.ndarray, degree: np.ndarray, config: dict, normalization) -> dict:
    output = {}
    patients = data["patient_id"].astype(str)
    for item in config["analysis"]["degree_strata"]:
        minimum = int(item["minimum"])
        maximum = item.get("maximum")
        mask = degree >= minimum
        if maximum is not None:
            mask &= degree <= int(maximum)
        name = str(item["name"])
        if not mask.any():
            output[name] = {"status": "not_applicable", "n_points": 0, "n_patients": 0, "valid_pathways": 0}
            continue
        metrics = compute_metric_bundle(
            smooth[mask], data["target_z"][mask], mean=normalization.mean, std=normalization.std,
            patient_ids=patients[mask], pathway_names=normalization.pathway_names,
        )
        output[name] = {
            "status": "computed",
            "n_points": int(mask.sum()),
            "n_patients": len(set(patients[mask].tolist())),
            "valid_pathways": metrics["pooled_pathway_pcc"]["valid_pathways"],
            "metrics": metrics,
        }
    return output


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_edges(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_batch(config: dict, batch_dir: str | Path) -> dict:
    batch = Path(batch_dir)
    output = batch / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    normalization = load_normalization_from_config(config)
    results = {}
    same_epoch = {}
    for run_dir in sorted(batch.glob("train_*_*")):
        best = run_dir / "raw" / "internal_best.npz"
        if not best.is_file():
            continue
        data = _load_npz(best)
        identities = list(zip(data["patient_id"].astype(str), data["slide_id"].astype(str), data["spot_id"].astype(str)))
        edge_path = run_dir / "raw" / "graph_edges_internal_val.csv"
        edges = _load_edges(edge_path) if edge_path.is_file() else []
        moran = residual_morans_i(data["pred_z"] - data["target_z"], identities, edges) if edges else None
        results[run_dir.name] = compute_metric_bundle(
            data["pred_z"], data["target_z"],
            mean=normalization.mean, std=normalization.std,
            patient_ids=data["patient_id"].astype(str), pathway_names=normalization.pathway_names,
            residual_moran=moran,
        )
        for epoch in config["analysis"]["same_epoch_comparisons"]:
            epoch_path = run_dir / "raw" / "epoch_predictions" / f"epoch_{int(epoch)}.npz"
            key = f"{run_dir.name}:epoch_{int(epoch)}"
            if epoch_path.is_file():
                epoch_data = _load_npz(epoch_path)
                same_epoch[key] = {"status": "available", "metrics": compute_metric_bundle(
                    epoch_data["pred_z"], epoch_data["target_z"],
                    mean=normalization.mean, std=normalization.std,
                    patient_ids=epoch_data["patient_id"].astype(str), pathway_names=normalization.pathway_names,
                )}
            else:
                same_epoch[key] = {"status": "unavailable", "reason": "not_recorded_not_interpolated"}

    smoothing = {}
    beta = float(config["analysis"]["fixed_prediction_smoothing"]["beta"])
    for seed in config["training"]["seeds"]:
        point_path = batch / f"train_{seed}_point" / "raw" / "internal_best.npz"
        edge_path = batch / f"train_{seed}_spatial" / "raw" / "graph_edges_internal_val.csv"
        if not point_path.is_file() or not edge_path.is_file():
            smoothing[str(seed)] = {"status": "unavailable", "reason": "point_prediction_or_matching_graph_missing"}
            continue
        data = _load_npz(point_path)
        identities = list(zip(data["patient_id"].astype(str), data["slide_id"].astype(str), data["spot_id"].astype(str)))
        edges = _load_edges(edge_path)
        smooth, degree = fixed_graph_smoothing(data["pred_z"], identities, edges, beta=beta)
        smooth_path = output / f"fixed_smoothing_seed_{seed}.npz"
        np.savez(smooth_path, pred_z=smooth, target_z=data["target_z"], degree=degree,
                 patient_id=data["patient_id"], slide_id=data["slide_id"], spot_id=data["spot_id"], beta=np.asarray(beta))
        smooth_metrics = compute_metric_bundle(
            smooth, data["target_z"], mean=normalization.mean, std=normalization.std,
            patient_ids=data["patient_id"].astype(str), pathway_names=normalization.pathway_names,
            residual_moran=residual_morans_i(smooth - data["target_z"], identities, edges),
        )
        point_metrics = results.get(f"train_{seed}_point")
        spatial_metrics = results.get(f"train_{seed}_spatial")
        smoothing[str(seed)] = {
            "status": "computed", "beta": beta, "used_for_selection": False,
            "path": str(smooth_path),
            "metrics": smooth_metrics,
            "delta_smoothing_minus_point": None if point_metrics is None else {
                "pooled_pathway_pcc": smooth_metrics["pooled_pathway_pcc"]["mean"] - point_metrics["pooled_pathway_pcc"]["mean"],
                "z_rmse": smooth_metrics["z_rmse"] - point_metrics["z_rmse"],
            },
            "delta_spatial_minus_smoothing": None if spatial_metrics is None else {
                "pooled_pathway_pcc": spatial_metrics["pooled_pathway_pcc"]["mean"] - smooth_metrics["pooled_pathway_pcc"]["mean"],
                "z_rmse": spatial_metrics["z_rmse"] - smooth_metrics["z_rmse"],
            },
            "degree_counts": {str(value): int(np.sum(degree == value)) for value in np.unique(degree)},
            "degree_strata": degree_strata_summary(data, smooth, degree, config, normalization),
        }
    payload = {
        "status": "completed",
        "selection_unchanged": True,
        "external_used_for_selection": False,
        "metrics": results,
        "same_epoch": same_epoch,
        "fixed_smoothing": smoothing,
        "note_zh": "本地派生分析不覆盖raw；缺少预定义窗口/mask或Top-k时明确记为不适用。",
    }
    write_json(output / "analysis.json", payload)
    fable_lines = [
        "# Fable 训练后排查",
        "",
        "本文件仅汇总可审计材料，不把代码存在或指标计算等同于科研结论。",
        f"- 正式端点指标文件数: {len(results)}",
        f"- 第15/25轮记录项: {len(same_epoch)}（缺失项不插值）",
        f"- 固定β=1平滑完成种子数: {sum(item['status'] == 'computed' for item in smoothing.values())}",
        "- 关系梯度、图覆盖和预测增量应结合各run的raw/diagnostics与precheck_report.json逐项解释。",
    ]
    (output / "fable_review_zh.md").write_text("\n".join(fable_lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 v2.1 回传后本地分析")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, package_dir=package_root())
    result = analyze_batch(config, args.batch_dir)
    print(json.dumps({"status": result["status"], "analysis_dir": str(args.batch_dir / "analysis")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
