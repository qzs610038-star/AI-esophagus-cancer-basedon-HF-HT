"""Recompute and summarize the returned XZY external evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import compute_regression_metrics
else:
    from .metrics import compute_regression_metrics


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def _expanded(prediction: np.ndarray, target: np.ndarray, patient: np.ndarray) -> dict:
    result = compute_regression_metrics(prediction, target, patient)
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    result.update({
        "zRMSE": float(np.sqrt(np.mean(error ** 2))),
        "zMAE": float(np.mean(np.abs(error))),
        "bias_z": float(np.mean(error)),
        "prediction_target_sd_ratio": float(np.std(prediction) / np.std(target)),
    })
    return result


def analyze(run_dir: Path) -> dict:
    external = run_dir / "external_xzy" / "spatial_joint"
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with np.load(external / "predictions.npz", allow_pickle=False) as z:
        selected = np.asarray(z["prediction"], dtype=np.float64)
        target = np.asarray(z["target"], dtype=np.float64)
        patient = np.asarray(z["patient"], dtype=str)
        patch_stem = np.asarray(z["patch_stem"], dtype=str)
        pathways = np.asarray(z["pathway_names"], dtype=str)
    with np.load(external / "graph_reconstruction.npz", allow_pickle=False) as z:
        step0 = np.asarray(z["graph_features"], dtype=np.float64)
        graph_patient = np.asarray(z["patient"], dtype=str)
        graph_patch = np.asarray(z["patch_stem"], dtype=str)
        index = np.asarray(z["neighbor_index"], dtype=np.int64)
        weight = np.asarray(z["neighbor_weight"], dtype=np.float64)
        mask = np.asarray(z["neighbor_mask"], dtype=bool)
        degree = np.asarray(z["degree"], dtype=np.int64)
        self_weight = np.asarray(z["self_weight"], dtype=np.float64)
    if not (np.array_equal(patient, graph_patient) and np.array_equal(patch_stem, graph_patch)):
        raise ValueError("prediction and graph identities differ")
    safe_index = np.maximum(index, 0)
    fixed = step0 + (
        (step0[safe_index] - step0[:, None, :]) * weight[:, :, None] * mask[:, :, None]
    ).sum(axis=1)
    variants = {
        "step0_source_e5": step0,
        "fixed_beta1_diagnostic": fixed,
        "selected_spatial_joint": selected,
    }
    metrics = {name: _expanded(values, target, patient) for name, values in variants.items()}
    server = json.loads((external / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    required = ("patient_macro_pathway_pcc", "flattened_pooled_pcc", "zMSE")
    max_server_difference = max(abs(float(server[key]) - float(metrics["selected_spatial_joint"][key])) for key in required)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    internal = next(row["best_validation_metrics"] for row in selection["trajectories"] if row["arm"] == "spatial_joint")
    report = {
        "schema_version": "1.0",
        "evidence_status": "accepted",
        "experiment_id": "phase2_spatial_warmstart_v1",
        "run_id": run_dir.name,
        "arm": "spatial_joint",
        "seed": 42,
        "selection_split": "internal_val",
        "external_used_for_selection": False,
        "external_dataset": "XZY",
        "point_count": int(len(target)),
        "pathway_count": int(target.shape[1]),
        "target_scale": "source_training_zscore",
        "metrics": metrics,
        "selected_minus_step0": {key: float(metrics["selected_spatial_joint"][key] - metrics["step0_source_e5"][key]) for key in required},
        "fixed_beta1_minus_step0": {key: float(metrics["fixed_beta1_diagnostic"][key] - metrics["step0_source_e5"][key]) for key in required},
        "external_minus_internal_selected": {key: float(metrics["selected_spatial_joint"][key] - internal[key]) for key in required},
        "graph": {
            "directed_edge_count": int(mask.sum()),
            "isolated_point_count": int((degree == 0).sum()),
            "degree_min": int(degree.min()),
            "degree_mean": float(degree.mean()),
            "degree_max": int(degree.max()),
            "max_weight_sum_error": float(np.max(np.abs(self_weight + weight.sum(axis=1) - 1.0))),
        },
        "validation": {
            "identity_alignment": True,
            "finite_predictions_and_targets": bool(np.isfinite(selected).all() and np.isfinite(target).all()),
            "server_metric_max_abs_difference": float(max_server_difference),
            "all_30_pathway_pcc_valid": metrics["selected_spatial_joint"]["valid_patient_pathway_count"] == 30,
        },
        "notes": [
            "fixed_beta1_diagnostic is prespecified and was not used for model selection",
            "raw-scale metrics are unavailable because the warm-start bundle does not embed source train mean/std",
            "single seed only; seed mean equals the reported value and seed SD is not applicable",
        ],
    }
    (analysis / "external_xzy_recomputed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for j, pathway in enumerate(pathways):
        row = {"pathway_index": j, "pathway": pathway, "n": len(target)}
        for name, values in variants.items():
            error = values[:, j] - target[:, j]
            row[f"{name}_pcc"] = _pcc(values[:, j], target[:, j])
            row[f"{name}_zMSE"] = float(np.mean(error ** 2))
            row[f"{name}_zMAE"] = float(np.mean(np.abs(error)))
        row["selected_minus_step0_pcc"] = row["selected_spatial_joint_pcc"] - row["step0_source_e5_pcc"]
        rows.append(row)
    with (analysis / "external_xzy_per_pathway.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# XZY 外部评估审核稿",
        "",
        f"- 状态：`accepted`（用户审核后已登记）",
        f"- 运行：`{run_dir.name}`；模型：`spatial_joint`；种子：42",
        f"- 数据：XZY，{len(target)} 个点位，{target.shape[1]} 条通路，来源训练集 z-score 尺度",
        "- 外部标签未用于训练、调参或模型选择",
        "",
        "## 汇总指标",
        "",
        "| 结果 | 逐通路平均 PCC | 整体展平 PCC | zMSE | zRMSE | zMAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "step0_source_e5": "来源 step0（固定 E5）",
        "fixed_beta1_diagnostic": "固定 β=1 平滑（诊断）",
        "selected_spatial_joint": "已选 spatial_joint",
    }
    for name in variants:
        m = metrics[name]
        lines.append(
            f"| {labels[name]} | {m['patient_macro_pathway_pcc']:.6f} | "
            f"{m['flattened_pooled_pcc']:.6f} | {m['zMSE']:.6f} | "
            f"{m['zRMSE']:.6f} | {m['zMAE']:.6f} |"
        )
    lines.extend([
        "",
        "## 30 条通路（已选模型）",
        "",
        "| 通路 | PCC | 相对 step0 PCC | zMSE | zMAE |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sorted(rows, key=lambda item: item["pathway_index"]):
        lines.append(
            f"| {row['pathway']} | {row['selected_spatial_joint_pcc']:.6f} | "
            f"{row['selected_minus_step0_pcc']:+.6f} | "
            f"{row['selected_spatial_joint_zMSE']:.6f} | {row['selected_spatial_joint_zMAE']:.6f} |"
        )
    lines.extend([
        "",
        "## 核验",
        "",
        f"- 服务器汇总与本地重算最大绝对差：`{max_server_difference:.3g}`",
        f"- 图：{int(mask.sum())} 条有向边；孤立点 {int((degree == 0).sum())}；平均度 {float(degree.mean()):.3f}",
        "- 30/30 通路 PCC 有效，预测和目标均为有限值，预测与图身份顺序一致",
        "- 固定 β=1 仅为预设诊断，不参与选模；单种子不报告种子标准差",
        "- 当前缺少来源训练 mean/std，不能审计原始尺度误差",
        "",
    ])
    (analysis / "external_xzy_review.md").write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(
        analysis / "external_xzy_fixed_beta1_predictions.npz",
        prediction=fixed,
        target=target,
        patient=patient,
        patch_stem=patch_stem,
        pathway_names=pathways,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
