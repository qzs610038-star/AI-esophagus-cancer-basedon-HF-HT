from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARMS = ("FBR", "RCC", "HCR", "CPGCR")
LOWER_IS_BETTER = {"z-MSE", "z-MAE", "raw MAE"}
sys.stdout.reconfigure(encoding="utf-8")


def arm_dir(arm: str) -> Path:
    matches = list(
        (ROOT / "project_state" / "inbox" / "W007" / "quarantine" / arm).glob(
            f"automation/returns/W007/*/{arm}"
        )
    )
    return matches[0]


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - np.square(y_true - y_pred).sum() / np.square(y_true - y_true.mean()).sum()


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


registry = json.loads((ROOT / "experiments" / "experiment_registry.json").read_text(encoding="utf-8"))
experiment = next(item for item in registry["experiments"] if item["id"] == "mpp2_cpgcr_probe_v001_20260811")

metrics = {}
histories = {}
predictions = {}
for arm in ARMS:
    current_dir = arm_dir(arm)
    metrics[arm] = json.loads((current_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions[(arm, "internal")] = pd.read_csv(current_dir / "internal_val_predictions.csv")
    predictions[(arm, "external")] = pd.read_csv(current_dir / "XZY_predictions.csv")
    if arm != "FBR":
        histories[arm] = pd.read_csv(current_dir / "training_history.csv")

pathways = [name.removeprefix("truth_z__") for name in predictions[("FBR", "external")].columns if name.startswith("truth_z__")]

# 图1：外部五项指标的逐臂排名（1 为该指标数值最优）。
metric_map = {
    "z-MSE": "pooled_z_mse",
    "z-MAE": "z_mae",
    "PCC": "pooled_pcc",
    "raw MAE": "raw_mae",
    "mean pathway R²": "mean_pathway_raw_r2",
}
rank_rows = []
for label, key in metric_map.items():
    values = pd.Series({arm: metrics[arm]["external"][key] for arm in ARMS})
    ranks = values.rank(method="min", ascending=label in LOWER_IS_BETTER).astype(int)
    for arm in ARMS:
        rank_rows.append({"metric": label, "arm": arm, "rank": int(ranks[arm]), "value": float(values[arm])})

# 图2：internal-val checkpoint 指标随 epoch 变化，FBR 为冻结参考线。
max_epoch = max(int(frame["epoch"].max()) for frame in histories.values())
training_rows = []
for epoch in range(1, max_epoch + 1):
    row = {"epoch": epoch, "FBR": metrics["FBR"]["internal"]["patient_balanced_z_mse"]}
    for arm in ("RCC", "HCR", "CPGCR"):
        hit = histories[arm].loc[histories[arm]["epoch"] == epoch, "internal_val_patient_balanced_z_mse"]
        row[arm] = None if hit.empty else float(hit.iloc[0])
    training_rows.append(row)

best_epochs = {}
for arm, frame in histories.items():
    target = metrics[arm]["internal"]["patient_balanced_z_mse"]
    index = (frame["internal_val_patient_balanced_z_mse"] - target).abs().idxmin()
    best_epochs[arm] = int(frame.loc[index, "epoch"])

# 图3：XZY 的跨患者偏移与预测振幅压缩；同时核对 z/raw R² 仿射不变性。
fbr_external = predictions[("FBR", "external")]
domain_rows = []
r2_invariance_diffs = []
for pathway in pathways:
    truth_z = fbr_external[f"truth_z__{pathway}"].to_numpy(float)
    pred_z = fbr_external[f"prediction_z__{pathway}"].to_numpy(float)
    truth_raw = fbr_external[f"truth_raw__{pathway}"].to_numpy(float)
    pred_raw = fbr_external[f"prediction_raw__{pathway}"].to_numpy(float)
    amplitude_ratio = float(pred_z.std(ddof=1) / truth_z.std(ddof=1))
    slope = float(np.cov(truth_z, pred_z, ddof=1)[0, 1] / np.var(truth_z, ddof=1))
    z_r2 = r2(truth_z, pred_z)
    raw_r2 = r2(truth_raw, pred_raw)
    r2_invariance_diffs.append(abs(z_r2 - raw_r2))
    domain_rows.append(
        {
            "pathway": pathway,
            "abs_truth_mean_z": float(abs(truth_z.mean())),
            "amplitude_ratio": amplitude_ratio,
            "regression_slope": slope,
            "pcc": pcc(truth_z, pred_z),
            "z_r2": z_r2,
            "raw_r2": raw_r2,
        }
    )

domain = pd.DataFrame(domain_rows)
summary_bars = [
    {
        "diagnostic": "均值偏移 > 0.5z 的通路",
        "percent": float((domain["abs_truth_mean_z"] > 0.5).mean() * 100),
        "numerator": int((domain["abs_truth_mean_z"] > 0.5).sum()),
        "denominator": len(domain),
    },
    {
        "diagnostic": "预测/真实振幅比（中位）",
        "percent": float(domain["amplitude_ratio"].median() * 100),
        "numerator": None,
        "denominator": None,
    },
    {
        "diagnostic": "回归斜率（中位）",
        "percent": float(domain["regression_slope"].median() * 100),
        "numerator": None,
        "denominator": None,
    },
]

external_ranges = {}
for label, key in metric_map.items():
    values = np.array([metrics[arm]["external"][key] for arm in ARMS], dtype=float)
    external_ranges[label] = float(values.max() - values.min())

metric_recompute_diffs = []
for arm in ARMS:
    for cohort in ("internal", "external"):
        frame = predictions[(arm, cohort)]
        truth_z = frame[[f"truth_z__{name}" for name in pathways]].to_numpy(float)
        pred_z = frame[[f"prediction_z__{name}" for name in pathways]].to_numpy(float)
        truth_raw = frame[[f"truth_raw__{name}" for name in pathways]].to_numpy(float)
        pred_raw = frame[[f"prediction_raw__{name}" for name in pathways]].to_numpy(float)
        recomputed = {
            "patient_balanced_z_mse": float(
                np.mean(
                    [np.square(truth_z[index] - pred_z[index]).mean() for _, index in frame.groupby("patient").groups.items()]
                )
            ),
            "pooled_z_mse": float(np.square(truth_z - pred_z).mean()),
            "z_mae": float(np.abs(truth_z - pred_z).mean()),
            "pooled_pcc": pcc(truth_z.ravel(), pred_z.ravel()),
            "raw_mae": float(np.abs(truth_raw - pred_raw).mean()),
            "mean_pathway_raw_r2": float(np.mean([r2(truth_raw[:, i], pred_raw[:, i]) for i in range(len(pathways))])),
        }
        metric_recompute_diffs.extend(abs(recomputed[key] - metrics[arm][cohort][key]) for key in recomputed)

analysis = {
    "experiment_id": experiment["id"],
    "evidence_status": experiment["evidence_status"],
    "analysis_status": experiment["analysis_status"],
    "seed": 42,
    "sample_counts": {
        "internal_spots": len(predictions[("FBR", "internal")]),
        "internal_patients": int(predictions[("FBR", "internal")]["patient"].nunique()),
        "external_spots": len(fbr_external),
        "external_patients": int(fbr_external["patient"].nunique()),
        "pathways": len(pathways),
    },
    "metrics": metrics,
    "external_metric_ranges": external_ranges,
    "max_abs_metric_recompute_difference": float(max(metric_recompute_diffs)),
    "best_epochs": best_epochs,
    "external_prediction_correlation_with_fbr": {
        arm: pcc(
            fbr_external[[f"prediction_z__{name}" for name in pathways]].to_numpy().ravel(),
            predictions[(arm, "external")][[f"prediction_z__{name}" for name in pathways]].to_numpy().ravel(),
        )
        for arm in ("RCC", "HCR", "CPGCR")
    },
    "external_mean_abs_prediction_change_from_fbr_z": {
        arm: float(
            np.abs(
                predictions[(arm, "external")][[f"prediction_z__{name}" for name in pathways]].to_numpy()
                - fbr_external[[f"prediction_z__{name}" for name in pathways]].to_numpy()
            ).mean()
        )
        for arm in ("RCC", "HCR", "CPGCR")
    },
    "zscore_diagnostics": {
        "pathways_abs_external_mean_gt_0_5z": int((domain["abs_truth_mean_z"] > 0.5).sum()),
        "median_amplitude_ratio": float(domain["amplitude_ratio"].median()),
        "median_regression_slope": float(domain["regression_slope"].median()),
        "max_abs_z_vs_raw_pathway_r2_difference": float(max(r2_invariance_diffs)),
    },
    "chart_data": {
        "external_ranks": rank_rows,
        "training_history": training_rows,
        "domain_summary": summary_bars,
        "pathway_diagnostics": domain_rows,
    },
}

(OUT / "analysis_summary.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: analysis[key] for key in ("sample_counts", "external_metric_ranges", "best_epochs", "external_prediction_correlation_with_fbr", "external_mean_abs_prediction_change_from_fbr_z", "zscore_diagnostics")}, ensure_ascii=False, indent=2))
