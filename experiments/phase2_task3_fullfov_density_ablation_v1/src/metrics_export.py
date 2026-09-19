"""Task-3 evaluation metrics and result tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from selection import patient_macro_metrics


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    return float(np.sum(xc * yc) / denom) if denom else float("nan")


def evaluate_metrics(
    pred_dense_z: np.ndarray,
    true_dense_z: np.ndarray,
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    patient_ids: Sequence[str],
    pathway_names: Sequence[str],
) -> tuple[dict, pd.DataFrame]:
    pred_z, true_z = np.asarray(pred_dense_z, dtype=np.float64), np.asarray(true_dense_z, dtype=np.float64)
    pred_r, true_r = np.asarray(pred_raw, dtype=np.float64), np.asarray(true_raw, dtype=np.float64)
    selection = patient_macro_metrics(pred_z, true_z, patient_ids, pathway_names=pathway_names)
    rows = []
    patients = np.asarray(list(patient_ids), dtype=object)
    for patient in sorted(set(patients.tolist())):
        mask = patients == patient
        for j, pathway in enumerate(pathway_names):
            rows.append({
                "patient_id": patient,
                "pathway": pathway,
                "n_points": int(mask.sum()),
                "pcc": _pearson(pred_z[mask, j], true_z[mask, j]),
                "z_mse": float(np.mean((pred_z[mask, j] - true_z[mask, j]) ** 2)),
                "raw_mae": float(np.mean(np.abs(pred_r[mask, j] - true_r[mask, j]))),
            })
    details = pd.DataFrame(rows)
    metrics = {
        "patient_macro_pathway_pcc": float(selection.patient_macro_pathway_pcc_measured),
        "patient_macro_pathway_pcc_selection": float(selection.patient_macro_pathway_pcc),
        "pooled_flattened_pcc": _pearson(pred_z.reshape(-1), true_z.reshape(-1)),
        "z_mse": float(np.mean((pred_z - true_z) ** 2)),
        "raw_mae": float(np.mean(np.abs(pred_r - true_r))),
        "n_points": int(pred_z.shape[0]),
        "n_pathways": int(pred_z.shape[1]),
        "n_patients": int(len(set(patients.tolist()))),
        "aggregation_note": "patient_macro_pathway_pcc为患者—通路等权；pooled_flattened_pcc为全部点位×通路展平后一次相关。",
    }
    return metrics, details


def save_evaluation(
    output_dir: str | Path,
    split: str,
    table,
    pred_dense_z: np.ndarray,
    true_dense_z: np.ndarray,
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
) -> dict:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    patients = [identity.patient_id for identity in table.identities]
    metrics, details = evaluate_metrics(pred_dense_z, true_dense_z, pred_raw, true_raw, patients, table.pathway_names)
    details.to_csv(root / f"{split}_per_patient_pathway.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        root / f"{split}_true_and_prediction.npz",
        pred_z=np.asarray(pred_dense_z, dtype=np.float32),
        true_z=np.asarray(true_dense_z, dtype=np.float32),
        pred_raw=np.asarray(pred_raw, dtype=np.float32),
        true_raw=np.asarray(true_raw, dtype=np.float32),
        patient_id=np.asarray(patients, dtype=str),
        slide_id=np.asarray([identity.slide_id for identity in table.identities], dtype=str),
        patch_id=np.asarray([identity.spot_id for identity in table.identities], dtype=str),
        x=np.asarray(table.x), y=np.asarray(table.y), pathway_names=np.asarray(table.pathway_names, dtype=str),
    )
    (root / f"{split}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def summarize_runs(run_summaries: list[dict], output_dir: str | Path) -> dict:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in run_summaries:
        for selector, split_map in run["evaluations"].items():
            for split, metrics in split_map.items():
                rows.append({"arm": run["arm"], "seed": run["seed"], "selector": selector, "split": split, **{k: metrics[k] for k in ("patient_macro_pathway_pcc", "pooled_flattened_pcc", "z_mse", "raw_mae")}})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "all_run_metrics.csv", index=False, encoding="utf-8-sig")
    grouped = frame.groupby(["arm", "selector", "split"], as_index=False).agg({metric: ["mean", "std"] for metric in ("patient_macro_pathway_pcc", "pooled_flattened_pcc", "z_mse", "raw_mae")})
    grouped.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in grouped.columns]
    grouped.to_csv(root / "three_seed_mean_std.csv", index=False, encoding="utf-8-sig")
    dense = frame[frame.arm == "dense"].rename(columns={metric: f"dense_{metric}" for metric in ("patient_macro_pathway_pcc", "pooled_flattened_pcc", "z_mse", "raw_mae")})
    paired = frame.merge(dense[["seed", "selector", "split"] + [f"dense_{m}" for m in ("patient_macro_pathway_pcc", "pooled_flattened_pcc", "z_mse", "raw_mae")]], on=["seed", "selector", "split"], how="left")
    for metric in ("patient_macro_pathway_pcc", "pooled_flattened_pcc", "z_mse", "raw_mae"):
        paired[f"delta_vs_dense_{metric}"] = paired[metric] - paired[f"dense_{metric}"]
    paired.to_csv(root / "paired_differences_vs_dense.csv", index=False, encoding="utf-8-sig")
    summary = {"n_training_runs": len(run_summaries), "expected_training_runs": 9, "external_used_for_selection": False, "files": ["all_run_metrics.csv", "three_seed_mean_std.csv", "paired_differences_vs_dense.csv"]}
    (root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

