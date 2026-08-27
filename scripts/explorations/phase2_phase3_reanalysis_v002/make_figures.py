"""Create paper-facing exploratory figures from the versioned RER6 tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = ["FBR", "RCC", "HCR", "CPGCR"]
ARM_COLORS = {
    "FBR": "#4C78A8",
    "RCC": "#F58518",
    "HCR": "#54A24B",
    "CPGCR": "#B279A2",
}


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_w007_figure(table: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("pcc", "PCC", True),
        ("ccc", "CCC", True),
        ("spearman", "Spearman", True),
        ("z_rmse", "z-RMSE", False),
        ("raw_r2", "raw R²", True),
    ]
    splits = [("internal_val", "Internal: 6 patients, equal weight"), ("XZY", "External case: 1 patient")]
    figure, axes = plt.subplots(2, 5, figsize=(15, 6.5), constrained_layout=True)
    for row_index, (split, split_label) in enumerate(splits):
        split_table = table.loc[table["split"].eq(split)].set_index("arm")
        for column_index, (column, label, higher_is_better) in enumerate(metrics):
            axis = axes[row_index, column_index]
            values = [float(split_table.loc[arm, column]) for arm in ARM_ORDER]
            axis.barh(
                np.arange(len(ARM_ORDER)),
                values,
                color=[ARM_COLORS[arm] for arm in ARM_ORDER],
                alpha=0.9,
            )
            axis.axvline(0.0, color="#555555", linewidth=0.7)
            axis.set_title(f"{label} ({'higher' if higher_is_better else 'lower'} better)", fontsize=9)
            axis.set_yticks(np.arange(len(ARM_ORDER)))
            axis.set_yticklabels(ARM_ORDER if column_index == 0 else [])
            axis.tick_params(labelsize=8)
            for y_position, value in enumerate(values):
                axis.text(value, y_position, f" {value:.3f}", va="center", fontsize=7)
            if column_index == 0:
                axis.set_ylabel(split_label, fontsize=9)
    figure.suptitle(
        "W007 patient-balanced pathway metrics\nexploratory_reanalysis / pending_user_review; no overall winner",
        fontsize=12,
    )
    _save(figure, output_path)


def make_w004_contrast_figure(table: pd.DataFrame, output_path: Path) -> None:
    ordered = table.copy()
    ordered["label"] = ordered["contrast"].map(
        {
            "ORD_MINUS_ABS": "Ordinal − absolute",
            "SPATIAL_IDENTITY_MINUS_SPATIAL_SHUFFLE": "Spatial identity − shuffle",
        }
    ) + " / " + ordered["task"]
    colors = ["#4C78A8" if contrast == "ORD_MINUS_ABS" else "#E45756" for contrast in ordered["contrast"]]
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    values = ordered["mean_per_fit_auc_difference"].to_numpy(dtype=float)
    axis.barh(np.arange(len(ordered)), values, color=colors)
    axis.axvline(0.0, color="#333333", linewidth=0.9)
    axis.set_yticks(np.arange(len(ordered)))
    axis.set_yticklabels(ordered["label"], fontsize=9)
    axis.set_xlabel("Mean paired per-fit AUROC difference")
    for y_position, value in enumerate(values):
        axis.text(value, y_position, f" {value:+.6f}", va="center", fontsize=9)
    axis.set_title(
        "W004 fixed contrasts\nCOMPATIBILITY_ONLY; repeated slide-level holdout; patient-nonindependent"
    )
    _save(figure, output_path)


def make_w004_diagnostic_figure(table: pd.DataFrame, output_path: Path) -> None:
    test_table = table.loc[table["split"].eq("test")].copy()
    arm_order = ["ABS", "ORD", "SPATIAL_IDENTITY", "SPATIAL_SHUFFLE"]
    arm_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_index, task in enumerate(["pCR", "MPR"]):
        task_table = test_table.loc[test_table["task"].eq(task)].set_index("arm")
        for column_index, (column, label, higher_is_better) in enumerate(
            [("auc", "AUROC", True), ("auprc", "AUPRC", True), ("brier", "Brier score", False)]
        ):
            axis = axes[row_index, column_index]
            values = [float(task_table.loc[arm, column]) for arm in arm_order]
            axis.bar(np.arange(len(arm_order)), values, color=arm_colors)
            axis.set_xticks(np.arange(len(arm_order)))
            axis.set_xticklabels(arm_order, rotation=25, ha="right", fontsize=8)
            axis.set_title(f"{task}: {label} ({'higher' if higher_is_better else 'lower'} better)", fontsize=9)
            axis.tick_params(labelsize=8)
            for x_position, value in enumerate(values):
                axis.text(x_position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    figure.suptitle(
        "W004 pooled test diagnostics\ndescriptive only; repeated rows are not independent patients",
        fontsize=12,
    )
    _save(figure, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    w007 = pd.read_csv(arguments.input_dir / "w007_pathway_metrics.csv")
    w007_summary = (
        w007.groupby(["arm", "split"], as_index=False)[
            ["pcc", "ccc", "spearman", "z_rmse", "raw_r2"]
        ]
        .mean()
    )
    w004_contrasts = pd.read_csv(arguments.input_dir / "w004_fixed_auc_differences.csv")
    w004_pooled = pd.read_csv(arguments.input_dir / "w004_pooled_metrics.csv")

    make_w007_figure(w007_summary, arguments.output_dir / "w007_patient_balanced_metrics.png")
    make_w004_contrast_figure(w004_contrasts, arguments.output_dir / "w004_fixed_auc_contrasts.png")
    make_w004_diagnostic_figure(w004_pooled, arguments.output_dir / "w004_test_diagnostics.png")


if __name__ == "__main__":
    main()
