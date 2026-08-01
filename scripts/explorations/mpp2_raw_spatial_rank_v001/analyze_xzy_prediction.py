"""Spatial fidelity analysis for the accepted repaired MPP2 XZY predictions.

This module appends a prediction-focused diagnostic below the existing v001
raw-score analysis.  It never trains a model or changes accepted evidence.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from analyze import (
    benjamini_hochberg,
    build_spatial_weights,
    compute_edge_contrasts,
    compute_geary_c,
    compute_moran_i,
    compute_residual_moran,
    compute_variogram_fast,
    compute_within_patient_percentile,
    permutation_test,
    precompute_variogram_pairs,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prediction_output_root(config):
    return Path(config["output_root"]) / "xzy_baseline_prediction"


def align_prediction_rows(original, bridge, tolerance=2e-6):
    """Attach audited barcodes/coordinates to the returned 60-column table.

    The returned historical CSV has no row identifier.  Alignment is accepted
    only when every true/pred vector matches the independently generated bridge
    table in the same row order within the declared CSV precision tolerance.
    """
    pathways = [column[5:] for column in original.columns if column.startswith("true_")]
    pred_pathways = [column[5:] for column in original.columns if column.startswith("pred_")]
    if not pathways or pathways != pred_pathways:
        raise ValueError("Returned CSV must contain ordered true_* and pred_* columns for the same pathways")
    required_bridge = ["patient_id", "patch_id", "x", "y"]
    if any(column not in bridge for column in required_bridge):
        raise ValueError("Coordinate bridge lacks patient_id/patch_id/x/y")
    if len(original) != len(bridge):
        raise ValueError("Returned CSV and coordinate bridge row counts differ")
    true_differences = []
    prediction_differences = []
    for pathway in pathways:
        for column in (f"true_{pathway}", f"pred_base_{pathway}"):
            if column not in bridge:
                raise ValueError(f"Coordinate bridge lacks {column}")
        true_differences.append(np.max(np.abs(
            original[f"true_{pathway}"].to_numpy(float)
            - bridge[f"true_{pathway}"].to_numpy(float)
        )))
        prediction_differences.append(np.max(np.abs(
            original[f"pred_{pathway}"].to_numpy(float)
            - bridge[f"pred_base_{pathway}"].to_numpy(float)
        )))
    max_true = float(np.max(true_differences))
    max_prediction = float(np.max(prediction_differences))
    if max_true > tolerance or max_prediction > tolerance:
        raise ValueError(
            f"Unproven row order: max true/pred differences are {max_true}/{max_prediction}, "
            f"tolerance={tolerance}"
        )
    expected_patch = "patch_x" + bridge["x"].astype(int).astype(str) + "_y" + bridge["y"].astype(int).astype(str)
    if not bridge["patch_id"].astype(str).equals(expected_patch):
        raise ValueError("Bridge patch_id is inconsistent with x/y")
    if bridge["patch_id"].duplicated().any():
        raise ValueError("Bridge contains duplicate patch_id")
    aligned = pd.DataFrame({
        "patient": bridge["patient_id"].astype(str).to_numpy(),
        "barcode": bridge["patch_id"].astype(str).to_numpy(),
        "x": bridge["x"].to_numpy(int),
        "y": bridge["y"].to_numpy(int),
    })
    for pathway in pathways:
        aligned[f"true_{pathway}"] = original[f"true_{pathway}"].to_numpy(float)
        aligned[f"pred_{pathway}"] = original[f"pred_{pathway}"].to_numpy(float)
    qc = {
        "row_count": int(len(aligned)),
        "pathway_count": int(len(pathways)),
        "patient_values": sorted(aligned["patient"].unique().tolist()),
        "duplicate_barcodes": int(aligned["barcode"].duplicated().sum()),
        "max_truth_abs_difference": max_true,
        "max_prediction_abs_difference": max_prediction,
        "alignment_tolerance": float(tolerance),
    }
    return aligned, qc


def compute_rank_fidelity(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("truth and prediction must be non-empty and row aligned")
    k = max(1, int(np.ceil(0.10 * len(truth))))
    truth_order = np.argsort(truth)
    prediction_order = np.argsort(prediction)

    def jaccard(left, right):
        union = left | right
        return float(len(left & right) / len(union)) if union else np.nan

    truth_pct = compute_within_patient_percentile(truth)
    prediction_pct = compute_within_patient_percentile(prediction)
    return {
        "spearman_rho": float(stats.spearmanr(truth, prediction).statistic),
        "kendall_tau": float(stats.kendalltau(truth, prediction).statistic),
        "percentile_mae": float(np.mean(np.abs(truth_pct - prediction_pct))),
        "top10_jaccard": jaccard(set(truth_order[-k:]), set(prediction_order[-k:])),
        "bottom10_jaccard": jaccard(set(truth_order[:k]), set(prediction_order[:k])),
        "top10_overlap_count": int(len(set(truth_order[-k:]) & set(prediction_order[-k:]))),
        "bottom10_overlap_count": int(len(set(truth_order[:k]) & set(prediction_order[:k]))),
        "set_size": int(k),
    }


def compute_coordinate_cv_r2(values, coords, block_pixels, n_splits=5):
    """Spatial-block CV R2 for a fixed quadratic coordinate-only ridge model."""
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    ranges = np.ptp(coords, axis=0)
    ranges[ranges == 0] = 1.0
    relative_coords = (coords - coords.min(axis=0)) / ranges
    features = PolynomialFeatures(2, include_bias=False).fit_transform(relative_coords)
    block_xy = np.floor((coords - coords.min(axis=0)) / float(block_pixels)).astype(int)
    _, groups = np.unique(block_xy, axis=0, return_inverse=True)
    n_blocks = int(len(np.unique(groups)))
    folds = min(int(n_splits), n_blocks)
    if folds < 2 or np.std(values) == 0:
        return {"cv_r2": np.nan, "n_blocks": n_blocks, "n_splits": folds}
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    predicted = cross_val_predict(
        estimator, features, values, groups=groups,
        cv=GroupKFold(n_splits=folds), method="predict",
    )
    return {
        "cv_r2": float(r2_score(values, predicted)),
        "n_blocks": n_blocks,
        "n_splits": folds,
    }


def classify_pathway_pattern(true_moran, pred_moran, error_moran, error_q):
    if not np.isfinite(error_q) or error_q >= 0.05 or error_moran <= 0:
        return "no_significant_structured_error"
    if pred_moran < true_moran:
        return "prediction_less_smooth_structured_error"
    return "prediction_more_smooth_structured_error"


def _spatial_field_metrics(values, coords, w, adj, step_size, variogram_pairs,
                           n_permutations, seed):
    moran, p_value, _ = permutation_test(values, w, adj, n_permutations, seed)
    edges = compute_edge_contrasts(values, adj, coords, step_size, seed)
    variogram = compute_variogram_fast(values, variogram_pairs, np.std(values))
    return {
        "moran_i": moran,
        "p_value": p_value,
        "geary_c": compute_geary_c(values, w, adj),
        "global_random_edge_ratio": edges["global_random_ratio"],
        "near_far_edge_ratio": edges["near_far_ratio"],
        "residual_moran": compute_residual_moran(values, coords, w),
        **{f"variogram_h{idx + 1}": value for idx, value in enumerate(variogram)},
    }


def _generate_figures(output_root, aligned, pathways, summary, coordinate_df, association_df):
    coords = aligned[["x", "y"]].to_numpy()
    selected = []
    for candidate in (
        summary.loc[summary["moran_delta_pred_minus_true"].idxmax(), "pathway"],
        summary.loc[summary["moran_delta_pred_minus_true"].idxmin(), "pathway"],
        summary.loc[summary["spearman_rho"].idxmin(), "pathway"],
    ):
        if candidate not in selected:
            selected.append(candidate)
    while len(selected) < 3:
        candidate = summary.sort_values("error_moran_i", ascending=False).iloc[len(selected)]["pathway"]
        if candidate not in selected:
            selected.append(candidate)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for row, pathway in enumerate(selected):
        truth = aligned[f"true_{pathway}"].to_numpy()
        prediction = aligned[f"pred_{pathway}"].to_numpy()
        fields = [(truth, "Truth"), (prediction, "Prediction"), (prediction - truth, "Signed error")]
        for col, (values, label) in enumerate(fields):
            cmap = "coolwarm" if label == "Signed error" else "viridis"
            scatter = axes[row, col].scatter(coords[:, 0], coords[:, 1], c=values, s=8, cmap=cmap)
            axes[row, col].set_title(f"{pathway}: {label}")
            axes[row, col].set_aspect("equal")
            axes[row, col].axis("off")
            fig.colorbar(scatter, ax=axes[row, col], fraction=0.046)
    fig.suptitle("F08: XZY Truth / Accepted-Baseline Prediction / Error Maps")
    plt.tight_layout()
    plt.savefig(output_root / "F08_truth_prediction_error_maps.png", dpi=250)
    plt.close()

    heat = summary.set_index("pathway")[["true_moran_i", "pred_moran_i", "error_moran_i", "abs_error_moran_i"]]
    plt.figure(figsize=(8, 12))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("F09: Spatial Autocorrelation of Truth, Prediction and Error")
    plt.tight_layout()
    plt.savefig(output_root / "F09_prediction_spatial_heatmap.png", dpi=250)
    plt.close()

    rank_sorted = summary.sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(rank_sorted["spearman_rho"], rank_sorted["pathway"], label="Spearman rho")
    ax.scatter(rank_sorted["top10_jaccard"], rank_sorted["pathway"], label="Top-10% Jaccard")
    ax.set_xlabel("Rank / hotspot fidelity")
    ax.set_title("F10: Within-XZY Relative-Rank Fidelity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "F10_rank_fidelity.png", dpi=250)
    plt.close()

    coord_plot = coordinate_df[coordinate_df["block_pixels"] == coordinate_df["block_pixels"].min()]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=coord_plot, x="field", y="cv_r2", order=["truth", "prediction", "error", "abs_error"])
    sns.stripplot(data=coord_plot, x="field", y="cv_r2", color="black", alpha=0.45, size=3,
                  order=["truth", "prediction", "error", "abs_error"])
    plt.axhline(0, color="grey", linestyle="--")
    plt.title("F11: Coordinate-Only Spatial-Block CV R2 (1024 px Blocks)")
    plt.tight_layout()
    plt.savefig(output_root / "F11_coordinate_content.png", dpi=250)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=summary, x="spearman_rho", y="pathway_pcc", ax=axes[0], ci=None)
    axes[0].set_title("Pathway PCC vs Rank Fidelity")
    sns.regplot(data=summary, x="error_moran_i", y="pathway_pcc", ax=axes[1], ci=None)
    axes[1].set_title("Pathway PCC vs Structured Error")
    fig.suptitle("F12: Predictive Performance and Spatial Fidelity")
    plt.tight_layout()
    plt.savefig(output_root / "F12_performance_spatial_links.png", dpi=250)
    plt.close()

    top_error = summary.nlargest(3, "error_moran_i")["pathway"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, pathway in zip(axes, top_error):
        error = aligned[f"pred_{pathway}"].to_numpy() - aligned[f"true_{pathway}"].to_numpy()
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=error, s=9, cmap="coolwarm")
        ax.set_title(f"{pathway}\nerror Moran={summary.set_index('pathway').loc[pathway, 'error_moran_i']:.2f}")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(scatter, ax=ax, fraction=0.046)
    fig.suptitle("F13: Most Spatially Structured Signed Errors")
    plt.tight_layout()
    plt.savefig(output_root / "F13_structured_error_maps.png", dpi=250)
    plt.close()


def run_prediction_analysis(config_path, mode="final", replace_existing=False):
    started = datetime.now(timezone.utc)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_prediction_output_root(config)
    if output_root.exists() and any(output_root.iterdir()) and not replace_existing:
        raise FileExistsError(f"Non-empty output exists: {output_root}; use --replace-existing explicitly")
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_path = Path(config["prediction_csv"])
    bridge_path = Path(config["coordinate_bridge_csv"])
    raw_path = Path(config["raw_xzy_csv"])
    v001_metrics_path = Path(config["current_v001_spatial_metrics"])
    observed_prediction_sha = sha256_file(prediction_path)
    observed_bridge_sha = sha256_file(bridge_path)
    if observed_prediction_sha != config["prediction_csv_sha256"]:
        raise ValueError("Returned prediction CSV SHA-256 does not match the frozen contract")
    if observed_bridge_sha != config["coordinate_bridge_sha256"]:
        raise ValueError("Coordinate bridge SHA-256 does not match the audited bridge")

    original = pd.read_csv(prediction_path)
    bridge = pd.read_csv(bridge_path)
    raw_xzy = pd.read_csv(raw_path)
    aligned, input_qc = align_prediction_rows(
        original, bridge, tolerance=float(config["row_alignment_tolerance"])
    )
    if not np.array_equal(aligned["barcode"].to_numpy(), raw_xzy["barcode"].astype(str).to_numpy()):
        raise ValueError("Audited bridge barcode order does not match the current XZY raw table")
    if input_qc["patient_values"] != ["XZY"]:
        raise ValueError("Prediction analysis is restricted to external patient XZY")

    pathways = [column[5:] for column in original.columns if column.startswith("true_")]
    if mode == "smoke":
        pathways = pathways[:3]
    n_permutations = int(config["n_permutations_smoke"] if mode == "smoke" else config["n_permutations_final"])
    coords = aligned[["x", "y"]].to_numpy(float)
    w, adj, step_size = build_spatial_weights(coords, "rook")
    variogram_pairs = precompute_variogram_pairs(coords, max_steps=5, step_size=step_size)

    input_qc.update({
        "prediction_csv_sha256": observed_prediction_sha,
        "coordinate_bridge_sha256": observed_bridge_sha,
        "raw_xzy_sha256": sha256_file(raw_path),
        "current_v001_spatial_metrics_sha256": sha256_file(v001_metrics_path),
        "rook_step_size": float(step_size),
        "rook_edge_count": int(adj.sum() / 2),
        "prediction_csv_has_row_identifiers": False,
        "coordinate_mapping_method": "rowwise true/pred equality to audited bridge, then exact bridge-to-raw barcode order",
    })

    seed = int(config["seed"])
    spatial_rows = []
    values_by_pathway = {}
    for pathway_index, pathway in enumerate(pathways):
        truth = aligned[f"true_{pathway}"].to_numpy(float)
        prediction = aligned[f"pred_{pathway}"].to_numpy(float)
        fields = {
            "truth": truth,
            "prediction": prediction,
            "error": prediction - truth,
            "abs_error": np.abs(prediction - truth),
        }
        values_by_pathway[pathway] = fields
        for field_index, (field, values) in enumerate(fields.items()):
            metrics = _spatial_field_metrics(
                values, coords, w, adj, step_size, variogram_pairs,
                n_permutations, seed + pathway_index * 10 + field_index,
            )
            spatial_rows.append({"pathway": pathway, "field": field, **metrics})
    spatial_df = pd.DataFrame(spatial_rows)
    spatial_df["q_value_bh_30_pathways"] = np.nan
    for field, group in spatial_df.groupby("field"):
        spatial_df.loc[group.index, "q_value_bh_30_pathways"] = benjamini_hochberg(group["p_value"].to_numpy())
    spatial_df.to_csv(output_root / "prediction_spatial_metrics.csv", index=False)

    coordinate_rows = []
    coordinate_blocks = config["coordinate_block_pixels"]
    if mode == "smoke":
        coordinate_blocks = coordinate_blocks[:1]
    for pathway in pathways:
        for block_pixels in coordinate_blocks:
            for field, values in values_by_pathway[pathway].items():
                result = compute_coordinate_cv_r2(
                    values, coords, float(block_pixels), int(config["coordinate_cv_folds"])
                )
                coordinate_rows.append({
                    "pathway": pathway, "field": field, "block_pixels": int(block_pixels), **result
                })
    coordinate_df = pd.DataFrame(coordinate_rows)
    coordinate_df.to_csv(output_root / "coordinate_content_cv.csv", index=False)

    spatial_index = spatial_df.set_index(["pathway", "field"])
    summary_rows = []
    for pathway in pathways:
        truth = values_by_pathway[pathway]["truth"]
        prediction = values_by_pathway[pathway]["prediction"]
        rank = compute_rank_fidelity(truth, prediction)
        true_moran = spatial_index.loc[(pathway, "truth"), "moran_i"]
        pred_moran = spatial_index.loc[(pathway, "prediction"), "moran_i"]
        error_moran = spatial_index.loc[(pathway, "error"), "moran_i"]
        error_q = spatial_index.loc[(pathway, "error"), "q_value_bh_30_pathways"]
        summary_rows.append({
            "pathway": pathway,
            "pathway_pcc": float(stats.pearsonr(truth, prediction).statistic),
            "pathway_mae_z": float(np.mean(np.abs(prediction - truth))),
            "pathway_r2": float(r2_score(truth, prediction)),
            "true_moran_i": float(true_moran),
            "pred_moran_i": float(pred_moran),
            "moran_delta_pred_minus_true": float(pred_moran - true_moran),
            "error_moran_i": float(error_moran),
            "error_q_bh_30_pathways": float(error_q),
            "abs_error_moran_i": float(spatial_index.loc[(pathway, "abs_error"), "moran_i"]),
            "error_residual_moran": float(spatial_index.loc[(pathway, "error"), "residual_moran"]),
            "pattern_class": classify_pathway_pattern(true_moran, pred_moran, error_moran, error_q),
            **rank,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "pathway_prediction_spatial_summary.csv", index=False)

    association_rows = []
    performance_columns = ["pathway_pcc", "pathway_r2", "pathway_mae_z"]
    spatial_columns = [
        "spearman_rho", "percentile_mae", "top10_jaccard",
        "moran_delta_pred_minus_true", "error_moran_i", "abs_error_moran_i",
    ]
    for performance in performance_columns:
        for spatial_metric in spatial_columns:
            rho, p_value = stats.spearmanr(summary_df[performance], summary_df[spatial_metric])
            association_rows.append({
                "performance_metric": performance, "spatial_metric": spatial_metric,
                "spearman_rho_across_30_pathways": float(rho), "p_value": float(p_value),
            })
    association_df = pd.DataFrame(association_rows)
    association_df["q_value_bh_all_associations"] = benjamini_hochberg(association_df["p_value"].to_numpy())
    association_df.to_csv(output_root / "performance_spatial_associations.csv", index=False)

    current_v001 = pd.read_csv(v001_metrics_path)
    current_xzy = current_v001[current_v001["patient"] == "XZY"].set_index("pathway")
    truth_moran_difference = summary_df.set_index("pathway")["true_moran_i"] - current_xzy.loc[pathways, "moran_i"]
    max_truth_moran_difference = float(truth_moran_difference.abs().max())
    if max_truth_moran_difference > 1e-6:
        raise ValueError("Returned truth field is inconsistent with current v001 XZY spatial metrics")

    coordinate_wide = coordinate_df.pivot_table(
        index=["pathway", "block_pixels"], columns="field", values="cv_r2"
    ).reset_index()
    coordinate_wide["prediction_minus_truth_cv_r2"] = coordinate_wide["prediction"] - coordinate_wide["truth"]
    decision = {
        "evidence_class": config["evidence_class"],
        "analysis_scope": "accepted repaired frozen MPP2 baseline; external XZY only",
        "input_prediction_sha256": observed_prediction_sha,
        "n_spots": int(len(aligned)),
        "n_pathways": int(len(pathways)),
        "current_v001_truth_moran_max_abs_difference": max_truth_moran_difference,
        "median_true_moran_i": float(summary_df["true_moran_i"].median()),
        "median_prediction_moran_i": float(summary_df["pred_moran_i"].median()),
        "prediction_more_smooth_count": int((summary_df["moran_delta_pred_minus_true"] > 0).sum()),
        "structured_signed_error_count": int(((summary_df["error_moran_i"] > 0) & (summary_df["error_q_bh_30_pathways"] < 0.05)).sum()),
        "structured_absolute_error_count": int(((spatial_df["field"] == "abs_error") & (spatial_df["moran_i"] > 0) & (spatial_df["q_value_bh_30_pathways"] < 0.05)).sum()),
        "median_truth_prediction_spearman": float(summary_df["spearman_rho"].median()),
        "median_percentile_mae": float(summary_df["percentile_mae"].median()),
        "median_top10_jaccard": float(summary_df["top10_jaccard"].median()),
        "coordinate_cv_r2_medians_by_field_and_grid": coordinate_df.groupby(["block_pixels", "field"])["cv_r2"].median().unstack().to_dict(orient="index"),
        "coordinate_inflation_median_by_grid": {
            str(int(grid)): float(group["prediction_minus_truth_cv_r2"].median())
            for grid, group in coordinate_wide.groupby("block_pixels")
        },
        "dual_view_implication": "prediction percentiles are pathway-specific diagnostics; blanket ordinal adoption is not supported by XZY rank fidelity alone",
        "spatial_regularization_implication": "structured regional errors require mechanism-aware testing; Moran smoothness alone does not authorize uniform smoothing",
        "coordinate_implication": "coordinate-only CV is descriptive; it does not prove coordinate leakage or causality",
        "does_not_change": [
            "accepted baseline performance metrics", "training authorization", "experiment registry status",
        ],
    }
    with open(output_root / "prediction_decision_summary.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    with open(output_root / "input_qc.json", "w", encoding="utf-8") as handle:
        json.dump(input_qc, handle, ensure_ascii=False, indent=2)

    _generate_figures(output_root, aligned, pathways, summary_df, coordinate_df, association_df)

    report = f"""# MPP2 accepted 基线 XZY 原始预测空间诊断报告

> 证据级别：`diagnostic_only`
> 输入：accepted 修复版冻结 MPP2 基线的用户手动回传原始预测 CSV
> 范围：XZY 1039 spots × {len(pathways)} pathways；不训练、不调参、不改变 accepted 指标

## 输入验真

- 原始预测 SHA-256：`{observed_prediction_sha}`。
- 原表没有 barcode；通过真值/预测逐行匹配到带坐标桥接表，再与 XZY 原始 barcode 顺序精确核对。
- 最大真值/预测桥接差：`{input_qc['max_truth_abs_difference']:.3g}` / `{input_qc['max_prediction_abs_difference']:.3g}`。
- 回传真值 Moran 与当前 v001 XZY 的最大绝对差：`{max_truth_moran_difference:.3g}`。

## 核心结果

- 真值 Moran 中位数：**{decision['median_true_moran_i']:.4f}**；预测 Moran 中位数：**{decision['median_prediction_moran_i']:.4f}**。
- 预测比真值更平滑：**{decision['prediction_more_smooth_count']}/{len(pathways)}** 通路。
- 有符号误差存在显著正空间结构：**{decision['structured_signed_error_count']}/{len(pathways)}**；绝对误差：**{decision['structured_absolute_error_count']}/{len(pathways)}**。
- 真值—预测患者内秩 Spearman 中位数：**{decision['median_truth_prediction_spearman']:.4f}**。
- 百分位 MAE 中位数：**{decision['median_percentile_mae']:.4f}**；top 10% Jaccard 中位数：**{decision['median_top10_jaccard']:.4f}**。

## 与当前双线方案的关系

- **S1/B1 患者内相对线**：真值百分位可稳定构造，不代表模型预测也能保留真实排序。本次直接测得的预测秩保真具有明显通路差异，因此不支持无差别使用全部 30 维预测百分位。
- **S2-S/A1-S 空间线**：误差具有区域性结构，不是简单独立白噪声。统一平滑可能掩盖区域偏差；应按通路区分“预测更不平滑”和“预测更平滑”后再设计消融。
- **相对坐标含量**：空间块 CV 的坐标-only R2 仅描述低频位置结构；即使预测或误差可被坐标解释，也不能据此宣称坐标泄漏或生物学原因。

## 明确边界

本报告只分析既有外部 XZY 预测，不使用 XZY 选择 checkpoint 或参数，不证明空间正则能提升性能，也不改变 Registry 中 accepted 基线结论。
"""
    (output_root / "human_prediction_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "mode": mode,
        "n_permutations": n_permutations,
        "seed": seed,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "code_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "inputs": input_qc,
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }
    with open(output_root / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    checksum_rows = []
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_rows.append(f"{sha256_file(path)}  {path.name}")
    (output_root / "checksums.sha256").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    print(f"Prediction analysis completed: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPP2 accepted-baseline XZY prediction spatial diagnostic")
    parser.add_argument("--config", default="scripts/explorations/mpp2_raw_spatial_rank_v001/prediction_config.json")
    parser.add_argument("--mode", choices=["smoke", "final"], default="final")
    parser.add_argument("--replace-existing", action="store_true")
    arguments = parser.parse_args()
    run_prediction_analysis(arguments.config, arguments.mode, arguments.replace_existing)
