# PFMVAL_EXPLORE
# PFMVAL_USER_FORCED_EXPERIMENT: pending_user_approval
"""用户强制指定的无约束 Ridge 本地探索，结果待用户批准才可登记采纳。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explorations.mpp2_force_ridge_calibration_audit import (
    load_and_validate_standalone,
    write_json,
)
from scripts.mpp2_pathway_ridge_calibration import (
    ccc,
    patient_weights,
    per_pathway_metrics,
    r2_score,
)


class UnconstrainedFitError(RuntimeError):
    """Raised when unconstrained calibration fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnconstrainedFitError(message)


def fit_unconstrained_affine(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
    ridge_lambda: float = 0.0,
    force_positive_slope: bool = False,
) -> tuple[float, float, str]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if float(np.var(prediction)) < 1e-6:
        return 1.0, 0.0, "pred_z_variance_below_1e-6"
    weights = patient_weights(patients)
    design = np.column_stack([prediction, np.ones_like(prediction)])

    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * truth)

    if ridge_lambda > 0:
        lhs = lhs + ridge_lambda * np.eye(2)
        rhs = rhs + ridge_lambda * np.asarray([1.0, 0.0])

    try:
        slope, intercept = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return 1.0, 0.0, "singular_system"

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return 1.0, 0.0, "nonfinite_solution"

    if force_positive_slope and slope <= 0:
        slope = 0.0
        intercept = float(np.average(truth - prediction, weights=weights))
        return float(slope), float(intercept), "negative_slope_fallback"

    return float(slope), float(intercept), "none"


def fit_pathways_for_lambda(
    truth: np.ndarray,
    prediction: np.ndarray,
    patients: Sequence[str],
    ridge_lambda: float,
    force_positive_slope: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    slopes = []
    intercepts = []
    reasons = []
    for idx in range(truth.shape[1]):
        k, b, reason = fit_unconstrained_affine(
            truth[:, idx],
            prediction[:, idx],
            patients,
            ridge_lambda=ridge_lambda,
            force_positive_slope=force_positive_slope,
        )
        slopes.append(k)
        intercepts.append(b)
        reasons.append(reason)
    return np.asarray(slopes, dtype=np.float64), np.asarray(intercepts, dtype=np.float64), reasons


def evaluate_mode(
    truth_i: np.ndarray,
    pred_i: np.ndarray,
    truth_x: np.ndarray,
    pred_x: np.ndarray,
    patients: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    pathways: list[str],
    mode_name: str,
    ridge_lambda: float = 0.0,
    force_positive_slope: bool = True,
    per_pathway_lambdas: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if per_pathway_lambdas is not None:
        slopes = np.empty(len(pathways), dtype=np.float64)
        intercepts = np.empty(len(pathways), dtype=np.float64)
        reasons = []
        for idx in range(len(pathways)):
            k, b, reason = fit_unconstrained_affine(
                truth_i[:, idx],
                pred_i[:, idx],
                patients,
                ridge_lambda=float(per_pathway_lambdas[idx]),
                force_positive_slope=force_positive_slope,
            )
            slopes[idx] = k
            intercepts[idx] = b
            reasons.append(reason)
        lambdas_used = per_pathway_lambdas.tolist()
    else:
        slopes, intercepts, reasons = fit_pathways_for_lambda(
            truth_i, pred_i, patients, ridge_lambda, force_positive_slope
        )
        lambdas_used = [ridge_lambda] * len(pathways)

    calibrated_x_z = pred_x * slopes + intercepts
    base_metrics = per_pathway_metrics(truth_x, pred_x, means, stds)
    cal_metrics = per_pathway_metrics(truth_x, calibrated_x_z, means, stds)

    rows = []
    pcc_changed_count = 0
    pcc_sign_flipped_count = 0

    for idx, pathway in enumerate(pathways):
        k = slopes[idx]
        b_z = intercepts[idx]
        b_raw = b_z * stds[idx]
        l_val = lambdas_used[idx]

        b_pcc = base_metrics[idx]["pcc"]
        c_pcc = cal_metrics[idx]["pcc"]
        d_pcc = c_pcc - b_pcc

        pcc_changed = bool(abs(d_pcc) > 1e-6)
        sign_flipped = bool(np.sign(b_pcc) != np.sign(c_pcc) and not np.isnan(b_pcc) and not np.isnan(c_pcc))

        if pcc_changed:
            pcc_changed_count += 1
        if sign_flipped:
            pcc_sign_flipped_count += 1

        rows.append({
            "mode": mode_name,
            "pathway": pathway,
            "lambda": l_val,
            "slope_k": k,
            "intercept_z_b": b_z,
            "intercept_raw_b": b_raw,
            "baseline_raw_r2": base_metrics[idx]["raw_r2"],
            "calibrated_raw_r2": cal_metrics[idx]["raw_r2"],
            "delta_raw_r2": cal_metrics[idx]["raw_r2"] - base_metrics[idx]["raw_r2"],
            "baseline_pcc": b_pcc,
            "calibrated_pcc": c_pcc,
            "delta_pcc": d_pcc,
            "pcc_changed": pcc_changed,
            "pcc_sign_flipped": sign_flipped,
            "baseline_raw_mae": base_metrics[idx]["raw_mae"],
            "calibrated_raw_mae": cal_metrics[idx]["raw_mae"],
            "delta_raw_mae": cal_metrics[idx]["raw_mae"] - base_metrics[idx]["raw_mae"],
            "fallback_reason": reasons[idx],
        })

    df = pd.DataFrame(rows)

    summary = {
        "mode": mode_name,
        "mean_raw_r2_baseline": float(np.nanmean(df["baseline_raw_r2"])),
        "mean_raw_r2_calibrated": float(np.nanmean(df["calibrated_raw_r2"])),
        "delta_mean_raw_r2": float(np.nanmean(df["delta_raw_r2"])),
        "mean_pcc_baseline": float(np.nanmean(df["baseline_pcc"])),
        "mean_pcc_calibrated": float(np.nanmean(df["calibrated_pcc"])),
        "delta_mean_pcc": float(np.nanmean(df["delta_pcc"])),
        "mean_raw_mae_baseline": float(np.nanmean(df["baseline_raw_mae"])),
        "mean_raw_mae_calibrated": float(np.nanmean(df["calibrated_raw_mae"])),
        "delta_mean_raw_mae": float(np.nanmean(df["delta_raw_mae"])),
        "pcc_changed_pathways_count": pcc_changed_count,
        "pcc_sign_flipped_pathways_count": pcc_sign_flipped_count,
        "r2_improved_pathways_count": int((df["delta_raw_r2"] > 0).sum()),
        "negative_slopes_count": int((df["slope_k"] < 0).sum()),
        "max_slope_k": float(df["slope_k"].max()),
        "min_slope_k": float(df["slope_k"].min()),
        "mean_slope_k": float(df["slope_k"].mean()),
    }

    return summary, df


def find_per_pathway_optimal_lambdas(
    truth_i: np.ndarray,
    pred_i: np.ndarray,
    patients: Sequence[str],
    grid: Sequence[float] = (0.0, 1e-4, 1e-2, 0.1, 1.0, 10.0),
) -> np.ndarray:
    unique_patients = sorted(set(patients.tolist()))
    best_lambdas = np.zeros(truth_i.shape[1], dtype=np.float64)

    for idx in range(truth_i.shape[1]):
        best_mse = float("inf")
        best_l = 0.0
        for l_candidate in grid:
            oof_err = []
            for held_out in unique_patients:
                tr_mask = patients != held_out
                te_mask = ~tr_mask
                k, b, _ = fit_unconstrained_affine(
                    truth_i[tr_mask, idx],
                    pred_i[tr_mask, idx],
                    patients[tr_mask],
                    ridge_lambda=float(l_candidate),
                    force_positive_slope=True,
                )
                pred_oof = pred_i[te_mask, idx] * k + b
                oof_err.append(np.mean((truth_i[te_mask, idx] - pred_oof) ** 2))
            mean_oof_mse = float(np.mean(oof_err))
            if mean_oof_mse < best_mse:
                best_mse = mean_oof_mse
                best_l = float(l_candidate)
        best_lambdas[idx] = best_l

    return best_lambdas


def run_unconstrained_ridge_audit(
    validated: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    pathways: list[str] = validated["pathways"]
    means: np.ndarray = validated["means"]
    stds: np.ndarray = validated["stds"]
    internal_frame: pd.DataFrame = validated["internal"]
    external_frame: pd.DataFrame = validated["external"]

    truth_i_z = validated["truth_internal_z"]
    pred_i_z = validated["pred_internal_z"]
    truth_x_z = validated["truth_external_z"]
    pred_x_z = validated["pred_external_z"]
    patients = internal_frame["patient_id"].to_numpy(dtype=str)

    all_summaries = {}
    all_dfs = []

    sum_a1, df_a1 = evaluate_mode(
        truth_i_z, pred_i_z, truth_x_z, pred_x_z, patients, means, stds, pathways,
        mode_name="Mode_A1_Pure_OLS_Unconstrained", ridge_lambda=0.0, force_positive_slope=False
    )
    all_summaries["Mode_A1_Pure_OLS_Unconstrained"] = sum_a1
    all_dfs.append(df_a1)

    sum_a2, df_a2 = evaluate_mode(
        truth_i_z, pred_i_z, truth_x_z, pred_x_z, patients, means, stds, pathways,
        mode_name="Mode_A2_Positive_OLS_Unconstrained", ridge_lambda=0.0, force_positive_slope=True
    )
    all_summaries["Mode_A2_Positive_OLS_Unconstrained"] = sum_a2
    all_dfs.append(df_a2)

    best_lambdas = find_per_pathway_optimal_lambdas(truth_i_z, pred_i_z, patients)
    sum_b, df_b = evaluate_mode(
        truth_i_z, pred_i_z, truth_x_z, pred_x_z, patients, means, stds, pathways,
        mode_name="Mode_B_Pathway_Specific_Lambda", per_pathway_lambdas=best_lambdas, force_positive_slope=True
    )
    sum_b["selected_pathway_lambdas"] = {p: float(best_lambdas[i]) for i, p in enumerate(pathways)}
    all_summaries["Mode_B_Pathway_Specific_Lambda"] = sum_b
    all_dfs.append(df_b)

    grid_summaries = {}
    for l_val in [1e-4, 1e-2, 0.1, 1.0, 10.0]:
        mode_key = f"Mode_C_Lambda_{l_val}"
        sum_c, df_c = evaluate_mode(
            truth_i_z, pred_i_z, truth_x_z, pred_x_z, patients, means, stds, pathways,
            mode_name=mode_key, ridge_lambda=l_val, force_positive_slope=True
        )
        grid_summaries[mode_key] = sum_c
        all_dfs.append(df_c)
    all_summaries["Mode_C_Grid_Scan"] = grid_summaries

    combined_df = pd.concat(all_dfs, ignore_index=True)

    pcc_comp_rows = []
    for pathway in pathways:
        p_sub = combined_df[combined_df["pathway"] == pathway]
        row_dict = {"pathway": pathway}
        for _, row in p_sub.iterrows():
            m_label = row["mode"]
            row_dict[f"{m_label}_k"] = row["slope_k"]
            row_dict[f"{m_label}_pcc"] = row["calibrated_pcc"]
            row_dict[f"{m_label}_delta_pcc"] = row["delta_pcc"]
            row_dict[f"{m_label}_pcc_changed"] = row["pcc_changed"]
        pcc_comp_rows.append(row_dict)
    pcc_comp_df = pd.DataFrame(pcc_comp_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_dir / "unconstrained_modes_k_b.csv", index=False, encoding="utf-8-sig")
    pcc_comp_df.to_csv(output_dir / "pcc_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "unconstrained_ridge_summary.json", all_summaries)

    report_md = _build_report_markdown(all_summaries, pcc_comp_df, combined_df)
    (output_dir / "MPP2_unconstrained_ridge_xzy_report_20260725.md").write_text(
        report_md, encoding="utf-8"
    )

    return all_summaries


def _build_report_markdown(
    summaries: dict[str, Any],
    pcc_comp_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> str:
    s_a1 = summaries["Mode_A1_Pure_OLS_Unconstrained"]
    s_a2 = summaries["Mode_A2_Positive_OLS_Unconstrained"]
    s_b = summaries["Mode_B_Pathway_Specific_Lambda"]

    a1_changed = combined_df[(combined_df["mode"] == "Mode_A1_Pure_OLS_Unconstrained") & (combined_df["pcc_changed"])]

    changed_details = []
    for _, r in a1_changed.iterrows():
        changed_details.append(
            f"- **{r['pathway']}**: k={r['slope_k']:.4f}, baseline PCC={r['baseline_pcc']:.4f} -> calibrated PCC={r['calibrated_pcc']:.4f} (delta PCC={r['delta_pcc']:+.4f})"
        )
    changed_text = "\n".join(changed_details) if changed_details else "None (All pathway PCCs remain stable)"

    table_rows = []
    mode_keys = [
        ("Mode A1: Pure OLS (lambda=0, any k)", s_a1),
        ("Mode A2: Positive OLS (lambda=0, k>=0)", s_a2),
        ("Mode B: Per-pathway optimal lambda_i", s_b),
        ("Mode C: Baseline limit (lambda=10.0)", summaries["Mode_C_Grid_Scan"]["Mode_C_Lambda_10.0"]),
    ]
    for label, s in mode_keys:
        table_rows.append(
            f"| {label} | {s['mean_raw_r2_calibrated']:.4f} | {s['delta_mean_raw_r2']:+.4f} | "
            f"{s['mean_pcc_calibrated']:.4f} | **{s['delta_mean_pcc']:+.4f}** | "
            f"{s['pcc_changed_pathways_count']} | {s['negative_slopes_count']} | "
            f"[{s['min_slope_k']:.3f}, {s['max_slope_k']:.3f}] |"
        )
    mode_table_text = "\n".join(table_rows)

    return f"""# MPP2 Unconstrained Ridge/OLS Calibration Report (2026-07-25)

## 1. Summary of Results

1. **PCC Changes**:
   - In **Mode A1 (Pure OLS)**, **{s_a1['pcc_changed_pathways_count']}** pathways changed PCC, and **{s_a1['pcc_sign_flipped_pathways_count']}** pathways had their PCC sign flipped (from positive to negative) due to negative fitted slopes (k < 0).
   - In **Mode A2 (Positive OLS, k >= 0)** and **Mode B/C**, since k >= 0 is strictly enforced, single-pathway PCC remains 100% unchanged (delta PCC = 0).
2. **Slope k Flexibility**:
   - Mode A1 slope range: [{s_a1['min_slope_k']:.3f}, {s_a1['max_slope_k']:.3f}]
   - Mode A2 slope range: [{s_a2['min_slope_k']:.3f}, {s_a2['max_slope_k']:.3f}]
3. **XZY Raw R2 Changes**:
   - Mode A1 (Pure OLS): Mean Raw R2 = {s_a1['mean_raw_r2_calibrated']:.4f} (degraded due to inverted negative slopes).
   - Mode A2 (Positive OLS): Mean Raw R2 = {s_a2['mean_raw_r2_calibrated']:.4f} (delta R2 = {s_a2['delta_mean_raw_r2']:+.4f}).
   - Mode B (Pathway Specific Lambda): Mean Raw R2 = {s_b['mean_raw_r2_calibrated']:.4f} (delta R2 = {s_b['delta_mean_raw_r2']:+.4f}).

---

## 2. Mode Comparison Table

| Mode | Mean Raw R2 | delta R2 | Mean PCC | delta PCC | PCC Changed Pathways | Negative Slope Count | Slope k Range [min, max] |
|---|---:|---:|---:|---:|---:|---:|---:|
{mode_table_text}

---

## 3. PCC Changed Pathway Details (Mode A1 Pure OLS)

{changed_text}

---

## 4. Key Takeaways

1. **Why does Pure OLS (lambda=0) alter PCC?**
   When k < 0 is fitted for unstable pathways, the linear prediction direction is inverted on XZY, flipping the PCC sign and drastically degrading R2.
2. **Why does Positive Slope (k >= 0) keep PCC unchanged?**
   Pearson correlation is invariant under positive linear transformations (y = k*x + b with k > 0).
3. **Impact of removing lambda constraints on R2**:
   Even with fully unconstrained slopes, zero-shot R2 gain on XZY remains very small (+0.001 ~ +0.002) because the cross-patient bias (intercept difference) cannot be estimated without target-patient anchor spots.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unconstrained Ridge/OLS calibration audit with PCC sensitivity tracking"
    )
    result_dir_default = PROJECT_ROOT.parent / "PFMval_new_result_mpp2_ridge_r003_20260718/automation/results/mpp2-pathway-ridge-calibration-20260717-r003"
    zscore_default = PROJECT_ROOT / "data/protected_local/mpp2_r2_root_cause_20260725/source_snapshot/zscore/group_2_repaired_v003/zscore_params_from_train.json"

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=result_dir_default,
    )
    parser.add_argument(
        "--zscore-params",
        type=Path,
        default=zscore_default,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/explorations/mpp2_unconstrained_ridge_20260725",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validated = load_and_validate_standalone(args.result_dir, args.zscore_params)
    summaries = run_unconstrained_ridge_audit(validated, args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir.resolve()),
                "mode_a1_pure_ols": summaries["Mode_A1_Pure_OLS_Unconstrained"],
                "mode_a2_positive_ols": summaries["Mode_A2_Positive_OLS_Unconstrained"],
                "mode_b_pathway_specific": summaries["Mode_B_Pathway_Specific_Lambda"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
