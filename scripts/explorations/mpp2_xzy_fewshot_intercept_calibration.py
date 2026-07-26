# PFMVAL_EXPLORE
# PFMVAL_USER_FORCED_EXPERIMENT: pending_user_approval
"""用户强制指定的 XZY few-shot intercept 本地探索，结果待用户批准才可登记采纳。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.explorations.mpp2_r2_root_cause_audit import (
    ccc,
    pcc,
    r2_score,
    sha256_file,
    validate_inputs,
)


class CalibrationError(RuntimeError):
    """Raised when the few-shot calibration contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def choose_anchor_indices(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    method: str,
) -> np.ndarray:
    """Choose an exact, deterministic calibration subset."""
    _require(0.0 < fraction < 1.0, "calibration fraction 必须位于 (0, 1)")
    _require(method in {"random", "spatial_stratified"}, "未知 sampling method")
    _require({"x", "y"}.issubset(frame.columns), "预测表缺少 x/y")
    n_rows = len(frame)
    _require(n_rows >= 4, "样本量不足")
    n_calibration = max(2, int(round(n_rows * fraction)))
    _require(n_calibration < n_rows, "校准集不得覆盖全部样本")
    rng = np.random.default_rng(seed)

    if method == "random":
        return np.sort(rng.choice(n_rows, size=n_calibration, replace=False))

    x_rank = frame["x"].rank(method="first")
    y_rank = frame["y"].rank(method="first")
    x_bin = pd.qcut(x_rank, q=min(4, n_rows), labels=False, duplicates="drop")
    y_bin = pd.qcut(y_rank, q=min(4, n_rows), labels=False, duplicates="drop")
    strata = pd.DataFrame({"x_bin": x_bin, "y_bin": y_bin})
    groups = [
        np.asarray(index, dtype=np.int64)
        for _, index in strata.groupby(["x_bin", "y_bin"], sort=True).groups.items()
    ]
    expected = np.asarray(
        [n_calibration * len(group) / n_rows for group in groups], dtype=float
    )
    allocation = np.floor(expected).astype(int)
    remaining = n_calibration - int(allocation.sum())
    order = sorted(
        range(len(groups)),
        key=lambda i: (-(expected[i] - allocation[i]), i),
    )
    for index in order[:remaining]:
        allocation[index] += 1

    chosen: list[int] = []
    for group, count in zip(groups, allocation, strict=True):
        if count:
            chosen.extend(rng.choice(group, size=int(count), replace=False).tolist())
    result = np.asarray(sorted(chosen), dtype=np.int64)
    _require(len(result) == n_calibration, "空间分层抽样数量异常")
    _require(len(np.unique(result)) == len(result), "空间分层抽样出现重复键")
    return result


def fit_intercepts(
    truth: np.ndarray,
    prediction: np.ndarray,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    """Fit one target-patient intercept per pathway using anchors only."""
    _require(truth.shape == prediction.shape, "truth/prediction 形状不一致")
    _require(truth.ndim == 2, "truth/prediction 必须为二维矩阵")
    _require(len(anchor_indices) >= 2, "至少需要两个校准锚点")
    _require(
        np.all((anchor_indices >= 0) & (anchor_indices < len(truth))),
        "校准索引越界",
    )
    _require(
        np.isfinite(truth).all() and np.isfinite(prediction).all(),
        "truth/prediction 含 NaN/Inf",
    )
    return np.mean(truth[anchor_indices] - prediction[anchor_indices], axis=0)


def apply_intercepts(prediction: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    _require(prediction.ndim == 2, "prediction 必须为二维矩阵")
    _require(prediction.shape[1] == len(intercepts), "intercept 通路数不一致")
    return prediction + intercepts


def _test_indices(n_rows: int, anchor_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(n_rows, dtype=bool)
    mask[anchor_indices] = False
    result = np.flatnonzero(mask)
    _require(
        len(np.intersect1d(anchor_indices, result)) == 0,
        "校准集与测试集发生重叠",
    )
    return result


def evaluate_split(
    truth: np.ndarray,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    anchor_indices: np.ndarray,
    raw_stds: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate baseline and calibrated predictions on held-out rows only."""
    _require(truth.shape == baseline.shape == calibrated.shape, "评估矩阵形状不一致")
    _require(len(raw_stds) == truth.shape[1], "raw std 通路数不一致")
    test_indices = _test_indices(len(truth), anchor_indices)
    _require(len(test_indices) >= 2, "留出测试集样本量不足")

    baseline_r2 = np.asarray(
        [
            r2_score(truth[test_indices, i], baseline[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    calibrated_r2 = np.asarray(
        [
            r2_score(truth[test_indices, i], calibrated[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    baseline_pcc = np.asarray(
        [
            pcc(truth[test_indices, i], baseline[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    calibrated_pcc = np.asarray(
        [
            pcc(truth[test_indices, i], calibrated[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    baseline_ccc = np.asarray(
        [
            ccc(truth[test_indices, i], baseline[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    calibrated_ccc = np.asarray(
        [
            ccc(truth[test_indices, i], calibrated[test_indices, i])
            for i in range(truth.shape[1])
        ]
    )
    baseline_mae = np.mean(
        np.abs(baseline[test_indices] - truth[test_indices]) * raw_stds,
        axis=0,
    )
    calibrated_mae = np.mean(
        np.abs(calibrated[test_indices] - truth[test_indices]) * raw_stds,
        axis=0,
    )
    return {
        "n_calibration": int(len(anchor_indices)),
        "n_test": int(len(test_indices)),
        "baseline_mean_r2": float(np.nanmean(baseline_r2)),
        "calibrated_mean_r2": float(np.nanmean(calibrated_r2)),
        "delta_mean_r2": float(np.nanmean(calibrated_r2 - baseline_r2)),
        "baseline_mean_pcc": float(np.nanmean(baseline_pcc)),
        "calibrated_mean_pcc": float(np.nanmean(calibrated_pcc)),
        "baseline_mean_ccc": float(np.nanmean(baseline_ccc)),
        "calibrated_mean_ccc": float(np.nanmean(calibrated_ccc)),
        "baseline_mean_raw_mae": float(np.nanmean(baseline_mae)),
        "calibrated_mean_raw_mae": float(np.nanmean(calibrated_mae)),
        "delta_mean_raw_mae": float(np.nanmean(calibrated_mae - baseline_mae)),
    }


def _per_pathway_rows(
    truth: np.ndarray,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    anchor_indices: np.ndarray,
    raw_stds: np.ndarray,
    pathways: list[str],
    intercepts: np.ndarray,
) -> list[dict[str, float | str]]:
    test_indices = _test_indices(len(truth), anchor_indices)
    rows: list[dict[str, float | str]] = []
    for index, pathway in enumerate(pathways):
        baseline_r2 = r2_score(truth[test_indices, index], baseline[test_indices, index])
        calibrated_r2 = r2_score(
            truth[test_indices, index], calibrated[test_indices, index]
        )
        rows.append(
            {
                "pathway": pathway,
                "intercept_z": float(intercepts[index]),
                "intercept_raw": float(intercepts[index] * raw_stds[index]),
                "baseline_raw_r2": baseline_r2,
                "calibrated_raw_r2": calibrated_r2,
                "delta_raw_r2": calibrated_r2 - baseline_r2,
                "baseline_raw_mae": float(
                    np.mean(
                        np.abs(
                            baseline[test_indices, index] - truth[test_indices, index]
                        )
                    )
                    * raw_stds[index]
                ),
                "calibrated_raw_mae": float(
                    np.mean(
                        np.abs(
                            calibrated[test_indices, index] - truth[test_indices, index]
                        )
                    )
                    * raw_stds[index]
                ),
            }
        )
    return rows


def _quantile(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, fraction), group in runs.groupby(
        ["sampling_method", "calibration_fraction"], sort=True
    ):
        rows.append(
            {
                "sampling_method": method,
                "calibration_fraction": fraction,
                "n_calibration": int(group["n_calibration"].iloc[0]),
                "n_test": int(group["n_test"].iloc[0]),
                "repeats": len(group),
                "baseline_mean_r2_median": float(
                    group["baseline_mean_r2"].median()
                ),
                "calibrated_mean_r2_median": float(
                    group["calibrated_mean_r2"].median()
                ),
                "delta_mean_r2_median": float(group["delta_mean_r2"].median()),
                "delta_mean_r2_p05": _quantile(group["delta_mean_r2"], 0.05),
                "delta_mean_r2_p95": _quantile(group["delta_mean_r2"], 0.95),
                "baseline_mean_raw_mae_median": float(
                    group["baseline_mean_raw_mae"].median()
                ),
                "calibrated_mean_raw_mae_median": float(
                    group["calibrated_mean_raw_mae"].median()
                ),
                "delta_mean_raw_mae_median": float(
                    group["delta_mean_raw_mae"].median()
                ),
                "calibrated_mean_ccc_median": float(
                    group["calibrated_mean_ccc"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_pathways(
    per_pathway: pd.DataFrame,
    method: str,
    fraction: float,
) -> pd.DataFrame:
    selected = per_pathway.loc[
        per_pathway["sampling_method"].eq(method)
        & per_pathway["calibration_fraction"].eq(fraction)
    ]
    _require(not selected.empty, "推荐条件缺少逐通路结果")
    rows: list[dict[str, Any]] = []
    for pathway, group in selected.groupby("pathway", sort=False):
        rows.append(
            {
                "pathway": pathway,
                "intercept_z_median": float(group["intercept_z"].median()),
                "intercept_z_p05": _quantile(group["intercept_z"], 0.05),
                "intercept_z_p95": _quantile(group["intercept_z"], 0.95),
                "intercept_raw_median": float(group["intercept_raw"].median()),
                "baseline_raw_r2_median": float(
                    group["baseline_raw_r2"].median()
                ),
                "calibrated_raw_r2_median": float(
                    group["calibrated_raw_r2"].median()
                ),
                "delta_raw_r2_median": float(group["delta_raw_r2"].median()),
                "delta_raw_r2_p05": _quantile(group["delta_raw_r2"], 0.05),
                "delta_raw_r2_p95": _quantile(group["delta_raw_r2"], 0.95),
                "baseline_raw_mae_median": float(
                    group["baseline_raw_mae"].median()
                ),
                "calibrated_raw_mae_median": float(
                    group["calibrated_raw_mae"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_recommendation(summary: pd.DataFrame) -> pd.Series:
    primary = summary.loc[summary["sampling_method"].eq("spatial_stratified")].copy()
    _require(not primary.empty, "缺少 spatial_stratified 结果")
    primary = primary.sort_values("calibration_fraction")
    maximum_gain = float(primary["delta_mean_r2_median"].max())
    eligible = primary.loc[
        (primary["delta_mean_r2_p05"] > 0)
        & (primary["delta_mean_r2_median"] >= 0.8 * maximum_gain)
    ]
    if eligible.empty:
        return primary.loc[primary["delta_mean_r2_median"].idxmax()]
    return eligible.iloc[0]


def run_analysis(
    validated: dict[str, Any],
    fractions: Iterable[float],
    repeats: int,
    seed: int,
    methods: Iterable[str],
) -> dict[str, Any]:
    _require(repeats >= 20, "repeats 至少为 20")
    fractions = sorted(set(float(item) for item in fractions))
    _require(bool(fractions), "至少提供一个 calibration fraction")
    frame: pd.DataFrame = validated["external"].reset_index(drop=True)
    truth = np.asarray(validated["truth_external_z"], dtype=np.float64)
    baseline = np.asarray(validated["pred_external_z"], dtype=np.float64)
    raw_stds = np.asarray(validated["stds"], dtype=np.float64)
    pathways: list[str] = validated["pathways"]

    run_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    split_cache: dict[tuple[str, float, int], np.ndarray] = {}
    for method in methods:
        for fraction in fractions:
            for repeat in range(repeats):
                repeat_seed = seed + repeat
                anchors = choose_anchor_indices(
                    frame, fraction, repeat_seed, method
                )
                split_cache[(method, fraction, repeat)] = anchors
                intercepts = fit_intercepts(truth, baseline, anchors)
                calibrated = apply_intercepts(baseline, intercepts)
                result = evaluate_split(
                    truth, baseline, calibrated, anchors, raw_stds
                )
                run_rows.append(
                    {
                        "sampling_method": method,
                        "calibration_fraction": fraction,
                        "repeat": repeat,
                        "seed": repeat_seed,
                        **result,
                    }
                )
                for row in _per_pathway_rows(
                    truth,
                    baseline,
                    calibrated,
                    anchors,
                    raw_stds,
                    pathways,
                    intercepts,
                ):
                    pathway_rows.append(
                        {
                            "sampling_method": method,
                            "calibration_fraction": fraction,
                            "repeat": repeat,
                            "seed": repeat_seed,
                            **row,
                        }
                    )

    runs = pd.DataFrame(run_rows)
    per_pathway = pd.DataFrame(pathway_rows)
    summary = _summarize_runs(runs)
    recommendation = _select_recommendation(summary)
    method = str(recommendation["sampling_method"])
    fraction = float(recommendation["calibration_fraction"])
    pathway_summary = summarize_pathways(per_pathway, method, fraction)
    candidates = runs.loc[
        runs["sampling_method"].eq(method)
        & runs["calibration_fraction"].eq(fraction)
    ].copy()
    median_gain = float(candidates["delta_mean_r2"].median())
    candidates["median_distance"] = (
        candidates["delta_mean_r2"] - median_gain
    ).abs()
    representative = candidates.sort_values(
        ["median_distance", "repeat"]
    ).iloc[0]
    repeat = int(representative["repeat"])
    anchors = split_cache[(method, fraction, repeat)]
    intercepts = fit_intercepts(truth, baseline, anchors)

    roles = np.full(len(frame), "test", dtype=object)
    roles[anchors] = "calibration"
    split_manifest = frame[["patient_id", "patch_id", "x", "y"]].copy()
    split_manifest["role"] = roles
    split_manifest["sampling_method"] = method
    split_manifest["calibration_fraction"] = fraction
    split_manifest["repeat"] = repeat
    split_manifest["seed"] = int(representative["seed"])
    _require(
        set(split_manifest["role"]) == {"calibration", "test"},
        "推荐 split 缺少 calibration/test",
    )

    intercept_table = pd.DataFrame(
        {
            "pathway": pathways,
            "intercept_z": intercepts,
            "intercept_raw": intercepts * raw_stds,
        }
    )
    return {
        "runs": runs,
        "per_pathway": per_pathway,
        "summary": summary,
        "pathway_summary": pathway_summary,
        "recommendation": recommendation.to_dict(),
        "representative": representative.drop(labels=["median_distance"]).to_dict(),
        "split_manifest": split_manifest,
        "intercepts": intercept_table,
    }


def _build_report(analysis: dict[str, Any]) -> str:
    summary: pd.DataFrame = analysis["summary"]
    pathway_summary: pd.DataFrame = analysis["pathway_summary"]
    recommendation = analysis["recommendation"]
    representative = analysis["representative"]
    positive_median = int((pathway_summary["delta_raw_r2_median"] > 0).sum())
    positive_p05 = int((pathway_summary["delta_raw_r2_p05"] > 0).sum())
    best = pathway_summary.nlargest(3, "delta_raw_r2_median")
    best_text = "、".join(
        f"{row.pathway} ({row.delta_raw_r2_median:+.3f})"
        for row in best.itertuples(index=False)
    )
    table_rows = []
    for row in summary.itertuples(index=False):
        table_rows.append(
            f"| {row.sampling_method} | {row.calibration_fraction:.0%} | "
            f"{row.n_calibration} | {row.calibrated_mean_r2_median:.4f} | "
            f"{row.delta_mean_r2_median:+.4f} | "
            f"[{row.delta_mean_r2_p05:+.4f}, {row.delta_mean_r2_p95:+.4f}] | "
            f"{row.delta_mean_raw_mae_median:+.2f} |"
        )
    return f"""# MPP2 XZY few-shot 仅截距校准（2026-07-25）

> **证据边界：这是 target-patient few-shot 本地诊断，不是零样本外测。**
> 校准参数只由每次 calibration anchors 的 XZY 真值拟合，并仅在不重叠的
> held-out spots 上评价。XZY 已被用于方法开发，不能继续充当未见外部测试集。

## 结论

- 推荐空间分层锚点比例：**{float(recommendation['calibration_fraction']):.0%}**
  （{int(recommendation['n_calibration'])} 个校准 spot，
  {int(recommendation['n_test'])} 个留出测试 spot）。
- 重复抽样的中位校准后 mean raw R²：
  **{float(recommendation['calibrated_mean_r2_median']):.4f}**；
  相对同一留出集基线的中位增量：
  **{float(recommendation['delta_mean_r2_median']):+.4f}**。
- ΔR² 的重复抽样 5%–95%区间：
  **[{float(recommendation['delta_mean_r2_p05']):+.4f},
  {float(recommendation['delta_mean_r2_p95']):+.4f}]**。
- raw MAE 中位变化：
  **{float(recommendation['delta_mean_raw_mae_median']):+.2f}**；
  负值代表误差下降。
- 30 条通路中有 **{positive_median}/30** 条的 ΔR² 中位数大于 0，
  其中 **{positive_p05}/30** 条在 5%分位仍大于 0；最大增益集中于
  {best_text}。
- 推荐代表性 split 使用 repeat={int(representative['repeat'])}、
  seed={int(representative['seed'])}；它仅用于复核，不是重新选择后的正式外测。

## 全部比例

| 抽样 | 锚点比例 | 锚点数 | 校准后 R² 中位数 | ΔR² 中位数 | ΔR² 5%–95% | Δ raw MAE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 判定

- 推荐规则预先固定为：在 `spatial_stratified` 中选择最小比例，同时满足
  `ΔR² p05 > 0` 且中位增益达到所有候选比例最大中位增益的 80%。
- `spatial_stratified` 表示按 X/Y 四分位形成空间网格后按比例抽取，
  用于降低少量锚点集中在单一区域的风险。
- 每次评价都在同一个 held-out 子集上比较 baseline 与 calibrated；
  calibration/test 的 `patient_id + patch_id` 完全不重叠。
- 截距 `b` 的含义是每条通路 calibration anchors 上
  `mean(true_z - pred_z)`；没有拟合斜率、Ridge 或其他模型参数。

## 风险

- 该结果不再代表 H&E-only 零样本能力，也不能进入当前 accepted 外测结论。
- 单个 XZY 患者无法证明对下一名患者同样有效；正式证据需要新的、未参与设计的患者。
- calibration anchors 仍可能受空间自相关和测序噪声影响；报告的重复抽样区间不能替代跨患者置信区间。
"""


def _build_one_click_prompt(
    result_dir: Path,
    zscore_params: Path,
    output_dir: Path,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> str:
    fractions_arg = ",".join(str(item) for item in fractions)
    command = (
        "python scripts/explorations/mpp2_xzy_fewshot_intercept_calibration.py "
        f'--result-dir "{result_dir}" '
        f'--zscore-params "{zscore_params}" '
        f'--output-dir "{output_dir}" '
        f'--fractions "{fractions_arg}" --repeats {repeats} --seed {seed}'
    )
    return f"""# 新对话一键执行提示词

请在 `D:\\AI空间转录病理研究\\PFMval_new` 完整执行 MPP2 XZY few-shot
仅截距校准复核。不要询问我是否继续，按以下边界执行到完成：

1. 读取 AGENTS.md、CURRENT_STATE.md 和 active 状态事实源，运行严格 start-check。
2. 不修改受保护 MPP 资产，不训练、不重新推理、不访问服务器、不更新
   Registry/Dashboard/accepted current state。
3. 先运行：
   `python -m unittest scripts.explorations.test_mpp2_xzy_fewshot_intercept_calibration`
4. 再运行：

```powershell
{command}
```

5. 校验 calibration/test 键无重叠、输入哈希、1,039 行与 30 通路、重复运行确定性，
   并读取 `MPP2_XZY_fewshot截距校准_20260725.md` 汇报最终数字。
6. 明确说明：这是使用 XZY 少量真值的 target-patient few-shot 结果，
   不是零样本外测，XZY 不能继续作为未见外部测试证据。

本任务本地已有基线预测和真值，因此**不需要服务器拉取或运行命令**。
"""


def write_outputs(
    analysis: dict[str, Any],
    validated: dict[str, Any],
    result_dir: Path,
    zscore_params: Path,
    output_dir: Path,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "split_metrics.csv": analysis["runs"],
        "per_pathway_split_metrics.csv": analysis["per_pathway"],
        "recommended_per_pathway_summary.csv": analysis["pathway_summary"],
        "fraction_summary.csv": analysis["summary"],
        "recommended_split_manifest.csv": analysis["split_manifest"],
        "recommended_intercepts.csv": analysis["intercepts"],
    }
    for name, frame in csv_outputs.items():
        frame.to_csv(
            output_dir / name,
            index=False,
            encoding="utf-8-sig",
            float_format="%.12g",
            lineterminator="\n",
        )

    manifest = {
        "schema_version": "1.0",
        "analysis_id": "mpp2_xzy_fewshot_intercept_20260725",
        "status": "local_target_patient_fewshot_nonaccepted",
        "disclaimer": (
            "XZY truth is used only in calibration anchors; results are evaluated "
            "on disjoint held-out XZY spots and are not zero-shot external evidence."
        ),
        "input": {
            "result_dir": str(result_dir.resolve()),
            "predictions_external_xzy_sha256": sha256_file(
                result_dir / "predictions_external_xzy.csv"
            ),
            "metrics_sha256": sha256_file(result_dir / "metrics.json"),
            "zscore_params": str(zscore_params.resolve()),
            "zscore_params_sha256": sha256_file(zscore_params),
            "external_rows": int(validated["preflight"]["external_rows"]),
            "pathways": int(validated["preflight"]["pathways"]),
        },
        "parameters": {
            "fractions": fractions,
            "repeats": repeats,
            "seed": seed,
            "sampling_methods": ["random", "spatial_stratified"],
            "selection_rule": (
                "smallest spatial_stratified fraction with delta R2 p05 > 0 "
                "and median gain >= 80% of maximum median gain"
            ),
        },
        "recommendation": analysis["recommendation"],
        "representative_split": analysis["representative"],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "MPP2_XZY_fewshot截距校准_20260725.md").write_text(
        _build_report(analysis), encoding="utf-8"
    )
    (output_dir / "新对话一键执行提示词.md").write_text(
        _build_one_click_prompt(
            result_dir,
            zscore_params,
            output_dir,
            fractions,
            repeats,
            seed,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local target-patient few-shot intercept calibration audit"
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--zscore-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        default="0.01,0.02,0.05,0.10,0.20",
        help="Comma-separated calibration fractions",
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fractions = [float(item) for item in args.fractions.split(",") if item.strip()]
    validated = validate_inputs(args.result_dir, args.zscore_params)
    analysis = run_analysis(
        validated,
        fractions=fractions,
        repeats=args.repeats,
        seed=args.seed,
        methods=["random", "spatial_stratified"],
    )
    write_outputs(
        analysis,
        validated,
        args.result_dir,
        args.zscore_params,
        args.output_dir,
        fractions,
        args.repeats,
        args.seed,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir.resolve()),
                "recommendation": analysis["recommendation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
