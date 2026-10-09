"""Create the Phase 2 second-batch static figures from copied source tables.

No model fitting or metric recomputation from predictions is performed here.
Figure 1 verifies summary mean/sample-SD values against the copied per-run CSV.
Figure 2 uses the six-model summary and task-2 summary, cross-checked against
the supplied descriptive comparison table.
"""
from pathlib import Path
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "exports"
OUT.mkdir(exist_ok=True)

INK = "#243247"
BLUE = "#3B77A5"
TEAL = "#268C84"
ORANGE = "#C7804C"
PURPLE = "#7869A7"
GREEN = "#648A64"
ROSE = "#B46E82"
GRAY = "#64748B"
GRID = "#E8EDF2"
MODEL_ORDER = ["UNI", "UNI2-h", "Virchow2", "H-optimus-0", "H-optimus-1", "Phikon-v2"]
MODEL_COLORS = {
    "UNI": BLUE, "UNI2-h": TEAL, "Virchow2": ORANGE,
    "H-optimus-0": PURPLE, "H-optimus-1": GREEN, "Phikon-v2": ROSE,
}
SPLIT_ORDER = ["internal_val", "external_test"]
SPLIT_LABEL = {"internal_val": "内部验证", "external_test": "外部 XZY"}
METRICS = {
    "macro": {
        "title": "患者—通路等权 PCC",
        "mean": "pcc_macro_mean", "sd": "pcc_macro_sd",
        "seed_aliases": ["pcc_macro", "macro_pcc"],
    },
    "pooled": {
        "title": "整体展平 pooled PCC",
        "mean": "pcc_pooled_mean", "sd": "pcc_pooled_sd",
        "seed_aliases": ["pcc_pooled", "pooled_pcc"],
    },
}
BASE_EVIDENCE = "accepted_phase2_uni2h_fullfov_spatial"
TASK2_EVIDENCE = "task2_gene_smoke_accepted_scoped"
TARGET_BASE = "barcode_repair_v003_30d"
TARGET_TASK2 = "319_gene_candidate_reconstructed_30d_provenance_pending"

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 11,
    "axes.unicode_minus": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": GRAY,
    "ytick.color": INK,
    "axes.edgecolor": "#B5BFCA",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


MODEL_ALIASES = {
    token(alias): name
    for name, aliases in {
        "UNI": ["uni"],
        "UNI2-h": ["uni2h", "uni2-h", "uni2_h"],
        "Virchow2": ["virchow2"],
        "H-optimus-0": ["hoptimus0", "h-optimus-0", "h_optimus_0"],
        "H-optimus-1": ["hoptimus1", "h-optimus-1", "h_optimus_1"],
        "Phikon-v2": ["phikonv2", "phikon-v2", "phikon_v2"],
    }.items()
    for alias in aliases
}
SPLIT_ALIASES = {
    "internalval": "internal_val", "internalvalidation": "internal_val",
    "validation": "internal_val", "val": "internal_val",
    "externaltest": "external_test", "external": "external_test",
    "test": "external_test", "xzy": "external_test",
}


def canonical_model(value):
    key = token(value)
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unrecognized model in source CSV: {value!r}")
    return MODEL_ALIASES[key]


def canonical_split(value):
    key = token(value)
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unrecognized split in source CSV: {value!r}")
    return SPLIT_ALIASES[key]


def require_file(name):
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f"Required copied source table is missing: {path}")
    return path


def read_inputs():
    summary = pd.read_csv(require_file("six_model_summary.csv"))
    seeds_raw = pd.read_csv(require_file("six_model_per_seed.csv"))
    task2 = pd.read_csv(require_file("task2_summary.csv"))
    comparison = pd.read_csv(require_file("baseline_descriptive_comparison.csv"))
    return summary, seeds_raw, task2, comparison


def long_seed_table(frame):
    required = {"model", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Per-seed table needs columns {sorted(required)}")
    seed_col = next((c for c in ["seed", "training_seed", "run_seed"] if c in frame.columns), None)
    if seed_col is None:
        raise ValueError("Per-seed table needs a seed, training_seed, or run_seed column")
    frame = frame.copy()
    if "arm" in frame.columns:
        arms = set(frame["arm"].dropna().astype(str).str.lower())
        if arms and arms != {"spatial"}:
            raise ValueError(f"Expected the fixed spatial arm; found arms {sorted(arms)}")
    frame["model"] = frame["model"].map(canonical_model)
    frame["split"] = frame["split"].map(canonical_split)
    frame["seed"] = pd.to_numeric(frame[seed_col], errors="raise")
    if "metric" in frame.columns and "value" in frame.columns:
        metric_map = {
            "macropcc": "macro", "pccmacro": "macro",
            "pooledpcc": "pooled", "pccpooled": "pooled",
        }
        frame["metric_key"] = frame["metric"].map(lambda x: metric_map.get(token(x)))
        if frame["metric_key"].isna().any():
            raise ValueError("Per-seed table has an unrecognized metric value")
        out = frame[["model", "split", "seed", "metric_key", "value"]].copy()
        out["value"] = pd.to_numeric(out["value"], errors="raise")
        out = out.pivot(index=["model", "split", "seed"], columns="metric_key", values="value").reset_index()
        out.columns.name = None
    else:
        macro_col = next((c for c in METRICS["macro"]["seed_aliases"] if c in frame.columns), None)
        pooled_col = next((c for c in METRICS["pooled"]["seed_aliases"] if c in frame.columns), None)
        if macro_col is None or pooled_col is None:
            raise ValueError("Per-seed table needs both macro and pooled PCC columns")
        out = frame[["model", "split", "seed", macro_col, pooled_col]].copy()
        out = out.rename(columns={macro_col: "macro", pooled_col: "pooled"})
        out["macro"] = pd.to_numeric(out["macro"], errors="raise")
        out["pooled"] = pd.to_numeric(out["pooled"], errors="raise")
    if out.duplicated(["model", "split", "seed"]).any():
        raise ValueError("Per-seed table contains duplicate model/split/seed rows")
    if not np.isfinite(out[["macro", "pooled"]].to_numpy(dtype=float)).all():
        raise ValueError("Per-seed PCC values must all be finite")
    return out


def validate_sources(summary, seeds, task2, comparison):
    required = {
        "model", "split", "n_runs",
        "pcc_macro_mean", "pcc_macro_sd",
        "pcc_pooled_mean", "pcc_pooled_sd",
    }
    if not required.issubset(summary.columns):
        raise ValueError(f"Six-model summary needs columns {sorted(required)}")
    summary = summary.copy()
    summary["model"] = summary["model"].map(canonical_model)
    summary["split"] = summary["split"].map(canonical_split)
    summary["n_runs"] = pd.to_numeric(summary["n_runs"], errors="raise")
    for col in ["pcc_macro_mean", "pcc_macro_sd", "pcc_pooled_mean", "pcc_pooled_sd"]:
        summary[col] = pd.to_numeric(summary[col], errors="raise")
    if summary.duplicated(["model", "split"]).any():
        raise ValueError("Six-model summary contains duplicate model/split rows")
    if set(summary["model"]) != set(MODEL_ORDER) or set(summary["split"]) != set(SPLIT_ORDER) or len(summary) != 12:
        raise ValueError("Six-model summary does not contain exactly six models × two splits")
    if not np.isfinite(summary[["pcc_macro_mean", "pcc_macro_sd", "pcc_pooled_mean", "pcc_pooled_sd"]].to_numpy()).all():
        raise ValueError("Six-model summary contains non-finite PCC values")
    checks = []
    for _, row in summary.iterrows():
        group = seeds[(seeds.model == row.model) & (seeds.split == row.split)].sort_values("seed")
        if len(group) != 3 or group["seed"].nunique() != 3 or int(row.n_runs) != 3:
            raise ValueError(f"Expected three runs for {row.model}/{row.split}; found {len(group)} seed rows")
        for metric in ["macro", "pooled"]:
            source_mean = float(row[METRICS[metric]["mean"]])
            source_sd = float(row[METRICS[metric]["sd"]])
            computed_mean = float(group[metric].mean())
            computed_sd = float(group[metric].std(ddof=1))
            if not np.isclose(source_mean, computed_mean, rtol=1e-8, atol=1e-9):
                raise ValueError(f"{metric} mean mismatch for {row.model}/{row.split}")
            if not np.isclose(source_sd, computed_sd, rtol=1e-8, atol=1e-9):
                raise ValueError(f"{metric} sample SD mismatch for {row.model}/{row.split}")
            checks.append({
                "model": row.model, "split": row.split, "metric": metric,
                "n_runs": 3, "summary_mean": source_mean, "per_seed_mean": computed_mean,
                "summary_sample_sd": source_sd, "per_seed_sample_sd": computed_sd,
            })
    if len(seeds) != 36:
        raise ValueError(f"Expected 36 per-seed rows; found {len(seeds)}")

    task_required = {"split", "n_runs", "pcc_macro", "pcc_pooled"}
    if not task_required.issubset(task2.columns):
        raise ValueError(f"Task-2 summary needs columns {sorted(task_required)}")
    task2 = task2.copy()
    task2["split"] = task2["split"].map(canonical_split)
    task2["n_runs"] = pd.to_numeric(task2["n_runs"], errors="raise")
    for col in ["pcc_macro", "pcc_pooled"]:
        task2[col] = pd.to_numeric(task2[col], errors="raise")
    if task2.duplicated(["split"]).any() or set(task2["split"]) != set(SPLIT_ORDER) or len(task2) != 2:
        raise ValueError("Task-2 summary must contain one row for each of the two splits")
    if not (task2["n_runs"] == 1).all():
        raise ValueError("Task-2 route is expected to have one run in each split")
    if "sample_sd" in task2.columns and task2["sample_sd"].notna().any():
        raise ValueError("Task-2 n=1 rows must not supply a sample SD")
    if "training_seed" in task2.columns and task2["training_seed"].notna().any():
        # A seed value may be recorded, but the figure does not imply multiple runs.
        task2["training_seed"] = task2["training_seed"].astype(str)

    comp_required = {
        "split", "evidence", "target", "n_runs",
        "pcc_macro_mean", "pcc_macro_sample_sd",
        "pcc_pooled_mean", "pcc_pooled_sample_sd",
    }
    if not comp_required.issubset(comparison.columns):
        raise ValueError(f"Descriptive comparison needs columns {sorted(comp_required)}")
    comparison = comparison.copy()
    comparison["split"] = comparison["split"].map(canonical_split)
    comparison["n_runs"] = pd.to_numeric(comparison["n_runs"], errors="raise")
    for col in ["pcc_macro_mean", "pcc_pooled_mean"]:
        comparison[col] = pd.to_numeric(comparison[col], errors="raise")
    for evidence, expected_n, target in [
        (BASE_EVIDENCE, 3, TARGET_BASE), (TASK2_EVIDENCE, 1, TARGET_TASK2)
    ]:
        subset = comparison[comparison.evidence == evidence]
        if len(subset) != 2 or set(subset.split) != set(SPLIT_ORDER):
            raise ValueError(f"Descriptive comparison is missing rows for {evidence}")
        if not (subset.n_runs == expected_n).all() or set(subset.target) != {target}:
            raise ValueError(f"Unexpected target or run count for {evidence}")
    base = comparison[comparison.evidence == BASE_EVIDENCE]
    for _, row in base.iterrows():
        six = summary[(summary.model == "UNI2-h") & (summary.split == row.split)].iloc[0]
        for metric, mean_col, sd_col in [
            ("macro", "pcc_macro_mean", "pcc_macro_sample_sd"),
            ("pooled", "pcc_pooled_mean", "pcc_pooled_sample_sd"),
        ]:
            if not np.isclose(float(row[mean_col]), float(six[METRICS[metric]["mean"]]), rtol=1e-8, atol=1e-9):
                raise ValueError(f"UNI2-h baseline mean disagrees across supplied summaries ({row.split}/{metric})")
            if not np.isclose(float(row[sd_col]), float(six[METRICS[metric]["sd"]]), rtol=1e-8, atol=1e-9):
                raise ValueError(f"UNI2-h baseline sample SD disagrees across supplied summaries ({row.split}/{metric})")
        checks.append({
            "cross_check": "UNI2-h baseline_descriptive_comparison vs six_model_summary",
            "split": row.split, "n_runs": 3,
            "pcc_macro_mean": float(six["pcc_macro_mean"]),
            "pcc_macro_sample_sd": float(six["pcc_macro_sd"]),
            "pcc_pooled_mean": float(six["pcc_pooled_mean"]),
            "pcc_pooled_sample_sd": float(six["pcc_pooled_sd"]),
        })
    task_comp = comparison[comparison.evidence == TASK2_EVIDENCE]
    for _, row in task_comp.iterrows():
        source = task2[task2.split == row.split].iloc[0]
        if int(row.n_runs) != 1:
            raise ValueError("Task-2 comparison row must be n=1")
        for metric, comp_col, task_col, comp_sd_col in [
            ("macro", "pcc_macro_mean", "pcc_macro", "pcc_macro_sample_sd"),
            ("pooled", "pcc_pooled_mean", "pcc_pooled", "pcc_pooled_sample_sd"),
        ]:
            if not np.isclose(float(row[comp_col]), float(source[task_col]), rtol=1e-8, atol=1e-9):
                raise ValueError(f"Task-2 {metric} value disagrees with task2_summary.csv ({row.split})")
            if pd.notna(row[comp_sd_col]):
                raise ValueError("Task-2 n=1 comparison row must not supply a sample SD")
        checks.append({
            "cross_check": "task2_summary vs baseline_descriptive_comparison",
            "split": row.split, "target": row.target, "n_runs": 1,
            "pcc_macro": float(source["pcc_macro"]),
            "pcc_pooled": float(source["pcc_pooled"]),
            "sample_sd": None,
        })

    return summary, seeds, task2, comparison, checks


def metric_xlim(metric, value_sets, step=0.02):
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in value_sets])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, step * 2)
    pad = max(step * 0.65, span * 0.12)
    low = math.floor((lo - pad) / step) * step
    high = math.ceil((hi + pad) / step) * step
    return round(low, 6), round(high, 6)


def axes_style(ax, limits, step=0.02):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B5BFCA")
    ax.spines["bottom"].set_color("#B5BFCA")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=10)
    ax.tick_params(axis="x", length=3, pad=5, labelsize=9)
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("PCC", fontsize=10, labelpad=5)


def panel_title(ax, letter, title):
    ax.set_title(title, fontsize=11.3, loc="left", pad=12)
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=INK)


def seed_values(seeds, model, split, metric):
    values = seeds[(seeds.model == model) & (seeds.split == split)].sort_values("seed")
    return values[metric].to_numpy(dtype=float)


def plot_six_encoders(summary, seeds):
    fig = plt.figure(figsize=(12.4, 9.0))
    fig.text(0.065, 0.962, "图 P2-3S｜固定空间臂下的六种编码器表现",
             fontsize=18.5, fontweight="bold", color=INK)
    fig.text(0.065, 0.929, "内部验证与外部 XZY；每种编码器均为 n=3 次运行。",
             fontsize=10.8, color=GRAY)
    legend = [
        Line2D([0], [0], marker="o", color=BLUE, lw=1.4, markersize=6,
               markerfacecolor="white", markeredgewidth=1.4,
               label="均值 ± 样本 SD"),
        Line2D([0], [0], marker="o", color=BLUE, lw=0, markersize=4,
               markerfacecolor=BLUE, alpha=0.45, label="逐种子值"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.062, 0.900),
               frameon=False, ncol=2, handletextpad=0.5, columnspacing=1.8, fontsize=9.5)
    gs = fig.add_gridspec(2, 2, left=0.19, right=0.965, bottom=0.19, top=0.79,
                          hspace=0.44, wspace=0.32)
    axis_ranges = {}
    panel_letters = [["A", "B"], ["C", "D"]]
    for row, metric in enumerate(["macro", "pooled"]):
        metric_values = []
        for split in SPLIT_ORDER:
            for model in MODEL_ORDER:
                r = summary[(summary.model == model) & (summary.split == split)].iloc[0]
                metric_values += [
                    seed_values(seeds, model, split, metric),
                    [float(r[METRICS[metric]["mean"]]) - float(r[METRICS[metric]["sd"]]),
                     float(r[METRICS[metric]["mean"]]) + float(r[METRICS[metric]["sd"]])],
                ]
        limits = metric_xlim(metric, metric_values)
        axis_ranges[metric] = list(limits)
        for col, split in enumerate(SPLIT_ORDER):
            ax = fig.add_subplot(gs[row, col])
            panel_title(ax, panel_letters[row][col],
                        f"{SPLIT_LABEL[split]}\n{METRICS[metric]['title']}")
            axes_style(ax, limits)
            ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
            ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.55)
            for yi, model in enumerate(MODEL_ORDER):
                r = summary[(summary.model == model) & (summary.split == split)].iloc[0]
                values = seed_values(seeds, model, split, metric)
                color = MODEL_COLORS[model]
                # Small vertical offsets separate individual runs; they carry no data scale.
                offsets = np.array([-0.105, 0.0, 0.105])
                ax.scatter(values, yi + offsets, s=20, color=color, alpha=0.48,
                           zorder=3, edgecolors="none")
                ax.errorbar(
                    float(r[METRICS[metric]["mean"]]), yi,
                    xerr=float(r[METRICS[metric]["sd"]]),
                    fmt="o", markersize=6.2, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.5, color=color,
                    capsize=3.3, elinewidth=1.35, capthick=1.05, zorder=4,
                )
    fig.text(0.065, 0.073,
             "患者—通路等权 PCC 为逐患者逐通路相关后等权平均；整体展平 pooled PCC 为全部点位×通路展平后的一次相关。",
             fontsize=9.2, color=GRAY)
    fig.text(0.065, 0.047,
             "空心标记与横向误差线为均值 ± 样本 SD；浅色点为逐种子值，纵向错开仅用于区分。",
             fontsize=9.2, color=GRAY)
    save_figure(fig, "Fig_P2_3S_six_encoders")
    return axis_ranges


def plot_gene_route(summary, task2):
    fig = plt.figure(figsize=(12.4, 8.65))
    fig.text(0.065, 0.962, "图 P2-5｜基线通路路线与任务二基因路线",
             fontsize=18.5, fontweight="bold", color=INK)
    fig.text(0.065, 0.929,
             "UNI2-h 基线：既有 30 条通路目标，n=3；任务二：319 基因背景 ssGSEA 目标，n=1。",
             fontsize=10.4, color=GRAY)
    gs = fig.add_gridspec(2, 2, left=0.245, right=0.965, bottom=0.20, top=0.83,
                          hspace=0.46, wspace=0.33)
    axis_ranges = {}
    panel_letters = [["A", "B"], ["C", "D"]]
    ylabels = ["UNI2-h 基线 · n=3", "任务二基因路线 · n=1"]
    route_colors = [BLUE, ORANGE]
    for row, metric in enumerate(["macro", "pooled"]):
        metric_values = []
        for split in SPLIT_ORDER:
            baseline = summary[(summary.model == "UNI2-h") & (summary.split == split)].iloc[0]
            gene = task2[task2.split == split].iloc[0]
            metric_values.extend([
                [float(baseline[METRICS[metric]["mean"]]) - float(baseline[METRICS[metric]["sd"]]),
                 float(baseline[METRICS[metric]["mean"]]) + float(baseline[METRICS[metric]["sd"]])],
                [float(gene[f"pcc_{'macro' if metric == 'macro' else 'pooled'}"])],
            ])
        limits = metric_xlim(metric, metric_values)
        axis_ranges[metric] = list(limits)
        for col, split in enumerate(SPLIT_ORDER):
            ax = fig.add_subplot(gs[row, col])
            panel_title(ax, panel_letters[row][col],
                        f"{SPLIT_LABEL[split]}\n{METRICS[metric]['title']}")
            axes_style(ax, limits, step=0.05 if metric == "macro" else 0.10)
            ax.set_yticks([0, 1], ylabels)
            ax.tick_params(axis="y", labelsize=9.3)
            ax.set_ylim(1.5, -0.5)
            baseline = summary[(summary.model == "UNI2-h") & (summary.split == split)].iloc[0]
            gene = task2[task2.split == split].iloc[0]
            base_mean = float(baseline[METRICS[metric]["mean"]])
            base_sd = float(baseline[METRICS[metric]["sd"]])
            gene_value = float(gene[f"pcc_{'macro' if metric == 'macro' else 'pooled'}"])
            ax.errorbar(
                base_mean, 0, xerr=base_sd, fmt="o", markersize=6.2,
                markerfacecolor="white", markeredgecolor=route_colors[0],
                markeredgewidth=1.5, color=route_colors[0], capsize=3.3,
                elinewidth=1.35, capthick=1.05, zorder=4,
            )
            ax.annotate(
                f"{base_mean:.3f} ± {base_sd:.4f}", (base_mean, 0),
                xytext=(-7, 9), textcoords="offset points", ha="right", va="bottom",
                fontsize=8.2, color=route_colors[0],
            )
            ax.scatter([gene_value], [1], marker="D", s=34, color=route_colors[1],
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.annotate(
                f"{gene_value:.3f}", (gene_value, 1),
                xytext=(7, -10), textcoords="offset points", ha="left", va="top",
                fontsize=8.2, color=route_colors[1],
            )
    fig.text(0.065, 0.072,
             "患者—通路等权 PCC 为逐患者逐通路相关后等权平均；整体展平 pooled PCC 为全部点位×通路展平后的一次相关。",
             fontsize=9.1, color=GRAY)
    fig.text(0.065, 0.045,
             "UNI2-h 基线显示均值 ± 样本 SD；任务二路线显示单次运行值。",
             fontsize=9.1, color=GRAY)
    save_figure(fig, "Fig_P2_5_gene_route")
    return axis_ranges


def save_figure(fig, stem):
    fig.canvas.draw()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_gallery():
    items = [
        ("Fig_P2_3S_six_encoders", "图 P2-3S｜固定空间臂下的六种编码器表现",
         "两种 PCC 分别展示内部验证与外部 XZY；六编码器均为 n=3。"),
        ("Fig_P2_5_gene_route", "图 P2-5｜基线通路路线与任务二基因路线",
         "标签标出各自目标定义与重复数。"),
    ]
    cards = []
    for stem, title, note in items:
        cards.append(
            f'<article><h2>{title}</h2><p>{note}</p>'
            f'<a href="exports/{stem}.png"><img src="exports/{stem}.png" alt="{title}"></a>'
            f'<p class="links"><a href="exports/{stem}.png">PNG</a> · '
            f'<a href="exports/{stem}.svg">SVG</a> · '
            f'<a href="exports/{stem}.pdf">PDF</a></p></article>'
        )
    html = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 第二批静态图</title>
<style>
body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#243247;background:#fff}
h1{font-size:24px} article{margin:32px 0 48px} h2{font-size:18px;margin-bottom:6px}
p{color:#64748b}.links{font-size:14px}a{color:#3b77a5}img{display:block;width:100%;height:auto;border:1px solid #e8edf2}
</style><body><h1>Phase 2 第二批静态科研图</h1>""" + "".join(cards) + "</body></html>"
    (ROOT / "图集.html").write_text(html, encoding="utf-8")


def main():
    summary_raw, seeds_raw, task2_raw, comparison_raw = read_inputs()
    seeds = long_seed_table(seeds_raw)
    summary, seeds, task2, comparison, checks = validate_sources(
        summary_raw, seeds, task2_raw, comparison_raw
    )
    fig1_ranges = plot_six_encoders(summary, seeds)
    fig2_ranges = plot_gene_route(summary, task2)
    write_gallery()
    render = {
        "status": "numeric_pass_visual_review_pending",
        "source_rows": {
            "six_model_summary": int(len(summary)),
            "six_model_per_seed": int(len(seeds)),
            "task2_summary": int(len(task2)),
            "baseline_descriptive_comparison": int(len(comparison)),
        },
        "numerical_checks": checks,
        "shared_pcc_axis_ranges": {
            "Fig_P2_3S_six_encoders": fig1_ranges,
            "Fig_P2_5_gene_route": fig2_ranges,
        },
        "rendered_formats": ["PNG (300 dpi)", "SVG (editable text)", "PDF (vector)"],
        "visual_review": {
            "status": "pending",
            "items": ["文字重叠", "文字越界或裁切", "图例与误差线可读性", "数值与面板标签"],
        },
    }
    (ROOT / "render_checks.json").write_text(
        json.dumps(render, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Rendered two figures after numerical checks; manual image review is pending.")


if __name__ == "__main__":
    main()
