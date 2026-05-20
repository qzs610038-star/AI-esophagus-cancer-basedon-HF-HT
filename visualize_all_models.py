"""
visualize_all_models.py
=======================
5 模型综合可视化对比报告

生成 2 张综合对比图：
  图1：单患者训练综合报告（2×2）
  图2：跨患者泛化综合报告（2×2）

5 模型：HisToGene、HisToGene-UNI、EGN-v1、EGN-v2、EGN-v2+UNI
配色：蓝、绿、红、橙、紫

使用方式：
  python visualize_all_models.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# 字体配置
# ─────────────────────────────────────────────
def _setup_font():
    """配置中文字体，优先 SimHei"""
    for font_name in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
        try:
            fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用 {font_name}")
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("[字体] 未找到中文字体，回退到 DejaVu Sans")

_setup_font()

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
from config_utils import get_project_root

BASE_DIR = get_project_root()
OUTPUT_DIR = os.path.join(BASE_DIR, "histogene", "checkpoints", "results_vis", "AllModels_comparison")

# 5 色方案
MODELS = ["HisToGene", "HisToGene-UNI", "EGN-v1", "EGN-v2", "EGN-v2+UNI"]
MODEL_COLORS = {
    "HisToGene":     "#4472C4",  # 蓝
    "HisToGene-UNI": "#70AD47",  # 绿
    "EGN-v1":        "#C00000",  # 红
    "EGN-v2":        "#ED7D31",  # 橙
    "EGN-v2+UNI":    "#7030A0",  # 紫
}

# 参数量（M）— 硬编码
PARAM_M = {
    "HisToGene":     70.6,
    "HisToGene-UNI": 4.0,
    "EGN-v1":        6.8,
    "EGN-v2":        3.0,
    "EGN-v2+UNI":    2.8,
}

DATASETS = ["HYZ15040", "JFX0729", "LMZ12939"]
DATASET_LABELS = ["HYZ15040", "JFX0729", "LMZ12939"]

# 跨患者模型（无 EGN-v1）
CROSS_MODELS = ["HisToGene", "HisToGene-UNI", "EGN-v2", "EGN-v2+UNI"]

# ─────────────────────────────────────────────
# 数据读取工具
# ─────────────────────────────────────────────
def _find_csv(directory, pattern):
    """在目录下查找匹配的CSV文件，返回第一个找到的路径或None"""
    matches = glob.glob(os.path.join(directory, pattern))
    if matches:
        return matches[0]
    return None


def _read_best_metrics(csv_path):
    """读取CSV，返回 best epoch（val_loss最小）对应的各项指标"""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    best_idx = df["val_loss"].idxmin()
    row = df.loc[best_idx]
    return {
        "epoch":     int(row["epoch"]),
        "train_pcc": float(row["train_pcc"]),
        "train_r2":  float(row["train_r2"]),
        "train_loss": float(row["train_loss"]),
        "val_pcc":   float(row["val_pcc"]),
        "val_r2":    float(row["val_r2"]),
        "val_loss":  float(row["val_loss"]),
    }


def _read_full_history(csv_path):
    """读取完整训练历史 DataFrame"""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _safe_mean(lst):
    """计算非None值的均值"""
    vals = [v for v in lst if v is not None]
    return np.mean(vals) if vals else None


# ─────────────────────────────────────────────
# 收集单患者训练数据
# ─────────────────────────────────────────────
def collect_single_patient_data():
    """从CSV文件中读取5模型×3数据集的最佳指标"""
    data = {}  # data[model][dataset] = metrics_dict

    # ── HisToGene ──
    data["HisToGene"] = {}
    for ds in DATASETS:
        csv_path = _find_csv(os.path.join(BASE_DIR, "histogene"), f"training_history_{ds}.csv")
        # 排除 UNI 文件（确保不含 _UNI）
        if csv_path and "_UNI" in os.path.basename(csv_path):
            csv_path = None
            # 搜索不含UNI的精确文件
            exact = os.path.join(BASE_DIR, "histogene", f"training_history_{ds}.csv")
            if os.path.exists(exact):
                csv_path = exact
        data["HisToGene"][ds] = _read_best_metrics(csv_path)

    # ── HisToGene-UNI ──
    data["HisToGene-UNI"] = {}
    for ds in DATASETS:
        # 优先查找 _UNI.csv，其次 _UNI_fixed.csv
        csv_path = _find_csv(os.path.join(BASE_DIR, "histogene"), f"training_history_{ds}_UNI.csv")
        if csv_path is None:
            csv_path = _find_csv(os.path.join(BASE_DIR, "histogene"), f"training_history_{ds}_UNI_fixed.csv")
        data["HisToGene-UNI"][ds] = _read_best_metrics(csv_path)

    # ── EGN-v1 ──
    data["EGN-v1"] = {}
    for ds in DATASETS:
        csv_path = _find_csv(os.path.join(BASE_DIR, "egnv1"), f"training_history_{ds}.csv")
        data["EGN-v1"][ds] = _read_best_metrics(csv_path)

    # ── EGN-v2 ──
    data["EGN-v2"] = {}
    for ds in DATASETS:
        csv_path = _find_csv(os.path.join(BASE_DIR, "egnv2"), f"training_history_{ds}.csv")
        if csv_path and "_UNI" in os.path.basename(csv_path):
            csv_path = None
            exact = os.path.join(BASE_DIR, "egnv2", f"training_history_{ds}.csv")
            if os.path.exists(exact):
                csv_path = exact
        data["EGN-v2"][ds] = _read_best_metrics(csv_path)

    # ── EGN-v2+UNI ──
    data["EGN-v2+UNI"] = {}
    for ds in DATASETS:
        csv_path = _find_csv(os.path.join(BASE_DIR, "egnv2"), f"training_history_{ds}_UNI.csv")
        data["EGN-v2+UNI"][ds] = _read_best_metrics(csv_path)

    # 打印收集到的数据
    for model in MODELS:
        print(f"\n[数据] {model}:")
        for ds in DATASETS:
            m = data[model].get(ds)
            if m:
                print(f"  {ds}: val_pcc={m['val_pcc']:.4f}, val_r2={m['val_r2']:.4f}, "
                      f"val_loss={m['val_loss']:.4f}, train_pcc={m['train_pcc']:.4f}, "
                      f"best_epoch={m['epoch']}")
            else:
                print(f"  {ds}: 无数据")

    return data


# ─────────────────────────────────────────────
# 收集跨患者泛化数据
# ─────────────────────────────────────────────
def collect_cross_patient_data(single_data):
    """读取4个模型的跨患者泛化数据"""
    cross_data = {}

    # HisToGene 原版跨患者
    csv_path = _find_csv(os.path.join(BASE_DIR, "histogene"),
                         "training_history_CrossPatient_JFX_LMZ_to_HYZ_orig.csv")
    cross_data["HisToGene"] = _read_best_metrics(csv_path)
    cross_data["HisToGene"] = _read_best_metrics(csv_path)
    if cross_data["HisToGene"]:
        cross_data["HisToGene"]["csv_path"] = csv_path

    # HisToGene-UNI 跨患者
    csv_path = _find_csv(os.path.join(BASE_DIR, "histogene"),
                         "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv")
    # 确保不是 _orig 文件
    if csv_path and "_orig" in os.path.basename(csv_path):
        exact = os.path.join(BASE_DIR, "histogene", "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv")
        if os.path.exists(exact):
            csv_path = exact
    cross_data["HisToGene-UNI"] = _read_best_metrics(csv_path)
    if cross_data["HisToGene-UNI"]:
        cross_data["HisToGene-UNI"]["csv_path"] = csv_path

    # EGN-v2 跨患者
    csv_path = _find_csv(os.path.join(BASE_DIR, "egnv2"),
                         "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv")
    if csv_path and "_UNI" in os.path.basename(csv_path):
        exact = os.path.join(BASE_DIR, "egnv2", "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv")
        if os.path.exists(exact):
            csv_path = exact
    cross_data["EGN-v2"] = _read_best_metrics(csv_path)
    if cross_data["EGN-v2"]:
        cross_data["EGN-v2"]["csv_path"] = csv_path

    # EGN-v2+UNI 跨患者
    csv_path = _find_csv(os.path.join(BASE_DIR, "egnv2"),
                         "training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv")
    cross_data["EGN-v2+UNI"] = _read_best_metrics(csv_path)
    if cross_data["EGN-v2+UNI"]:
        cross_data["EGN-v2+UNI"]["csv_path"] = csv_path

    print("\n[跨患者数据]:")
    for model in CROSS_MODELS:
        m = cross_data.get(model)
        if m:
            print(f"  {model}: test_pcc(val_pcc)={m['val_pcc']:.4f}, test_r2={m['val_r2']:.4f}, "
                  f"test_loss={m['val_loss']:.4f}, train_pcc={m['train_pcc']:.4f}, "
                  f"best_epoch={m['epoch']}")
        else:
            print(f"  {model}: 无数据")

    return cross_data


# ═════════════════════════════════════════════
# 图1：单患者训练综合报告
# ═════════════════════════════════════════════

def plot_single_patient_figure(single_data):
    """生成单患者训练综合报告 2×2 图"""

    # ── 提取数据 ──
    val_pcc_grid = {}   # model -> [HYZ, JFX, LMZ]
    val_r2_grid = {}
    val_loss_grid = {}
    train_pcc_grid = {}

    for model in MODELS:
        val_pcc_grid[model] = [single_data[model][ds]["val_pcc"] if single_data[model][ds] else None for ds in DATASETS]
        val_r2_grid[model] = [single_data[model][ds]["val_r2"] if single_data[model][ds] else None for ds in DATASETS]
        val_loss_grid[model] = [single_data[model][ds]["val_loss"] if single_data[model][ds] else None for ds in DATASETS]
        train_pcc_grid[model] = [single_data[model][ds]["train_pcc"] if single_data[model][ds] else None for ds in DATASETS]

    # ── 创建画布 ──
    fig = plt.figure(figsize=(18, 15), dpi=300)
    fig.suptitle(
        "五模型单患者训练综合对比报告",
        fontsize=20, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.945,
        "数据集：食管癌 | 模型：HisToGene / HisToGene-UNI / EGN-v1 / EGN-v2 / EGN-v2+UNI",
        ha="center", fontsize=10, color="gray",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # ──────── 子图1.1：Val PCC 分组柱状图 ────────
    _plot_val_pcc_bar(ax1, val_pcc_grid)

    # ──────── 子图1.2：模型综合性能雷达图 ────────
    _plot_radar_single(ax2, val_pcc_grid, val_r2_grid, val_loss_grid, train_pcc_grid)

    # ──────── 子图1.3：过拟合分析散点图 ────────
    _plot_overfit_scatter(ax3, train_pcc_grid, val_pcc_grid)

    # ──────── 子图1.4：参数量 vs 性能气泡图 ────────
    _plot_param_bubble(ax4, val_pcc_grid, val_r2_grid)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "all_models_single_patient.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"\n[完成] 图1已保存到: {out_path}")


def _plot_val_pcc_bar(ax, val_pcc_grid):
    """子图1.1：Val PCC 分组柱状图"""
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    bar_width = 0.14
    x = np.arange(n_datasets)

    for i, model in enumerate(MODELS):
        offset = (i - n_models / 2 + 0.5) * bar_width
        vals = val_pcc_grid[model]
        bars = ax.bar(
            x + offset, [v if v is not None else 0 for v in vals],
            width=bar_width,
            color=MODEL_COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold",
                    color=MODEL_COLORS[model],
                )
            else:
                rect = FancyBboxPatch(
                    (bar.get_x() + 0.01, 0.01),
                    bar.get_width() - 0.02, 0.05,
                    boxstyle="round,pad=0.01",
                    linewidth=1.2, edgecolor="gray",
                    facecolor="#f0f0f0", linestyle="--", zorder=4,
                )
                ax.add_patch(rect)
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 0.035,
                    "N/A", ha="center", va="center",
                    fontsize=6, color="gray", style="italic", zorder=5,
                )
                bar.set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=10)
    ax.set_ylabel("Val PCC", fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.80, 0.1))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    ax.set_title("① 各模型在三个数据集上的验证集 PCC", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_radar_single(ax, val_pcc_grid, val_r2_grid, val_loss_grid, train_pcc_grid):
    """子图1.2：模型综合性能雷达图"""
    cat_keys = ["Val PCC", "Val R2", "1-Val Loss", "参数效率", "过拟合控制"]
    cat_labels = [
        "Val PCC\n(3集均值)", "Val R2\n(3集均值)", "1-Val Loss\n(3集均值)",
        "参数效率\n(PCC/参数M)", "过拟合控制\n(1-|gap|/max_gap)"
    ]
    n_cats = len(cat_keys)

    # 计算各模型各维度值
    model_vals = {}
    max_gap = 0
    for model in MODELS:
        mean_pcc = _safe_mean(val_pcc_grid[model])
        mean_r2 = _safe_mean(val_r2_grid[model])
        mean_loss = _safe_mean(val_loss_grid[model])
        mean_train = _safe_mean(train_pcc_grid[model])
        gap = abs(mean_train - mean_pcc) if mean_train is not None and mean_pcc is not None else 0
        max_gap = max(max_gap, gap)
        pcc_per_param = mean_pcc / PARAM_M[model] if mean_pcc else 0
        model_vals[model] = {
            "Val PCC": mean_pcc if mean_pcc else 0,
            "Val R2": mean_r2 if mean_r2 else 0,
            "1-Val Loss": (1 - mean_loss) if mean_loss else 0,
            "参数效率": pcc_per_param,
            "过拟合控制": 1 - gap / max_gap if max_gap > 0 else 1,  # 先存原始值
        }

    # 归一化到 0-1
    all_vals = {key: [model_vals[m][key] for m in MODELS] for key in cat_keys}
    norm_ranges = {}
    for key in cat_keys:
        vals = all_vals[key]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:
            norm_ranges[key] = (v_min - 0.1, v_max + 0.1)
        else:
            margin = (v_max - v_min) * 0.1
            norm_ranges[key] = (v_min - margin, v_max + margin)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6, color="gray")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=7)

    for model in MODELS:
        norm_vals = []
        for key in cat_keys:
            raw = model_vals[model][key]
            lo, hi = norm_ranges[key]
            norm_v = (raw - lo) / (hi - lo) if hi != lo else 0.5
            norm_vals.append(norm_v)
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=4,
                color=MODEL_COLORS[model], label=model, zorder=3)
        ax.fill(angles, norm_vals, alpha=0.06, color=MODEL_COLORS[model])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=7, framealpha=0.9)
    ax.set_title("② 模型综合性能雷达图", fontsize=12, fontweight="bold", pad=20)


def _plot_overfit_scatter(ax, train_pcc_grid, val_pcc_grid):
    """子图1.3：过拟合分析散点图"""
    markers = {"HYZ15040": "o", "JFX0729": "s", "LMZ12939": "^"}

    lim_min, lim_max = -0.05, 1.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1, label="理想拟合 (y=x)")

    # 过拟合区域填充
    ax.fill_between(
        [lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
        alpha=0.04, color="red", zorder=0,
    )
    ax.text(0.85, 0.12, "过拟合区", fontsize=8, color="red", alpha=0.5,
            ha="center", style="italic", transform=ax.transAxes)

    for model in MODELS:
        for j, ds in enumerate(DATASETS):
            tr = train_pcc_grid[model][j]
            va = val_pcc_grid[model][j]
            if tr is None or va is None:
                continue
            ax.scatter(
                tr, va,
                marker=markers[ds], s=90,
                c=MODEL_COLORS[model],
                edgecolors="white", linewidths=0.8,
                zorder=3, alpha=0.85,
            )

    # 手动构建图例
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m],
               markersize=8, label=m) for m in MODELS
    ]
    ds_handles = [
        Line2D([0], [0], marker=markers[ds], color="w",
               markerfacecolor="gray", markersize=7, label=ds) for ds in DATASETS
    ]
    legend1 = ax.legend(handles=model_handles, loc="upper left", fontsize=7,
                        title="模型", title_fontsize=7, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=ds_handles, loc="lower right", fontsize=6,
              title="数据集", title_fontsize=6, framealpha=0.9)

    ax.set_xlabel("Train PCC", fontsize=10)
    ax.set_ylabel("Val PCC", fontsize=10)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("③ 过拟合分析（Train PCC vs Val PCC）", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_param_bubble(ax, val_pcc_grid, val_r2_grid):
    """子图1.4：参数量 vs 性能气泡图"""
    for model in MODELS:
        mean_pcc = _safe_mean(val_pcc_grid[model])
        mean_r2 = _safe_mean(val_r2_grid[model])
        p = PARAM_M[model]
        if mean_pcc is None:
            continue
        bubble_size = max(mean_r2 * 1500, 50) if mean_r2 else 50

        ax.scatter(
            p, mean_pcc,
            s=bubble_size,
            c=MODEL_COLORS[model],
            alpha=0.7, edgecolors="white", linewidths=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{model}\n(Val PCC={mean_pcc:.3f})",
            (p, mean_pcc),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center", fontsize=7, fontweight="bold",
            color=MODEL_COLORS[model],
        )

    ax.annotate(
        "← 理想区域\n(参数少 · 性能高)",
        xy=(4, 0.55), fontsize=8, color="gray", style="italic",
        ha="left", va="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("参数量 (M)", fontsize=10)
    ax.set_ylabel("平均 Val PCC", fontsize=10)
    ax.set_xlim(1.5, 120)
    ax.set_ylim(0.15, 0.65)
    ax.set_title("④ 参数量 vs 性能气泡图", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 气泡大小图例
    for r2_val, label in [(0.05, "R2~0.05"), (0.20, "R2~0.20"), (0.35, "R2~0.35")]:
        ax.scatter([], [], s=max(r2_val * 1500, 50), c="gray", alpha=0.3,
                   edgecolors="white", linewidths=1, label=label)
    ax.legend(loc="lower right", fontsize=7, title="气泡大小=平均Val R2",
              title_fontsize=7, framealpha=0.9, scatterpoints=1)


# ═════════════════════════════════════════════
# 图2：跨患者泛化综合报告
# ═════════════════════════════════════════════

def plot_cross_patient_figure(single_data, cross_data):
    """生成跨患者泛化综合报告 2×2 图"""

    fig = plt.figure(figsize=(18, 15), dpi=300)
    fig.suptitle(
        "跨患者泛化综合对比报告",
        fontsize=20, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.945,
        "训练集：JFX0729 + LMZ12939 → 测试集：HYZ15040 | 模型：HisToGene / HisToGene-UNI / EGN-v2 / EGN-v2+UNI",
        ha="center", fontsize=10, color="gray",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # ──────── 子图2.1：单患者 vs 跨患者 PCC 衰减对比 ────────
    _plot_pcc_decay_bar(ax1, single_data, cross_data)

    # ──────── 子图2.2：跨患者泛化多指标雷达图 ────────
    _plot_radar_cross(ax2, single_data, cross_data)

    # ──────── 子图2.3：跨患者训练曲线对比 ────────
    _plot_cross_training_curves(ax3, cross_data)

    # ──────── 子图2.4：UNI特征提升效果对比 ────────
    _plot_uni_improvement(ax4, single_data, cross_data)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "all_models_cross_patient.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"[完成] 图2已保存到: {out_path}")


def _plot_pcc_decay_bar(ax, single_data, cross_data):
    """子图2.1：单患者 vs 跨患者 PCC 衰减对比"""
    x = np.arange(len(CROSS_MODELS))
    bar_width = 0.30

    # 单患者 HYZ15040 的 Val PCC
    single_pccs = []
    cross_pccs = []
    for model in CROSS_MODELS:
        sp = single_data[model].get("HYZ15040")
        cp = cross_data.get(model)
        single_pccs.append(sp["val_pcc"] if sp else 0)
        cross_pccs.append(cp["val_pcc"] if cp else 0)

    decay_pcts = []
    for s, c in zip(single_pccs, cross_pccs):
        if s > 0:
            decay_pcts.append((c - s) / s * 100)
        else:
            decay_pcts.append(0)

    for i, model in enumerate(CROSS_MODELS):
        # 深色=单患者，浅色=跨患者
        dark_color = MODEL_COLORS[model]
        # 生成浅色
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(dark_color)
        light_color = mcolors.to_hex(tuple(min(1, c + 0.35) for c in rgb))

        bar_s = ax.bar(
            x[i] - bar_width / 2, single_pccs[i],
            width=bar_width, color=dark_color,
            edgecolor="white", linewidth=0.8, zorder=3,
            label="单患者 Val PCC" if i == 0 else None,
        )
        bar_c = ax.bar(
            x[i] + bar_width / 2, cross_pccs[i],
            width=bar_width, color=light_color,
            edgecolor="white", linewidth=0.8, zorder=3,
            label="跨患者 Test PCC" if i == 0 else None,
        )
        bs = bar_s[0]
        bc = bar_c[0]
        ax.text(
            bs.get_x() + bs.get_width() / 2, bs.get_height() + 0.008,
            f"{single_pccs[i]:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=dark_color,
        )
        ax.text(
            bc.get_x() + bc.get_width() / 2, bc.get_height() + 0.008,
            f"{cross_pccs[i]:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=light_color,
        )
        # 衰减百分比
        mid_x = (bs.get_x() + bc.get_x() + bc.get_width()) / 2
        top_y = max(single_pccs[i], cross_pccs[i]) + 0.045
        ax.text(
            mid_x, top_y,
            f"{decay_pcts[i]:+.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF2F2",
                      edgecolor="red", linewidth=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CROSS_MODELS, fontsize=9, fontweight="bold")
    ax.set_ylabel("PCC", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_yticks(np.arange(0, 0.75, 0.1))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("① 单患者 vs 跨患者 PCC 性能衰减对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_radar_cross(ax, single_data, cross_data):
    """子图2.2：跨患者泛化多指标雷达图"""
    cat_keys = ["Test PCC", "Test R2", "1-Test Loss", "参数效率", "过拟合控制"]
    cat_labels = [
        "Test PCC", "Test R2", "1-Test Loss",
        "参数效率\n(PCC/参数M)", "过拟合控制\n(1-gap/max_gap)"
    ]
    n_cats = len(cat_keys)

    # 计算各维度原始值
    model_vals = {}
    max_gap = 0
    for model in CROSS_MODELS:
        cp = cross_data.get(model)
        sp = single_data[model].get("HYZ15040")
        if cp is None or sp is None:
            model_vals[model] = {key: 0 for key in cat_keys}
            continue
        test_pcc = cp["val_pcc"]
        test_r2 = cp["val_r2"]
        test_loss = cp["val_loss"]
        train_pcc = cp["train_pcc"]
        gap = abs(train_pcc - test_pcc)
        max_gap = max(max_gap, gap)
        model_vals[model] = {
            "Test PCC": test_pcc,
            "Test R2": test_r2,
            "1-Test Loss": 1 - test_loss,
            "参数效率": test_pcc / PARAM_M[model],
            "过拟合控制": 1 - gap,  # 后续再归一化
        }

    # 归一化到 0-1
    all_vals = {key: [model_vals[m][key] for m in CROSS_MODELS] for key in cat_keys}
    norm_ranges = {}
    for key in cat_keys:
        vals = all_vals[key]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:
            norm_ranges[key] = (v_min - 0.1, v_max + 0.1)
        else:
            margin = (v_max - v_min) * 0.1
            norm_ranges[key] = (v_min - margin, v_max + margin)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6, color="gray")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=7)

    for model in CROSS_MODELS:
        norm_vals = []
        for key in cat_keys:
            raw = model_vals[model][key]
            lo, hi = norm_ranges[key]
            norm_v = (raw - lo) / (hi - lo) if hi != lo else 0.5
            norm_vals.append(norm_v)
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=4,
                color=MODEL_COLORS[model], label=model, zorder=3)
        ax.fill(angles, norm_vals, alpha=0.06, color=MODEL_COLORS[model])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=7, framealpha=0.9)
    ax.set_title("② 跨患者泛化多指标雷达图", fontsize=12, fontweight="bold", pad=20)


def _plot_cross_training_curves(ax, cross_data):
    """子图2.3：跨患者训练曲线对比"""
    # 跨患者CSV路径
    csv_paths = {
        "HisToGene": os.path.join(BASE_DIR, "histogene",
                                  "training_history_CrossPatient_JFX_LMZ_to_HYZ_orig.csv"),
        "HisToGene-UNI": os.path.join(BASE_DIR, "histogene",
                                       "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
        "EGN-v2": os.path.join(BASE_DIR, "egnv2",
                                "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
        "EGN-v2+UNI": os.path.join(BASE_DIR, "egnv2",
                                     "training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv"),
    }

    ax2 = ax.twinx()

    for model in CROSS_MODELS:
        csv_path = csv_paths.get(model)
        if csv_path is None or not os.path.exists(csv_path):
            print(f"[警告] 未找到训练历史文件: {csv_path}")
            continue

        df = _read_full_history(csv_path)
        if df is None:
            continue

        epochs = df["epoch"].values
        test_pccs = df["val_pcc"].values  # 跨患者中 val_pcc 即 test_pcc

        # 生成浅色用于 loss 曲线
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(MODEL_COLORS[model])
        light_color = mcolors.to_hex(tuple(min(1, c + 0.35) for c in rgb))

        # Test PCC 曲线（左Y轴）
        ax.plot(epochs, test_pccs, "-o", linewidth=1.8, markersize=3,
                color=MODEL_COLORS[model], label=f"{model} Test PCC", zorder=3)

        # Train Loss 曲线（右Y轴）
        ax2.plot(epochs, df["train_loss"].values, "--", linewidth=1.2, markersize=2,
                 color=light_color, label=f"{model} Train Loss", zorder=2, alpha=0.7)

        # 标注最佳 Epoch
        cp = cross_data.get(model)
        if cp:
            best_ep = cp["epoch"]
            best_idx = np.where(epochs == best_ep)[0]
            if len(best_idx) > 0:
                best_pcc = test_pccs[best_idx[0]]
                ax.axvline(x=best_ep, color=MODEL_COLORS[model], linestyle=":", alpha=0.5, linewidth=1)
                ax.annotate(
                    f"Best Ep.{best_ep}\nPCC={best_pcc:.4f}",
                    xy=(best_ep, best_pcc),
                    xytext=(best_ep + 2, best_pcc + 0.03),
                    fontsize=6, color=MODEL_COLORS[model], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[model], lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=MODEL_COLORS[model], alpha=0.8),
                )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Test PCC", fontsize=10, color="#4472C4")
    ax2.set_ylabel("Train Loss", fontsize=10, color="gray")
    ax.tick_params(axis="y", labelcolor="#4472C4")
    ax2.tick_params(axis="y", labelcolor="gray")

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=6, framealpha=0.9)

    ax.set_title("③ 跨患者训练曲线对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)


def _plot_uni_improvement(ax, single_data, cross_data):
    """子图2.4：UNI特征提升效果对比"""
    # 4组对比：HisToGene vs HisToGene-UNI (单患者+跨患者), EGN-v2 vs EGN-v2+UNI (单患者+跨患者)
    groups = ["HisToGene\n→ HisToGene-UNI", "EGN-v2\n→ EGN-v2+UNI"]
    pair_models = [
        ("HisToGene", "HisToGene-UNI"),
        ("EGN-v2", "EGN-v2+UNI"),
    ]

    bar_width = 0.18
    x = np.arange(len(groups))

    # 4种柱子：原版单患者、UNI单患者、原版跨患者、UNI跨患者
    orig_single_pccs = []
    uni_single_pccs = []
    orig_cross_pccs = []
    uni_cross_pccs = []

    for orig_model, uni_model in pair_models:
        # 单患者 HYZ15040
        sp_orig = single_data[orig_model].get("HYZ15040")
        sp_uni = single_data[uni_model].get("HYZ15040")
        orig_single_pccs.append(sp_orig["val_pcc"] if sp_orig else 0)
        uni_single_pccs.append(sp_uni["val_pcc"] if sp_uni else 0)

        # 跨患者
        cp_orig = cross_data.get(orig_model)
        cp_uni = cross_data.get(uni_model)
        orig_cross_pccs.append(cp_orig["val_pcc"] if cp_orig else 0)
        uni_cross_pccs.append(cp_uni["val_pcc"] if cp_uni else 0)

    # 绘制
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    labels = ["原版-单患者", "UNI-单患者", "原版-跨患者", "UNI-跨患者"]
    data_sets = [orig_single_pccs, uni_single_pccs, orig_cross_pccs, uni_cross_pccs]
    hatch_patterns = ["", "///", "", "///"]
    edge_styles = ["-", "-", "--", "--"]

    # 配色
    bar_colors = ["#4472C4", "#70AD47", "#9DC3E6", "#A9D18E"]  # 蓝/绿/浅蓝/浅绿

    for i, (offset, data, label, hatch) in enumerate(zip(offsets, data_sets, labels, hatch_patterns)):
        bars = ax.bar(
            x + offset, data,
            width=bar_width, color=bar_colors[i],
            edgecolor="white", linewidth=0.8, zorder=3,
            label=label, hatch=hatch,
        )
        for j, (bar, v) in enumerate(zip(bars, data)):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold",
                )

    # 标注提升百分比
    for j in range(len(groups)):
        # 单患者提升
        if orig_single_pccs[j] > 0:
            pct_single = (uni_single_pccs[j] - orig_single_pccs[j]) / orig_single_pccs[j] * 100
            top_y = max(orig_single_pccs[j], uni_single_pccs[j]) + 0.04
            ax.text(
                x[j] - bar_width, top_y,
                f"↑{pct_single:+.1f}%",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="green",
            )
        # 跨患者提升
        if orig_cross_pccs[j] > 0:
            pct_cross = (uni_cross_pccs[j] - orig_cross_pccs[j]) / orig_cross_pccs[j] * 100
            top_y = max(orig_cross_pccs[j], uni_cross_pccs[j]) + 0.04
            ax.text(
                x[j] + bar_width, top_y,
                f"↑{pct_cross:+.1f}%",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="green",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, fontweight="bold")
    ax.set_ylabel("PCC", fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.80, 0.1))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    ax.set_title("④ UNI特征提升效果对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("五模型综合可视化对比报告")
    print("=" * 60)

    # 1. 收集单患者数据
    print("\n[步骤1] 收集单患者训练数据...")
    single_data = collect_single_patient_data()

    # 2. 收集跨患者数据
    print("\n[步骤2] 收集跨患者泛化数据...")
    cross_data = collect_cross_patient_data(single_data)

    # 3. 生成图1：单患者训练综合报告
    print("\n[步骤3] 生成图1：单患者训练综合报告...")
    plot_single_patient_figure(single_data)

    # 4. 生成图2：跨患者泛化综合报告
    print("\n[步骤4] 生成图2：跨患者泛化综合报告...")
    plot_cross_patient_figure(single_data, cross_data)

    print("\n" + "=" * 60)
    print("全部完成！输出目录：")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
