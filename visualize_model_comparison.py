"""
visualize_model_comparison.py
=============================
多模型训练结果对比可视化脚本（组会汇报用）

生成 2×2 综合对比图表，包含：
  1. Val PCC 分组柱状图
  2. 模型平均性能雷达图
  3. 过拟合分析散点图（Train PCC vs Val PCC）
  4. 参数量 vs 性能气泡图

使用方式：
  python visualize_model_comparison.py
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────
# 字体配置：优先使用 Microsoft YaHei，回退到 SimHei
# ─────────────────────────────────────────────
def _setup_font():
    """配置中文字体，避免乱码"""
    for font_name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
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
# 数据定义
# ─────────────────────────────────────────────
datasets = ["HYZ15040", "JFX0729", "LMZ12939", "MultiPatient_3ST"]
dataset_labels = ["HYZ15040", "JFX0729", "LMZ12939", "Multi\nPatient_3ST"]

models = ["HisToGene", "HisToGene-UNI", "EGN-v1", "EGN-v2"]
model_colors = {
    "HisToGene":     "#4472C4",  # 蓝色
    "HisToGene-UNI": "#70AD47",  # 绿色
    "EGN-v1":        "#C00000",  # 红色
    "EGN-v2":        "#ED7D31",  # 橙色
}

# Val PCC
val_pcc = {
    "HisToGene":     [0.5164, 0.6041, 0.5287, 0.5569],
    "HisToGene-UNI": [0.5336, 0.6114, 0.5385, None],  # MultiPatient无数据
    "EGN-v1":        [0.2289, 0.3141, 0.2165, 0.2460],
    "EGN-v2":        [0.4048, 0.4449, 0.3832, 0.4250],
}

# Val R²
val_r2 = {
    "HisToGene":     [0.2257, 0.3521, 0.2533, 0.3027],
    "HisToGene-UNI": [0.2821, 0.3742, 0.2781, None],
    "EGN-v1":        [0.0327, 0.0389, 0.0088, 0.0576],
    "EGN-v2":        [0.1571, 0.1679, 0.1344, 0.1644],
}

# Val Loss
val_loss = {
    "HisToGene":     [0.2869, 0.2782, 0.2949, 0.2850],
    "HisToGene-UNI": [0.2687, 0.2725, 0.2872, None],
    "EGN-v1":        [0.3483, 0.3784, 0.3738, 0.3799],
    "EGN-v2":        [0.3098, 0.3476, 0.3335, 0.3324],
}

# Train PCC
train_pcc = {
    "HisToGene":     [0.7238, 0.7955, 0.7135, 0.7620],
    "HisToGene-UNI": [0.80,   0.8442, 0.8288, None],
    "EGN-v1":        [0.1428, 0.0899, 0.0200, 0.0795],
    "EGN-v2":        [0.4423, 0.4985, 0.4034, 0.4723],
}

# 过拟合Gap = Train PCC - Val PCC
overfit_gap = {
    "HisToGene":     [0.2074, 0.1914, 0.1848, 0.2051],
    "HisToGene-UNI": [0.27,   0.2328, 0.2903, None],
    "EGN-v1":        [-0.0861, -0.2242, -0.1965, -0.1665],
    "EGN-v2":        [0.0375, 0.0536, 0.0202, 0.0473],
}

# 参数量（M）
param_m = {
    "HisToGene":     70.6,
    "HisToGene-UNI": 4.0,
    "EGN-v1":        6.8,
    "EGN-v2":        3.0,
}


def _safe_mean(lst):
    """计算非None值的均值"""
    vals = [v for v in lst if v is not None]
    return np.mean(vals) if vals else None


# ─────────────────────────────────────────────
# 图1：Val PCC 分组柱状图
# ─────────────────────────────────────────────
def plot_val_pcc_bar(ax):
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.18
    x = np.arange(n_datasets)

    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_width
        vals = val_pcc[model]
        bars = ax.bar(
            x + offset, [v if v is not None else 0 for v in vals],
            width=bar_width,
            color=model_colors[model],
            label=model,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        # 在柱顶标注数值
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    color=model_colors[model],
                )
            else:
                # 虚线框标注无数据
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
                    fontsize=7, color="gray", style="italic", zorder=5,
                )
                bar.set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=9)
    ax.set_ylabel("Val PCC", fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0, 0.75, 0.1))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_title("① 各模型 Val PCC 对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# 图2：模型平均性能雷达图
# ─────────────────────────────────────────────
def plot_radar(ax):
    categories = [
        "Val PCC", "Val R²", "1-Val Loss",
        "参数效率\n(PCC/参数M)", "过拟合控制\n(1-|Gap|)"
    ]
    n_cats = len(categories)

    # 计算各模型各维度均值
    model_means = {}
    for model in models:
        mean_pcc = _safe_mean(val_pcc[model])
        mean_r2 = _safe_mean(val_r2[model])
        mean_loss = _safe_mean(val_loss[model])
        mean_gap = _safe_mean(overfit_gap[model])
        pcc_per_param = mean_pcc / param_m[model] if mean_pcc else 0
        model_means[model] = {
            "Val PCC":  mean_pcc,
            "Val R²":   mean_r2,
            "1-Val Loss": 1 - mean_loss if mean_loss else 0,
            "参数效率": pcc_per_param,
            "过拟合控制": 1 - abs(mean_gap) if mean_gap is not None else 0,
        }

    # 归一化到0-1
    all_vals = {cat: [] for cat in categories}
    for model in models:
        for i, cat in enumerate(categories):
            key = ["Val PCC", "Val R²", "1-Val Loss", "参数效率", "过拟合控制"][i]
            all_vals[cat].append(model_means[model][key])

    norm_ranges = {}
    for cat in categories:
        vals = all_vals[cat]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:
            norm_ranges[cat] = (v_min - 0.1, v_max + 0.1)
        else:
            margin = (v_max - v_min) * 0.1
            norm_ranges[cat] = (v_min - margin, v_max + margin)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6, color="gray")
    ax.set_ylim(0, 1.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)

    for model in models:
        norm_vals = []
        for i, cat in enumerate(categories):
            key = ["Val PCC", "Val R²", "1-Val Loss", "参数效率", "过拟合控制"][i]
            raw = model_means[model][key]
            lo, hi = norm_ranges[cat]
            norm_v = (raw - lo) / (hi - lo) if hi != lo else 0.5
            norm_vals.append(norm_v)
        norm_vals += norm_vals[:1]

        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=4,
                color=model_colors[model], label=model, zorder=3)
        ax.fill(angles, norm_vals, alpha=0.08, color=model_colors[model])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=7, framealpha=0.9)
    ax.set_title("② 模型综合性能雷达图", fontsize=12, fontweight="bold", pad=20)


# ─────────────────────────────────────────────
# 图3：过拟合分析散点图
# ─────────────────────────────────────────────
def plot_overfitting_scatter(ax):
    markers = {"HYZ15040": "o", "JFX0729": "s", "LMZ12939": "^", "MultiPatient_3ST": "D"}
    dataset_colors = {
        "HYZ15040": "#5B9BD5", "JFX0729": "#A5A5A5",
        "LMZ12939": "#FFC000", "MultiPatient_3ST": "#FF6384",
    }

    # 画 y=x 虚线
    lim_min, lim_max = -0.05, 1.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1, label="无过拟合线 (y=x)")

    # 在y=x线下方区域标注"过拟合区"
    ax.fill_between(
        [lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
        alpha=0.04, color="red", zorder=0,
    )
    ax.text(0.85, 0.15, "过拟合区", fontsize=8, color="red", alpha=0.5,
            ha="center", style="italic", transform=ax.transAxes)

    for model in models:
        for j, ds in enumerate(datasets):
            tr = train_pcc[model][j]
            va = val_pcc[model][j]
            if tr is None or va is None:
                continue
            ax.scatter(
                tr, va,
                marker=markers[ds], s=90,
                c=model_colors[model],
                edgecolors="white", linewidths=0.8,
                zorder=3, alpha=0.85,
            )

    # 手动构建图例
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m],
               markersize=8, label=m) for m in models
    ]
    ds_handles = [
        Line2D([0], [0], marker=markers[ds], color="w",
               markerfacecolor="gray", markersize=7, label=ds) for ds in datasets
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


# ─────────────────────────────────────────────
# 图4：参数量 vs 性能气泡图
# ─────────────────────────────────────────────
def plot_param_bubble(ax):
    for model in models:
        mean_pcc = _safe_mean(val_pcc[model])
        mean_r2 = _safe_mean(val_r2[model])
        p = param_m[model]
        if mean_pcc is None:
            continue
        bubble_size = max(mean_r2 * 1500, 50)  # 气泡大小

        ax.scatter(
            p, mean_pcc,
            s=bubble_size,
            c=model_colors[model],
            alpha=0.7, edgecolors="white", linewidths=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{model}\n(Val PCC={mean_pcc:.3f})",
            (p, mean_pcc),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center", fontsize=8, fontweight="bold",
            color=model_colors[model],
        )

    # 标注理想区域
    ax.annotate(
        "← 理想区域\n(参数少 · 性能高)",
        xy=(5, 0.55), fontsize=8, color="gray", style="italic",
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
    for r2_val, label in [(0.05, "R²≈0.05"), (0.20, "R²≈0.20"), (0.35, "R²≈0.35")]:
        ax.scatter([], [], s=max(r2_val * 1500, 50), c="gray", alpha=0.3,
                   edgecolors="white", linewidths=1, label=label)
    ax.legend(loc="lower right", fontsize=7, title="气泡大小=平均Val R²",
              title_fontsize=7, framealpha=0.9, scatterpoints=1)


# ─────────────────────────────────────────────
# 主图绘制
# ─────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(16, 14), dpi=150)
    fig.suptitle(
        "空间转录组预测模型训练结果对比分析",
        fontsize=18, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.945,
        "数据集：食管癌 | 模型：HisToGene / HisToGene-UNI / EGN-v1 / EGN-v2",
        ha="center", fontsize=10, color="gray",
    )

    # 2×2 布局，图2用极坐标
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_val_pcc_bar(ax1)
    plot_radar(ax2)
    plot_overfitting_scatter(ax3)
    plot_param_bubble(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 保存
    output_dir = r"d:\AI空间转录病理研究\PFMval_new\histogene\results_vis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison_report.png")
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[完成] 图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
