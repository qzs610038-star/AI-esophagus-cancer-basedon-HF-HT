"""
visualize_cross_patient.py
==========================
跨患者泛化训练结果对比可视化脚本

生成 2×2 综合对比图表，包含：
  1. 单患者 vs 跨患者 PCC 性能衰减对比（分组柱状图）
  2. 跨患者泛化多指标雷达图
  3. 训练曲线对比（双Y轴，读取实际训练历史CSV）
  4. 过拟合分析散点图（含箭头标注性能变化）

使用方式：
  python visualize_cross_patient.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────
# 字体配置
# ─────────────────────────────────────────────
def _setup_font():
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
# 数据定义
# ─────────────────────────────────────────────
BASE_DIR = r"d:\AI空间转录病理研究\PFMval_new"

# 配色方案（与 visualize_model_comparison.py 一致）
COLOR_HISTO_UNI = "#70AD47"   # 绿色 — HisToGene-UNI 单患者
COLOR_HISTO_UNI_LIGHT = "#A9D18E"  # 浅绿 — HisToGene-UNI 跨患者
COLOR_EGNV2 = "#ED7D31"      # 橙色 — EGN-v2 单患者
COLOR_EGNV2_LIGHT = "#F4B183"     # 浅橙 — EGN-v2 跨患者

# 蓝色系用于子图1中 HisToGene-UNI 的替代配色（按需求用蓝色系）
BLUE_DARK = "#4472C4"
BLUE_LIGHT = "#9DC3E6"
ORANGE_DARK = "#ED7D31"
ORANGE_LIGHT = "#F4B183"

# ──── 跨患者泛化结果（JFX+LMZ → HYZ） ────
cross_patient = {
    "HisToGene-UNI": {
        "test_pcc": 0.3946, "test_r2": 0.1270, "test_loss": 0.3331,
        "best_epoch": 3, "total_epochs": 18, "train_pcc": 0.7876,
    },
    "EGN-v2": {
        "test_pcc": 0.1950, "test_r2": 0.0287, "test_loss": 0.3700,
        "best_epoch": 16, "total_epochs": 31, "train_pcc": 0.2985,
    },
}

# ──── 单患者基线（HYZ15040 train/val split） ────
single_patient = {
    "HisToGene-UNI": {
        "val_pcc": 0.5336, "val_r2": 0.2821, "val_loss": 0.2687,
        "best_epoch": 3, "total_epochs": 18, "train_pcc": 0.7988,
    },
    "EGN-v2": {
        "val_pcc": 0.4048, "val_r2": 0.1571, "val_loss": 0.3098,
        "best_epoch": 36, "total_epochs": 51, "train_pcc": 0.4423,
    },
}

models = ["HisToGene-UNI", "EGN-v2"]

# ─────────────────────────────────────────────
# 子图1：单患者 vs 跨患者 PCC 性能衰减对比
# ─────────────────────────────────────────────
def plot_pcc_decay_bar(ax):
    x = np.arange(len(models))
    bar_width = 0.30

    single_pccs = [single_patient[m]["val_pcc"] for m in models]
    cross_pccs = [cross_patient[m]["test_pcc"] for m in models]
    decay_pcts = [
        (cross_patient[m]["test_pcc"] - single_patient[m]["val_pcc"]) / single_patient[m]["val_pcc"] * 100
        for m in models
    ]

    dark_colors = [BLUE_DARK, ORANGE_DARK]
    light_colors = [BLUE_LIGHT, ORANGE_LIGHT]

    for i in range(len(models)):
        # 单患者柱
        bar_s = ax.bar(
            x[i] - bar_width / 2, single_pccs[i],
            width=bar_width, color=dark_colors[i],
            edgecolor="white", linewidth=0.8, zorder=3,
            label="单患者 Val PCC" if i == 0 else None,
        )
        # 跨患者柱
        bar_c = ax.bar(
            x[i] + bar_width / 2, cross_pccs[i],
            width=bar_width, color=light_colors[i],
            edgecolor="white", linewidth=0.8, zorder=3,
            label="跨患者 Test PCC" if i == 0 else None,
        )
        bs = bar_s[0]
        bc = bar_c[0]
        # 数值标注
        ax.text(
            bs.get_x() + bs.get_width() / 2, bs.get_height() + 0.008,
            f"{single_pccs[i]:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=dark_colors[i],
        )
        ax.text(
            bc.get_x() + bc.get_width() / 2, bc.get_height() + 0.008,
            f"{cross_pccs[i]:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=light_colors[i],
        )
        # 衰减百分比标注
        mid_x = (bs.get_x() + bc.get_x() + bc.get_width()) / 2
        top_y = max(single_pccs[i], cross_pccs[i]) + 0.045
        ax.text(
            mid_x, top_y,
            f"{decay_pcts[i]:+.1f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF2F2", edgecolor="red", linewidth=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("PCC", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_yticks(np.arange(0, 0.75, 0.1))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("① 单患者 vs 跨患者 PCC 性能衰减对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# 子图2：跨患者泛化多指标雷达图
# ─────────────────────────────────────────────
def plot_radar(ax):
    # 显示标签（可含换行）和查找键（不含换行）
    cat_keys = ["Test PCC", "Test R2", "1-Test Loss", "参数效率", "过拟合控制"]
    cat_labels = [
        "Test PCC", "Test R2", "1-Test Loss",
        "参数效率\n(PCC/参数M)", "过拟合控制\n(1-gap/max_gap)"
    ]
    n_cats = len(cat_keys)

    # 参数量（M）
    param_m = {"HisToGene-UNI": 4.0, "EGN-v2": 3.0}

    # 计算原始值
    raw_data = {}
    max_gap = 0
    for m in models:
        cp = cross_patient[m]
        sp = single_patient[m]
        gap = sp["train_pcc"] - cp["test_pcc"]
        max_gap = max(max_gap, gap)
        raw_data[m] = {
            "Test PCC": cp["test_pcc"],
            "Test R2": cp["test_r2"],
            "1-Test Loss": 1 - cp["test_loss"],
            "参数效率": cp["test_pcc"] / param_m[m],
            "过拟合控制": 1 - gap,  # 先存，后面再归一化
        }

    # 归一化到 0-1
    all_vals = {key: [raw_data[m][key] for m in models] for key in cat_keys}
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
    ax.set_xticklabels(cat_labels, fontsize=8)

    radar_colors = {"HisToGene-UNI": BLUE_DARK, "EGN-v2": ORANGE_DARK}
    for m in models:
        norm_vals = []
        for key in cat_keys:
            raw = raw_data[m][key]
            lo, hi = norm_ranges[key]
            norm_v = (raw - lo) / (hi - lo) if hi != lo else 0.5
            norm_vals.append(norm_v)
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=4,
                color=radar_colors[m], label=m, zorder=3)
        ax.fill(angles, norm_vals, alpha=0.08, color=radar_colors[m])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0.9)
    ax.set_title("② 跨患者泛化多指标雷达图", fontsize=12, fontweight="bold", pad=20)


# ─────────────────────────────────────────────
# 子图3：训练曲线对比（双Y轴）
# ─────────────────────────────────────────────
def plot_training_curves(ax):
    csv_paths = {
        "HisToGene-UNI": os.path.join(BASE_DIR, "histogene", "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
        "EGN-v2": os.path.join(BASE_DIR, "egnv2", "training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
    }

    ax2 = ax.twinx()

    curve_colors = {"HisToGene-UNI": BLUE_DARK, "EGN-v2": ORANGE_DARK}
    loss_colors = {"HisToGene-UNI": BLUE_LIGHT, "EGN-v2": ORANGE_LIGHT}

    for m in models:
        csv_path = csv_paths[m]
        if not os.path.exists(csv_path):
            print(f"[警告] 未找到训练历史文件: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        epochs = df["epoch"].values
        test_pccs = df["val_pcc"].values  # cross-patient中val_pcc即test_pcc
        train_losses = df["train_loss"].values

        # Test PCC 曲线（左Y轴）
        ax.plot(epochs, test_pccs, "-o", linewidth=1.8, markersize=3,
                color=curve_colors[m], label=f"{m} Test PCC", zorder=3)

        # Train Loss 曲线（右Y轴）
        ax2.plot(epochs, train_losses, "--", linewidth=1.5, markersize=2,
                 color=loss_colors[m], label=f"{m} Train Loss", zorder=2, alpha=0.8)

        # 标注最佳 Epoch
        best_ep = cross_patient[m]["best_epoch"]
        best_idx = np.where(epochs == best_ep)[0]
        if len(best_idx) > 0:
            best_pcc = test_pccs[best_idx[0]]
            ax.axvline(x=best_ep, color=curve_colors[m], linestyle=":", alpha=0.5, linewidth=1)
            ax.annotate(
                f"Best Ep.{best_ep}\nPCC={best_pcc:.4f}",
                xy=(best_ep, best_pcc),
                xytext=(best_ep + 2, best_pcc + 0.03),
                fontsize=7, color=curve_colors[m], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=curve_colors[m], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=curve_colors[m], alpha=0.8),
            )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Test PCC", fontsize=10, color=BLUE_DARK)
    ax2.set_ylabel("Train Loss", fontsize=10, color="gray")
    ax.tick_params(axis="y", labelcolor=BLUE_DARK)
    ax2.tick_params(axis="y", labelcolor="gray")

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7, framealpha=0.9)

    ax.set_title("③ 跨患者训练曲线对比", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)


# ─────────────────────────────────────────────
# 子图4：过拟合分析散点图
# ─────────────────────────────────────────────
def plot_overfitting_scatter(ax):
    # 四个数据点
    points = {
        "HisToGene-UNI\n(单患者)": {
            "train": single_patient["HisToGene-UNI"]["train_pcc"],
            "test": single_patient["HisToGene-UNI"]["val_pcc"],
            "color": BLUE_DARK, "marker": "o",
        },
        "HisToGene-UNI\n(跨患者)": {
            "train": cross_patient["HisToGene-UNI"]["train_pcc"],
            "test": cross_patient["HisToGene-UNI"]["test_pcc"],
            "color": BLUE_LIGHT, "marker": "o",
        },
        "EGN-v2\n(单患者)": {
            "train": single_patient["EGN-v2"]["train_pcc"],
            "test": single_patient["EGN-v2"]["val_pcc"],
            "color": ORANGE_DARK, "marker": "s",
        },
        "EGN-v2\n(跨患者)": {
            "train": cross_patient["EGN-v2"]["train_pcc"],
            "test": cross_patient["EGN-v2"]["test_pcc"],
            "color": ORANGE_LIGHT, "marker": "s",
        },
    }

    lim_min, lim_max = -0.05, 1.0

    # 画 y=x 对角线
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1, label="理想拟合 (y=x)")

    # 过拟合区域填充
    ax.fill_between(
        [lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
        alpha=0.04, color="red", zorder=0,
    )
    ax.text(0.82, 0.12, "过拟合区", fontsize=8, color="red", alpha=0.5,
            ha="center", style="italic", transform=ax.transAxes)

    # 绘制散点
    for name, pt in points.items():
        ax.scatter(
            pt["train"], pt["test"],
            marker=pt["marker"], s=120,
            c=pt["color"], edgecolors="white", linewidths=1,
            zorder=4, alpha=0.9, label=name,
        )

    # 从单患者指向跨患者的箭头
    arrow_pairs = [
        ("HisToGene-UNI\n(单患者)", "HisToGene-UNI\n(跨患者)"),
        ("EGN-v2\n(单患者)", "EGN-v2\n(跨患者)"),
    ]
    for src_name, dst_name in arrow_pairs:
        src = points[src_name]
        dst = points[dst_name]
        dx = dst["train"] - src["train"]
        dy = dst["test"] - src["test"]
        ax.annotate(
            "", xy=(dst["train"], dst["test"]),
            xytext=(src["train"], src["test"]),
            arrowprops=dict(
                arrowstyle="->", color="red", lw=1.5,
                connectionstyle="arc3,rad=0.15", alpha=0.7,
            ),
            zorder=3,
        )
        # 标注性能变化
        pcc_change = dst["test"] - src["test"]
        mid_x = (src["train"] + dst["train"]) / 2
        mid_y = (src["test"] + dst["test"]) / 2
        ax.text(
            mid_x + 0.02, mid_y + 0.015,
            f"{pcc_change:+.3f}",
            fontsize=8, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="red", alpha=0.7, linewidth=0.6),
            zorder=5,
        )

    ax.set_xlabel("Train PCC", fontsize=10)
    ax.set_ylabel("Test / Val PCC", fontsize=10)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("④ 过拟合分析（Train vs Test/Val PCC）", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)


# ─────────────────────────────────────────────
# 主图绘制
# ─────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(16, 14), dpi=300)
    fig.suptitle(
        "跨患者泛化训练结果对比分析",
        fontsize=18, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.945,
        "训练集：JFX0729 + LMZ12939 → 测试集：HYZ15040 | 模型：HisToGene-UNI / EGN-v2",
        ha="center", fontsize=10, color="gray",
    )

    # 2×2 布局
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_pcc_decay_bar(ax1)
    plot_radar(ax2)
    plot_training_curves(ax3)
    plot_overfitting_scatter(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 保存
    output_dir = os.path.join(BASE_DIR, "histogene", "checkpoints", "results_vis", "CrossPatient_comparison")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cross_patient_comparison.png")
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"[完成] 图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
