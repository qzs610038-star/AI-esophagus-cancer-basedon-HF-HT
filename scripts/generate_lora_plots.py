"""MPP2 LoRA r001 的历史可视化复现脚本。

输入路径和 SHA-256 均固定指向 2026-07-13 的已回传预测包；本文件用于复现当时
报告图，不是当前训练入口、实验结果登记器或 accepted evidence 的来源。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy.stats import pearsonr

# ─────────────────────────────────────────────
# 字体与基本配置：临床学术亮色
# ─────────────────────────────────────────────
def setup_plt_style():
    for font_name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(
                matplotlib.font_manager.FontProperties(family=font_name),
                fallback_to_default=False
            )
            rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
        except Exception:
            continue
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["font.size"] = 10
    rcParams["axes.edgecolor"] = "#333333"
    rcParams["axes.labelcolor"] = "#333333"
    rcParams["xtick.color"] = "#333333"
    rcParams["ytick.color"] = "#333333"

setup_plt_style()

# 颜色定义
C_BLUE = "#2b5c8f"    # 改善 (色盲友好蓝)
C_ORANGE = "#d95f02"  # 退化/变差 (色盲友好橙)
C_GREY = "#7f7f7f"    # 对照/基线 (中 grey)
C_LIGHT_GREY = "#f0f0f0"
C_SENSITIVE = "#a6a6a6" # 敏感性结果 (浅灰)

# 建立输出目录
out_dir = r"D:\AI空间转录病理研究\PFMval_new\01_指南与解读\分析报告\report_figures\lora_vis_20260713"
os.makedirs(out_dir, exist_ok=True)

import hashlib

# 预测数据路径
s0_val_path = r"D:\AI空间转录病理研究\PFMval_new_prediction_supplement_20260712\automation\results\mpp2-paired-smoke-prediction-supplement-20260712-r001\s0_predictions_internal_val.csv"
s1_val_path = r"D:\AI空间转录病理研究\PFMval_new_prediction_supplement_20260712\automation\results\mpp2-paired-smoke-prediction-supplement-20260712-r001\s1_predictions_internal_val.csv"
s0_ext_path = r"D:\AI空间转录病理研究\PFMval_new_prediction_supplement_20260712\automation\results\mpp2-paired-smoke-prediction-supplement-20260712-r001\s0_predictions_external_xzy.csv"
s1_ext_path = r"D:\AI空间转录病理研究\PFMval_new_prediction_supplement_20260712\automation\results\mpp2-paired-smoke-prediction-supplement-20260712-r001\s1_predictions_external_xzy.csv"

# 权威哈希声明 (对应 mpp2-paired-smoke-prediction-supplement-20260712-r001/sha256_manifest.json)
EXPECTED_HASHES = {
    s0_ext_path: "b5972c69d14fe4dde57c33eb32c4423523d8f7bac9d95c21e7424cddcbe3deee",
    s0_val_path: "fab7798511b9bb80ce172f0f961d9e3517ec14b3e8f31183c40453e0f26a1b57",
    s1_ext_path: "78a69e69a7b412d3f94c649c8425b157c3cc04e5cd3ffa071af5307fb9d7886b",
    s1_val_path: "1ae94c7fa5ab8489012d1a077f4a64233a44f0ed3523062904a18c746c46ae8a"
}

def verify_file_sha256(file_path, expected_sha):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_sha = sha256_hash.hexdigest()
    if actual_sha != expected_sha:
        print(f"Error: Hash mismatch for {file_path}.\nExpected: {expected_sha}\nActual: {actual_sha}", file=sys.stderr)
        sys.exit(1)

# 校验数据是否存在与哈希正确性
for p in [s0_val_path, s1_val_path, s0_ext_path, s1_ext_path]:
    if not os.path.exists(p):
        print(f"Error: File not found {p}", file=sys.stderr)
        sys.exit(1)
    verify_file_sha256(p, EXPECTED_HASHES[p])
print("All 4 input CSV files verified successfully (SHA-256 match).")

# 读取数据
df_s0_val = pd.read_csv(s0_val_path)
df_s1_val = pd.read_csv(s1_val_path)
df_s0_ext = pd.read_csv(s0_ext_path)
df_s1_ext = pd.read_csv(s1_ext_path)

pathways = [col[5:] for col in df_s0_val.columns if col.startswith("true_")]

# 中英文通路对照映射
pathway_cn = {
    "tls": "三级淋巴结构 (tls)",
    "tgfb": "TGF-β 通路 (tgfb)",
    "emt": "上皮-间质转化 (emt)",
    "hypoxia": "缺氧通路 (hypoxia)",
    "mhc": "MHC 复合体 (mhc)",
    "icp": "免疫检查点 (icp)",
    "ifng": "IFN-γ 反应 (ifng)",
    "toxic": "细胞毒性评分 (toxic)",
    "Glycolysis": "糖酵解 (Glycolysis)",
    "Inflammatory_Response": "炎症反应 (Inflammatory)",
    "IL6_JAK_STAT3": "IL6/JAK/STAT3 (IL6_JAK)",
    "P53_Pathway": "P53 通路 (P53)",
    "DNA_Damage_Response": "DNA 损伤反应 (DDR)",
    "Complement": "补体系统 (Complement)",
    "Coagulation": "凝血级联 (Coagulation)",
    "Oxidative_Phosphorylation": "氧化磷酸化 (OxPhos)",
    "Reactive_Oxygen_Species": "活性氧通路 (ROS)",
    "Wound_Healing": "伤口愈合 (Wound_Healing)",
    "Fibrosis": "纤维化 (Fibrosis)",
    "MYC_Targets": "MYC 靶基因 (MYC)",
    "E2F_Targets": "E2F 靶基因 (E2F)",
    "G2M_Checkpoint": "G2M 检查点 (G2M)",
    "Mitotic_Spindle": "有丝分裂纺锤体 (Spindle)",
    "Unfolded_Protein_Response": "未折叠蛋白反应 (UPR)",
    "mTOR_Signaling": "mTOR 信号 (mTOR)",
    "Interferon_Alpha": "IFN-α 反应 (ifna)",
    "Angiogenesis": "血管生成 (Angio)",
    "Apoptosis": "细胞凋亡 (Apoptosis)",
    "TNF_Signaling": "TNF-α 信号 (TNF)",
    "ECM_Organization": "细胞外基质重塑 (ECM)"
}

# ══════════════════════════════════════════════
#  图表 1：不同 MPP 采样方案的表现对比图 (点图，无连线)
# ══════════════════════════════════════════════
def draw_plot1(ax_pcc=None, ax_mae=None, ax_r2=None):
    mpps_labels = [
        "MPP1\n(最高分/100%步长)\n[混合背景]",
        "MPP2\n(0.27um/100%步长)\n[主线方案]",
        "MPP3\n(0.27um/50%步长)\n[重叠背景]",
        "MPP4\n(0.54um/100%步长)\n[20x规格背景]",
        "MPP5\n(0.54um/50%步长)\n[重叠背景]"
    ]
    mpps = [f"MPP{i}" for i in range(1, 6)]
    int_pcc = [0.759747, 0.797100, 0.822454, 0.820883, 0.834697]
    ext_pcc = [0.690000, 0.654900, 0.643600, 0.615100, 0.607200]
    raw_mae = [1360.9619, 1176.2114, 1209.3894, 1053.1715, 1045.6602]
    raw_r2 = [-0.3719, -0.0880, -0.1265, 0.1090, 0.0811]

    standalone = False
    if ax_pcc is None:
        standalone = True
        fig, (ax_pcc, ax_mae, ax_r2) = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

    # 1. PCC 哑铃图
    for i in range(len(mpps)):
        ax_pcc.plot([int_pcc[i], ext_pcc[i]], [i, i], color="#cccccc", linestyle="-", zorder=1)
    ax_pcc.scatter(int_pcc, range(len(mpps)), color=C_BLUE, marker="o", s=90, label="内部验证集 (Internal Val)", zorder=2)
    ax_pcc.scatter(ext_pcc, range(len(mpps)), color=C_ORANGE, marker="s", s=90, label="外部测试集 (External XZY)", zorder=2)
    ax_pcc.set_yticks(range(len(mpps)))
    ax_pcc.set_yticklabels(mpps)
    ax_pcc.set_title("A. 内部与外部 PCC 数值对比", fontsize=11, fontweight="bold")
    ax_pcc.set_xlabel("Pearson 相关系数 (PCC)")
    ax_pcc.grid(axis="x", linestyle="--", alpha=0.4)
    ax_pcc.legend(loc="lower left", fontsize=9)

    # 2. Raw MAE 散点图 (不画连线)
    ax_mae.scatter(mpps, raw_mae, color=C_BLUE, marker="D", s=80, zorder=2)
    ax_mae.set_title("B. Raw MAE 分散点图 (越低越好)", fontsize=11, fontweight="bold")
    ax_mae.set_ylabel("原始 MAE (Raw MAE)")
    ax_mae.grid(linestyle="--", alpha=0.4)

    # 3. Raw R2 散点图 (不画连线)
    ax_r2.scatter(mpps, raw_r2, color=C_BLUE, marker="X", s=90, zorder=2)
    ax_r2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_r2.set_title("C. Raw R² 分散点图 (越高越好)", fontsize=11, fontweight="bold")
    ax_r2.set_ylabel("原始 R² (Raw R²)")
    ax_r2.grid(linestyle="--", alpha=0.4)
    ax_r2.set_xticks(range(len(mpps)))
    ax_r2.set_xticklabels(mpps_labels, fontsize=8.5)

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart1_mpp_comparison.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "chart1_mpp_comparison.pdf"))
        plt.savefig(os.path.join(out_dir, "chart1_mpp_comparison.svg"))
        plt.close()

# ══════════════════════════════════════════════
#  图表 2：LoRA (S1) vs 冻结基线 (S0) 核心指标对比图
# ══════════════════════════════════════════════
def draw_plot2(ax_abs=None, ax_pcc_d=None, ax_mae_d=None, ax_r2_d=None):
    # 数据定义
    mpp2_baseline_pcc = 0.6549
    s0_pcc = 0.644546
    s1_pcc = 0.646439
    pcc_gate = 0.6749  # 0.6549 + 0.02

    delta_pcc = 0.001893
    delta_mae_pct = -0.6959  # %
    delta_r2 = 0.009022

    standalone = False
    if ax_abs is None:
        standalone = True
        fig = plt.figure(figsize=(11, 5.5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.8, 1, 1, 1])
        ax_abs = fig.add_subplot(gs[0])
        ax_pcc_d = fig.add_subplot(gs[1])
        ax_mae_d = fig.add_subplot(gs[2])
        ax_r2_d = fig.add_subplot(gs[3])

    # A. 绝对 PCC 对比柱状图
    labels = ["正式 MPP2\n基线", "S0 冻结\n继续微调", "S1 LoRA\nr=8 烟雾测试"]
    values = [mpp2_baseline_pcc, s0_pcc, s1_pcc]
    bars = ax_abs.bar(labels, values, color=[C_GREY, C_GREY, C_BLUE], width=0.45, edgecolor="#333333")
    ax_abs.axhline(pcc_gate, color=C_ORANGE, linestyle="--", linewidth=1.5, label=f"候选有效性线 (PCC={pcc_gate:.4f})")
    ax_abs.set_ylabel("外部测试集 PCC")
    ax_abs.set_title("A. 绝对 PCC 对比\n(未达到预设有效性线)", fontsize=11, fontweight="bold")
    ax_abs.set_ylim(0.58, 0.70)
    for bar in bars:
        height = bar.get_height()
        ax_abs.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    ax_abs.legend(loc="lower left", fontsize=8)
    ax_abs.grid(axis="y", linestyle="--", alpha=0.4)

    # B. 配对变化率的独立小图
    # B1. PCC变化量
    ax_pcc_d.bar(["S1 - S0"], [delta_pcc], color=C_BLUE, width=0.35, edgecolor="#333333")
    ax_pcc_d.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_pcc_d.set_title("B1. ΔPCC\n(绝对配对增量)", fontsize=9.5, fontweight="bold")
    ax_pcc_d.set_ylim(-0.005, 0.005)
    ax_pcc_d.text(0, delta_pcc + 0.0003, f"+{delta_pcc:.4f}", ha='center', va='bottom', fontsize=9, fontweight="bold")
    ax_pcc_d.grid(axis="y", linestyle="--", alpha=0.4)

    # B2. Raw MAE 变化率
    ax_mae_d.bar(["S1 - S0"], [delta_mae_pct], color=C_BLUE, width=0.35, edgecolor="#333333")
    ax_mae_d.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_mae_d.set_title("B2. ΔRaw MAE\n(相对变动 %)", fontsize=9.5, fontweight="bold")
    ax_mae_d.set_ylim(-1.5, 0.5)
    ax_mae_d.text(0, delta_mae_pct - 0.1, f"{delta_mae_pct:.3f}%", ha='center', va='top', fontsize=9, fontweight="bold")
    ax_mae_d.grid(axis="y", linestyle="--", alpha=0.4)

    # B3. R2 变化量
    ax_r2_d.bar(["S1 - S0"], [delta_r2], color=C_BLUE, width=0.35, edgecolor="#333333")
    ax_r2_d.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_r2_d.set_title("B3. ΔRaw R²\n(绝对配对增量)", fontsize=9.5, fontweight="bold")
    ax_r2_d.set_ylim(-0.02, 0.02)
    ax_r2_d.text(0, delta_r2 + 0.001, f"+{delta_r2:.4f}", ha='center', va='bottom', fontsize=9, fontweight="bold")
    ax_r2_d.grid(axis="y", linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart2_lora_comparison.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "chart2_lora_comparison.pdf"))
        plt.savefig(os.path.join(out_dir, "chart2_lora_comparison.svg"))
        plt.close()

# ══════════════════════════════════════════════
#  图表 3：患者级 PCC 与标准化 MAE 变化横向点图
# ══════════════════════════════════════════════
def draw_plot3(ax_pcc=None, ax_mae=None):
    patients = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
    s0_pcc = [0.647404, 0.742965, 0.659530, 0.779775, 0.786082, 0.865072]
    s1_pcc = [0.643119, 0.741386, 0.651777, 0.781628, 0.784006, 0.865550]
    s0_mae = [0.411509, 0.486236, 0.437854, 0.496353, 0.476735, 0.441815]
    s1_mae = [0.415657, 0.486975, 0.441379, 0.497379, 0.483061, 0.445857]

    delta_pcc = [s1 - s0 for s1, s0 in zip(s1_pcc, s0_pcc)]
    delta_mae = [s1 - s0 for s1, s0 in zip(s1_mae, s0_mae)] # 正代表变大/变差，负代表变小/改善

    standalone = False
    if ax_pcc is None:
        standalone = True
        fig, (ax_pcc, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.5))

    y_pos = np.arange(len(patients))

    # A. ΔPCC 散点图
    ax_pcc.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    colors_pcc = [C_BLUE if d >= 0 else C_ORANGE for d in delta_pcc]
    ax_pcc.scatter(delta_pcc, y_pos, color=colors_pcc, s=80, edgecolor="#333333", zorder=3)
    ax_pcc.set_yticks(y_pos)
    ax_pcc.set_yticklabels(patients)
    ax_pcc.set_xlabel("PCC 绝对变化量 (S1 - S0)")
    ax_pcc.set_title("A. 内部验证集患者切片 ΔPCC\n(仅 TGC/ZHZ 改善，其余退化)", fontsize=11, fontweight="bold")
    ax_pcc.grid(linestyle="--", alpha=0.4)
    # 增加数值标签
    for i, d in enumerate(delta_pcc):
        ha = 'left' if d >= 0 else 'right'
        offset = 0.0005 if d >= 0 else -0.0005
        ax_pcc.text(d + offset, i, f"{d:+.4f}", va='center', ha=ha, fontsize=8.5, fontweight="bold")
    ax_pcc.set_xlim(-0.015, 0.015)

    # B. Δ标准化 MAE 散点图
    ax_mae.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    # 既然均大于0且MAE越大越差，这里均画橙色
    colors_mae = [C_BLUE if d <= 0 else C_ORANGE for d in delta_mae]
    ax_mae.scatter(delta_mae, y_pos, color=colors_mae, s=80, edgecolor="#333333", zorder=3)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels([]) # 共享左侧患者标签
    ax_mae.set_xlabel("标准化 MAE 绝对变化量 (S1 - S0)")
    ax_mae.set_title("B. 内部验证集患者切片 Δ标准化 MAE\n(6位患者的MAE均小幅增加/退化)", fontsize=11, fontweight="bold")
    ax_mae.grid(linestyle="--", alpha=0.4)
    for i, d in enumerate(delta_mae):
        ha = 'left' if d >= 0 else 'right'
        offset = 0.0002 if d >= 0 else -0.0002
        ax_mae.text(d + offset, i, f"{d:+.4f}", va='center', ha=ha, fontsize=8.5, fontweight="bold")
    ax_mae.set_xlim(-0.002, 0.010)

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart3_patient_slope.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "chart3_patient_slope.pdf"))
        plt.savefig(os.path.join(out_dir, "chart3_patient_slope.svg"))
        plt.close()

# ══════════════════════════════════════════════
#  图表 4：External XZY 空间 Bootstrap 置信区间 (删除 R2)
# ══════════════════════════════════════════════
def draw_plot4(ax_pcc=None, ax_mae=None):
    grids = [
        "1024像素网格\n(67个空间簇)",
        "2048像素网格\n(21个空间簇)",
        "4096像素网格\n(8个空间簇)\n[低簇数敏感性]"
    ]

    # PCC
    pcc_mean = 0.00189
    pcc_cis = [
        [-0.00190, 0.00599],
        [-0.00291, 0.00729],
        [-0.00350, 0.00691]
    ]

    # Raw MAE
    mae_mean = -8.45
    mae_cis = [
        [-14.94, -2.09],
        [-17.79, -0.42],
        [-19.91, -1.77]
    ]

    standalone = False
    if ax_pcc is None:
        standalone = True
        fig, (ax_pcc, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.5))

    y_pos = np.arange(len(grids))

    # A. Delta PCC 置信区间 (4096 像素网格为敏感性结果，用灰色虚线绘制)
    ax_pcc.axvline(0, color="black", linestyle="-", linewidth=0.8)
    for i in range(len(grids)):
        low, high = pcc_cis[i]
        is_sensitive = (i == 2)
        color = C_SENSITIVE if is_sensitive else C_ORANGE
        linestyle = "--" if is_sensitive else "-"
        marker = "x" if is_sensitive else "o"
        ax_pcc.plot([low, high], [i, i], color=color, linestyle=linestyle, linewidth=2)
        ax_pcc.scatter(pcc_mean, i, color=color, marker=marker, s=60, zorder=3)
    ax_pcc.set_yticks(y_pos)
    ax_pcc.set_yticklabels(grids)
    ax_pcc.set_title(r"A. $\Delta$ PCC 95% 置信区间 (跨0线/不显著)", fontsize=10.5, fontweight="bold")
    ax_pcc.set_xlabel("PCC 绝对变化量 (S1 - S0)")
    ax_pcc.set_ylim(-0.5, 2.5)
    ax_pcc.grid(axis="x", linestyle="--", alpha=0.4)

    # B. Delta Raw MAE 置信区间
    ax_mae.axvline(0, color="black", linestyle="-", linewidth=0.8)
    for i in range(len(grids)):
        low, high = mae_cis[i]
        is_sensitive = (i == 2)
        color = C_SENSITIVE if is_sensitive else C_BLUE
        linestyle = "--" if is_sensitive else "-"
        marker = "x" if is_sensitive else "o"
        ax_mae.plot([low, high], [i, i], color=color, linestyle=linestyle, linewidth=2)
        ax_mae.scatter(mae_mean, i, color=color, marker=marker, s=60, zorder=3)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels([]) # 共享y轴标签
    ax_mae.set_title(r"B. $\Delta$ Raw MAE 95% 置信区间 (均在0线下方)", fontsize=10.5, fontweight="bold")
    ax_mae.set_xlabel("Raw MAE 绝对变化量 (S1 - S0)")
    ax_mae.set_ylim(-0.5, 2.5)
    ax_mae.grid(axis="x", linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart4_bootstrap_ci.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "chart4_bootstrap_ci.pdf"))
        plt.savefig(os.path.join(out_dir, "chart4_bootstrap_ci.svg"))
        plt.close()

# ══════════════════════════════════════════════
#  图表 5：30个通路 External PCC 下降与改善分布 (附录)
# ══════════════════════════════════════════════
def draw_plot5(ax=None):
    pathway_metrics = []
    for pw in pathways:
        t0 = df_s0_ext["true_" + pw].values
        p0 = df_s0_ext["pred_" + pw].values
        r0, _ = pearsonr(t0, p0)

        t1 = df_s1_ext["true_" + pw].values
        p1 = df_s1_ext["pred_" + pw].values
        r1, _ = pearsonr(t1, p1)

        delta_r = r1 - r0
        pathway_metrics.append({
            "pathway": pw,
            "s0_pcc": r0,
            "s1_pcc": r1,
            "delta_pcc": delta_r
        })

    df_pw = pd.DataFrame(pathway_metrics).sort_values("delta_pcc", ascending=True)

    standalone = False
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=(9, 10))

    y_pos = np.arange(len(df_pw))
    colors = [C_BLUE if d >= 0 else C_ORANGE for d in df_pw["delta_pcc"]]

    # 纵轴显示中英文对照
    labels_with_cn = [f"{pathway_cn.get(p, p)}" for p in df_pw["pathway"]]

    bars = ax.barh(y_pos, df_pw["delta_pcc"], color=colors, height=0.6, edgecolor="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_with_cn, fontsize=8.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("外部测试集 PCC 变动量 (S1 - S0)")
    ax.set_title("外部测试集各通路 PCC 变化分布\n(4个通路改善，26个通路下降)", fontsize=11, fontweight="bold")
    ax_limit = max(abs(df_pw["delta_pcc"].min()), abs(df_pw["delta_pcc"].max())) * 1.25
    ax.set_xlim(-ax_limit, ax_limit)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        text_x = width + 0.002 if width >= 0 else width - 0.002
        ha = "left" if width >= 0 else "right"
        ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{width:+.4f}",
                va='center', ha=ha, fontsize=8, color=C_BLUE if width >= 0 else C_ORANGE, fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart5_pathway_comparison.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "chart5_pathway_comparison.pdf"))
        plt.savefig(os.path.join(out_dir, "chart5_pathway_comparison.svg"))
        plt.close()

# ══════════════════════════════════════════════
#  数据收集与保存逻辑 (只执行一次，避免重复)
# ══════════════════════════════════════════════
def generate_summary_data():
    summary_list = []

    # 1. MPP Comparison
    mpps = [f"MPP{i}" for i in range(1, 6)]
    int_pcc = [0.759747, 0.797100, 0.822454, 0.820883, 0.834697]
    ext_pcc = [0.690000, 0.654900, 0.643600, 0.615100, 0.607200]
    raw_mae = [1360.9619, 1176.2114, 1209.3894, 1053.1715, 1045.6602]
    raw_r2 = [-0.3719, -0.0880, -0.1265, 0.1090, 0.0811]
    for i, m in enumerate(mpps):
        summary_list.append({
            "Chart": "Chart1_MPP_Comparison", "Group/Key": m, "SubKey": "N/A",
            "PCC_Internal": int_pcc[i], "PCC_External": ext_pcc[i],
            "Raw_MAE": raw_mae[i], "Raw_R2": raw_r2[i], "CI_Lower": np.nan, "CI_Upper": np.nan
        })

    # 2. LoRA vs S0
    summary_list.extend([
        {"Chart": "Chart2_LoRA_vs_S0", "Group/Key": "MPP2_Baseline_PCC", "SubKey": "N/A", "PCC_External": 0.6549, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan},
        {"Chart": "Chart2_LoRA_vs_S0", "Group/Key": "S0_PCC", "SubKey": "N/A", "PCC_External": 0.644546, "Raw_MAE": 1215.284, "Raw_R2": -0.144913, "CI_Lower": np.nan, "CI_Upper": np.nan},
        {"Chart": "Chart2_LoRA_vs_S0", "Group/Key": "S1_PCC", "SubKey": "N/A", "PCC_External": 0.646439, "Raw_MAE": 1206.826, "Raw_R2": -0.135891, "CI_Lower": np.nan, "CI_Upper": np.nan},
        {"Chart": "Chart2_LoRA_vs_S0", "Group/Key": "PCC_Gate", "SubKey": "N/A", "PCC_External": 0.6749, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan},
        {"Chart": "Chart2_LoRA_vs_S0", "Group/Key": "S1-S0_Delta", "SubKey": "N/A", "PCC_External": 0.001893, "Raw_MAE": -8.458, "Raw_R2": 0.009022, "CI_Lower": np.nan, "CI_Upper": np.nan}
    ])

    # 3. Patient Slope
    patients = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
    s0_pcc = [0.647404, 0.742965, 0.659530, 0.779775, 0.786082, 0.865072]
    s1_pcc = [0.643119, 0.741386, 0.651777, 0.781628, 0.784006, 0.865550]
    s0_mae = [0.411509, 0.486236, 0.437854, 0.496353, 0.476735, 0.441815]
    s1_mae = [0.415657, 0.486975, 0.441379, 0.497379, 0.483061, 0.445857]
    for i, p in enumerate(patients):
        summary_list.append({
            "Chart": "Chart3_Patient_Delta", "Group/Key": p, "SubKey": "PCC",
            "PCC_Internal": s1_pcc[i] - s0_pcc[i], "PCC_External": np.nan, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan
        })
        summary_list.append({
            "Chart": "Chart3_Patient_Delta", "Group/Key": p, "SubKey": "Std_MAE",
            "PCC_Internal": s1_mae[i] - s0_mae[i], "PCC_External": np.nan, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan
        })

    # 4. Bootstrap CI (仅 PCC & Raw MAE, 无 R2)
    grids = ["1024", "2048", "4096"]
    pcc_cis = [[-0.00190, 0.00599], [-0.00291, 0.00729], [-0.00350, 0.00691]]
    mae_cis = [[-14.94, -2.09], [-17.79, -0.42], [-19.91, -1.77]]
    for i, g in enumerate(grids):
        summary_list.append({
            "Chart": "Chart4_Bootstrap_CI", "Group/Key": f"Grid_{g}", "SubKey": "Delta_PCC",
            "PCC_External": 0.00189, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": pcc_cis[i][0], "CI_Upper": pcc_cis[i][1]
        })
        summary_list.append({
            "Chart": "Chart4_Bootstrap_CI", "Group/Key": f"Grid_{g}", "SubKey": "Delta_Raw_MAE",
            "PCC_External": np.nan, "Raw_MAE": -8.45, "Raw_R2": np.nan, "CI_Lower": mae_cis[i][0], "CI_Upper": mae_cis[i][1]
        })

    # 5. Pathway comparison
    for pw in pathways:
        t0 = df_s0_ext["true_" + pw].values
        p0 = df_s0_ext["pred_" + pw].values
        r0, _ = pearsonr(t0, p0)
        t1 = df_s1_ext["true_" + pw].values
        p1 = df_s1_ext["pred_" + pw].values
        r1, _ = pearsonr(t1, p1)
        summary_list.append({
            "Chart": "Chart5_Pathway_Delta_PCC", "Group/Key": pw, "SubKey": "N/A",
            "PCC_External": r1 - r0, "Raw_MAE": np.nan, "Raw_R2": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan
        })

    df_sum = pd.DataFrame(summary_list)
    df_sum.to_csv(os.path.join(out_dir, "plot_data_summary.csv"), index=False, encoding="utf-8-sig")
    print("plot_data_summary.csv generated successfully with zero duplicate logic.")

# ══════════════════════════════════════════════
#  生成各独立子图与综合看板
# ══════════════════════════════════════════════
print("Generating standalone charts...")
draw_plot1()
draw_plot2()
draw_plot3()
draw_plot4()
draw_plot5()

print("Generating combined dashboard...")
# 重新排版综合看板：包含医生版核心结论卡片 + 图 1 + 图 2 + 患者一致性图 + 空间敏感性图。
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 1.1, 1], width_ratios=[1, 1])

# 1. 顶部：医生版核心结论卡片 (占满顶部 gs[0, :])
ax_text = fig.add_subplot(gs[0, :])
ax_text.axis("off")
card_text = (
    "研究性汇报版核心实验结论摘要：\n"
    "本次基于 seed 42、3个训练周期的配对探索实验表明，LoRA 在工程构建上能够稳定运行（峰值显存约 4.24 GiB）。\n"
    "在预测表现上，外部测试 PCC 仅微增 +0.0019，未能达到预设的 +0.0200 候选有效性门槛线。\n"
    "尽管外部测试集上的平均原始偏差 (MAE) 有约 -0.7% 的微幅稳定下降（其空间置信区间完全低于 0 轴），\n"
    "但内部验证集上的 6 位患者其标准化 MAE 均呈现小幅增加（泛化降级），且 30 个通路中只有 4 个在外部测试集上表现改善。\n"
    "在通过空间相关性修正后，探索性自适应置信区间分析表明 PCC 和 R² 的提升均不具备统计学显著性（置信区间包含 0 轴）。\n"
    "综上，当前 LoRA 配置满足工程可行性，但尚无充分医学与统计学证据支持进入正式的大规模超参数微调训练。"
)
ax_text.text(0.01, 0.5, card_text, transform=ax_text.transAxes, fontsize=10.5, fontweight="bold",
             va="center", ha="left", color="#333333",
             bbox=dict(facecolor="#fcf8e3", edgecolor="#faebcc", boxstyle="round,pad=0.8", linewidth=1))

# 2. 中左：图 1 (MPP 方案权衡，占用 gs[1, 0]，包含哑铃图和散点图)
gs_mpp = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.35)
ax_pcc1 = fig.add_subplot(gs_mpp[0, 0])
ax_mae1 = fig.add_subplot(gs_mpp[1, 0])
ax_r21 = fig.add_subplot(gs_mpp[2, 0])
draw_plot1(ax_pcc1, ax_mae1, ax_r21)

# 3. 中右：图 2 (LoRA 对比，包含绝对 PCC 柱图与配对变化卡片，占用 gs[1, 1])
gs_lora = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 1], wspace=0.35)
ax_abs2 = fig.add_subplot(gs_lora[0, 0])
ax_pcc_d2 = fig.add_subplot(gs_lora[0, 1])
ax_mae_d2 = fig.add_subplot(gs_lora[0, 2])
ax_r2_d2 = fig.add_subplot(gs_lora[0, 3])
draw_plot2(ax_abs2, ax_pcc_d2, ax_mae_d2, ax_r2_d2)

# 4. 底左：图 3 (患者一致性，包含患者 ΔPCC 和 Δ标准化 MAE，占用 gs[2, 0])
gs_patient = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.3)
ax_pcc3 = fig.add_subplot(gs_patient[0, 0])
ax_mae3 = fig.add_subplot(gs_patient[0, 1])
draw_plot3(ax_pcc3, ax_mae3)

# 5. 底右：图 4 (空间置信区间，包含 PCC & Raw MAE 置信区间，占用 gs[2, 1])
gs_boot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], wspace=0.3)
ax_pcc4 = fig.add_subplot(gs_boot[0, 0])
ax_mae4 = fig.add_subplot(gs_boot[0, 1])
draw_plot4(ax_pcc4, ax_mae4)

# 装饰综合看板
fig.suptitle("MPP 与 LoRA 空间转录组预测结果综合分析看板 (临床学术亮色版)", fontsize=15, fontweight="bold", y=0.98)
fig.text(0.5, 0.015, "数据说明：1. PCC 越高越好，MAE 越低越好。 2. 方案 3 的 MAE 变化为标准化 Z-score 尺度，方案 4 变化为原始 ssGSEA 绝对尺度。\n3. 方案 4 中 4096 像素网格下由于自相关空间簇数量低（仅 8 个），其置信区间仅作为低簇数敏感性分析参考，结论须保守解释。",
         ha="center", fontsize=9.5, style="italic", bbox=dict(facecolor='#f9f9f9', edgecolor='#cccccc', boxstyle='round,pad=0.4'))

plt.subplots_adjust(top=0.94, bottom=0.07, left=0.07, right=0.95, hspace=0.4, wspace=0.28)
plt.savefig(os.path.join(out_dir, "combined_dashboard.png"), dpi=300)
plt.savefig(os.path.join(out_dir, "combined_dashboard.pdf"))
plt.savefig(os.path.join(out_dir, "combined_dashboard.svg"))
plt.close()

# 6. 生成汇总数据表格
generate_summary_data()

print("All tasks completed successfully!")
