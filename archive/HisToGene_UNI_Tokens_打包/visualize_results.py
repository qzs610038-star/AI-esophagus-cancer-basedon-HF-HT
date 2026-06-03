"""
visualize_results.py
====================
PFMval 项目通用可视化脚本

功能：
  1. 训练曲线可视化（Loss / MAE / R² / PCC）
  2. 模型参数展示面板（深色代码块风格）
  3. 逐通路指标表格（颜色编码）
  4. 逐通路 PCC 柱状图
  5. 综合报告图（合并以上4部分）

使用方式：
  # 命令行
  python visualize_results.py --model_name "HisToGene" \
      --history_csv histogene/training_history.csv \
      --predictions_csv histogene/infer_results/predictions.csv \
      --output_dir histogene/results_vis \
      --params '{"batch_size":64,"epochs":150,"lr":"1e-4"}'

  # Python 函数调用
  from visualize_results import generate_full_report
  generate_full_report(model_name="HisToGene", ...)
"""

import os
import json
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# ─────────────────────────────────────────────
# 字体配置：优先使用 Microsoft YaHei，回退到 SimHei
# ─────────────────────────────────────────────
def _setup_font():
    """配置中文字体，避免乱码"""
    for font_name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(
                matplotlib.font_manager.FontProperties(family=font_name),
                fallback_to_default=False
            )
            rcParams["font.family"] = font_name
            break
        except Exception:
            continue
    rcParams["axes.unicode_minus"] = False   # 解决负号显示问题

_setup_font()

# ─────────────────────────────────────────────
# 颜色主题
# ─────────────────────────────────────────────
COLOR_TRAIN   = "#4C9BE8"   # 训练集蓝色
COLOR_VAL     = "#E86F4C"   # 验证集橙红色
COLOR_BEST    = "#2ECC71"   # 最优点绿色
BG_DARK       = "#1E1E2E"   # 参数面板深色背景
BG_PANEL      = "#2B2B3B"   # 代码行背景
TEXT_LIGHT    = "#CDD6F4"   # 浅色文字
TEXT_ACCENT   = "#89DCEB"   # 高亮关键字颜色（青色）
TEXT_VALUE    = "#A6E3A1"   # 值颜色（绿色）

# 红→绿渐变色映射（用于表格着色）
RED_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "rg", ["#E74C3C", "#F39C12", "#2ECC71"], N=256
)

DPI = 150   # 输出分辨率


# ══════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════

def _warn(msg: str):
    """打印警告信息"""
    print(f"[WARNING] {msg}")


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def _load_history(csv_path: str) -> pd.DataFrame | None:
    """
    读取训练历史 CSV，返回 DataFrame。
    列格式可能包含 train_mape 等额外列，统一只取所需列。
    """
    if not os.path.isfile(csv_path):
        _warn(f"训练历史文件不存在：{csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        # 删除最后一行（可能为空行）
        df = df.dropna(subset=["epoch"])
        df["epoch"] = df["epoch"].astype(int)
        return df
    except Exception as e:
        _warn(f"读取训练历史失败：{e}")
        return None


def _load_predictions(csv_path: str) -> pd.DataFrame | None:
    """
    读取推理结果 CSV。
    期望格式：patch_id, true_xxx, ..., pred_xxx, ...
    """
    if csv_path is None or not os.path.isfile(csv_path):
        if csv_path is not None:
            _warn(f"推理结果文件不存在：{csv_path}")
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        _warn(f"读取推理结果失败：{e}")
        return None


def _calc_per_pathway_metrics(df_pred: pd.DataFrame) -> pd.DataFrame | None:
    """
    根据 predictions.csv 计算每条通路的 MSE / MAE / R² / PCC。
    返回 DataFrame，列：pathway, MSE, MAE, R2, PCC。
    """
    if df_pred is None:
        return None

    # 推断通路名：从 true_xxx 列名中提取 xxx
    true_cols = [c for c in df_pred.columns if c.startswith("true_")]
    pathways = [c[5:] for c in true_cols]

    if not pathways:
        _warn("predictions.csv 中未找到 true_* 列，跳过逐通路指标计算")
        return None

    rows = []
    for pw in pathways:
        tc = f"true_{pw}"
        pc = f"pred_{pw}"
        if tc not in df_pred.columns or pc not in df_pred.columns:
            continue
        y_true = df_pred[tc].values
        y_pred = df_pred[pc].values

        mse  = float(np.mean((y_true - y_pred) ** 2))
        mae  = float(np.mean(np.abs(y_true - y_pred)))

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # PCC
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pcc = float("nan")

        rows.append({"pathway": pw, "MSE": mse, "MAE": mae, "R2": r2, "PCC": pcc})

    if not rows:
        return None

    df_metrics = pd.DataFrame(rows)

    # 添加 mean 行
    mean_row = {
        "pathway": "mean",
        "MSE":  df_metrics["MSE"].mean(),
        "MAE":  df_metrics["MAE"].mean(),
        "R2":   df_metrics["R2"].mean(),
        "PCC":  df_metrics["PCC"].mean(),
    }
    df_metrics = pd.concat(
        [pd.DataFrame([mean_row]), df_metrics], ignore_index=True
    )
    return df_metrics


def _extract_metrics_from_history(df_hist: pd.DataFrame, pathways: list[str]) -> pd.DataFrame | None:
    """
    当没有 predictions.csv 时，从训练历史最后一行提取汇总验证指标，
    生成只含 mean 行的简化指标表（无逐通路分解）。
    """
    if df_hist is None:
        return None
    last = df_hist.iloc[-1]
    rows = [{
        "pathway": "mean (val)",
        "MSE":  float(last.get("val_loss", float("nan"))),
        "MAE":  float(last.get("val_mae",  float("nan"))),
        "R2":   float(last.get("val_r2",   float("nan"))),
        "PCC":  float(last.get("val_pcc",  float("nan"))),
    }]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
#  Part 1：训练曲线
# ══════════════════════════════════════════════

def plot_training_curves(df_hist: pd.DataFrame, output_path: str, model_name: str = ""):
    """
    绘制 2×2 训练曲线子图：Loss / MAE / R² / PCC。
    最佳 val_pcc 所在 epoch 用竖虚线标注。
    """
    epochs = df_hist["epoch"].values

    # 4个指标配置：(列名后缀, 显示名称, y轴方向是否越高越好)
    metrics = [
        ("loss", "Loss",   False),
        ("mae",  "MAE",    False),
        ("r2",   "R²",     True),
        ("pcc",  "PCC",    True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"{model_name}  训练曲线" if model_name else "训练曲线",
        fontsize=16, fontweight="bold", y=0.98
    )

    # 找最佳 val_pcc epoch
    best_epoch = None
    best_pcc   = None
    if "val_pcc" in df_hist.columns:
        idx = df_hist["val_pcc"].idxmax()
        best_epoch = df_hist.loc[idx, "epoch"]
        best_pcc   = df_hist.loc[idx, "val_pcc"]

    for ax, (suffix, label, higher_better) in zip(axes.flatten(), metrics):
        train_col = f"train_{suffix}"
        val_col   = f"val_{suffix}"

        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

        if train_col in df_hist.columns:
            ax.plot(epochs, df_hist[train_col].values,
                    color=COLOR_TRAIN, linewidth=1.8,
                    label="Train", zorder=3)
        if val_col in df_hist.columns:
            ax.plot(epochs, df_hist[val_col].values,
                    color=COLOR_VAL, linewidth=1.8,
                    label="Val", zorder=3)

        # 标注最佳 val_pcc epoch（对所有子图均标注）
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color=COLOR_BEST,
                       linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
            ymin, ymax = ax.get_ylim()
            ax.text(
                best_epoch + 0.3, ymin + (ymax - ymin) * 0.03,
                f"best\nE{best_epoch}",
                color=COLOR_BEST, fontsize=7.5, va="bottom", zorder=4
            )

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 训练曲线已保存：{output_path}")


# ══════════════════════════════════════════════
#  Part 2：模型参数面板
# ══════════════════════════════════════════════

def plot_params_panel(params: dict, output_path: str, model_name: str = ""):
    """
    在深色背景上渲染参数列表，类似 IDE 代码块风格。
    左侧行号，右侧 key = value 格式。
    """
    if not params:
        _warn("参数字典为空，跳过参数面板")
        return

    items = list(params.items())
    n_lines = len(items)

    # 自适应图片高度
    fig_h = max(2.5, 0.45 * n_lines + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    # 标题
    title_text = f"# {model_name}  Model Config" if model_name else "# Model Config"
    ax.text(0.02, 0.97, title_text,
            transform=ax.transAxes,
            fontsize=11, color=TEXT_ACCENT,
            fontfamily="monospace", fontweight="bold",
            va="top")

    # 参数行
    line_h = 1.0 / (n_lines + 2)
    for i, (key, val) in enumerate(items):
        y_pos = 0.90 - i * (0.82 / max(n_lines, 1))

        # 行背景条（交替颜色）
        bg_color = BG_PANEL if i % 2 == 0 else BG_DARK
        rect = mpatches.FancyBboxPatch(
            (0.01, y_pos - 0.025), 0.98, 0.048,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor="none",
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)

        # 行号
        ax.text(0.035, y_pos,
                f"{i+1:>3}",
                transform=ax.transAxes,
                fontsize=9.5, color="#6C7086",
                fontfamily="monospace", va="center", zorder=2)

        # key
        ax.text(0.09, y_pos,
                str(key),
                transform=ax.transAxes,
                fontsize=9.5, color=TEXT_LIGHT,
                fontfamily="monospace", va="center", zorder=2)

        # = 号
        ax.text(0.38, y_pos,
                "=",
                transform=ax.transAxes,
                fontsize=9.5, color="#CBA6F7",
                fontfamily="monospace", va="center", zorder=2)

        # value（绿色）
        ax.text(0.42, y_pos,
                str(val),
                transform=ax.transAxes,
                fontsize=9.5, color=TEXT_VALUE,
                fontfamily="monospace", va="center", zorder=2)

    plt.tight_layout(pad=0.3)
    _ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"[OK] 参数面板已保存：{output_path}")


# ══════════════════════════════════════════════
#  Part 3：逐通路指标表格
# ══════════════════════════════════════════════

def plot_metrics_table(df_metrics: pd.DataFrame, output_path: str, model_name: str = ""):
    """
    绘制逐通路指标表格，R² 和 PCC 列使用红→绿渐变色编码。
    第一行为 mean 汇总行。
    """
    if df_metrics is None or df_metrics.empty:
        _warn("指标数据为空，跳过指标表格")
        return

    n_rows = len(df_metrics)
    col_labels = ["Pathway", "MSE", "MAE", "R²", "PCC"]
    col_keys   = ["pathway", "MSE", "MAE", "R2", "PCC"]

    fig_h = max(3.0, 0.4 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("#F8F9FA")
    ax.axis("off")

    ax.set_title(
        f"{model_name}  逐通路指标" if model_name else "逐通路指标",
        fontsize=13, fontweight="bold", pad=10
    )

    # 构建表格数据
    cell_text  = []
    cell_colors = []
    for _, row in df_metrics.iterrows():
        row_vals   = []
        row_colors = []
        for ck in col_keys:
            v = row[ck]
            if isinstance(v, float):
                row_vals.append(f"{v:.4f}")
            else:
                row_vals.append(str(v))
            row_colors.append("#FFFFFF")   # 默认白色，后面覆盖
        cell_text.append(row_vals)
        cell_colors.append(row_colors)

    # 对 R² 列（index 3）和 PCC 列（index 4）做颜色编码
    for color_col_idx, col_key in [(3, "R2"), (4, "PCC")]:
        vals = df_metrics[col_key].values.astype(float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        vmin, vmax = valid.min(), valid.max()
        v_range = vmax - vmin if vmax > vmin else 1.0

        for ri, v in enumerate(vals):
            if np.isnan(v):
                continue
            norm_v = (v - vmin) / v_range      # 0~1
            rgba = RED_GREEN_CMAP(norm_v)
            # 调低透明度，不要太深
            light_rgba = (rgba[0], rgba[1], rgba[2], 0.45)
            cell_colors[ri][color_col_idx] = light_rgba

    # mean 行用浅金色背景
    for ci in range(len(col_keys)):
        cell_colors[0][ci] = "#FFF3CD"

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)

    # 表头加粗
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 指标表格已保存：{output_path}")


# ══════════════════════════════════════════════
#  Part 4：逐通路 PCC 柱状图
# ══════════════════════════════════════════════

def plot_pcc_barplot(df_metrics: pd.DataFrame, output_path: str, model_name: str = ""):
    """
    横轴通路名，纵轴 PCC，柱子颜色按 PCC 大小从红到绿着色。
    排除 mean 行，添加均值水平线。
    """
    if df_metrics is None or df_metrics.empty:
        _warn("指标数据为空，跳过 PCC 柱状图")
        return

    # 过滤掉 mean/mean(val) 行，只保留具体通路
    df_pw = df_metrics[~df_metrics["pathway"].str.startswith("mean")].copy()
    if df_pw.empty:
        _warn("没有逐通路数据，跳过 PCC 柱状图（仅有 mean 行）")
        return

    pathways = df_pw["pathway"].tolist()
    pcc_vals  = df_pw["PCC"].values.astype(float)
    n = len(pathways)

    # 颜色映射
    pcc_min = pcc_vals[~np.isnan(pcc_vals)].min() if not np.all(np.isnan(pcc_vals)) else 0.0
    pcc_max = pcc_vals[~np.isnan(pcc_vals)].max() if not np.all(np.isnan(pcc_vals)) else 1.0
    pcc_range = pcc_max - pcc_min if pcc_max > pcc_min else 1.0
    bar_colors = [RED_GREEN_CMAP((v - pcc_min) / pcc_range)
                  if not np.isnan(v) else (0.7, 0.7, 0.7, 1.0)
                  for v in pcc_vals]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(np.arange(n), pcc_vals, color=bar_colors,
                  width=0.6, edgecolor="white", linewidth=0.8, zorder=3)

    # 在柱子顶部标注数值
    for bar, v in zip(bars, pcc_vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333"
            )

    # 均值水平线
    mean_pcc = float(np.nanmean(pcc_vals))
    ax.axhline(y=mean_pcc, color="#E74C3C", linestyle="--",
               linewidth=1.5, label=f"Mean PCC = {mean_pcc:.4f}", zorder=4)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(pathways, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("PCC", fontsize=11)
    ax.set_title(
        f"{model_name}  逐通路 PCC" if model_name else "逐通路 PCC",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
    ax.set_ylim(bottom=min(0, float(np.nanmin(pcc_vals)) - 0.05))

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] PCC 柱状图已保存：{output_path}")


# ══════════════════════════════════════════════
#  Part 5：综合报告图
# ══════════════════════════════════════════════

def plot_full_report(
    df_hist: pd.DataFrame | None,
    df_metrics: pd.DataFrame | None,
    params: dict,
    output_path: str,
    model_name: str = "",
):
    """
    将4个可视化部分合成一张综合报告图（A4 纵向比例约 1:1.4）。
    布局：
      行1：参数面板（左宽）+ 最佳指标摘要（右窄）
      行2：训练曲线（2×2 小图，展开为4列）
      行3：指标表格（左宽）+ PCC 柱状图（右窄）
    """
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#F0F0F0")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.6, 3.0, 3.5],
        hspace=0.35
    )

    # ── 行 1：参数面板 + 摘要 ──────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[2, 1], wspace=0.25
    )
    ax_params  = fig.add_subplot(gs_top[0])
    ax_summary = fig.add_subplot(gs_top[1])

    # 参数面板
    ax_params.set_facecolor(BG_DARK)
    ax_params.axis("off")
    title_text = f"# {model_name}  Model Config" if model_name else "# Model Config"
    ax_params.text(0.03, 0.96, title_text,
                   transform=ax_params.transAxes,
                   fontsize=10, color=TEXT_ACCENT,
                   fontfamily="monospace", fontweight="bold",
                   va="top")
    items = list(params.items())
    n_p = len(items)
    for i, (k, v) in enumerate(items):
        y = 0.88 - i * (0.82 / max(n_p, 1))
        bg = BG_PANEL if i % 2 == 0 else BG_DARK
        rect = mpatches.FancyBboxPatch(
            (0.01, y - 0.042), 0.98, 0.06,
            boxstyle="round,pad=0.004",
            facecolor=bg, edgecolor="none",
            transform=ax_params.transAxes, zorder=1
        )
        ax_params.add_patch(rect)
        ax_params.text(0.04, y, f"{i+1:>2}",
                       transform=ax_params.transAxes,
                       fontsize=8.5, color="#6C7086",
                       fontfamily="monospace", va="center", zorder=2)
        ax_params.text(0.10, y, str(k),
                       transform=ax_params.transAxes,
                       fontsize=8.5, color=TEXT_LIGHT,
                       fontfamily="monospace", va="center", zorder=2)
        ax_params.text(0.50, y, "=",
                       transform=ax_params.transAxes,
                       fontsize=8.5, color="#CBA6F7",
                       fontfamily="monospace", va="center", zorder=2)
        ax_params.text(0.56, y, str(v),
                       transform=ax_params.transAxes,
                       fontsize=8.5, color=TEXT_VALUE,
                       fontfamily="monospace", va="center", zorder=2)
    # 参数面板外框
    for sp in ax_params.spines.values():
        sp.set_edgecolor(BG_DARK)

    # 最佳指标摘要
    ax_summary.set_facecolor("#FFFFFF")
    ax_summary.axis("off")
    ax_summary.set_title("Best Val Metrics", fontsize=11,
                          fontweight="bold", pad=6, color="#2C3E50")
    summary_lines = []
    if df_hist is not None:
        last = df_hist.iloc[-1]
        if "val_pcc" in df_hist.columns:
            best_idx  = df_hist["val_pcc"].idxmax()
            best_row  = df_hist.loc[best_idx]
            summary_lines.append(f"Best Epoch : {int(best_row['epoch'])}")
            summary_lines.append(f"Val PCC    : {best_row['val_pcc']:.4f}")
            if "val_r2"  in df_hist.columns:
                summary_lines.append(f"Val R²     : {best_row['val_r2']:.4f}")
            if "val_mae" in df_hist.columns:
                summary_lines.append(f"Val MAE    : {best_row['val_mae']:.4f}")
            if "val_loss" in df_hist.columns:
                summary_lines.append(f"Val Loss   : {best_row['val_loss']:.4f}")
            summary_lines.append("")
            summary_lines.append(f"Last Epoch : {int(last['epoch'])}")
            summary_lines.append(f"Val PCC    : {last['val_pcc']:.4f}")
        if df_metrics is not None and not df_metrics.empty:
            mean_row = df_metrics[df_metrics["pathway"].str.startswith("mean")].iloc[0]
            summary_lines.append("")
            summary_lines.append(f"Mean PCC   : {mean_row['PCC']:.4f}")
            summary_lines.append(f"Mean R²    : {mean_row['R2']:.4f}")

    for li, line in enumerate(summary_lines):
        ax_summary.text(
            0.08, 0.90 - li * 0.085, line,
            transform=ax_summary.transAxes,
            fontsize=9, color="#2C3E50",
            fontfamily="monospace", va="top"
        )

    # ── 行 2：训练曲线 ──────────────────────────
    if df_hist is not None:
        gs_curves = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1],
            hspace=0.40, wspace=0.30
        )
        metrics_cfg = [
            ("loss", "Loss"),
            ("mae",  "MAE"),
            ("r2",   "R²"),
            ("pcc",  "PCC"),
        ]
        epochs = df_hist["epoch"].values
        best_epoch = None
        if "val_pcc" in df_hist.columns:
            best_epoch = int(df_hist.loc[df_hist["val_pcc"].idxmax(), "epoch"])

        for idx, (suffix, label) in enumerate(metrics_cfg):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs_curves[row, col])
            ax.set_facecolor("#FFFFFF")
            ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")

            tcol, vcol = f"train_{suffix}", f"val_{suffix}"
            if tcol in df_hist.columns:
                ax.plot(epochs, df_hist[tcol].values,
                        color=COLOR_TRAIN, linewidth=1.6,
                        label="Train", zorder=3)
            if vcol in df_hist.columns:
                ax.plot(epochs, df_hist[vcol].values,
                        color=COLOR_VAL, linewidth=1.6,
                        label="Val", zorder=3)
            if best_epoch is not None:
                ax.axvline(x=best_epoch, color=COLOR_BEST,
                           linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
                ylims = ax.get_ylim()
                ax.text(best_epoch + 0.2, ylims[0] + (ylims[1]-ylims[0])*0.03,
                        f"E{best_epoch}", color=COLOR_BEST,
                        fontsize=7, va="bottom", zorder=4)

            ax.set_title(label, fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.legend(fontsize=7.5, framealpha=0.8)
            ax.tick_params(labelsize=7.5)
    else:
        ax_nc = fig.add_subplot(outer[1])
        ax_nc.axis("off")
        ax_nc.text(0.5, 0.5, "训练历史数据不可用",
                   transform=ax_nc.transAxes,
                   ha="center", va="center", fontsize=14, color="#999999")

    # ── 行 3：指标表格 + PCC 柱状图 ─────────────
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.2, 1.8], wspace=0.25
    )
    ax_tbl = fig.add_subplot(gs_bottom[0])
    ax_bar = fig.add_subplot(gs_bottom[1])

    # 指标表格
    ax_tbl.axis("off")
    ax_tbl.set_title("逐通路指标", fontsize=10.5,
                      fontweight="bold", pad=6)
    if df_metrics is not None and not df_metrics.empty:
        col_labels = ["Pathway", "MSE", "MAE", "R²", "PCC"]
        col_keys   = ["pathway", "MSE", "MAE", "R2", "PCC"]

        cell_text   = []
        cell_colors = []
        for _, row in df_metrics.iterrows():
            rv, rc = [], []
            for ck in col_keys:
                v = row[ck]
                rv.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                rc.append("#FFFFFF")
            cell_text.append(rv)
            cell_colors.append(rc)

        # R²/PCC 颜色编码
        for ci, ck in [(3, "R2"), (4, "PCC")]:
            vals = df_metrics[ck].values.astype(float)
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            vmin, vmax = valid.min(), valid.max()
            vr = vmax - vmin if vmax > vmin else 1.0
            for ri, v in enumerate(vals):
                if not np.isnan(v):
                    rgba = RED_GREEN_CMAP((v - vmin) / vr)
                    cell_colors[ri][ci] = (rgba[0], rgba[1], rgba[2], 0.45)
        for ci in range(len(col_keys)):
            cell_colors[0][ci] = "#FFF3CD"

        tbl = ax_tbl.table(
            cellText=cell_text, colLabels=col_labels,
            cellColours=cell_colors,
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if r == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
    else:
        ax_tbl.text(0.5, 0.5, "指标数据不可用",
                    transform=ax_tbl.transAxes,
                    ha="center", va="center", fontsize=11, color="#999999")

    # PCC 柱状图
    ax_bar.set_facecolor("#FFFFFF")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
    ax_bar.set_title("逐通路 PCC", fontsize=10.5,
                      fontweight="bold", pad=6)

    if df_metrics is not None and not df_metrics.empty:
        df_pw = df_metrics[~df_metrics["pathway"].str.startswith("mean")]
        if not df_pw.empty:
            pathways  = df_pw["pathway"].tolist()
            pcc_vals  = df_pw["PCC"].values.astype(float)
            n_pw      = len(pathways)
            pmin = pcc_vals[~np.isnan(pcc_vals)].min() if not np.all(np.isnan(pcc_vals)) else 0.0
            pmax = pcc_vals[~np.isnan(pcc_vals)].max() if not np.all(np.isnan(pcc_vals)) else 1.0
            pr   = pmax - pmin if pmax > pmin else 1.0
            bcolors = [RED_GREEN_CMAP((v - pmin) / pr)
                       if not np.isnan(v) else (0.7, 0.7, 0.7, 1.0)
                       for v in pcc_vals]

            bars = ax_bar.bar(np.arange(n_pw), pcc_vals,
                              color=bcolors, width=0.6,
                              edgecolor="white", linewidth=0.8, zorder=3)
            for bar, v in zip(bars, pcc_vals):
                if not np.isnan(v):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.003,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color="#333333"
                    )
            mean_pcc = float(np.nanmean(pcc_vals))
            ax_bar.axhline(y=mean_pcc, color="#E74C3C", linestyle="--",
                           linewidth=1.4,
                           label=f"Mean={mean_pcc:.4f}", zorder=4)
            ax_bar.set_xticks(np.arange(n_pw))
            ax_bar.set_xticklabels(pathways, rotation=30,
                                   ha="right", fontsize=8.5)
            ax_bar.set_ylabel("PCC", fontsize=9)
            ax_bar.legend(fontsize=8)
            ax_bar.set_ylim(bottom=min(0, float(np.nanmin(pcc_vals)) - 0.05))
        else:
            ax_bar.text(0.5, 0.5, "无逐通路数据",
                        transform=ax_bar.transAxes,
                        ha="center", va="center", fontsize=11, color="#999999")
    else:
        ax_bar.text(0.5, 0.5, "指标数据不可用",
                    transform=ax_bar.transAxes,
                    ha="center", va="center", fontsize=11, color="#999999")

    # 大标题
    fig.suptitle(
        f"{model_name}  训练结果综合报告" if model_name else "训练结果综合报告",
        fontsize=17, fontweight="bold", y=0.995
    )

    _ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="#F0F0F0")
    plt.close(fig)
    print(f"[OK] 综合报告已保存：{output_path}")


# ══════════════════════════════════════════════
#  主入口：generate_full_report
# ══════════════════════════════════════════════

def generate_full_report(
    model_name: str = "",
    history_csv: str | None = None,
    predictions_csv: str | None = None,
    output_dir: str = "results_vis",
    params: dict | None = None,
    pathways: list[str] | None = None,
    prefix: str = "",
    actual_output_dir: str | None = None,
) -> str:
    """
    生成完整可视化报告，包含：
      - training_curves.png
      - params_panel.png
      - metrics_table.png
      - pcc_barplot.png
      - full_report.png

    参数
    ----
    model_name        : 模型名称（用于标题）
    history_csv       : 训练历史 CSV 路径
    predictions_csv   : 推理结果 CSV 路径（可选，如无则从历史提取）
    output_dir        : 输出目录（当 actual_output_dir 为 None 时创建时间戳子文件夹）
    params            : 模型参数字典
    pathways          : 通路名称列表（当 predictions_csv 为 None 时备用）
    prefix            : 时间戳子文件夹前缀（如数据集名称），默认为空
    actual_output_dir : 已有的输出目录路径（可选，如指定则直接使用，不创建新的时间戳子文件夹）

    返回
    ----
    str: 实际输出目录路径
    """
    if params is None:
        params = {}

    # 确定实际输出目录
    if actual_output_dir is None:
        # 创建时间戳子文件夹（可选前缀）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{prefix}_{timestamp}" if prefix else timestamp
        actual_output_dir = os.path.join(output_dir, folder_name)
    _ensure_dir(actual_output_dir)
    print(f"[INFO] 可视化结果将保存到: {actual_output_dir}")

    # 加载数据
    df_hist = _load_history(history_csv) if history_csv else None
    df_pred = _load_predictions(predictions_csv)

    # 计算逐通路指标
    if df_pred is not None:
        df_metrics = _calc_per_pathway_metrics(df_pred)
    elif df_hist is not None:
        df_metrics = _extract_metrics_from_history(df_hist, pathways or [])
    else:
        df_metrics = None

    # ── 各图生成 ──────────────────────────────
    if df_hist is not None:
        plot_training_curves(
            df_hist,
            os.path.join(actual_output_dir, "training_curves.png"),
            model_name
        )

    if params:
        plot_params_panel(
            params,
            os.path.join(actual_output_dir, "params_panel.png"),
            model_name
        )

    if df_metrics is not None:
        plot_metrics_table(
            df_metrics,
            os.path.join(actual_output_dir, "metrics_table.png"),
            model_name
        )
        plot_pcc_barplot(
            df_metrics,
            os.path.join(actual_output_dir, "pcc_barplot.png"),
            model_name
        )

    # 综合报告
    plot_full_report(
        df_hist,
        df_metrics,
        params,
        os.path.join(actual_output_dir, "full_report.png"),
        model_name
    )

    print(f"\n[完成] 所有图片已保存至 {os.path.abspath(actual_output_dir)}")
    return actual_output_dir


# ══════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="PFMval 可视化脚本 - 生成训练结果可视化图片"
    )
    parser.add_argument("--model_name",      default="",   help="模型名称")
    parser.add_argument("--history_csv",     default=None, help="训练历史 CSV 路径")
    parser.add_argument("--predictions_csv", default=None, help="推理结果 CSV 路径（可选）")
    parser.add_argument("--output_dir",      default="results_vis", help="输出目录")
    parser.add_argument("--params",          default="{}", help="JSON 格式的模型参数字典")
    parser.add_argument("--pathways",        default=None, help="逗号分隔的通路名称列表")
    parser.add_argument("--prefix",          default="",   help="时间戳子文件夹前缀（如数据集名称）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # 解析 params JSON
    try:
        params_dict = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"[ERROR] --params 解析失败：{e}")
        params_dict = {}

    # 解析 pathways
    pathways_list = None
    if args.pathways:
        pathways_list = [p.strip() for p in args.pathways.split(",")]

    generate_full_report(
        model_name=args.model_name,
        history_csv=args.history_csv,
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        params=params_dict,
        pathways=pathways_list,
        prefix=args.prefix,
    )
