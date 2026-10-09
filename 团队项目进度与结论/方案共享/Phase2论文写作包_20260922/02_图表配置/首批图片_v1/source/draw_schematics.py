"""Render editable Phase 2 schematic figures without using experimental imagery."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports"

INK = "#243247"
BLUE = "#3B77A5"
TEAL = "#268C84"
ORANGE = "#C7804C"
PURPLE = "#7869A7"
LIGHTGRAY = "#E8EDF2"
WHITE = "#FFFFFF"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, text, *, edge=INK, face=WHITE, fontsize=11, lw=1.7, radius=0.018,
        weight="normal", color=INK, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight=weight, linespacing=1.35, zorder=z + 1)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.8, rad=0.0, label=None, label_offset=(0, 0.015),
          style="-|>", z=1):
    p = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, linewidth=lw,
                        color=color, connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(p)
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1],
                label, ha="center", va="bottom", fontsize=9.5, color=INK, zorder=z + 1)
    return p


def panel_tag(ax, letter, title, x, y):
    ax.text(x, y, letter, ha="left", va="center", fontsize=14, fontweight="bold", color=WHITE,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=INK, edgecolor=INK))
    ax.text(x + 0.042, y, title, ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)


def divider(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=LIGHTGRAY, linewidth=1.2, zorder=0)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": 300},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT / f"{stem}.{ext}", facecolor=WHITE, **kwargs)
    plt.close(fig)


def fig_p2_1():
    fig = plt.figure(figsize=(12, 8.5), facecolor=WHITE)
    ax = fig.add_axes([0.035, 0.04, 0.93, 0.90])
    clean(ax)
    ax.text(0.0, 1.015, "图 P2-1｜Phase 2 数据流程与评估边界", fontsize=18, fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 0.985, "固定30条通路｜完整视野图像输入｜训练、内部验证与外部评价", fontsize=10.5, color=INK, va="top")

    # A: cohort and split
    panel_tag(ax, "A", "数据集合与角色", 0.01, 0.925)
    box(ax, 0.03, 0.710, 0.19, 0.115, "开发队列\n6 名患者", edge=BLUE, face="#F3F8FC", fontsize=12, weight="bold")
    box(ax, 0.03, 0.535, 0.19, 0.105, "训练集\n9,472 点位", edge=TEAL, face="#F0F9F7", fontsize=11.5, weight="bold")
    box(ax, 0.03, 0.385, 0.19, 0.105, "内部验证\n1,078 点位", edge=BLUE, face="#F3F8FC", fontsize=11.5, weight="bold")
    arrow(ax, (0.125, 0.710), (0.125, 0.642), color=INK)
    # Parallel split from the development cohort: do not route through training.
    ax.plot([0.125, 0.006, 0.006], [0.682, 0.682, 0.438], color=INK, lw=1.8, zorder=1)
    arrow(ax, (0.006, 0.438), (0.029, 0.438), color=INK)
    box(ax, 0.03, 0.195, 0.19, 0.110, "外部 XZY\n1 名既有患者｜1,039 点位", edge=ORANGE, face="#FDF6F0", fontsize=10.5, weight="bold")
    ax.text(0.125, 0.162, "外部单患者", ha="center", va="top", fontsize=9.5, color=ORANGE, fontweight="bold")

    # B: primary data process
    panel_tag(ax, "B", "全视野输入至点位级通路预测", 0.275, 0.925)
    xs = [0.275, 0.425, 0.575, 0.725, 0.865]
    ws = [0.115, 0.115, 0.115, 0.115, 0.105]
    texts = [
        "完整正方形\nH&E图像块",
        "缩放\n224 × 224\n双三次插值",
        "冻结病理\n图像编码器",
        "空间残差\nH / C / B",
        "30 维\n通路 z 分数",
    ]
    colors = [ORANGE, ORANGE, BLUE, PURPLE, TEAL]
    faces = ["#FDF6F0", "#FDF6F0", "#F3F8FC", "#F7F5FB", "#F0F9F7"]
    for x, w, t, c, f in zip(xs, ws, texts, colors, faces):
        box(ax, x, 0.665, w, 0.135, t, edge=c, face=f, fontsize=10.5, weight="bold")
    for i in range(4):
        arrow(ax, (xs[i] + ws[i], 0.732), (xs[i + 1], 0.732), color=INK)
    ax.text(0.273, 0.625, "每个点位保留身份：患者 / 切片 / 点位坐标", fontsize=10, color=INK, va="top")

    # C: fitting and selection boundary
    panel_tag(ax, "C", "训练、内部选模与外部冻结评估", 0.275, 0.535)
    box(ax, 0.285, 0.370, 0.205, 0.120, "训练集拟合\n模型参数与 z 标准化参数", edge=TEAL, face="#F0F9F7", fontsize=11, weight="bold")
    box(ax, 0.545, 0.370, 0.205, 0.120, "内部验证\n配方与检查点选择", edge=BLUE, face="#F3F8FC", fontsize=11, weight="bold")
    box(ax, 0.805, 0.370, 0.155, 0.120, "冻结模型\n外部 XZY 评价", edge=ORANGE, face="#FDF6F0", fontsize=10.5, weight="bold")
    arrow(ax, (0.490, 0.430), (0.545, 0.430), color=INK, label="训练后")
    arrow(ax, (0.750, 0.430), (0.805, 0.430), color=INK, label="冻结后")
    arrow(ax, (0.648, 0.370), (0.648, 0.320), color=BLUE, style="-")
    ax.text(0.648, 0.297, "仅内部信息用于选择", ha="center", va="top", fontsize=9.5, color=BLUE)
    ax.text(0.883, 0.297, "XZY 不参与选择", ha="center", va="top", fontsize=9.5, color=ORANGE)

    # D: anchored annotation and interface
    panel_tag(ax, "D", "固定目标与下游接口", 0.275, 0.195)
    box(ax, 0.285, 0.075, 0.295, 0.085, "固定30条通路及其顺序\n标准化参数沿用训练集", edge=INK, face=LIGHTGRAY, fontsize=10.5, weight="bold")
    box(ax, 0.680, 0.075, 0.280, 0.085, "Phase 2 输出：点位级 30 维通路 z 分数\n作为后续患者级任务的候选输入接口", edge=PURPLE, face="#F7F5FB", fontsize=10.5, weight="bold")
    arrow(ax, (0.580, 0.117), (0.680, 0.117), color=INK)
    ax.text(0.01, 0.015, "内部验证为开发患者的留出点；外部XZY为既有单患者评价。图像与坐标的物理尺度另见方法资料。",
            fontsize=9.5, color=INK, va="bottom")
    save(fig, "Fig_P2_1_data_flow")


def node(ax, x, y, *, color=TEAL, label=None, center=False):
    ax.add_patch(Circle((x, y), 0.018 if center else 0.013, facecolor=color if center else WHITE,
                        edgecolor=color, linewidth=1.8, zorder=3))
    if label:
        ax.text(x, y - 0.040, label, ha="center", va="top", fontsize=8.5, color=INK)


def fig_p2_2():
    fig = plt.figure(figsize=(12, 8.5), facecolor=WHITE)
    ax = fig.add_axes([0.035, 0.04, 0.93, 0.90])
    clean(ax)
    ax.text(0.0, 1.015, "图 P2-2｜全视野空间残差模型与三臂比较边界", fontsize=18, fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 0.985, "以UNI2-h为例：冻结编码器，训练共享映射、中心读出与空间残差头。邻域仅作结构示意。", fontsize=10.5, color=INK, va="top")

    # A: Center branch
    panel_tag(ax, "A", "中心点基础预测", 0.01, 0.925)
    box(ax, 0.03, 0.665, 0.145, 0.135, "UNI2-h\nCLS 1,536 维", edge=BLUE, face="#F3F8FC", fontsize=11.5, weight="bold")
    box(ax, 0.235, 0.665, 0.145, 0.135, "H：共享映射\n256 维表示", edge=PURPLE, face="#F7F5FB", fontsize=11, weight="bold")
    box(ax, 0.440, 0.665, 0.135, 0.135, "C：中心读出\n30 维基础预测", edge=TEAL, face="#F0F9F7", fontsize=10.5, weight="bold")
    arrow(ax, (0.175, 0.732), (0.235, 0.732), color=INK)
    arrow(ax, (0.380, 0.732), (0.440, 0.732), color=INK)
    ax.text(0.102, 0.622, "编码器冻结", ha="center", va="top", fontsize=9.5, color=BLUE)
    ax.text(0.405, 0.622, "H、C与B从头训练", ha="center", va="top", fontsize=9.5, color=PURPLE)

    # B: spatial residual
    panel_tag(ax, "B", "同切片邻域与空间残差", 0.01, 0.545)
    # mini graph
    node(ax, 0.105, 0.385, color=TEAL, label="中心点", center=True)
    for x, y in [(0.050, 0.445), (0.145, 0.462), (0.182, 0.390), (0.062, 0.325), (0.152, 0.305)]:
        node(ax, x, y, color=PURPLE, label=None)
        arrow(ax, (x, y), (0.105, 0.385), color=PURPLE, lw=1.1, style="->", z=2)
    ax.text(0.105, 0.265, "同切片邻居\n半径 2 个原生网格步长\n最多 12 个", ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.35)
    box(ax, 0.235, 0.330, 0.160, 0.115, "加权差分 u\nΣ a_i (H邻居 − H中心)", edge=PURPLE, face="#F7F5FB", fontsize=10.5, weight="bold")
    box(ax, 0.455, 0.330, 0.120, 0.115, "B：空间头\n30 维修正", edge=ORANGE, face="#FDF6F0", fontsize=10.5, weight="bold")
    arrow(ax, (0.195, 0.385), (0.235, 0.385), color=INK)
    arrow(ax, (0.395, 0.385), (0.455, 0.385), color=INK)
    ax.text(0.315, 0.292, "边权：距离 + CLS 相似性", ha="center", va="top", fontsize=9.5, color=INK)

    # C: sum output and isolated
    panel_tag(ax, "C", "输出与孤立点", 0.635, 0.925)
    box(ax, 0.660, 0.665, 0.125, 0.135, "C\n30 维基础预测", edge=TEAL, face="#F0F9F7", fontsize=11, weight="bold")
    box(ax, 0.830, 0.665, 0.125, 0.135, "B\n30 维修正", edge=ORANGE, face="#FDF6F0", fontsize=11, weight="bold")
    ax.text(0.804, 0.732, "+", fontsize=23, color=INK, va="center", ha="center")
    box(ax, 0.735, 0.515, 0.155, 0.115, "最终预测 C + B\n30 维通路 z 分数", edge=INK, face=LIGHTGRAY, fontsize=10.5, weight="bold")
    arrow(ax, (0.722, 0.665), (0.785, 0.630), color=INK)
    arrow(ax, (0.892, 0.665), (0.842, 0.630), color=INK)
    ax.text(0.812, 0.475, "孤立点：无可用邻居时，B = 0，仅输出 C", ha="center", va="top", fontsize=9.5, color=INK)

    # D: arm comparison matrix
    panel_tag(ax, "D", "三臂比较矩阵", 0.620, 0.405)
    x0, y0 = 0.620, 0.045
    colw = [0.085, 0.105, 0.105, 0.070]
    rowh = 0.075
    headers = ["模型臂", "配方", "B 空间头", "训练\n方式"]
    rows = [
        ["point", "point-07", "关闭", "从头训练"],
        ["no_b", "spatial-11", "关闭", "从头训练"],
        ["spatial", "spatial-11", "启用", "从头训练"],
    ]
    x = x0
    for w, h in zip(colw, headers):
        box(ax, x, y0 + 3 * rowh, w, rowh, h, edge=INK, face=LIGHTGRAY, fontsize=8.8, weight="bold", radius=0.008, lw=1.1)
        x += w
    for r, row in enumerate(rows):
        x = x0
        y = y0 + (2 - r) * rowh
        for c, (w, value) in enumerate(zip(colw, row)):
            face = "#FDF6F0" if (r == 2 and c == 2) else WHITE
            edge = ORANGE if (r == 2 and c == 2) else INK
            box(ax, x, y, w, rowh, value, edge=edge, face=face, fontsize=8.9, radius=0.006, lw=1.0)
            x += w
    ax.text(0.802, 0.018, "spatial 与 no_b 同配方，比较 B 的增量。", ha="center", va="top", fontsize=9.2, color=INK)
    ax.text(0.01, 0.015, "同切片有向一跳图；图半径以原生网格步长定义。", fontsize=9.5, color=INK, va="bottom")
    save(fig, "Fig_P2_2_model_structure")


def main():
    setup()
    fig_p2_1()
    fig_p2_2()


if __name__ == "__main__":
    main()
