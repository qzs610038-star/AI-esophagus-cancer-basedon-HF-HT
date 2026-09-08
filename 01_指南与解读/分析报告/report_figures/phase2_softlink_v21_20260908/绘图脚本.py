import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 基础配置与路径定义
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV_PATH = os.path.join(PROJECT_ROOT,
    'experiments',
    'results',
    'phase2_softlink_local_v2',
    '20260908_005325_810_8d306c10',
    'analysis',
    'metrics_by_seed.csv',
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT,
    '01_指南与解读',
    '分析报告',
    'report_figures',
    'phase2_softlink_v21_20260908',
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[(df['endpoint'] == 'best') & df['arm'].isin(['point', 'relation', 'spatial', 'joint'])]
assert len(df) == 24

# 字体与排版设置
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
    'Arial Unicode MS',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

# 2. 面板指标定义 (共 10 个面板)
PANELS = [
  {
    "split": "internal_val",
    "col": "pcc",
    "title": "内部宏平均PCC",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "pcc",
    "title": "外部宏平均PCC",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "internal_val",
    "col": "flattened_pcc",
    "title": "内部展平PCC",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "flattened_pcc",
    "title": "外部展平PCC",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "ccc",
    "title": "外部CCC",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "internal_val",
    "col": "raw_r2",
    "title": "内部R²",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "raw_r2",
    "title": "外部R²",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "spearman",
    "title": "外部Spearman",
    "higher_better": True,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "z_rmse",
    "title": "外部zRMSE",
    "higher_better": False,
    "fmt": "{:.4f}"
  },
  {
    "split": "XZY",
    "col": "raw_mae",
    "title": "外部rawMAE",
    "higher_better": False,
    "fmt": "{:.1f}"
  }
]

ARMS = ['point', 'relation', 'spatial', 'joint']
ARM_LABELS = [
    '单点 (Point)',
    '关系 (Relation)',
    '空间 (Spatial)',
    '联合 (Joint)',
]
COLORS = ['#3274A1', '#E1812C', '#3B925F', '#C03D3E']

fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300)
axes = axes.flatten()

for idx, panel in enumerate(PANELS):
  ax = axes[idx]
  split = panel['split']
  col = panel['col']
  higher = panel['higher_better']
  fmt = panel['fmt']

  sub_df = df[df['split'] == split]

  all_y_vals = []
  all_low_bounds = []
  all_top_bounds = []

  for a_idx, arm in enumerate(ARMS):
    arm_data = sub_df[sub_df['arm'] == arm][col].dropna().values
    assert len(arm_data) == 3

    mean_val = np.mean(arm_data)
    sd_val = np.std(arm_data, ddof=1)
    color = COLORS[a_idx]

    # 1. 散点打点 (轻微横向抖动以避免重叠)
    jitter = np.linspace(-0.12, 0.12, len(arm_data))
    ax.scatter(
        a_idx + jitter,
        arm_data,
        color=color,
        alpha=0.75,
        s=55,
        edgecolors='black',
        linewidths=0.6,
        zorder=3,
    )

    # 2. 均值菱形与样本标准差误差棒 (SD, 非置信区间)
    ax.errorbar(
        a_idx,
        mean_val,
        yerr=sd_val,
        fmt='D',
        color=color,
        ecolor=color,
        markersize=9,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        zorder=4,
        label=ARM_LABELS[a_idx] if idx == 0 else '',
    )

    top_val = max(np.max(arm_data), mean_val + sd_val)
    all_y_vals.extend(arm_data)
    all_low_bounds.append(mean_val - sd_val)
    all_top_bounds.append((a_idx, mean_val, top_val))

  # 3. 动态范围计算与标签定位 (确保不截断且标签不覆盖数据点)
  y_min_data = min(np.min(all_y_vals), min(all_low_bounds))
  y_max_data = max([t[2] for t in all_top_bounds])
  y_span = y_max_data - y_min_data if y_max_data != y_min_data else 1.0

  y_lim_bottom = y_min_data - 0.12 * y_span
  y_lim_top = y_max_data + 0.28 * y_span
  ax.set_ylim(y_lim_bottom, y_lim_top)
  if col == 'raw_r2' and y_lim_bottom < 0 < y_lim_top:
    ax.axhline(0, color='#777777', linestyle=':', linewidth=1)

  # 绘制数值标签 (置于数据及误差棒顶端之上)
  for a_idx, mean_val, top_val in all_top_bounds:
    label_y = top_val + 0.06 * y_span
    ax.text(
        a_idx,
        label_y,
        fmt.format(mean_val),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color=COLORS[a_idx],
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
        ),
    )

  # 标题与刻度装饰
  direction_str = '↑ 越高越好' if higher else '↓ 越低越好'
  ax.set_title(
      f"{panel['title']}\n({direction_str})",
      fontsize=12,
      fontweight='bold',
      pad=10,
  )
  ax.set_xticks(range(len(ARMS)))
  ax.set_xticklabels(ARM_LABELS, fontsize=10, rotation=15)
  ax.grid(True, linestyle='--', alpha=0.4, axis='y')
  ax.set_xlim(-0.5, len(ARMS) - 0.5)

# 图底注脚与整体说明
plt.suptitle(
    'Phase 2 软连接：空间贡献主要收益，关系与联合未见明确增量',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)
fig.text(
    0.5,
    0.01,
    '注：散点代表种子 42/43/44 单次运行，实心菱形为算术均值，误差棒为样本标准差' 
    ' (SD，非 CI)。内部验证为 6 例患者等权平均，外部测试集 XZY 为单例患者' 
    ' (n=1)。\n内部 R²：0.1976 → 0.1746；残差 Moran’s I：0.1509 → 0.1729，空间臂并非全指标改善。'
    ' 数据源：metrics_by_seed.csv，正式最佳端点。',
    ha='center',
    fontsize=10,
    color='#444444',
)

plt.tight_layout(rect=[0, 0.055, 1, 0.95])

# 输出 PNG 与 SVG
png_path = os.path.join(OUTPUT_DIR, '关键结果总览.png')
svg_path = os.path.join(OUTPUT_DIR, '关键结果总览.svg')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(svg_path, bbox_inches='tight')
plt.close()
print(f'绘图完成，图片已输出至:\n  {png_path}\n  {svg_path}')
