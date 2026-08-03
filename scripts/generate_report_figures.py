import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局绘图风格 - Publication Quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

# 创建输出目录
out_dir = r"D:\AI空间转录病理研究\PFMval_new\01_指南与解读\分析报告\report_figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 图 1：Huber Loss vs MSE Loss 高端对比图
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.2]})

# 配色
color_mse = '#1E88E5'   # 专业蓝色
color_huber = '#E53935' # 专业红色
color_accent = '#43A047' # 绿色

# 左图：全局 Pooled PCC 对比
categories = ['MSE Loss\n(Baseline)', 'Huber Loss\n(delta=1.0)']
pcc_values = [0.6548958, 0.6526907]

bars = ax1.bar(categories, pcc_values, color=[color_mse, color_huber], width=0.45, edgecolor='black', linewidth=1.2, zorder=3)
ax1.set_ylim(0.640, 0.662)
ax1.set_ylabel('External XZY Pooled PCC', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('(A) Global Pooled PCC Comparison', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 添加数据标签
for bar, val in zip(bars, pcc_values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.0005, f'{val:.4f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加 Δ 标注线
ax1.annotate('', xy=(1, 0.6526907), xytext=(1, 0.6548958),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(1.08, 0.6538, r'$\Delta = -0.0022$' + '\n(Degraded)', 
         va='center', ha='left', fontsize=10, color='#D32F2F', fontweight='bold')

# 右图：30 条通路变化方向分布
pathway_counts = [17, 13]
labels = ['Improved\n(17 Path)', 'Degraded\n(13 Path)']
colors = ['#4CAF50', '#FF9800']

bars2 = ax2.barh(labels, pathway_counts, color=colors, height=0.45, edgecolor='black', linewidth=1.2, zorder=3)
ax2.set_xlim(0, 22)
ax2.set_xlabel('Number of Pathways (Total = 30)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('(B) 30-Pathway PCC Directional Delta', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

for bar, count in zip(bars2, pathway_counts):
    ax2.text(count + 0.5, bar.get_y() + bar.get_height()/2, f'{count} / 30 ({count/30*100:.1f}%)', 
             va='center', ha='left', fontsize=11, fontweight='bold')

# 总体注释
fig.suptitle('Figure 1: Performance Comparison of MSE vs Huber Loss on External Challenge Set', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path1 = os.path.join(out_dir, "fig1_huber_vs_mse.png")
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 图 2：基线数据修复前后对比图
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

stages = ['Before Repair\n(Contaminated)', 'After Repair\n(Repaired Baseline)']
c_before = '#78909C'
c_after = '#2E7D32'

# 1. PCC
pcc_vals = [0.6489, 0.6549]
bars1 = ax1.bar(stages, pcc_vals, color=[c_before, c_after], width=0.45, edgecolor='black', linewidth=1.2, zorder=3)
ax1.set_ylim(0.63, 0.67)
ax1.set_title('Pooled PCC (Higher is better)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, val in zip(bars1, pcc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax1.text(0.5, 0.665, r'$\Delta = +0.0060 \uparrow$', ha='center', fontsize=11, color='#2E7D32', fontweight='bold')

# 2. Raw MAE
mae_vals = [1209.93, 1176.21]
bars2 = ax2.bar(stages, mae_vals, color=[c_before, c_after], width=0.45, edgecolor='black', linewidth=1.2, zorder=3)
ax2.set_ylim(1100, 1250)
ax2.set_title('Raw MAE (Lower is better)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, val in zip(bars2, mae_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
ax2.text(0.5, 1230, r'$\Delta = -33.72 \downarrow$', ha='center', fontsize=11, color='#2E7D32', fontweight='bold')

# 3. Raw R2
r2_vals = [-0.1554, -0.0880]
bars3 = ax3.bar(stages, r2_vals, color=[c_before, c_after], width=0.45, edgecolor='black', linewidth=1.2, zorder=3)
ax3.set_ylim(-0.20, 0.02)
ax3.axhline(0, color='black', linewidth=1, linestyle='-')
ax3.set_title('Raw R² (Higher is better)', fontsize=12, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, val in zip(bars3, r2_vals):
    y_pos = val + 0.005 if val < 0 else val + 0.002
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax3.text(0.5, -0.02, r'$\Delta = +0.0674 \uparrow$', ha='center', fontsize=11, color='#2E7D32', fontweight='bold')

fig.suptitle('Figure 2: Baseline Model Performance Before and After Data Contamination Repair', 
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig_path2 = os.path.join(out_dir, "fig2_baseline_repair_comparison.png")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()

print("Figures successfully generated at:", out_dir)
