import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# 1. 字体配置：载入微软雅黑，确保中文与学术图表排版无乱码
font_path = r'C:/Windows/Fonts/msyh.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 2. 工作路径与数据读取
base_dir = Path(r'D:/AI空间转录病理研究/PFMval_new')
summary_path = base_dir / 'experiments/results/phase2_softlink_local_v2/20260908_005325_810_8d306c10/analysis/pathway_summary_20260908.csv'
deltas_path = base_dir / 'experiments/results/phase2_softlink_local_v2/20260908_005325_810_8d306c10/analysis/pathway_paired_deltas_20260908.csv'
out_dir = base_dir / '01_指南与解读/分析报告/report_figures/phase2_softlink_v21_20260908'
out_dir.mkdir(parents=True, exist_ok=True)

# 严谨校验输入文件存在性
assert summary_path.exists(), f"未找到文件: {summary_path}"
assert deltas_path.exists(), f"未找到文件: {deltas_path}"

df_summary = pd.read_csv(summary_path)
df_deltas = pd.read_csv(deltas_path)

# 严格断言240行，严禁插值或补模拟数据
assert len(df_summary) == 240, f"df_summary 行数预期为 240，实际读取到 {len(df_summary)}"
assert len(df_deltas) == 240, f"df_deltas 行数预期为 240，实际读取到 {len(df_deltas)}"

# 3. 统一排序基准：按外部独立测试集 (XZY) 的 spatial - point PCC 增益降序排列
df_sp_xzy = df_deltas[
    (df_deltas['split'] == 'XZY') &
    (df_deltas['arm'] == 'spatial') &
    (df_deltas['reference'] == 'point')
]
assert len(df_sp_xzy) == 30, f"XZY 数据集 spatial-point 对比记录预期为 30 条，实际为 {len(df_sp_xzy)}"
pathway_order = df_sp_xzy.sort_values(by='pcc_delta_mean', ascending=False)['pathway'].tolist()

# ==========================================================================
# 图1：通路四臂PCC全景 (内部验证集 vs 外部测试集)
# ==========================================================================
arms = ['point', 'relation', 'spatial', 'joint']
arm_labels = ['Point 臂', 'Relation 臂', 'Spatial 臂', 'Joint 臂']

piv_int = df_summary[df_summary['split'] == 'internal_val'].pivot(
    index='pathway', columns='arm', values='pcc_mean'
).reindex(index=pathway_order, columns=arms)

piv_xzy = df_summary[df_summary['split'] == 'XZY'].pivot(
    index='pathway', columns='arm', values='pcc_mean'
).reindex(index=pathway_order, columns=arms)

# 共享色彩标度范围
vmin_pcc = min(piv_int.min().min(), piv_xzy.min().min())
vmax_pcc = max(piv_int.max().max(), piv_xzy.max().max())

fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(18, 16), sharey=True)

im1 = ax1_1.imshow(piv_int.values, cmap='YlGnBu', vmin=vmin_pcc, vmax=vmax_pcc, aspect='auto')
im2 = ax1_2.imshow(piv_xzy.values, cmap='YlGnBu', vmin=vmin_pcc, vmax=vmax_pcc, aspect='auto')

# 单元格数值标注（3位小数，动态高对比度文字）
for i in range(30):
    for j in range(4):
        v1 = piv_int.iloc[i, j]
        c1 = 'white' if v1 > (vmin_pcc + vmax_pcc) * 0.55 else 'black'
        ax1_1.text(j, i, f"{v1:.3f}", ha='center', va='center', color=c1, fontsize=9.5, fontweight='semibold')
        
        v2 = piv_xzy.iloc[i, j]
        c2 = 'white' if v2 > (vmin_pcc + vmax_pcc) * 0.55 else 'black'
        ax1_2.text(j, i, f"{v2:.3f}", ha='center', va='center', color=c2, fontsize=9.5, fontweight='semibold')

# 坐标轴与网格划分
ax1_1.set_xticks(range(4))
ax1_1.set_xticklabels(arm_labels, fontsize=11, fontweight='bold')
ax1_1.set_yticks(range(30))
ax1_1.set_yticklabels(pathway_order, fontsize=10)
ax1_1.set_title("内部验证集 (internal_val)\n四臂通路 PCC 均值", fontsize=13, fontweight='bold', pad=14)

ax1_2.set_xticks(range(4))
ax1_2.set_xticklabels(arm_labels, fontsize=11, fontweight='bold')
ax1_2.set_title("外部独立测试集 (XZY)\n四臂通路 PCC 均值", fontsize=13, fontweight='bold', pad=14)

for ax in [ax1_1, ax1_2]:
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 30, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

fig1.subplots_adjust(left=0.18, right=0.88, top=0.86, bottom=0.08, wspace=0.12)
cbar_ax1 = fig1.add_axes([0.90, 0.25, 0.02, 0.5])
cbar1 = fig1.colorbar(im1, cax=cbar_ax1)
cbar1.set_label("该通路 PCC 均值 (患者等权、3种子平均)", fontsize=11, labelpad=10)

fig1.suptitle("图1：内部验证集与外部测试集（XZY）通路四臂PCC全景", fontsize=16, fontweight='bold', y=0.96)
fig1.text(0.5, 0.93, "标注：该通路 PCC（患者等权、种子平均）；外部仅1名患者；按外部空间增益排序", ha='center', fontsize=11, color='#333333')

fig1.savefig(out_dir / "通路四臂PCC全景.png", dpi=180, bbox_inches='tight')
fig1.savefig(out_dir / "通路四臂PCC全景.svg", bbox_inches='tight')
plt.close(fig1)

# ==========================================================================
# 图2：通路收益与误差对照 (4个并排 30行×3列 热图)
# ==========================================================================
df_deltas['comp'] = df_deltas['arm'] + '-' + df_deltas['reference']
comps = ['relation-point', 'spatial-point', 'joint-spatial']
comp_labels = ['relation - point', 'spatial - point', 'joint - spatial']

p_int_pcc = df_deltas[df_deltas['split'] == 'internal_val'].pivot(
    index='pathway', columns='comp', values='pcc_delta_mean'
).reindex(index=pathway_order, columns=comps)

p_xzy_pcc = df_deltas[df_deltas['split'] == 'XZY'].pivot(
    index='pathway', columns='comp', values='pcc_delta_mean'
).reindex(index=pathway_order, columns=comps)

p_xzy_ccc = df_deltas[df_deltas['split'] == 'XZY'].pivot(
    index='pathway', columns='comp', values='ccc_delta_mean'
).reindex(index=pathway_order, columns=comps)

# 误差改善量：zRMSE取负差，正数始终表示模型改善（预测误差下降）
p_xzy_rmse = - df_deltas[df_deltas['split'] == 'XZY'].pivot(
    index='pathway', columns='comp', values='z_rmse_delta_mean'
).reindex(index=pathway_order, columns=comps)

# 对称色标界限计算
max_pcc_delta = max(np.nanmax(np.abs(p_int_pcc.values)), np.nanmax(np.abs(p_xzy_pcc.values)))
norm_pcc = TwoSlopeNorm(vmin=-max_pcc_delta, vcenter=0, vmax=max_pcc_delta)

max_ccc_delta = np.nanmax(np.abs(p_xzy_ccc.values))
norm_ccc = TwoSlopeNorm(vmin=-max_ccc_delta, vcenter=0, vmax=max_ccc_delta)

max_rmse_delta = np.nanmax(np.abs(p_xzy_rmse.values))
norm_rmse = TwoSlopeNorm(vmin=-max_rmse_delta, vcenter=0, vmax=max_rmse_delta)

fig2 = plt.figure(figsize=(22, 19))
gs = fig2.add_gridspec(2, 4, height_ratios=[25, 1], hspace=0.16, wspace=0.18, left=0.15, right=0.96, top=0.86, bottom=0.09)

ax2_1 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[0, 1], sharey=ax2_1)
ax2_3 = fig2.add_subplot(gs[0, 2], sharey=ax2_1)
ax2_4 = fig2.add_subplot(gs[0, 3], sharey=ax2_1)

im2_1 = ax2_1.imshow(p_int_pcc.values, cmap='RdBu_r', norm=norm_pcc, aspect='auto')
im2_2 = ax2_2.imshow(p_xzy_pcc.values, cmap='RdBu_r', norm=norm_pcc, aspect='auto')
im2_3 = ax2_3.imshow(p_xzy_ccc.values, cmap='RdBu_r', norm=norm_ccc, aspect='auto')
im2_4 = ax2_4.imshow(p_xzy_rmse.values, cmap='RdBu_r', norm=norm_rmse, aspect='auto')

panels_cfg = [
    (ax2_1, p_int_pcc, norm_pcc, "内部验证集 ΔPCC\n(PCC 差量)"),
    (ax2_2, p_xzy_pcc, norm_pcc, "外部测试集 (XZY) ΔPCC\n(PCC 差量)"),
    (ax2_3, p_xzy_ccc, norm_ccc, "外部测试集 (XZY) ΔCCC\n(CCC 差量)"),
    (ax2_4, p_xzy_rmse, norm_rmse, "外部测试集 (XZY) -ΔzRMSE\n(zRMSE 改善量)")
]

for ax, data_mat, norm, title in panels_cfg:
    ax.set_xticks(range(3))
    ax.set_xticklabels(comp_labels, rotation=25, ha='right', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 30, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    for r in range(30):
        for c in range(3):
            val = data_mat.iloc[r, c]
            is_extreme = abs(val) > (norm.vmax * 0.55)
            text_color = 'white' if is_extreme else 'black'
            ax.text(c, r, f"{val:+.3f}", ha='center', va='center', color=text_color, fontsize=8.5, fontweight='semibold')

ax2_1.set_yticks(range(30))
ax2_1.set_yticklabels(pathway_order, fontsize=10)

for ax in [ax2_2, ax2_3, ax2_4]:
    ax.tick_params(labelleft=False)

# 底部水平独立色标配置
cax_pcc = fig2.add_subplot(gs[1, 0:2])
cb_pcc = fig2.colorbar(im2_1, cax=cax_pcc, orientation='horizontal')
cb_pcc.set_label("PCC 增益量 (ΔPCC, 统一对称色标, 正值=性能提升)", fontsize=10, fontweight='bold')

cax_ccc = fig2.add_subplot(gs[1, 2])
cb_ccc = fig2.colorbar(im2_3, cax=cax_ccc, orientation='horizontal')
cb_ccc.set_label("CCC 增益量 (ΔCCC, 对称色标, 正值=一致性改善)", fontsize=10, fontweight='bold')

cax_rmse = fig2.add_subplot(gs[1, 3])
cb_rmse = fig2.colorbar(im2_4, cax=cax_rmse, orientation='horizontal')
cb_rmse.set_label("zRMSE 改善量 (-ΔzRMSE, 对称色标, 正值=误差降低)", fontsize=10, fontweight='bold')

fig2.suptitle("图2：消融与联合配置在内部与外部测试集的多指标收益与误差改善全景对照", fontsize=16, fontweight='bold', y=0.96)
fig2.text(0.5, 0.925, "说明：3种子均值；外部仅1名患者；正值表示改善，不代表统计显著", ha='center', fontsize=11, color='#333333')

fig2.savefig(out_dir / "通路收益与误差对照.png", dpi=180, bbox_inches='tight')
fig2.savefig(out_dir / "通路收益与误差对照.svg", bbox_inches='tight')
plt.close(fig2)

print("全景图与对比图生成完毕，格式包含 PNG (180 DPI) 与矢量 SVG。")
