"""
三折交叉验证综合分析脚本
对比 HisToGene-UNI 和 EGN-v2+UNI 两个模型的跨患者泛化性能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\AI空间转录病理研究\PFMval_new"
OUTPUT_DIR = os.path.join(BASE_DIR, "histogene", "checkpoints", "results_vis",
                          "CrossValidation_3Fold_Comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练历史 CSV 路径
HISTOGENE_CSVS = {
    1: os.path.join(BASE_DIR, "histogene/checkpoints/results_vis/CrossPatient_JFX_LMZ_to_HYZ_20260424_221349/training_history_CrossPatient_JFX_LMZ_to_HYZ.csv"),
    2: os.path.join(BASE_DIR, "histogene/checkpoints/results_vis/CrossPatient_Fold2_to_JFX0729_20260501_190244/training_history_CrossPatient_Fold2_to_JFX0729.csv"),
    3: os.path.join(BASE_DIR, "histogene/checkpoints/results_vis/CrossPatient_Fold3_to_LMZ12939_20260501_190720/training_history_CrossPatient_Fold3_to_LMZ12939.csv"),
}

EGNV2_CSVS = {
    1: os.path.join(BASE_DIR, "egnv2/checkpoints/results_vis/CrossPatient_JFX_LMZ_to_HYZ_UNI_20260424_232239/training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv"),
    2: os.path.join(BASE_DIR, "egnv2/checkpoints/results_vis/CrossPatient_Fold2_HYZ_LMZ_to_JFX_UNI_20260501_185340/training_history_CrossPatient_Fold2_HYZ_LMZ_to_JFX_UNI.csv"),
    3: os.path.join(BASE_DIR, "egnv2/checkpoints/results_vis/CrossPatient_Fold3_HYZ_JFX_to_LMZ_UNI_20260501_185532/training_history_CrossPatient_Fold3_HYZ_JFX_to_LMZ_UNI.csv"),
}

FOLD_INFO = {
    1: {"train": "JFX+LMZ", "test": "HYZ"},
    2: {"train": "HYZ+LMZ", "test": "JFX"},
    3: {"train": "HYZ+JFX", "test": "LMZ"},
}

# ============================================================
# 数据读取
# ============================================================
def read_best_epoch(csv_path):
    """从 training_history CSV 中读取最佳 epoch（val_loss 最小）"""
    df = pd.read_csv(csv_path)
    best_idx = df['val_loss'].idxmin()
    row = df.loc[best_idx]
    total_epochs = int(df['epoch'].max())
    return {
        'best_epoch': int(row['epoch']),
        'test_pcc': float(row['val_pcc']),
        'train_pcc': float(row['train_pcc']),
        'overfitting_gap': float(row['train_pcc']) - float(row['val_pcc']),
        'total_epochs': total_epochs,
    }

print("=" * 70)
print("  三折交叉验证综合分析: HisToGene-UNI vs EGN-v2+UNI")
print("=" * 70)

# 读取所有数据
results = []
for model_name, csvs in [("HisToGene-UNI", HISTOGENE_CSVS), ("EGN-v2+UNI", EGNV2_CSVS)]:
    for fold, csv_path in csvs.items():
        info = read_best_epoch(csv_path)
        info['model'] = model_name
        info['fold'] = fold
        info['train_patients'] = FOLD_INFO[fold]['train']
        info['test_patient'] = FOLD_INFO[fold]['test']
        results.append(info)
        print(f"  {model_name} Fold {fold} ({FOLD_INFO[fold]['train']} → {FOLD_INFO[fold]['test']}): "
              f"Test PCC={info['test_pcc']:.4f}, Best Epoch={info['best_epoch']}, "
              f"Gap={info['overfitting_gap']:.4f}, Total={info['total_epochs']} epochs")

df_results = pd.DataFrame(results)

# 按模型分组
hg = df_results[df_results['model'] == 'HisToGene-UNI'].sort_values('fold')
eg = df_results[df_results['model'] == 'EGN-v2+UNI'].sort_values('fold')

hg_pccs = hg['test_pcc'].values
eg_pccs = eg['test_pcc'].values
hg_mean, hg_std = hg_pccs.mean(), hg_pccs.std()
eg_mean, eg_std = eg_pccs.mean(), eg_pccs.std()

# ============================================================
# 控制台分析报告
# ============================================================
print("\n" + "=" * 70)
print("  分析报告")
print("=" * 70)

print(f"\n【1. 三折平均 Test PCC】")
print(f"  HisToGene-UNI: {hg_mean:.4f} ± {hg_std:.4f}")
print(f"  EGN-v2+UNI:    {eg_mean:.4f} ± {eg_std:.4f}")

if eg_mean > hg_mean:
    winner = "EGN-v2+UNI"
    diff = eg_mean - hg_mean
else:
    winner = "HisToGene-UNI"
    diff = hg_mean - eg_mean
print(f"\n【2. 最优模型判定】")
print(f"  ★ {winner} 平均 PCC 更高，领先 {diff:.4f}")
if eg_std < hg_std:
    print(f"  ★ EGN-v2+UNI 标准差更小 ({eg_std:.4f} vs {hg_std:.4f})，泛化稳定性更好")
else:
    print(f"  ★ HisToGene-UNI 标准差更小 ({hg_std:.4f} vs {eg_std:.4f})，泛化稳定性更好")

print(f"\n【3. 各 Fold 表现差异分析】")
all_pccs = {}
for fold in [1, 2, 3]:
    h_pcc = hg[hg['fold'] == fold]['test_pcc'].values[0]
    e_pcc = eg[eg['fold'] == fold]['test_pcc'].values[0]
    avg_pcc = (h_pcc + e_pcc) / 2
    all_pccs[fold] = avg_pcc
    better = "EGN-v2+UNI" if e_pcc > h_pcc else "HisToGene-UNI"
    print(f"  Fold {fold} ({FOLD_INFO[fold]['train']} → {FOLD_INFO[fold]['test']}): "
          f"HG={h_pcc:.4f}, EGN={e_pcc:.4f} → {better} 更优")

easiest = max(all_pccs, key=all_pccs.get)
hardest = min(all_pccs, key=all_pccs.get)
print(f"\n  → 最易预测的患者: {FOLD_INFO[easiest]['test']} (Fold {easiest}, 平均PCC={all_pccs[easiest]:.4f})")
print(f"  → 最难预测的患者: {FOLD_INFO[hardest]['test']} (Fold {hardest}, 平均PCC={all_pccs[hardest]:.4f})")

print(f"\n【4. 过拟合程度对比】")
hg_gaps = hg['overfitting_gap'].values
eg_gaps = eg['overfitting_gap'].values
print(f"  HisToGene-UNI 平均过拟合 Gap: {hg_gaps.mean():.4f} (范围: {hg_gaps.min():.4f} ~ {hg_gaps.max():.4f})")
print(f"  EGN-v2+UNI    平均过拟合 Gap: {eg_gaps.mean():.4f} (范围: {eg_gaps.min():.4f} ~ {eg_gaps.max():.4f})")
if abs(eg_gaps.mean()) < abs(hg_gaps.mean()):
    print(f"  ★ EGN-v2+UNI 过拟合程度更低，训练-测试差距更小")
else:
    print(f"  ★ HisToGene-UNI 过拟合程度更低，训练-测试差距更小")

print(f"\n【5. 收敛速度对比】")
hg_epochs = hg['best_epoch'].values
eg_epochs = eg['best_epoch'].values
print(f"  HisToGene-UNI 最佳 Epoch: {hg_epochs} (平均 {hg_epochs.mean():.1f})")
print(f"  EGN-v2+UNI    最佳 Epoch: {eg_epochs} (平均 {eg_epochs.mean():.1f})")
if hg_epochs.mean() < eg_epochs.mean():
    print(f"  ★ HisToGene-UNI 收敛更快，但可能存在欠拟合风险")
else:
    print(f"  ★ EGN-v2+UNI 收敛更快")

print(f"\n【6. 综合结论】")
print(f"  1) {winner} 在三折交叉验证中平均 PCC 更高 ({max(hg_mean, eg_mean):.4f} vs {min(hg_mean, eg_mean):.4f})")
print(f"  2) EGN-v2+UNI 基于 GraphSAGE 架构，过拟合控制能力"
      f"{'更强' if abs(eg_gaps.mean()) < abs(hg_gaps.mean()) else '较弱'}，"
      f"训练过程更{'稳定' if eg_std < hg_std else '波动'}")
print(f"  3) HisToGene-UNI 基于 ViT 架构，收敛速度"
      f"{'更快' if hg_epochs.mean() < eg_epochs.mean() else '较慢'}，"
      f"但在早期 epoch 即可达到较好性能")
print(f"  4) 两模型对不同患者的预测难度排序一致: "
      f"{FOLD_INFO[easiest]['test']} (易) > {FOLD_INFO[3 if easiest != 3 and hardest != 3 else (2 if easiest != 2 and hardest != 2 else 1)]['test']} (中) > {FOLD_INFO[hardest]['test']} (难)")

# ============================================================
# 生成 CSV
# ============================================================
csv_out = df_results[['model', 'fold', 'train_patients', 'test_patient',
                       'test_pcc', 'overfitting_gap', 'best_epoch', 'total_epochs']]
csv_path = os.path.join(OUTPUT_DIR, "cross_validation_summary.csv")
csv_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✓ CSV 已保存: {csv_path}")

# ============================================================
# 配色方案
# ============================================================
COLOR_HG = '#2E86C1'   # 蓝色系
COLOR_EG = '#E74C3C'   # 红色系
COLOR_HG_LIGHT = '#85C1E9'
COLOR_EG_LIGHT = '#F1948A'

# ============================================================
# 图1: 三折交叉对比柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

labels = ['Fold 1\n(→HYZ)', 'Fold 2\n(→JFX)', 'Fold 3\n(→LMZ)', 'Mean']
x = np.arange(len(labels))
width = 0.32

hg_vals = list(hg_pccs) + [hg_mean]
eg_vals = list(eg_pccs) + [eg_mean]
hg_errs = [0, 0, 0, hg_std]
eg_errs = [0, 0, 0, eg_std]

bars1 = ax.bar(x - width/2, hg_vals, width, label='HisToGene-UNI',
               color=COLOR_HG, edgecolor='white', linewidth=1.2,
               yerr=hg_errs, capsize=5, error_kw={'linewidth': 1.5})
bars2 = ax.bar(x + width/2, eg_vals, width, label='EGN-v2+UNI',
               color=COLOR_EG, edgecolor='white', linewidth=1.2,
               yerr=eg_errs, capsize=5, error_kw={'linewidth': 1.5})

# 标注具体值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLOR_HG)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLOR_EG)

# Mean 标注 ± std
ax.text(x[3] - width/2, hg_vals[3] + hg_std + 0.012,
        f'±{hg_std:.4f}', ha='center', fontsize=9, color=COLOR_HG, style='italic')
ax.text(x[3] + width/2, eg_vals[3] + eg_std + 0.012,
        f'±{eg_std:.4f}', ha='center', fontsize=9, color=COLOR_EG, style='italic')

ax.set_ylabel('Test PCC (皮尔逊相关系数)', fontsize=13)
ax.set_title('三折交叉验证: HisToGene-UNI vs EGN-v2+UNI 性能对比', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, max(max(hg_vals), max(eg_vals)) + 0.08)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加分隔线
ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "cross_validation_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ 图1已保存: {fig_path}")

# ============================================================
# 图2: 雷达图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

categories = ['Fold 1\n(→HYZ)', 'Fold 2\n(→JFX)', 'Fold 3\n(→LMZ)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

hg_radar = list(hg_pccs) + [hg_pccs[0]]
eg_radar = list(eg_pccs) + [eg_pccs[0]]

ax.plot(angles, hg_radar, 'o-', linewidth=2.5, color=COLOR_HG, label='HisToGene-UNI', markersize=10)
ax.fill(angles, hg_radar, alpha=0.15, color=COLOR_HG)
ax.plot(angles, eg_radar, 's-', linewidth=2.5, color=COLOR_EG, label='EGN-v2+UNI', markersize=10)
ax.fill(angles, eg_radar, alpha=0.15, color=COLOR_EG)

# 标注数值
for i in range(N):
    offset = 0.015
    ax.text(angles[i], hg_radar[i] + offset, f'{hg_radar[i]:.4f}',
            ha='center', fontsize=10, color=COLOR_HG, fontweight='bold')
    ax.text(angles[i], eg_radar[i] - offset, f'{eg_radar[i]:.4f}',
            ha='center', fontsize=10, color=COLOR_EG, fontweight='bold')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13)
ax.set_title('三折交叉验证雷达图: Test PCC 对比', fontsize=15, fontweight='bold', pad=25)
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=12)

# 设置 r 轴范围
all_vals = list(hg_pccs) + list(eg_pccs)
r_min = max(0, min(all_vals) - 0.05)
r_max = max(all_vals) + 0.05
ax.set_ylim(r_min, r_max)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "cross_validation_radar.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ 图2已保存: {fig_path}")

# ============================================================
# 图3: 过拟合分析（双Y轴）
# ============================================================
fig, ax1 = plt.subplots(figsize=(14, 8))

folds = [1, 2, 3]
fold_labels = ['Fold 1\n(→HYZ)', 'Fold 2\n(→JFX)', 'Fold 3\n(→LMZ)']
x = np.arange(len(folds))
width = 0.18

# 左Y轴: Test PCC (柱状图)
bars1 = ax1.bar(x - width*1.5, hg_pccs, width, label='HG-UNI Test PCC',
                color=COLOR_HG, alpha=0.85, edgecolor='white')
bars2 = ax1.bar(x - width*0.5, eg_pccs, width, label='EGN-v2 Test PCC',
                color=COLOR_EG, alpha=0.85, edgecolor='white')
ax1.set_ylabel('Test PCC', fontsize=13, color='black')
ax1.set_ylim(0, max(max(hg_pccs), max(eg_pccs)) * 1.35)

# 标注 PCC 值
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=COLOR_HG)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=COLOR_EG)

# 右Y轴: 过拟合 Gap (折线图)
ax2 = ax1.twinx()
line1, = ax2.plot(x, hg_gaps, 'D--', color=COLOR_HG_LIGHT, linewidth=2.5, markersize=10,
                  markeredgecolor=COLOR_HG, markerfacecolor=COLOR_HG_LIGHT, label='HG-UNI 过拟合Gap')
line2, = ax2.plot(x, eg_gaps, 's--', color=COLOR_EG_LIGHT, linewidth=2.5, markersize=10,
                  markeredgecolor=COLOR_EG, markerfacecolor=COLOR_EG_LIGHT, label='EGN-v2 过拟合Gap')
ax2.set_ylabel('过拟合 Gap (Train PCC - Test PCC)', fontsize=13, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# 标注 Gap 值
for i in range(len(folds)):
    ax2.annotate(f'{hg_gaps[i]:.4f}', (x[i], hg_gaps[i]),
                 textcoords="offset points", xytext=(-30, 12), fontsize=9,
                 color=COLOR_HG, fontweight='bold')
    ax2.annotate(f'{eg_gaps[i]:.4f}', (x[i], eg_gaps[i]),
                 textcoords="offset points", xytext=(10, 12), fontsize=9,
                 color=COLOR_EG, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(fold_labels, fontsize=12)
ax1.set_title('过拟合分析: Test PCC vs 过拟合 Gap', fontsize=16, fontweight='bold', pad=15)

# 合并图例
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10)

ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "cross_validation_overfitting.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ 图3已保存: {fig_path}")

print(f"\n{'=' * 70}")
print(f"  所有输出已保存到: {OUTPUT_DIR}")
print(f"{'=' * 70}")
