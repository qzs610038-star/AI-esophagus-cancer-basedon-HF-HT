"""
单患者 vs 多患者联合训练对比分析脚本
生成4个可视化图表和详细分析报告
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义配色方案
COLORS = {
    'HYZ15040': '#4C72B0',
    'JFX0729': '#DD8452', 
    'LMZ12939': '#55A868',
    'MultiPatient': '#C44E52'
}

# 文件路径配置
BASE_DIR = r"d:\AI空间转录病理研究\PFMval_new\histogene"
OUTPUT_DIR = os.path.join(BASE_DIR, "results_vis", "单患者vs联合训练对比")

# 训练历史文件
TRAINING_HISTORY_FILES = {
    'HYZ15040': os.path.join(BASE_DIR, "training_history_HYZ15040.csv"),
    'JFX0729': os.path.join(BASE_DIR, "training_history_JFX0729.csv"),
    'LMZ12939': os.path.join(BASE_DIR, "training_history_LMZ12939.csv"),
    'MultiPatient': os.path.join(BASE_DIR, "training_history_MultiPatient_3ST.csv")
}

# Predictions文件
PREDICTIONS_FILES = {
    'HYZ15040': os.path.join(BASE_DIR, "checkpoints", "results_vis", "HYZ15040_20260416_213453", "predictions.csv"),
    'JFX0729': os.path.join(BASE_DIR, "checkpoints", "results_vis", "JFX0729_20260416_224437", "predictions.csv"),
    'LMZ12939': os.path.join(BASE_DIR, "checkpoints", "results_vis", "LMZ12939_20260417_114425", "predictions.csv"),
    'MultiPatient': os.path.join(BASE_DIR, "checkpoints", "results_vis", "MultiPatient_3ST_20260417_140522", "predictions.csv")
}


def load_training_history(file_path):
    """加载训练历史数据"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return None
    return pd.read_csv(file_path)


def extract_best_metrics(df):
    """从训练历史中提取最佳epoch的指标"""
    if df is None or df.empty:
        return None
    
    # 找到val_pcc最大的epoch
    best_idx = df['val_pcc'].idxmax()
    best_row = df.loc[best_idx]
    
    return {
        'best_epoch': int(best_row['epoch']),
        'total_epochs': len(df),
        'val_pcc': best_row['val_pcc'],
        'val_r2': best_row['val_r2'],
        'val_loss': best_row['val_loss'],
        'val_mae': best_row['val_mae'],
        'train_pcc': best_row['train_pcc'],
        'train_r2': best_row['train_r2'],
        'overfit_gap': best_row['train_pcc'] - best_row['val_pcc']
    }


def load_predictions(file_path):
    """加载预测结果"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return None
    return pd.read_csv(file_path)


def calculate_pathway_pcc(df):
    """计算每个通路的PCC"""
    if df is None or df.empty:
        return None
    
    # 提取所有通路名（从列名中提取）
    pathways = []
    for col in df.columns:
        if col.startswith('true_'):
            pathway = col.replace('true_', '')
            pathways.append(pathway)
    
    pcc_results = {}
    for pathway in pathways:
        true_col = f'true_{pathway}'
        pred_col = f'pred_{pathway}'
        
        if true_col in df.columns and pred_col in df.columns:
            # 去除NaN值
            mask = df[true_col].notna() & df[pred_col].notna()
            if mask.sum() > 1:
                pcc, _ = pearsonr(df.loc[mask, true_col], df.loc[mask, pred_col])
                pcc_results[pathway] = pcc
            else:
                pcc_results[pathway] = np.nan
    
    return pcc_results


def create_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")


def plot_overall_comparison(all_history, all_metrics):
    """图1: 总体性能对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle('单患者 vs 多患者联合训练 - 总体性能对比', fontsize=16, fontweight='bold')
    
    # 子图1: Val PCC 训练曲线
    ax1 = axes[0, 0]
    for name, df in all_history.items():
        if df is not None:
            linestyle = '--' if name == 'MultiPatient' else '-'
            linewidth = 2.5 if name == 'MultiPatient' else 1.5
            ax1.plot(df['epoch'], df['val_pcc'], label=name, color=COLORS[name], 
                    linestyle=linestyle, linewidth=linewidth)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation PCC')
    ax1.set_title('Validation PCC 训练曲线')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Val Loss 训练曲线
    ax2 = axes[0, 1]
    for name, df in all_history.items():
        if df is not None:
            linestyle = '--' if name == 'MultiPatient' else '-'
            linewidth = 2.5 if name == 'MultiPatient' else 1.5
            ax2.plot(df['epoch'], df['val_loss'], label=name, color=COLORS[name],
                    linestyle=linestyle, linewidth=linewidth)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss 训练曲线')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 最佳指标柱状图
    ax3 = axes[1, 0]
    names = list(all_metrics.keys())
    val_pccs = [all_metrics[n]['val_pcc'] for n in names]
    val_r2s = [all_metrics[n]['val_r2'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, val_pccs, width, label='Val PCC', color='steelblue')
    bars2 = ax3.bar(x + width/2, val_r2s, width, label='Val R²', color='coral')
    
    ax3.set_ylabel('Score')
    ax3.set_title('最佳 Validation PCC vs R²')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 子图4: 过拟合程度
    ax4 = axes[1, 1]
    gaps = [all_metrics[n]['overfit_gap'] for n in names]
    colors_list = [COLORS[n] for n in names]
    
    bars = ax4.bar(names, gaps, color=colors_list, alpha=0.8)
    ax4.set_ylabel('Train PCC - Val PCC')
    ax4.set_title('过拟合程度 (Overfitting Gap)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'overall_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_pathway_pcc_comparison(all_pathway_pcc):
    """图2: 逐通路PCC四方案对比"""
    # 获取所有通路并排序（按联合训练PCC降序）
    all_pathways = set()
    for pcc_dict in all_pathway_pcc.values():
        all_pathways.update(pcc_dict.keys())
    
    # 按联合训练PCC排序
    multi_pcc = all_pathway_pcc.get('MultiPatient', {})
    sorted_pathways = sorted(all_pathways, key=lambda x: multi_pcc.get(x, 0), reverse=True)
    
    # 准备数据
    data = []
    for pathway in sorted_pathways:
        row = {'Pathway': pathway}
        for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
            row[name] = all_pathway_pcc.get(name, {}).get(pathway, np.nan)
        data.append(row)
    
    df_plot = pd.DataFrame(data)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(22, 8), dpi=150)
    
    x = np.arange(len(sorted_pathways))
    width = 0.2
    
    # 绘制分组柱状图
    bars1 = ax.bar(x - 1.5*width, df_plot['HYZ15040'], width, label='HYZ15040', color=COLORS['HYZ15040'])
    bars2 = ax.bar(x - 0.5*width, df_plot['JFX0729'], width, label='JFX0729', color=COLORS['JFX0729'])
    bars3 = ax.bar(x + 0.5*width, df_plot['LMZ12939'], width, label='LMZ12939', color=COLORS['LMZ12939'])
    bars4 = ax.bar(x + 1.5*width, df_plot['MultiPatient'], width, label='MultiPatient (联合训练)', 
                   color=COLORS['MultiPatient'], hatch='//', edgecolor='black', linewidth=0.5)
    
    # 添加平均线
    for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
        mean_pcc = df_plot[name].mean()
        ax.axhline(y=mean_pcc, color=COLORS[name], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(sorted_pathways)-0.5, mean_pcc, f'{name}均值: {mean_pcc:.3f}', 
                fontsize=8, color=COLORS[name], va='center')
    
    ax.set_xlabel('通路名称', fontsize=12)
    ax.set_ylabel('PCC (Pearson Correlation)', fontsize=12)
    ax.set_title('逐通路 PCC 四方案对比 (按联合训练PCC降序排列)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_pathways, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.6, 0.8)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'pathway_pcc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")
    
    return df_plot, sorted_pathways


def plot_improvement_analysis(df_plot, sorted_pathways):
    """图3: 联合训练增益分析"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), dpi=150)
    fig.suptitle('联合训练 vs 单患者训练 - 增益分析', fontsize=16, fontweight='bold')
    
    # 计算单患者平均PCC
    single_avg = df_plot[['HYZ15040', 'JFX0729', 'LMZ12939']].mean(axis=1)
    improvement = df_plot['MultiPatient'] - single_avg
    
    # 上半部分：增益柱状图
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in improvement]
    bars = ax1.bar(range(len(sorted_pathways)), improvement, color=colors, alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('通路名称')
    ax1.set_ylabel('PCC 差异 (联合训练 - 单患者平均)')
    ax1.set_title('每个通路的联合训练增益 (正=提升, 负=下降)')
    ax1.set_xticks(range(len(sorted_pathways)))
    ax1.set_xticklabels(sorted_pathways, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 标注最大提升和最大下降
    max_idx = improvement.idxmax()
    min_idx = improvement.idxmin()
    ax1.annotate(f'最大提升\n{sorted_pathways[max_idx]}\n+{improvement[max_idx]:.3f}',
                xy=(max_idx, improvement[max_idx]), xytext=(max_idx, improvement[max_idx]+0.05),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, ha='center', color='green', fontweight='bold')
    ax1.annotate(f'最大下降\n{sorted_pathways[min_idx]}\n{improvement[min_idx]:.3f}',
                xy=(min_idx, improvement[min_idx]), xytext=(min_idx, improvement[min_idx]-0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, ha='center', color='red', fontweight='bold')
    
    # 下半部分：散点图
    ax2 = axes[1]
    ax2.scatter(single_avg, df_plot['MultiPatient'], alpha=0.6, s=80, c='steelblue')
    
    # 对角线参考线
    min_val = min(single_avg.min(), df_plot['MultiPatient'].min()) - 0.05
    max_val = max(single_avg.max(), df_plot['MultiPatient'].max()) + 0.05
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (无差异线)', linewidth=1.5)
    
    # 标注特殊点
    for i, pathway in enumerate(sorted_pathways):
        if i == max_idx or i == min_idx or abs(improvement[i]) > 0.1:
            ax2.annotate(pathway, (single_avg.iloc[i], df_plot['MultiPatient'].iloc[i]),
                        fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('单患者平均 PCC')
    ax2.set_ylabel('联合训练 PCC')
    ax2.set_title('联合训练 PCC vs 单患者平均 PCC (高于对角线=联合训练更好)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'improvement_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")
    
    return improvement


def plot_full_report(all_history, all_metrics, df_plot, sorted_pathways):
    """图4: 综合报告大图"""
    fig = plt.figure(figsize=(20, 16), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.5], hspace=0.3, wspace=0.25)
    
    fig.suptitle('单患者 vs 多患者联合训练 - 综合对比报告', fontsize=18, fontweight='bold', y=0.98)
    
    # 顶部：指标表格
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['训练方案', '样本数', '总Epoch', '最佳Epoch', 'Val PCC', 'Val R²', 'Val Loss', 'Train PCC', '过拟合Gap']
    
    for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
        metrics = all_metrics[name]
        # 获取样本数
        pred_file = PREDICTIONS_FILES[name]
        if os.path.exists(pred_file):
            sample_count = len(pd.read_csv(pred_file))
        else:
            sample_count = '-'
        
        table_data.append([
            name,
            sample_count,
            metrics['total_epochs'],
            metrics['best_epoch'],
            f"{metrics['val_pcc']:.4f}",
            f"{metrics['val_r2']:.4f}",
            f"{metrics['val_loss']:.4f}",
            f"{metrics['train_pcc']:.4f}",
            f"{metrics['overfit_gap']:.4f}"
        ])
    
    table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 高亮联合训练行
    for i in range(len(headers)):
        table[(4, i)].set_facecolor('#FFE4E1')
        table[(4, i)].set_text_props(fontweight='bold')
    
    ax_table.set_title('总体指标对比', fontsize=14, fontweight='bold', pad=20)
    
    # 中间左：Val PCC曲线
    ax1 = fig.add_subplot(gs[1, 0])
    for name, df in all_history.items():
        if df is not None:
            linestyle = '--' if name == 'MultiPatient' else '-'
            linewidth = 2.5 if name == 'MultiPatient' else 1.5
            ax1.plot(df['epoch'], df['val_pcc'], label=name, color=COLORS[name],
                    linestyle=linestyle, linewidth=linewidth)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation PCC')
    ax1.set_title('Validation PCC 训练曲线对比')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 中间右：Val Loss曲线
    ax2 = fig.add_subplot(gs[1, 1])
    for name, df in all_history.items():
        if df is not None:
            linestyle = '--' if name == 'MultiPatient' else '-'
            linewidth = 2.5 if name == 'MultiPatient' else 1.5
            ax2.plot(df['epoch'], df['val_loss'], label=name, color=COLORS[name],
                    linestyle=linestyle, linewidth=linewidth)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss 训练曲线对比')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 底部：逐通路PCC对比
    ax3 = fig.add_subplot(gs[2, :])
    
    x = np.arange(len(sorted_pathways))
    width = 0.2
    
    ax3.bar(x - 1.5*width, df_plot['HYZ15040'], width, label='HYZ15040', color=COLORS['HYZ15040'])
    ax3.bar(x - 0.5*width, df_plot['JFX0729'], width, label='JFX0729', color=COLORS['JFX0729'])
    ax3.bar(x + 0.5*width, df_plot['LMZ12939'], width, label='LMZ12939', color=COLORS['LMZ12939'])
    ax3.bar(x + 1.5*width, df_plot['MultiPatient'], width, label='MultiPatient (联合训练)',
            color=COLORS['MultiPatient'], hatch='//', edgecolor='black', linewidth=0.5)
    
    # 添加平均线
    for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
        mean_pcc = df_plot[name].mean()
        ax3.axhline(y=mean_pcc, color=COLORS[name], linestyle='--', alpha=0.4, linewidth=1)
    
    ax3.set_xlabel('通路名称', fontsize=11)
    ax3.set_ylabel('PCC (Pearson Correlation)', fontsize=11)
    ax3.set_title('逐通路 PCC 对比 (按联合训练PCC降序排列)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sorted_pathways, rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(-0.6, 0.8)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_report.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {os.path.join(OUTPUT_DIR, 'full_report.png')}")


def generate_text_report(all_metrics, all_pathway_pcc, improvement):
    """生成文本分析报告"""
    report = []
    report.append("=" * 80)
    report.append("单患者 vs 多患者联合训练 - 对比分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # 总体指标对比
    report.append("【一、总体指标对比】")
    report.append("-" * 60)
    for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
        m = all_metrics[name]
        report.append(f"\n{name}:")
        report.append(f"  - 最佳Epoch: {m['best_epoch']}/{m['total_epochs']}")
        report.append(f"  - Val PCC: {m['val_pcc']:.4f}")
        report.append(f"  - Val R²: {m['val_r2']:.4f}")
        report.append(f"  - Val Loss: {m['val_loss']:.4f}")
        report.append(f"  - Train PCC: {m['train_pcc']:.4f}")
        report.append(f"  - 过拟合Gap: {m['overfit_gap']:.4f}")
    
    # 单患者平均 vs 联合训练
    single_val_pcc = np.mean([all_metrics[n]['val_pcc'] for n in ['HYZ15040', 'JFX0729', 'LMZ12939']])
    multi_val_pcc = all_metrics['MultiPatient']['val_pcc']
    report.append(f"\n单患者平均 Val PCC: {single_val_pcc:.4f}")
    report.append(f"联合训练 Val PCC: {multi_val_pcc:.4f}")
    report.append(f"差异: {multi_val_pcc - single_val_pcc:+.4f}")
    report.append("")
    
    # 逐通路分析
    report.append("【二、逐通路PCC分析】")
    report.append("-" * 60)
    
    # 计算各方案平均PCC
    for name in ['HYZ15040', 'JFX0729', 'LMZ12939', 'MultiPatient']:
        avg_pcc = np.mean(list(all_pathway_pcc[name].values()))
        report.append(f"{name} 平均通路PCC: {avg_pcc:.4f}")
    
    single_avg_pcc = np.mean([
        np.mean(list(all_pathway_pcc[n].values())) 
        for n in ['HYZ15040', 'JFX0729', 'LMZ12939']
    ])
    multi_avg_pcc = np.mean(list(all_pathway_pcc['MultiPatient'].values()))
    report.append(f"\n单患者平均通路PCC: {single_avg_pcc:.4f}")
    report.append(f"联合训练平均通路PCC: {multi_avg_pcc:.4f}")
    report.append(f"差异: {multi_avg_pcc - single_avg_pcc:+.4f}")
    report.append("")
    
    # 增益分析
    report.append("【三、联合训练增益分析】")
    report.append("-" * 60)
    
    # 获取通路名称列表（从improvement的index获取）
    pathway_names = list(improvement.index)
    
    # 找出提升最大和下降最大的通路
    sorted_imp = improvement.sort_values(ascending=False)
    report.append("\n提升最大的5个通路:")
    for i, (idx, val) in enumerate(sorted_imp.head(5).items()):
        pathway = pathway_names[idx]
        report.append(f"  {i+1}. {pathway}: +{val:.4f}")
    
    report.append("\n下降最多的5个通路:")
    for i, (idx, val) in enumerate(sorted_imp.tail(5).items()):
        pathway = pathway_names[idx]
        report.append(f"  {i+1}. {pathway}: {val:.4f}")
    
    # 统计提升/下降通路数
    improved_count = (improvement > 0).sum()
    declined_count = (improvement < 0).sum()
    report.append(f"\n统计:")
    report.append(f"  - 提升通路数: {improved_count}/{len(improvement)} ({improved_count/len(improvement)*100:.1f}%)")
    report.append(f"  - 下降通路数: {declined_count}/{len(improvement)} ({declined_count/len(improvement)*100:.1f}%)")
    
    report.append("")
    report.append("【四、关键结论】")
    report.append("-" * 60)
    
    if multi_val_pcc > single_val_pcc:
        report.append(f"✓ 联合训练在总体Val PCC上优于单患者平均 (+{multi_val_pcc - single_val_pcc:.4f})")
    else:
        report.append(f"✗ 联合训练在总体Val PCC上劣于单患者平均 ({multi_val_pcc - single_val_pcc:.4f})")
    
    if multi_avg_pcc > single_avg_pcc:
        report.append(f"✓ 联合训练在平均通路PCC上优于单患者平均 (+{multi_avg_pcc - single_avg_pcc:.4f})")
    else:
        report.append(f"✗ 联合训练在平均通路PCC上劣于单患者平均 ({multi_avg_pcc - single_avg_pcc:.4f})")
    
    # 过拟合分析
    single_overfit = np.mean([all_metrics[n]['overfit_gap'] for n in ['HYZ15040', 'JFX0729', 'LMZ12939']])
    multi_overfit = all_metrics['MultiPatient']['overfit_gap']
    if multi_overfit < single_overfit:
        report.append(f"✓ 联合训练过拟合程度更低 ({multi_overfit:.4f} vs {single_overfit:.4f})")
    else:
        report.append(f"✗ 联合训练过拟合程度更高 ({multi_overfit:.4f} vs {single_overfit:.4f})")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"已保存: {report_path}")
    
    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("单患者 vs 多患者联合训练对比分析")
    print("=" * 60)
    
    # 创建输出目录
    create_output_dir()
    
    # 加载训练历史
    print("\n[1/5] 加载训练历史数据...")
    all_history = {}
    for name, file_path in TRAINING_HISTORY_FILES.items():
        all_history[name] = load_training_history(file_path)
        if all_history[name] is not None:
            print(f"  ✓ {name}: {len(all_history[name])} epochs")
    
    # 提取最佳指标
    print("\n[2/5] 提取最佳指标...")
    all_metrics = {}
    for name, df in all_history.items():
        all_metrics[name] = extract_best_metrics(df)
        if all_metrics[name]:
            m = all_metrics[name]
            print(f"  ✓ {name}: Best Epoch={m['best_epoch']}, Val PCC={m['val_pcc']:.4f}")
    
    # 加载预测结果并计算通路PCC
    print("\n[3/5] 计算逐通路PCC...")
    all_pathway_pcc = {}
    for name, file_path in PREDICTIONS_FILES.items():
        df = load_predictions(file_path)
        if df is not None:
            all_pathway_pcc[name] = calculate_pathway_pcc(df)
            print(f"  ✓ {name}: {len(all_pathway_pcc[name])} pathways, {len(df)} samples")
    
    # 生成图表
    print("\n[4/5] 生成可视化图表...")
    
    # 图1: 总体对比
    plot_overall_comparison(all_history, all_metrics)
    
    # 图2: 逐通路PCC对比
    df_plot, sorted_pathways = plot_pathway_pcc_comparison(all_pathway_pcc)
    
    # 图3: 增益分析
    improvement = plot_improvement_analysis(df_plot, sorted_pathways)
    
    # 图4: 综合报告
    plot_full_report(all_history, all_metrics, df_plot, sorted_pathways)
    
    # 生成文本报告
    print("\n[5/5] 生成分析报告...")
    report_text = generate_text_report(all_metrics, all_pathway_pcc, improvement)
    
    # 输出报告到控制台
    print("\n" + "=" * 60)
    print("分析报告摘要")
    print("=" * 60)
    print(report_text)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
