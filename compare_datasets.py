# -*- coding: utf-8 -*-
"""
HisToGene 三数据集训练结果对比可视化
生成 HYZ15040、JFX0729、LMZ12939 三个数据集的对比图表
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============== 配置 ==============
PROJECT_ROOT = r"D:\AI空间转录病理研究\PFMval_new"
HISTOGENE_DIR = os.path.join(PROJECT_ROOT, "histogene")
OUTPUT_DIR = os.path.join(HISTOGENE_DIR, "results_vis", "三数据集对比")

# 数据集配置
DATASETS = {
    "HYZ15040": {
        "color": "#4C72B0",  # 蓝色
        "training_history": os.path.join(HISTOGENE_DIR, "training_history_HYZ15040.csv"),
    },
    "JFX0729": {
        "color": "#DD8452",  # 橙色
        "training_history": os.path.join(HISTOGENE_DIR, "training_history_JFX0729.csv"),
    },
    "LMZ12939": {
        "color": "#55A868",  # 绿色
        "training_history": os.path.join(HISTOGENE_DIR, "training_history_LMZ12939.csv"),
    }
}

# 8个原始通路和22个新增通路
ORIGINAL_PATHWAYS = ['tls', 'tgfb', 'emt', 'hypoxia', 'mhc', 'icp', 'ifng', 'toxic']

# ============== 中文字体设置 ==============
def setup_chinese_font():
    """设置中文字体"""
    # 尝试多种中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'KaiTi']
    
    for font_name in chinese_fonts:
        try:
            # 查找字体文件
            font_paths = fm.findfont(font_name)
            if font_paths and 'DEJAVU' not in font_paths.upper():
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用中文字体: {font_name}")
                return True
        except:
            continue
    
    # 如果找不到中文字体，使用系统字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用默认中文字体设置")
    return True

# ============== 数据加载 ==============
def find_predictions_csv(dataset_name):
    """查找数据集对应的predictions.csv文件"""
    # 在 results_vis 目录下查找
    pattern = os.path.join(HISTOGENE_DIR, "checkpoints", "results_vis", f"{dataset_name}_*", "predictions.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    # 备用查找
    pattern2 = os.path.join(HISTOGENE_DIR, f"**/{dataset_name}*", "predictions.csv")
    matches2 = glob.glob(pattern2, recursive=True)
    if matches2:
        return matches2[0]
    
    return None

def load_training_history():
    """加载所有数据集的训练历史"""
    histories = {}
    for name, config in DATASETS.items():
        path = config["training_history"]
        if os.path.exists(path):
            df = pd.read_csv(path)
            histories[name] = df
            print(f"加载 {name} 训练历史: {len(df)} epochs")
        else:
            print(f"警告: 找不到 {path}")
    return histories

def load_predictions():
    """加载所有数据集的预测结果"""
    predictions = {}
    for name in DATASETS.keys():
        path = find_predictions_csv(name)
        if path:
            df = pd.read_csv(path)
            predictions[name] = df
            print(f"加载 {name} 预测结果: {len(df)} 样本")
        else:
            print(f"警告: 找不到 {name} 的 predictions.csv")
    return predictions

def calculate_pathway_pcc(pred_df):
    """计算每个通路的PCC"""
    # 获取所有通路名
    columns = pred_df.columns.tolist()
    pathways = []
    for col in columns:
        if col.startswith('true_'):
            pathway = col.replace('true_', '')
            pathways.append(pathway)
    
    pcc_values = {}
    for pathway in pathways:
        true_col = f'true_{pathway}'
        pred_col = f'pred_{pathway}'
        if true_col in pred_df.columns and pred_col in pred_df.columns:
            true_vals = pred_df[true_col].values
            pred_vals = pred_df[pred_col].values
            # 移除NaN
            mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
            if mask.sum() > 2:
                pcc, _ = pearsonr(true_vals[mask], pred_vals[mask])
                pcc_values[pathway] = pcc
    return pcc_values

# ============== 图表生成 ==============
def plot_overall_comparison(histories, predictions, output_path):
    """图1: 总体性能对比面板"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HisToGene 三数据集总体性能对比', fontsize=16, fontweight='bold')
    
    # 子图1: Val Loss 训练曲线
    ax1 = axes[0, 0]
    for name, config in DATASETS.items():
        if name in histories:
            df = histories[name]
            sample_n = len(predictions.get(name, []))
            label = f"{name} (n={sample_n})"
            
            # 绘制曲线
            ax1.plot(df['epoch'], df['val_loss'], color=config['color'], label=label, linewidth=2)
            
            # 标注最佳epoch (最小val_loss)
            best_idx = df['val_loss'].idxmin()
            best_epoch = df.loc[best_idx, 'epoch']
            best_loss = df.loc[best_idx, 'val_loss']
            ax1.scatter([best_epoch], [best_loss], color=config['color'], s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('验证损失曲线', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Val PCC 训练曲线
    ax2 = axes[0, 1]
    for name, config in DATASETS.items():
        if name in histories:
            df = histories[name]
            sample_n = len(predictions.get(name, []))
            label = f"{name} (n={sample_n})"
            
            ax2.plot(df['epoch'], df['val_pcc'], color=config['color'], label=label, linewidth=2)
            
            # 标注最佳epoch (最大val_pcc)
            best_idx = df['val_pcc'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_pcc = df.loc[best_idx, 'val_pcc']
            ax2.scatter([best_epoch], [best_pcc], color=config['color'], s=100, zorder=5, marker='*')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation PCC', fontsize=12)
    ax2.set_title('验证PCC曲线', fontsize=13)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: Val R² 训练曲线
    ax3 = axes[1, 0]
    for name, config in DATASETS.items():
        if name in histories:
            df = histories[name]
            sample_n = len(predictions.get(name, []))
            label = f"{name} (n={sample_n})"
            
            ax3.plot(df['epoch'], df['val_r2'], color=config['color'], label=label, linewidth=2)
            
            # 标注最佳epoch (最大val_r2)
            best_idx = df['val_r2'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_r2 = df.loc[best_idx, 'val_r2']
            ax3.scatter([best_epoch], [best_r2], color=config['color'], s=100, zorder=5, marker='*')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation R²', fontsize=12)
    ax3.set_title('验证R²曲线', fontsize=13)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 最佳epoch关键指标柱状图
    ax4 = axes[1, 1]
    metrics_data = {'数据集': [], 'PCC': [], 'R²': [], 'MAE': []}
    
    for name in DATASETS.keys():
        if name in histories:
            df = histories[name]
            best_idx = df['val_pcc'].idxmax()
            metrics_data['数据集'].append(name)
            metrics_data['PCC'].append(df.loc[best_idx, 'val_pcc'])
            metrics_data['R²'].append(df.loc[best_idx, 'val_r2'])
            metrics_data['MAE'].append(df.loc[best_idx, 'val_mae'])
    
    x = np.arange(len(metrics_data['数据集']))
    width = 0.25
    
    colors = ['#4C72B0', '#DD8452', '#55A868']
    
    bars1 = ax4.bar(x - width, metrics_data['PCC'], width, label='PCC', color=colors[0], alpha=0.8)
    bars2 = ax4.bar(x, metrics_data['R²'], width, label='R²', color=colors[1], alpha=0.8)
    bars3 = ax4.bar(x + width, metrics_data['MAE'], width, label='MAE', color=colors[2], alpha=0.8)
    
    ax4.set_xlabel('数据集', fontsize=12)
    ax4.set_ylabel('指标值', fontsize=12)
    ax4.set_title('最佳Epoch关键指标对比', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_data['数据集'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_path}")

def plot_pathway_pcc_comparison(pathway_pcc_all, output_path):
    """图2: 逐通路PCC三数据集对比"""
    # 获取所有通路并按平均PCC排序
    all_pathways = list(pathway_pcc_all[list(pathway_pcc_all.keys())[0]].keys())
    
    # 计算每个通路的平均PCC并排序
    avg_pcc = {}
    for pathway in all_pathways:
        pccs = [pathway_pcc_all[ds][pathway] for ds in pathway_pcc_all.keys() if pathway in pathway_pcc_all[ds]]
        avg_pcc[pathway] = np.mean(pccs) if pccs else 0
    
    sorted_pathways = sorted(all_pathways, key=lambda x: avg_pcc.get(x, 0), reverse=True)
    
    # 区分原始通路和新增通路
    original_pathways = [p for p in sorted_pathways if p.lower() in [o.lower() for o in ORIGINAL_PATHWAYS]]
    new_pathways = [p for p in sorted_pathways if p.lower() not in [o.lower() for o in ORIGINAL_PATHWAYS]]
    
    # 按原始通路在前，新增通路在后的顺序排列（各自按PCC排序）
    original_pathways = sorted(original_pathways, key=lambda x: avg_pcc.get(x, 0), reverse=True)
    new_pathways = sorted(new_pathways, key=lambda x: avg_pcc.get(x, 0), reverse=True)
    final_pathways = original_pathways + new_pathways
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x = np.arange(len(final_pathways))
    width = 0.25
    
    colors = {'HYZ15040': '#4C72B0', 'JFX0729': '#DD8452', 'LMZ12939': '#55A868'}
    
    # 绘制分组柱状图
    for i, (ds, color) in enumerate(colors.items()):
        pcc_values = [pathway_pcc_all[ds].get(p, 0) for p in final_pathways]
        bars = ax.bar(x + i * width, pcc_values, width, label=ds, color=color, alpha=0.8)
    
    # 添加原始通路和新增通路的分隔线
    if len(original_pathways) > 0 and len(new_pathways) > 0:
        sep_pos = len(original_pathways) - 0.5
        ax.axvline(x=sep_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(sep_pos, ax.get_ylim()[1] * 0.95, '← 原始通路 | 新增通路 →', 
                ha='center', va='top', fontsize=10, color='red')
    
    # 添加水平参考线（各数据集整体平均PCC）
    for ds, color in colors.items():
        avg = np.mean(list(pathway_pcc_all[ds].values()))
        ax.axhline(y=avg, color=color, linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text(len(final_pathways) - 0.5, avg + 0.01, f'{ds}均值:{avg:.3f}', 
                ha='right', va='bottom', fontsize=9, color=color)
    
    ax.set_xlabel('通路', fontsize=12)
    ax.set_ylabel('PCC', fontsize=12)
    ax.set_title('逐通路PCC三数据集对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(final_pathways, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_path}")

def plot_pathway_heatmap(pathway_pcc_all, output_path):
    """图3: 逐通路PCC热力图"""
    # 获取所有通路
    all_pathways = list(pathway_pcc_all[list(pathway_pcc_all.keys())[0]].keys())
    
    # 计算平均PCC并排序
    avg_pcc = {}
    for pathway in all_pathways:
        pccs = [pathway_pcc_all[ds][pathway] for ds in pathway_pcc_all.keys()]
        avg_pcc[pathway] = np.mean(pccs) if pccs else 0
    
    sorted_pathways = sorted(all_pathways, key=lambda x: avg_pcc.get(x, 0), reverse=True)
    
    # 构建热力图数据
    data = []
    for pathway in sorted_pathways:
        row = [pathway_pcc_all[ds].get(pathway, 0) for ds in DATASETS.keys()]
        data.append(row)
    
    df_heatmap = pd.DataFrame(data, index=sorted_pathways, columns=list(DATASETS.keys()))
    
    fig, ax = plt.subplots(figsize=(8, 14))
    
    # 使用绿色渐变色系
    cmap = sns.color_palette("YlGn", as_cmap=True)
    
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap=cmap, 
                vmin=0, vmax=1, ax=ax, 
                cbar_kws={'label': 'PCC', 'shrink': 0.8},
                annot_kws={'size': 9})
    
    ax.set_title('逐通路PCC热力图', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('数据集', fontsize=12)
    ax.set_ylabel('通路', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_path}")

def plot_full_comparison_report(histories, predictions, pathway_pcc_all, output_path):
    """图4: 综合大图"""
    fig = plt.figure(figsize=(20, 16))
    
    # 添加总标题
    fig.suptitle('HisToGene 三数据集训练结果对比报告', fontsize=18, fontweight='bold', y=0.98)
    
    # ========== 顶部：总体指标表格 ==========
    # 创建一个子图用于表格
    ax_table = fig.add_axes([0.05, 0.88, 0.9, 0.08])
    ax_table.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['数据集', '样本数', '训练Epoch', '最佳PCC', '最佳R²', '最佳MAE', '最终Loss']
    
    for name in DATASETS.keys():
        if name in histories:
            df = histories[name]
            sample_n = len(predictions.get(name, []))
            best_idx = df['val_pcc'].idxmax()
            row = [
                name,
                str(sample_n),
                str(len(df)),
                f"{df.loc[best_idx, 'val_pcc']:.4f}",
                f"{df.loc[best_idx, 'val_r2']:.4f}",
                f"{df.loc[best_idx, 'val_mae']:.4f}",
                f"{df.loc[best_idx, 'val_loss']:.4f}"
            ]
            table_data.append(row)
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center',
                          colColours=['#E8E8E8']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # ========== 上半部分：训练曲线 ==========
    # Val Loss 曲线
    ax1 = fig.add_axes([0.08, 0.58, 0.4, 0.25])
    for name, config in DATASETS.items():
        if name in histories:
            df = histories[name]
            ax1.plot(df['epoch'], df['val_loss'], color=config['color'], label=name, linewidth=2)
            best_idx = df['val_loss'].idxmin()
            best_epoch = df.loc[best_idx, 'epoch']
            best_loss = df.loc[best_idx, 'val_loss']
            ax1.scatter([best_epoch], [best_loss], color=config['color'], s=80, zorder=5, marker='*')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Validation Loss', fontsize=11)
    ax1.set_title('验证损失曲线', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Val PCC 曲线
    ax2 = fig.add_axes([0.55, 0.58, 0.4, 0.25])
    for name, config in DATASETS.items():
        if name in histories:
            df = histories[name]
            ax2.plot(df['epoch'], df['val_pcc'], color=config['color'], label=name, linewidth=2)
            best_idx = df['val_pcc'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_pcc = df.loc[best_idx, 'val_pcc']
            ax2.scatter([best_epoch], [best_pcc], color=config['color'], s=80, zorder=5, marker='*')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation PCC', fontsize=11)
    ax2.set_title('验证PCC曲线', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # ========== 下半部分：逐通路PCC柱状图 ==========
    ax3 = fig.add_axes([0.05, 0.08, 0.9, 0.42])
    
    # 获取所有通路
    all_pathways = list(pathway_pcc_all[list(pathway_pcc_all.keys())[0]].keys())
    
    # 计算平均PCC并排序
    avg_pcc = {}
    for pathway in all_pathways:
        pccs = [pathway_pcc_all[ds][pathway] for ds in pathway_pcc_all.keys()]
        avg_pcc[pathway] = np.mean(pccs)
    
    sorted_pathways = sorted(all_pathways, key=lambda x: avg_pcc[x], reverse=True)
    
    # 区分原始通路和新增通路
    original_pathways = [p for p in sorted_pathways if p.lower() in [o.lower() for o in ORIGINAL_PATHWAYS]]
    new_pathways = [p for p in sorted_pathways if p.lower() not in [o.lower() for o in ORIGINAL_PATHWAYS]]
    original_pathways = sorted(original_pathways, key=lambda x: avg_pcc[x], reverse=True)
    new_pathways = sorted(new_pathways, key=lambda x: avg_pcc[x], reverse=True)
    final_pathways = original_pathways + new_pathways
    
    x = np.arange(len(final_pathways))
    width = 0.25
    
    colors = {'HYZ15040': '#4C72B0', 'JFX0729': '#DD8452', 'LMZ12939': '#55A868'}
    
    for i, (ds, color) in enumerate(colors.items()):
        pcc_values = [pathway_pcc_all[ds].get(p, 0) for p in final_pathways]
        ax3.bar(x + i * width, pcc_values, width, label=ds, color=color, alpha=0.8)
    
    # 添加分隔线
    if len(original_pathways) > 0 and len(new_pathways) > 0:
        sep_pos = len(original_pathways) - 0.5
        ax3.axvline(x=sep_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.text(sep_pos, 0.95, '← 原始通路 | 新增通路 →', ha='center', va='top', fontsize=10, color='red')
    
    # 添加水平参考线
    for ds, color in colors.items():
        avg = np.mean(list(pathway_pcc_all[ds].values()))
        ax3.axhline(y=avg, color=color, linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax3.set_xlabel('通路', fontsize=11)
    ax3.set_ylabel('PCC', fontsize=11)
    ax3.set_title('逐通路PCC三数据集对比', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(final_pathways, rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper right', ncol=3)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.0)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_path}")

# ============== 主函数 ==============
def main():
    print("=" * 60)
    print("HisToGene 三数据集训练结果对比可视化")
    print("=" * 60)
    
    # 设置中文字体
    setup_chinese_font()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("\n[1] 加载训练历史数据...")
    histories = load_training_history()
    
    print("\n[2] 加载预测结果...")
    predictions = load_predictions()
    
    print("\n[3] 计算逐通路PCC...")
    pathway_pcc_all = {}
    for name, pred_df in predictions.items():
        pathway_pcc_all[name] = calculate_pathway_pcc(pred_df)
        print(f"  {name}: {len(pathway_pcc_all[name])} 个通路")
    
    # 生成图表
    print("\n[4] 生成图表...")
    
    # 图1: 总体性能对比面板
    plot_overall_comparison(
        histories, predictions,
        os.path.join(OUTPUT_DIR, "overall_comparison.png")
    )
    
    # 图2: 逐通路PCC对比
    plot_pathway_pcc_comparison(
        pathway_pcc_all,
        os.path.join(OUTPUT_DIR, "pathway_pcc_comparison.png")
    )
    
    # 图3: 逐通路PCC热力图
    plot_pathway_heatmap(
        pathway_pcc_all,
        os.path.join(OUTPUT_DIR, "pathway_heatmap.png")
    )
    
    # 图4: 综合大图
    plot_full_comparison_report(
        histories, predictions, pathway_pcc_all,
        os.path.join(OUTPUT_DIR, "full_comparison_report.png")
    )
    
    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 打印生成的文件列表
    print("\n生成的文件:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            filepath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
