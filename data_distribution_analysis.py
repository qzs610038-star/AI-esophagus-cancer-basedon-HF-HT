# -*- coding: utf-8 -*-
"""
数据分布分析与可视化脚本

用途：
    对 HYZ15040_ssGSEA_scores.csv 中的基因集评分数据进行全面的统计分析
    和可视化展示，包括描述性统计、正态性检验、异常值检测以及多种图表生成。

输入：
    - HYZ15040_ssGSEA_scores.csv: 包含 patch_id 和 8 个基因集评分的数据文件

输出：
    - analysis_output/statistics_summary.csv: 统计结果汇总表
    - analysis_output/histograms.png: 各列直方图（含正态拟合曲线）
    - analysis_output/qq_plots.png: QQ图（正态概率图）
    - analysis_output/boxplots.png: 箱线图对比
    - analysis_output/skew_kurtosis.png: 偏度和峰度对比柱状图
    - analysis_output/correlation_heatmap.png: 相关性热力图

作者：自动生成
日期：2026-04-11
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = r'd:\AI空间转录病理研究\PFMval_new\analysis_output'


def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")


def load_data(file_path):
    """
    读取CSV数据文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        DataFrame: 包含数据的DataFrame
    """
    df = pd.read_csv(file_path)
    print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")
    return df


def calculate_statistics(df, numeric_cols):
    """
    计算各数值列的详细统计量
    
    Args:
        df: DataFrame
        numeric_cols: 数值列名称列表
        
    Returns:
        dict: 包含各列统计结果的字典
    """
    stats_dict = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        n = len(data)
        
        # 基本统计量
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val
        
        # 偏度和峰度
        skew_val = skew(data)
        kurt_val = kurtosis(data)  #  excess kurtosis (Fisher's definition)
        
        # 正态性检验 - Shapiro-Wilk (样本量上限5000)
        if n <= 5000:
            shapiro_stat, shapiro_p = shapiro(data)
        else:
            # 对于大样本，随机抽取5000个进行Shapiro检验
            sample_data = data.sample(n=5000, random_state=42)
            shapiro_stat, shapiro_p = shapiro(sample_data)
        
        # D'Agostino-Pearson 正态性检验
        dagostino_stat, dagostino_p = normaltest(data)
        
        # 异常值检测 (1.5×IQR法则)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / n * 100
        
        # 均值与中位数差值
        mean_median_diff = mean_val - median_val
        
        stats_dict[col] = {
            '样本量': n,
            '均值': mean_val,
            '中位数': median_val,
            '标准差': std_val,
            '最小值': min_val,
            '最大值': max_val,
            '范围': range_val,
            '偏度': skew_val,
            '峰度': kurt_val,
            'Shapiro-Wilk统计量': shapiro_stat,
            'Shapiro-Wilk_p值': shapiro_p,
            'D\'Agostino统计量': dagostino_stat,
            'D\'Agostino_p值': dagostino_p,
            '异常值数量': outlier_count,
            '异常值比例(%)': outlier_ratio,
            '均值-中位数': mean_median_diff
        }
    
    return stats_dict


def print_statistics_table(stats_dict):
    """打印统计结果汇总表"""
    print("\n" + "="*100)
    print("统计结果汇总表")
    print("="*100)
    
    # 创建DataFrame便于展示
    stats_df = pd.DataFrame(stats_dict).T
    
    # 打印简化版表格
    display_cols = ['样本量', '均值', '中位数', '标准差', '偏度', '峰度', '异常值比例(%)']
    print("\n基本统计量:")
    print(stats_df[display_cols].round(4).to_string())
    
    return stats_df


def save_statistics_csv(stats_dict):
    """将统计结果保存为CSV文件"""
    stats_df = pd.DataFrame(stats_dict).T
    output_path = os.path.join(OUTPUT_DIR, 'statistics_summary.csv')
    stats_df.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n统计结果已保存至: {output_path}")
    return stats_df


def plot_histograms(df, numeric_cols, stats_dict):
    """
    绘制各列直方图（含正态拟合曲线）
    
    Args:
        df: DataFrame
        numeric_cols: 数值列名称列表
        stats_dict: 统计结果字典
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        # 绘制直方图
        n_bins, bins, patches = ax.hist(data, bins=50, density=True, 
                                         alpha=0.7, color='steelblue', edgecolor='white')
        
        # 计算正态分布拟合曲线
        mean_val = stats_dict[col]['均值']
        std_val = stats_dict[col]['标准差']
        x = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x, mean_val, std_val)
        
        # 绘制正态拟合曲线
        ax.plot(x, normal_curve, 'r--', linewidth=2, label='Normal Fit')
        
        # 标注均值和中位数竖线
        ax.axvline(stats_dict[col]['均值'], color='blue', linestyle='-', 
                   linewidth=2, label=f'Mean={mean_val:.2f}')
        ax.axvline(stats_dict[col]['中位数'], color='green', linestyle='--', 
                   linewidth=2, label=f'Median={stats_dict[col]["中位数"]:.2f}')
        
        # 标题包含偏度和峰度
        skew_val = stats_dict[col]['偏度']
        kurt_val = stats_dict[col]['峰度']
        ax.set_title(f'{col}\nSkew={skew_val:.3f}, Kurt={kurt_val:.3f}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'histograms.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"直方图已保存至: {output_path}")
    plt.close()


def plot_qq_plots(df, numeric_cols):
    """
    绘制QQ图（正态概率图）
    
    Args:
        df: DataFrame
        numeric_cols: 数值列名称列表
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        # 绘制QQ图
        stats.probplot(data, dist="norm", plot=ax)
        
        ax.set_title(f'{col} - Q-Q Plot', fontsize=10)
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')
        ax.get_lines()[1].set_linestyle('--')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'qq_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"QQ图已保存至: {output_path}")
    plt.close()


def plot_boxplots(df, numeric_cols):
    """
    绘制箱线图对比（标准化后的数据）
    
    Args:
        df: DataFrame
        numeric_cols: 数值列名称列表
    """
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    # 标准化数据以便对比
    standardized_data = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # 绘制箱线图
    bp = ax.boxplot([standardized_data[col].dropna() for col in numeric_cols],
                    labels=numeric_cols, patch_artist=True)
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(numeric_cols)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置异常值样式
    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5, markersize=4)
    
    ax.set_title('Boxplots of Standardized Gene Set Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gene Sets', fontsize=12)
    ax.set_ylabel('Standardized Values (Z-score)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'boxplots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"箱线图已保存至: {output_path}")
    plt.close()


def plot_skew_kurtosis(stats_dict, numeric_cols):
    """
    绘制偏度和峰度对比柱状图
    
    Args:
        stats_dict: 统计结果字典
        numeric_cols: 数值列名称列表
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    skew_values = [stats_dict[col]['偏度'] for col in numeric_cols]
    kurt_values = [stats_dict[col]['峰度'] for col in numeric_cols]
    
    # 偏度对比图
    ax1 = axes[0]
    colors_skew = ['green' if abs(s) < 0.5 else 'orange' if abs(s) < 1 else 'red' for s in skew_values]
    bars1 = ax1.bar(numeric_cols, skew_values, color=colors_skew, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Normal (Skew=0)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate (±0.5)')
    ax1.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='High (±1)')
    ax1.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_title('Skewness Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Gene Sets')
    ax1.set_ylabel('Skewness')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars1, skew_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # 峰度对比图
    ax2 = axes[1]
    colors_kurt = ['green' if abs(k) < 0.5 else 'orange' if abs(k) < 1 else 'red' for k in kurt_values]
    bars2 = ax2.bar(numeric_cols, kurt_values, color=colors_kurt, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Normal (Kurt=0)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate (±0.5)')
    ax2.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='High (±1)')
    ax2.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title('Kurtosis Comparison (Excess)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Gene Sets')
    ax2.set_ylabel('Kurtosis')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars2, kurt_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'skew_kurtosis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"偏度峰度图已保存至: {output_path}")
    plt.close()


def plot_correlation_heatmap(df, numeric_cols):
    """
    绘制相关性热力图
    
    Args:
        df: DataFrame
        numeric_cols: 数值列名称列表
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # 计算皮尔逊相关系数矩阵
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Pearson Correlation Heatmap\n(Gene Set Scores)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"相关性热力图已保存至: {output_path}")
    plt.close()
    
    return corr_matrix


def print_normality_conclusion(stats_dict, numeric_cols):
    """打印正态性检验结论"""
    print("\n" + "="*100)
    print("正态性检验结论")
    print("="*100)
    print("\n检验方法说明:")
    print("- Shapiro-Wilk检验: 适用于小样本(n≤5000)，p>0.05表示符合正态分布")
    print("- D'Agostino-Pearson检验: 基于偏度和峰度，p>0.05表示符合正态分布")
    print("- 偏度: 衡量分布不对称性，|偏度|<0.5近似对称，0.5-1中等偏态，>1高度偏态")
    print("- 峰度: 衡量分布尾部厚度，峰度=0为正态分布，>0尖峰，<0平峰")
    
    print("\n" + "-"*100)
    print(f"{'基因集':<10} {'Shapiro_p':<12} {'D\'Agostino_p':<15} {'偏度':<10} {'峰度':<10} {'结论':<20}")
    print("-"*100)
    
    for col in numeric_cols:
        shapiro_p = stats_dict[col]['Shapiro-Wilk_p值']
        dagostino_p = stats_dict[col]['D\'Agostino_p值']
        skew_val = stats_dict[col]['偏度']
        kurt_val = stats_dict[col]['峰度']
        
        # 判断结论
        if shapiro_p > 0.05 and dagostino_p > 0.05 and abs(skew_val) < 0.5:
            conclusion = "近似正态分布"
        elif abs(skew_val) > 1 or abs(kurt_val) > 1:
            conclusion = "明显非正态分布"
        else:
            conclusion = "轻度偏离正态"
        
        print(f"{col:<10} {shapiro_p:<12.2e} {dagostino_p:<15.2e} {skew_val:<10.3f} {kurt_val:<10.3f} {conclusion:<20}")
    
    print("-"*100)


def main():
    """主函数"""
    print("="*100)
    print("数据分布分析与可视化")
    print("="*100)
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 数据文件路径
    data_file = r'd:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores.csv'
    
    # 加载数据
    df = load_data(data_file)
    
    # 自动识别数值列（排除patch_id）
    numeric_cols = [col for col in df.columns if col != 'patch_id']
    print(f"\n识别的数值列: {numeric_cols}")
    
    # 计算统计量
    print("\n正在计算统计量...")
    stats_dict = calculate_statistics(df, numeric_cols)
    
    # 打印统计结果汇总表
    stats_df = print_statistics_table(stats_dict)
    
    # 保存统计结果到CSV
    save_statistics_csv(stats_dict)
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    
    print("\n1. 绘制直方图...")
    plot_histograms(df, numeric_cols, stats_dict)
    
    print("2. 绘制QQ图...")
    plot_qq_plots(df, numeric_cols)
    
    print("3. 绘制箱线图...")
    plot_boxplots(df, numeric_cols)
    
    print("4. 绘制偏度峰度对比图...")
    plot_skew_kurtosis(stats_dict, numeric_cols)
    
    print("5. 绘制相关性热力图...")
    corr_matrix = plot_correlation_heatmap(df, numeric_cols)
    
    # 打印正态性检验结论
    print_normality_conclusion(stats_dict, numeric_cols)
    
    # 打印完成信息
    print("\n" + "="*100)
    print("分析完成！")
    print("="*100)
    print(f"\n所有输出文件保存在: {OUTPUT_DIR}")
    print("生成的文件:")
    print("  - statistics_summary.csv: 统计结果汇总表")
    print("  - histograms.png: 各列直方图（含正态拟合曲线）")
    print("  - qq_plots.png: QQ图（正态概率图）")
    print("  - boxplots.png: 箱线图对比")
    print("  - skew_kurtosis.png: 偏度和峰度对比柱状图")
    print("  - correlation_heatmap.png: 相关性热力图")


if __name__ == "__main__":
    main()
