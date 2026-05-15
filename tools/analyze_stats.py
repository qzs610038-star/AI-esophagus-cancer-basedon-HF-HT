import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"d:\AI空间转录病理研究\PFMval_new\HYZ15040_ssGSEA_scores.csv")

# 获取数值列（排除第一列 patch_id）
cols = df.columns[1:]

for col in cols:
    data = df[col].dropna()
    print(f"\n=== {col} ===")
    print(f"样本量: {len(data)}")
    print(f"均值: {data.mean():.4f}")
    print(f"中位数: {data.median():.4f}")
    print(f"标准差: {data.std():.4f}")
    print(f"最小值: {data.min():.4f}")
    print(f"最大值: {data.max():.4f}")
    print(f"偏度(skewness): {data.skew():.4f}")
    print(f"峰度(kurtosis): {data.kurtosis():.4f}")
    # Shapiro-Wilk 正态性检验（取前5000个样本，因为该检验有样本量限制）
    sample = data.sample(min(5000, len(data)), random_state=42)
    stat_sw, p_sw = stats.shapiro(sample)
    print(f"Shapiro-Wilk检验: stat={stat_sw:.6f}, p-value={p_sw:.2e}")
    # D'Agostino-Pearson 正态性检验
    stat_dp, p_dp = stats.normaltest(data)
    print(f"D'Agostino-Pearson检验: stat={stat_dp:.4f}, p-value={p_dp:.2e}")
    # 分位数
    print(f"25%分位: {data.quantile(0.25):.4f}")
    print(f"75%分位: {data.quantile(0.75):.4f}")
    print(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}")
    # 异常值数量（基于1.5*IQR规则）
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
    print(f"异常值数量(1.5*IQR): {outliers} ({outliers/len(data)*100:.2f}%)")
