"""
从混合CSV中提取仅包含HYZ15040补丁的干净CSV
这是一个修复脚本（只新增文件，不修改现有文件）
"""
import os
import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

# 加载原始混合CSV
csv_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
df_mixed = pd.read_csv(csv_file)

print(f"原始混合CSV: {len(df_mixed)} 行")
print(f"列数: {len(df_mixed.columns)}")
print(f"首列名: {df_mixed.columns[0]}")

# 获取HYZ15040的有效补丁
val_patches = set([f.replace('.png', '') for f in os.listdir(
    base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"
) if f.lower().endswith('.png')])

train_patches = set([f.replace('.png', '') for f in os.listdir(
    base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "train_patches"
) if f.lower().endswith('.png')])

all_hyz_patches = val_patches | train_patches

print(f"\nHYZ15040数据集的补丁数:")
print(f"  val_patches: {len(val_patches)}")
print(f"  train_patches: {len(train_patches)}")
print(f"  总计: {len(all_hyz_patches)}")

# 过滤CSV，只保留HYZ15040的补丁
csv_stems = set([str(pid).replace('.png', '') for pid in df_mixed.iloc[:, 0]])
hyz_patches_in_csv = all_hyz_patches & csv_stems

print(f"\nCSV中与HYZ15040匹配的行数: {len(hyz_patches_in_csv)}")

# 创建过滤后的DataFrame
df_clean = df_mixed[df_mixed.iloc[:, 0].apply(
    lambda x: str(x).replace('.png', '') in all_hyz_patches
)]

print(f"清洁CSV后的行数: {len(df_clean)}")

# 保存为新文件（不修改原文件）
output_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore_clean.csv"
df_clean.to_csv(output_file, index=False)

print(f"\n✓ 清洁CSV已保存至:")
print(f"  {output_file}")

print(f"\n修复建议:")
print(f"1. 在train_uni.py中修改标签CSV路径为 HYZ15040_ssGSEA_zscore_clean.csv")
print(f"2. 或在命令行中添加: --labels_csv {output_file}")
print(f"3. 重新运行训练，验证集应该会加载265个样本而不是17个")
