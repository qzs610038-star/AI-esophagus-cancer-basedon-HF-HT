"""
深度诊断：检查CSV中是否包含多个患者混合的数据
"""
import os
import pandas as pd
from pathlib import Path
import re

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

# 加载CSV
csv_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
df = pd.read_csv(csv_file)

print("=" * 70)
print("诊断：CSV中的补丁来源分析")
print("=" * 70)
print(f"CSV文件: {csv_file.name}")
print(f"总行数: {len(df)}")

# 分析补丁ID的分布
patch_ids = df.iloc[:, 0].tolist()

# 尝试从补丁坐标提取患者信息（如果有的话）
# 但patch_id格式是patch_x*_y*，没有患者前缀
# 让我们检查每个patients的val数据
val_patches_dirs = {
    'HYZ15040': base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches",
    'JFX0729': base_dir / "data_new_3ST" / "patch_noov_spilt" / "JFX0729_noov_split" / "val_patches",
    'LMZ12939': base_dir / "data_new_3ST" / "patch_noov_spilt" / "LMZ12939_noov_split" / "val_patches",
}

print("\n各患者的val_patches统计:")
all_patches = set()
for patient, path in val_patches_dirs.items():
    if path.exists():
        patches = set([f.replace('.png', '') for f in os.listdir(path) if f.lower().endswith('.png')])
        all_patches.update(patches)
        print(f"  {patient}: {len(patches)} patches")
    else:
        print(f"  {patient}: 目录不存在")

# 检查CSV中的补丁与各患者的patches的交集
csv_stems = set([pid.replace('.png', '') for pid in patch_ids])
print(f"\nCSV中的patch总数: {len(csv_stems)}")

print("\nCSV中的补丁与各患者val_patches的交集:")
for patient, path in val_patches_dirs.items():
    if path.exists():
        patches = set([f.replace('.png', '') for f in os.listdir(path) if f.lower().endswith('.png')])
        intersection = csv_stems & patches
        print(f"  CSV ∩ {patient} val_patches: {len(intersection)}")
        
        # 验证这些补丁是否真的都在CSV中
        if len(intersection) > 0:
            # 检查第一个几个
            sample = list(intersection)[:3]
            for s in sample:
                if s in csv_stems:
                    print(f"    ✓ {s} 在CSV中")
                else:
                    print(f"    ✗ {s} 不在CSV中")

# 关键诊断：检查CSV是否只包含HYZ15040的补丁
print("\n" + "=" * 70)
print("关键发现:")
print("=" * 70)

# 获取HYZ15040的val_patches
hyz_val_patches = set([f.replace('.png', '') for f in os.listdir(val_patches_dirs['HYZ15040']) if f.lower().endswith('.png')])
print(f"HYZ15040 val_patches: {len(hyz_val_patches)}")
print(f"CSV中的补丁总数: {len(csv_stems)}")
print(f"HYZ15040 val_patches ∩ CSV: {len(hyz_val_patches & csv_stems)}")
print(f"CSV中但不在HYZ15040 val_patches中: {len(csv_stems - hyz_val_patches)}")

# 这很可能意味着CSV是所有三个患者的数据混合！
if len(csv_stems - hyz_val_patches) > 0:
    print("\n!!! 问题发现 !!!")
    print("CSV文件包含了不属于HYZ15040的补丁，说明CSV是多个患者的数据混合！")
    print("这导致dataset_uni加载时：")
    print("  1. patches_dir中有265个HYZ15040的val补丁")
    print("  2. CSV中只有其中一部分与HYZ15040匹配（因为CSV大部分是其他患者的数据）")
    print("  3. cache中因为特征提取时使用的patches_dir导致的数据来源也混乱")
    
    # 估算实际应该有多少补丁
    # 通过检查cache的命名
    cache_val_dir = base_dir / "uni2h_cache" / "HYZ15040" / "val"
    cache_stems = set([f.replace('.pt', '') for f in os.listdir(cache_val_dir) if f.lower().endswith('.pt')])
    
    hyz_in_cache = hyz_val_patches & cache_stems
    print(f"\n进一步诊断：")
    print(f"  HYZ15040 val_patches中有多少在cache中: {len(hyz_in_cache)}")
    print(f"  这就是为什么最终只有{len(hyz_val_patches & csv_stems & hyz_in_cache)}个样本！")
