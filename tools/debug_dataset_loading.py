import os
import sys
from pathlib import Path
import pandas as pd

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

# 加载val_patches
val_patches_dir = base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"
val_patches = {}
for fname in sorted(os.listdir(val_patches_dir)):
    if fname.lower().endswith('.png'):
        stem = fname.replace('.png', '')
        val_patches[stem] = fname

print(f"Val patches count: {len(val_patches)}")
print(f"First 5 val patches: {list(val_patches.keys())[:5]}")

# 加载CSV并构建label_map（复制dataset_uni.py逻辑）
csv_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
df = pd.read_csv(csv_file)
id_col = df.columns[0]
print(f"\nCSV id_col: {id_col}")

label_map = {}
for _, row in df.iterrows():
    stem = str(row[id_col]).replace('.png', '')
    label_map[stem] = row

print(f"Label_map size: {len(label_map)}")
print(f"First 5 label_map keys: {list(label_map.keys())[:5]}")

# 加载cache
cache_val_dir = base_dir / "uni2h_cache" / "HYZ15040" / "val"
cached_stems = set()
for fname in os.listdir(cache_val_dir):
    if fname.lower().endswith('.pt'):
        cached_stems.add(fname[:-3])  # 去掉.pt后缀

print(f"\nCache val stems count: {len(cached_stems)}")
print(f"First 5 cache stems: {sorted(list(cached_stems))[:5]}")

# 模拟dataset_uni.py的加载逻辑（从第59-71行）
matching_samples = []
print(f"\n开始匹配...")
for fname in sorted(os.listdir(val_patches_dir)):
    if not fname.lower().endswith('.png'):
        continue
    stem = fname.replace('.png', '')
    
    # 仅保留缓存目录和CSV中都有对应项的patch
    if stem not in label_map:
        print(f"  SKIP: {stem} - 不在label_map中")
        continue
    if stem not in cached_stems:
        print(f"  SKIP: {stem} - 不在cached_stems中")
        continue
    
    matching_samples.append(stem)

print(f"\nMatching samples: {len(matching_samples)}")
print(f"First 10 matching samples: {matching_samples[:10]}")

# 检查是否有样本只在patches中但不在CSV或cache中
only_in_patches = set(val_patches.keys()) - set(label_map.keys()) - set(cached_stems)
only_in_patches_but_csv = set(val_patches.keys()) - set(label_map.keys())
only_in_patches_but_cache = set(val_patches.keys()) - set(cached_stems)

print(f"\n诊断:")
print(f"  val_patches中有但CSV和cache都没有: {len(only_in_patches)}")
print(f"  val_patches中有但CSV中没有: {len(only_in_patches_but_csv)}")
print(f"  val_patches中有但cache中没有: {len(only_in_patches_but_cache)}")
