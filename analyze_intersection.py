import os
from pathlib import Path

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

# 加载三层数据
val_patches = set()
val_patches_dir = base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"
for f in os.listdir(val_patches_dir):
    if f.lower().endswith(".png"):
        val_patches.add(f.replace(".png", ""))

csv_stems = set()
csv_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
with open(csv_file, "r", encoding="utf-8") as f:
    next(f)  # 跳过header
    for line in f:
        parts = line.strip().split(",")
        patch_id = parts[0].replace(".png", "")
        csv_stems.add(patch_id)

cache_val_stems = set()
cache_val_dir = base_dir / "uni2h_cache" / "HYZ15040" / "val"
for f in os.listdir(cache_val_dir):
    if f.lower().endswith(".pt"):
        cache_val_stems.add(f.replace(".pt", ""))

print("=" * 70)
print("三层数据统计（Val）:")
print("=" * 70)
print(f"  patches_dir PNG count: {len(val_patches)}")
print(f"  CSV stems count: {len(csv_stems)}")
print(f"  cache .pt count: {len(cache_val_stems)}")

# 计算交集
intersection_1_2 = val_patches & csv_stems
intersection_1_3 = val_patches & cache_val_stems
intersection_2_3 = csv_stems & cache_val_stems
intersection_all = val_patches & csv_stems & cache_val_stems

print("\n" + "=" * 70)
print("三层交集分析（Val）:")
print("=" * 70)
print(f"  patches ∩ CSV: {len(intersection_1_2)}")
print(f"  patches ∩ cache: {len(intersection_1_3)}")
print(f"  CSV ∩ cache: {len(intersection_2_3)}")
print(f"  patches ∩ CSV ∩ cache: {len(intersection_all)}")

# 找出缺失的部分
missing_from_csv = val_patches - csv_stems
missing_from_cache = val_patches - cache_val_stems
missing_from_intersection = val_patches - intersection_all

print("\n" + "=" * 70)
print("缺失分析（Val）:")
print("=" * 70)
print(f"  patches中有但CSV中没有: {len(missing_from_csv)}")
if len(missing_from_csv) > 0:
    print(f"    Examples (up to 10): {sorted(list(missing_from_csv))[:10]}")

print(f"\n  patches中有但cache中没有: {len(missing_from_cache)}")
if len(missing_from_cache) > 0:
    print(f"    Examples (up to 10): {sorted(list(missing_from_cache))[:10]}")

print(f"\n  patches中有但三层交集中没有: {len(missing_from_intersection)}")
if len(missing_from_intersection) > 0:
    print(f"    Examples (up to 10): {sorted(list(missing_from_intersection))[:10]}")

# Train数据集的同样分析
print("\n\n" + "=" * 70)
print("Train数据集分析:")
print("=" * 70)

train_patches = set()
train_patches_dir = base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "train_patches"
for f in os.listdir(train_patches_dir):
    if f.lower().endswith(".png"):
        train_patches.add(f.replace(".png", ""))

cache_train_stems = set()
cache_train_dir = base_dir / "uni2h_cache" / "HYZ15040" / "train"
for f in os.listdir(cache_train_dir):
    if f.lower().endswith(".pt"):
        cache_train_stems.add(f.replace(".pt", ""))

print(f"  patches_dir PNG count: {len(train_patches)}")
print(f"  CSV stems count: {len(csv_stems)}")
print(f"  cache .pt count: {len(cache_train_stems)}")

train_intersection_1_2 = train_patches & csv_stems
train_intersection_1_3 = train_patches & cache_train_stems
train_intersection_2_3 = csv_stems & cache_train_stems
train_intersection_all = train_patches & csv_stems & cache_train_stems

print(f"\n  patches ∩ CSV: {len(train_intersection_1_2)}")
print(f"  patches ∩ cache: {len(train_intersection_1_3)}")
print(f"  CSV ∩ cache: {len(train_intersection_2_3)}")
print(f"  patches ∩ CSV ∩ cache: {len(train_intersection_all)}")

