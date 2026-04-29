import os
from pathlib import Path

base_dir = r"d:\AI空间转录病理研究\PFMval_new"

# 1. 列出val_patches前20个PNG文件名
val_patches_dir = Path(base_dir) / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"
print("=" * 60)
print("1. val_patches 前20个PNG文件名:")
print("=" * 60)
if val_patches_dir.exists():
    pngs = sorted([f for f in os.listdir(val_patches_dir) if f.lower().endswith('.png')])[:20]
    for f in pngs:
        print(f"  {f}")
    print(f"Total val_patches PNG count: {len([f for f in os.listdir(val_patches_dir) if f.lower().endswith('.png')])}")
else:
    print(f"目录不存在: {val_patches_dir}")

# 2. 列出uni2h_cache/val前20个.pt文件名
cache_val_dir = Path(base_dir) / "uni2h_cache" / "HYZ15040" / "val"
print("\n" + "=" * 60)
print("2. uni2h_cache/HYZ15040/val 前20个.pt文件名:")
print("=" * 60)
if cache_val_dir.exists():
    pts = sorted([f for f in os.listdir(cache_val_dir) if f.lower().endswith('.pt')])[:20]
    for f in pts:
        print(f"  {f}")
    print(f"Total cache val .pt count: {len([f for f in os.listdir(cache_val_dir) if f.lower().endswith('.pt')])}")
else:
    print(f"目录不存在: {cache_val_dir}")

# 3. 读取CSV前20行，查看patch_id列的格式
csv_file = Path(base_dir) / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
print("\n" + "=" * 60)
print("3. ssGSEA_zscore CSV 前20行 patch_id 列:")
print("=" * 60)
if csv_file.exists():
    with open(csv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                print(f"Header: {line.strip()[:100]}")
            elif i <= 20:
                parts = line.strip().split(',')
                print(f"  {parts[0]}")
            else:
                break
    # 总行数统计
    with open(csv_file, 'r', encoding='utf-8') as f:
        total_rows = len(f.readlines())
    print(f"Total CSV rows (including header): {total_rows}")
else:
    print(f"文件不存在: {csv_file}")
