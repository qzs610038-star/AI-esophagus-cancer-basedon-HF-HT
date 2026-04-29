"""
完全模拟train_uni.py的加载逻辑
"""
import sys
import os
from pathlib import Path

# 加入项目路径
sys.path.insert(0, r"d:\AI空间转录病理研究\PFMval_new")

from histogene.dataset_uni import HisToGeneUNIDataset
from config_utils import load_config, get_data_paths

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

print("=" * 70)
print("模拟train_uni.py的加载逻辑")
print("=" * 70)

# 模拟train_uni.py中的参数设置
config = load_config()
data_paths = get_data_paths(config)

# 这些是默认从config.yaml读取的
train_patches_dir = data_paths["train_patches_dir"]
val_patches_dir = data_paths["val_patches_dir"]
labels_csv = data_paths["labels_csv_zscore"]

print(f"\n从config.yaml读取的路径:")
print(f"  train_patches_dir: {train_patches_dir}")
print(f"  val_patches_dir: {val_patches_dir}")
print(f"  labels_csv: {labels_csv}")

# 模拟缓存目录（从train_uni.py第305-306行）
dataset_name = "HYZ15040"
train_cache_dir = str(Path(base_dir) / "uni2h_cache" / dataset_name / "train")
val_cache_dir = str(Path(base_dir) / "uni2h_cache" / dataset_name / "val")

print(f"\n推断的缓存目录:")
print(f"  train_cache_dir: {train_cache_dir}")
print(f"  val_cache_dir: {val_cache_dir}")

# 现在加载验证集（这里会打印出样本数）
print(f"\n" + "=" * 70)
print("加载验证集...")
print("=" * 70)

val_dataset = HisToGeneUNIDataset(
    feature_cache_dir=val_cache_dir,
    patches_dir=val_patches_dir,
    labels_csv=labels_csv,
    n_pos=128,
)

print(f"\n验证集大小: {len(val_dataset)}")
print(f"\n如果上面显示17，说明问题在dataset_uni.py的加载逻辑中")
print(f"如果上面显示265，说明问题出在train_uni.py的其他地方（如多患者模式误判）")
