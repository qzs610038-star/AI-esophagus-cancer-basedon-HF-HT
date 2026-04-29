import sys
sys.path.insert(0, r"d:\AI空间转录病理研究\PFMval_new")

from histogene.dataset_uni import HisToGeneUNIDataset
from pathlib import Path

base_dir = Path(r"d:\AI空间转录病理研究\PFMval_new")

print("=" * 70)
print("验证清洁CSV是否能加载正确的样本数")
print("=" * 70)

# 使用新的干净CSV
val_dataset = HisToGeneUNIDataset(
    feature_cache_dir=str(base_dir / "uni2h_cache" / "HYZ15040" / "val"),
    patches_dir=str(base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"),
    labels_csv=str(base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore_clean.csv"),
    n_pos=128,
)

print(f"\n验证集样本数: {len(val_dataset)}")
print(f"✓ 预期: 265个样本")
print(f"✓ 实际: {len(val_dataset)}个样本")

if len(val_dataset) == 265:
    print("\n✓ 成功！清洁CSV能正确加载所有265个验证样本")
else:
    print(f"\n✗ 失败！样本数不符")
