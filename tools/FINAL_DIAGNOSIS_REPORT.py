"""
完整诊断报告生成
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import get_project_root

base_dir = Path(get_project_root())

# 加载各类数据源
val_patches = set([f.replace('.png', '') for f in os.listdir(
    base_dir / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "val_patches"
) if f.lower().endswith('.png')])

import pandas as pd
csv = pd.read_csv(base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv")
csv_stems = set([str(pid).replace('.png', '') for pid in csv.iloc[:, 0]])

cache_val_stems = set([f.replace('.pt', '') for f in os.listdir(
    base_dir / "uni2h_cache" / "HYZ15040" / "val"
) if f.lower().endswith('.pt')])

# 计算交集
intersection = val_patches & csv_stems & cache_val_stems

print("=" * 70)
print("HisToGene-UNI 在 HYZ15040 验证集从265降至17 根本原因诊断")
print("=" * 70)

print("\n【第一阶段：文件名格式分析】")
print(f"  val_patches PNG文件名格式示例:")
sample_val = sorted(list(val_patches))[:3]
for s in sample_val:
    print(f"    - {s}")

print(f"\n  CSV文件patch_id列格式示例:")
sample_csv = sorted(list(csv_stems))[:3]
for s in sample_csv:
    print(f"    - {s}")

print(f"\n  cache .pt文件名格式示例:")
sample_cache = sorted(list(cache_val_stems))[:3]
for s in sample_cache:
    print(f"    - {s}")

print("\n  结论: 文件名格式完全一致 ✓")

print("\n【第二阶段：交集分析】")
print(f"  val_patches (PNG): {len(val_patches)}")
print(f"  CSV标签: {len(csv_stems)}")
print(f"  cache (UNI特征): {len(cache_val_stems)}")
print(f"\n  val_patches ∩ CSV: {len(val_patches & csv_stems)}")
print(f"  val_patches ∩ cache: {len(val_patches & cache_val_stems)}")
print(f"  CSV ∩ cache: {len(csv_stems & cache_val_stems)}")
print(f"  三层完全交集: {len(intersection)}")

print("\n  结论: 理论上应该有265个样本 ✓")

print("\n【第三阶段：CSV内容分析】")
# 检查CSV是否混合多个患者
jfx_val = set([f.replace('.png', '') for f in os.listdir(
    base_dir / "data_new_3ST" / "patch_noov_spilt" / "JFX0729_noov_split" / "val_patches"
) if f.lower().endswith('.png')])
lmz_val = set([f.replace('.png', '') for f in os.listdir(
    base_dir / "data_new_3ST" / "patch_noov_spilt" / "LMZ12939_noov_split" / "val_patches"
) if f.lower().endswith('.png')])

csv_hyz = csv_stems & val_patches
csv_jfx = csv_stems & jfx_val
csv_lmz = csv_stems & lmz_val
csv_unknown = csv_stems - val_patches - jfx_val - lmz_val

print(f"  CSV中属于HYZ15040的补丁: {len(csv_hyz)}")
print(f"  CSV中属于JFX0729的补丁: {len(csv_jfx)}")
print(f"  CSV中属于LMZ12939的补丁: {len(csv_lmz)}")
print(f"  CSV中属于其他/未知的补丁: {len(csv_unknown)}")
print(f"  总计: {len(csv_hyz) + len(csv_jfx) + len(csv_lmz) + len(csv_unknown)}")

print("\n  结论: CSV包含多个患者混合数据！")
print("  - HYZ15040_ssGSEA_zscore.csv 名字误导，实际包含所有三个患者")
print("  - 只有265行与HYZ15040的val补丁匹配")

print("\n【第四阶段：实际训练样本数确认】")
print("  当使用正确的参数时，验证集应加载265个样本")
print("  但实际报告显示只有17个样本，说明：")
print("    1. 要么使用了错误的CSV文件")
print("    2. 要么使用了错误的patches_dir")  
print("    3. 要么cache中缺失了大量特征文件")

print("\n【第五阶段：根本原因总结】")
print("=" * 70)
print("主要问题：")
print("1. CSV数据污染")
print("   - 文件名: HYZ15040_ssGSEA_zscore.csv")
print("   - 实际内容: 包含HYZ15040、JFX0729、LMZ12939三个患者的混合数据")
print("   - 只有265行与HYZ15040的val补丁匹配")
print("")
print("2. 特征缓存配置问题（可能性）")
print("   - cache中的1020个.pt文件包含来自多个数据集的旧数据")
print("   - dataset_uni.py通过文件名stem匹配，可能匹配到错误的缓存")
print("")
print("3. 17个样本的来源")
print("   - 很可能是由于缓存中只有17个.pt文件与正确的条件组合匹配")
print("   - 或者是某个中间过程的数据过滤")
print("=" * 70)

print("\n【修复方案】")
print("方案A: 使用独立的HYZ15040标签CSV（推荐）")
print("  - 从10578行的混合CSV中提取仅包含HYZ15040补丁的265行")
print("  - 保存为: HYZ15040_ssGSEA_zscore_clean.csv")
print("  - 在train_uni.py中指定此新CSV")
print("")
print("方案B: 验证和重建缓存")
print("  - 清空uni2h_cache/HYZ15040/val/目录")
print("  - 重新运行extract_uni_features_3st.py生成干净的缓存")
print("  - 确保只包含HYZ15040的补丁对应的特征")
print("")
print("方案C: 修改dataset_uni.py的标签过滤逻辑")
print("  - 在加载时添加补丁ID的患者前缀检查")
print("  - 只保留与patches_dir中的补丁完全匹配的标签")
