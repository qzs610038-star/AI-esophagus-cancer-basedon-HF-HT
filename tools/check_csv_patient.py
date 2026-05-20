import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import get_project_root

base_dir = Path(get_project_root())

csv_file = base_dir / "data_new_3ST" / "ssGSEA_zscore" / "HYZ15040_ssGSEA_zscore.csv"
print(f"Reading: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# 查看patch_id列，提取患者代码
patch_ids = df.iloc[:, 0]
print(f"\nFirst 20 patch IDs:")
for i, pid in enumerate(patch_ids.head(20)):
    print(f"  {i}: {pid}")

# 统计每个患者的patch数
import re
patient_counts = {}
for pid in patch_ids:
    match = re.search(r'(HYZ\d+|JFX\d+|LMZ\d+|patch_x\d+_y\d+)', str(pid))
    if match:
        extracted = match.group(1)
        # 只统计患者代码
        if any(c in extracted for c in ['HYZ', 'JFX', 'LMZ']):
            patient = extracted
        else:
            # 可能没有患者前缀
            patient = "unknown"
    else:
        patient = "unknown"
    patient_counts[patient] = patient_counts.get(patient, 0) + 1

print(f"\nPatient distribution in CSV:")
for patient, count in sorted(patient_counts.items(), key=lambda x: -x[1]):
    print(f"  {patient}: {count}")
