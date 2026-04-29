import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

print('\n====== 单患者训练 ======\n')

# HisToGene
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_pcc'].idxmax()
        best_row = df.loc[best_idx]
        print(f'HisToGene {ds}: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# HisToGene-UNI
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}_UNI.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}_UNI_fixed.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_pcc'].idxmax()
        best_row = df.loc[best_idx]
        print(f'HisToGene-UNI {ds}: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# EGN-v1
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv1', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_pcc'].idxmax()
        best_row = df.loc[best_idx]
        print(f'EGN-v1 {ds}: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# EGN-v2
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv2', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_pcc'].idxmax()
        best_row = df.loc[best_idx]
        print(f'EGN-v2 {ds}: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# EGN-v2+UNI
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv2', f'training_history_{ds}_UNI.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_pcc'].idxmax()
        best_row = df.loc[best_idx]
        print(f'EGN-v2+UNI {ds}: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

print('\n====== 跨患者泛化 ======\n')

# HisToGene 原版
csv_path = os.path.join(BASE_DIR, 'histogene', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_orig.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    best_idx = df['val_pcc'].idxmax()
    best_row = df.loc[best_idx]
    print(f'HisToGene CrossPatient (orig): epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# HisToGene-UNI
csv_path = os.path.join(BASE_DIR, 'histogene', 'training_history_CrossPatient_JFX_LMZ_to_HYZ.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    best_idx = df['val_pcc'].idxmax()
    best_row = df.loc[best_idx]
    print(f'HisToGene-UNI CrossPatient: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# EGN-v2
csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    best_idx = df['val_pcc'].idxmax()
    best_row = df.loc[best_idx]
    print(f'EGN-v2 CrossPatient: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')

# EGN-v2+UNI
csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    best_idx = df['val_pcc'].idxmax()
    best_row = df.loc[best_idx]
    print(f'EGN-v2+UNI CrossPatient: epoch={int(best_row["epoch"])}, val_pcc={best_row["val_pcc"]:.4f}')
