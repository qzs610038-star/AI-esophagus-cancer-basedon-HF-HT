import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

print('\n====== 最后一行（最终epoch）数据 ======\n')

# HisToGene
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        print(f'HisToGene {ds}: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# HisToGene-UNI
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}_UNI.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'histogene', f'training_history_{ds}_UNI_fixed.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        print(f'HisToGene-UNI {ds}: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# EGN-v1
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv1', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        print(f'EGN-v1 {ds}: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# EGN-v2
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv2', f'training_history_{ds}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        print(f'EGN-v2 {ds}: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# EGN-v2+UNI
for ds in ['HYZ15040', 'JFX0729', 'LMZ12939']:
    csv_path = os.path.join(BASE_DIR, 'egnv2', f'training_history_{ds}_UNI.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        print(f'EGN-v2+UNI {ds}: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

print('\n====== 跨患者最后一行 ======\n')

# HisToGene CrossPatient orig
csv_path = os.path.join(BASE_DIR, 'histogene', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_orig.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    print(f'HisToGene CrossPatient (orig): last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# HisToGene-UNI CrossPatient
csv_path = os.path.join(BASE_DIR, 'histogene', 'training_history_CrossPatient_JFX_LMZ_to_HYZ.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    print(f'HisToGene-UNI CrossPatient: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# EGN-v2 CrossPatient
csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    print(f'EGN-v2 CrossPatient: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')

# EGN-v2+UNI CrossPatient
csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    print(f'EGN-v2+UNI CrossPatient: last_epoch={int(last_row["epoch"])}, last_val_pcc={last_row["val_pcc"]:.4f}')
