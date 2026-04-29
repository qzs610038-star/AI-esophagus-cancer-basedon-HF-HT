import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv')
df = pd.read_csv(csv_path)

# 找到epoch 16的行
epoch_16_row = df[df['epoch'] == 16]
if len(epoch_16_row) > 0:
    row = epoch_16_row.iloc[0]
    print(f'Epoch 16: val_pcc={row["val_pcc"]:.4f}')
else:
    print('找不到epoch 16')

# 显示所有行的epoch和val_pcc
print('\n所有行的epoch和val_pcc：')
for idx, row in df.iterrows():
    print(f'Epoch {int(row["epoch"])}: val_pcc={row["val_pcc"]:.4f}')
