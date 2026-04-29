import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ.csv')
df = pd.read_csv(csv_path)

print('所有行的epoch和val_pcc：')
for idx, row in df.iterrows():
    print(f'Epoch {int(row["epoch"])}: val_pcc={row["val_pcc"]:.4f}')

best_idx = df['val_pcc'].idxmax()
best_row = df.loc[best_idx]
print(f'\n最佳: Epoch {int(best_row["epoch"])}: val_pcc={best_row["val_pcc"]:.4f}')
