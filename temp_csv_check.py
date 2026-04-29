import pandas as pd
import os

BASE_DIR = r'd:\AI空间转录病理研究\PFMval_new'

csv_path = os.path.join(BASE_DIR, 'egnv2', 'training_history_CrossPatient_JFX_LMZ_to_HYZ_UNI.csv')
df = pd.read_csv(csv_path)

print('最后5行：')
print(df.tail(5).to_string())

print('\n最佳epoch行：')
best_idx = df['val_pcc'].idxmax()
best_row = df.loc[best_idx]
print(f'Epoch {int(best_row["epoch"])}: val_pcc={best_row["val_pcc"]:.4f}')

print(f'\n最后一行epoch: {int(df.iloc[-1]["epoch"])}, val_pcc={df.iloc[-1]["val_pcc"]:.4f}')
