import numpy as np
import pandas as pd
import os

# --- 1. 配置路径 ---
# 替换为你的实际路径
patches_dir = r"D:\AIPatho\ESCC_aipatho\patch\HYZ15040\val_patches" # 你想检查的文件夹 (val_patches 或 train_patches)
labels_csv_path = r"D:\AIPatho\ESCC_aipatho\HYZ15040_ssGSEA_scores.csv"
labels_npy_path = r"D:\AIPatho\hhy\ESCC\openmidnight\data\val_labels.npy"   # 对应的 .npy 标签文件
# 如果检查训练集，则用:
# patches_dir = r"D:\AIPatho\ESCC_aipatho\patch\HYZ15040\train_patches"
# labels_npy_path = r"D:\AIPatho\hhy\ESCC\openmidnight\train_labels.npy"

# CSV 中标签列的起始索引 (例如 M=13 对应索引 12)
M_index = 12

print(f"Checking alignment for: {patches_dir}")
print(f"Labels CSV: {labels_csv_path}")
print(f"Loaded Labels .npy: {labels_npy_path}")

# --- 2. 加载 CSV 数据 ---
print("\n--- Loading CSV data ---")
labels_df = pd.read_csv(labels_csv_path)
# 假设第一列是 patch 名称
csv_patch_names = labels_df.iloc[:, 0].astype(str).tolist() # 确保名称是字符串
targets_from_csv_all = labels_df.iloc[:, M_index:M_index+8].values.astype(np.float32)

print(f"Total rows in CSV: {len(csv_patch_names)}")
print(f"Targets shape from CSV: {targets_from_csv_all.shape}")

# --- 3. 加载 .npy 标签数据 ---
print("\n--- Loading .npy labels ---")
loaded_labels = np.load(labels_npy_path)
print(f"Loaded labels shape: {loaded_labels.shape}")

# --- 4. 获取图像文件夹中的文件名 (不含扩展名) ---
print("\n--- Loading patch names from folder ---")
all_files_in_folder = os.listdir(patches_dir)
png_files = sorted([f for f in all_files_in_folder if f.lower().endswith('.png')]) # 排序以确保顺序
patch_names_from_folder = [os.path.splitext(f)[0] for f in png_files]
print(f"Total PNG files in folder: {len(patch_names_from_folder)}")

# --- 5. 复现特征提取时的对齐逻辑 ---
print("\n--- Reproducing alignment logic (Folder -> CSV lookup) ---")
retrieved_labels_from_csv = []
missing_in_csv_count = 0
found_count = 0

for patch_name in patch_names_from_folder:
    # 在 CSV 名称列表中查找当前 patch 名称
    try:
        csv_idx = csv_patch_names.index(patch_name)
        # 找到后，取出对应的标签
        corresponding_label = targets_from_csv_all[csv_idx]
        retrieved_labels_from_csv.append(corresponding_label)
        found_count += 1
        # Optional: Print first few matches for verification
        if found_count <= 5:
             print(f"  Found '{patch_name}' at CSV index {csv_idx}, label (first 3): {corresponding_label[:3]}")
    except ValueError:
        # 如果在 CSV 中找不到该名称
        print(f"  [WARNING] Patch name '{patch_name}' not found in CSV!")
        missing_in_csv_count += 1

print(f"\nAlignment check summary:")
print(f"  - Patches found in folder and matched in CSV: {found_count}")
print(f"  - Patches in folder but NOT found in CSV: {missing_in_csv_count}")
print(f"  - Expected length of retrieved labels: {found_count}")

if missing_in_csv_count > 0:
    print("\n[CRITICAL ERROR] Some patch files were not found in the CSV. The .npy files might be invalid or correspond to a different dataset.")
    exit(1) # 退出检查

if found_count != len(loaded_labels):
    print(f"\n[MISMATCH ERROR] Number of matched labels ({found_count}) does not match length of loaded .npy labels ({len(loaded_labels)}).")
    exit(1) # 退出检查

# --- 6. 比较 "复现的" 标签与加载的 .npy 标签 ---
print("\n--- Comparing retrieved CSV labels vs loaded .npy labels ---")
retrieved_labels_array = np.array(retrieved_labels_from_csv)

# 使用 np.allclose 比较浮点数数组
are_equal = np.allclose(retrieved_labels_array, loaded_labels, rtol=1e-05, atol=1e-08)

if are_equal:
    print(f"[SUCCESS] The labels in {labels_npy_path} match the labels looked up from the CSV based on folder filenames.")
    print(f"The order and values are consistent. Alignment is CORRECT.")
else:
    print(f"[FAILURE] The labels in {labels_npy_path} DO NOT match the labels looked up from the CSV.")
    print(f"There is a misalignment in order or values.")
    # Optional: Print first few differences to investigate
    diff_mask = ~np.isclose(retrieved_labels_array, loaded_labels, rtol=1e-05, atol=1e-08)
    if np.any(diff_mask):
        first_diff_indices = np.where(diff_mask)[0][:5] # Show first 5 differing indices
        print(f"First few differing indices and values (CSV vs .npy):")
        for idx in first_diff_indices:
            print(f"  Index {idx}: CSV={retrieved_labels_array[idx]}, .npy={loaded_labels[idx]}")


print("\n--- Verification Complete ---")
