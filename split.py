import numpy as np
import os
import re
import shutil
from sklearn.model_selection import train_test_split


def parse_coordinates_from_filename(filename):
    """
    从文件名（如 'patch_x4641_y16969.png' 或 'patch_x4641_y16969'）中解析出 x, y 坐标。
    """
    match = re.search(r'patch_x(\d+)_y(\d+)', filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    else:
        print(f"Warning: Could not parse coordinates from filename: {filename}")
        return None, None


def find_valid_indices_to_exclude(patch_filenames, val_size_fraction, distance_threshold, random_state=42):
    """
    根据坐标距离和阈值，找出应被排除在验证集之外的索引（即与验证集patch太相近的patch索引）。
    这部分逻辑与之前相同，但现在基于文件名列表。
    """
    coordinates = []
    valid_indices_map = {}  # 建立有效文件名索引到原始文件名列表索引的映射
    for i, name in enumerate(patch_filenames):
        # 确保处理 .png 扩展名
        name_without_ext = os.path.splitext(name)[0]
        x, y = parse_coordinates_from_filename(name_without_ext)
        if x is not None and y is not None:
            coordinates.append([x, y])
            valid_indices_map[len(coordinates) - 1] = i  # 有效索引 -> 原始列表索引
        else:
            pass  # 无法解析坐标的文件会被跳过

    coordinates = np.array(coordinates)
    n_valid_patches = len(coordinates)

    if n_valid_patches == 0:
        print("Warning: No patches with valid coordinates found. Cannot perform distance-based split.")
        return list(range(len(patch_filenames)))  # 所有原始索引都应被排除

    print(f"Found {n_valid_patches} patches with valid coordinates out of {len(patch_filenames)} total.")

    target_val_size = int(n_valid_patches * val_size_fraction)
    print(f"Attempting to select {target_val_size} patches for validation based on {distance_threshold}px threshold.")

    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(n_valid_patches)

    selected_for_val = []
    excluded_from_val = set()

    for current_idx_in_shuffled in shuffled_indices:
        current_coord = coordinates[current_idx_in_shuffled]

        too_close = False
        for selected_idx in selected_for_val:
            selected_coord = coordinates[selected_idx]
            distance = np.linalg.norm(current_coord - selected_coord)
            if distance < distance_threshold:
                too_close = True
                # 记录这个与已选验证集patch太近的 *有效* 索引
                excluded_from_val.add(current_idx_in_shuffled)
                break  # 找到一个就够了，无需继续检查

        # 如果当前patch不与任何已选patch太近
        if not too_close:
            # 检查验证集是否已达到目标大小
            if len(selected_for_val) < target_val_size:
                selected_for_val.append(current_idx_in_shuffled)
            else:
                # 验证集已满，停止选择，但继续将剩余patch标记为排除
                excluded_from_val.add(current_idx_in_shuffled)

    print(f"Selected {len(selected_for_val)} patches for validation.")
    print(f"Excluded {len(excluded_from_val)} patches from validation due to proximity.")

    # 将有效patch的索引转换回原始patch列表的索引
    original_indices_to_exclude = set()
    # 首先，所有无法解析坐标的patch都应该被排除在验证集之外
    for i, name in enumerate(patch_filenames):
        name_without_ext = os.path.splitext(name)[0]
        x, y = parse_coordinates_from_filename(name_without_ext)
        if x is None or y is None:
            original_indices_to_exclude.add(i)

    # 然后，将基于距离排除的有效patch索引转换回原始列表索引
    for valid_idx in excluded_from_val:
        original_idx = valid_indices_map[valid_idx]
        original_indices_to_exclude.add(original_idx)

    return list(original_indices_to_exclude)


def main():
    # --- 配置 ---
    patches_dir = r"D:\PycharmProjects\AIPath-data\patch\HYZ15040" # 数据集路径============================
    val_size_fraction = 0.1  # 验证集比例 1/10
    distance_threshold_px = 350  # 距离阈值
    random_state = 42

    # --- 直接读取文件夹中的 PNG 文件 ---
    all_files = os.listdir(patches_dir)
    patch_filenames = [f for f in all_files if f.lower().endswith('.png')]
    print(f"Total number of PNG patches found in directory: {len(patch_filenames)}")

    # --- 查找应排除在验证集之外的索引 ---
    indices_to_exclude_from_val = find_valid_indices_to_exclude(
        patch_filenames, val_size_fraction, distance_threshold_px, random_state
    )

    # --- 创建所有 patch 的原始索引列表 ---
    all_original_indices = list(range(len(patch_filenames)))

    # --- 确定验证集索引 ---
    val_indices_in_list = sorted(list(set(all_original_indices) - set(indices_to_exclude_from_val)))

    # --- 精确控制验证集大小 ---
    target_val_count = int(len(patch_filenames) * val_size_fraction)
    if len(val_indices_in_list) > target_val_count:
        np.random.seed(random_state)
        # 从候选验证集索引中随机选择目标数量
        selected_val_indices = np.random.choice(val_indices_in_list, size=target_val_count, replace=False)
        val_indices_in_list = sorted(selected_val_indices.tolist())

    # --- 确定训练集索引 ---
    train_indices_in_list = sorted(list(set(all_original_indices) - set(val_indices_in_list)))

    print("\n--- Dataset Split Summary ---")
    print(f"Total patches: {len(patch_filenames)}")
    print(
        f"Training patches: {len(train_indices_in_list)} ({len(train_indices_in_list) / len(patch_filenames) * 100:.2f}%)")
    print(
        f"Validation patches: {len(val_indices_in_list)} ({len(val_indices_in_list) / len(patch_filenames) * 100:.2f}%)")
    print(f"Distance threshold used: {distance_threshold_px}px")
    print(f"Target validation size: ~{target_val_count} patches")

    # --- 创建目标文件夹 ---
    train_folder_path = os.path.join(patches_dir, 'train_patches')
    val_folder_path = os.path.join(patches_dir, 'val_patches')

    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(val_folder_path, exist_ok=True)

    print(f"\nCreating folders: {train_folder_path}")
    print(f"Creating folders: {val_folder_path}")

    # --- 移动文件 ---
    moved_train_count = 0
    moved_val_count = 0

    for idx in train_indices_in_list:
        patch_name = patch_filenames[idx]
        src_path = os.path.join(patches_dir, patch_name)
        dst_path = os.path.join(train_folder_path, patch_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)  # 使用 move，如果想保留原文件用 copy
            moved_train_count += 1
        else:
            print(f"Warning: Source file does not exist (should not happen if list was accurate): {src_path}")

    for idx in val_indices_in_list:
        patch_name = patch_filenames[idx]
        src_path = os.path.join(patches_dir, patch_name)
        dst_path = os.path.join(val_folder_path, patch_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)  # 使用 move，如果想保留原文件用 copy
            moved_val_count += 1
        else:
            print(f"Warning: Source file does not exist (should not happen if list was accurate): {src_path}")

    print(f"\nMoved {moved_train_count} files to {train_folder_path}")
    print(f"Moved {moved_val_count} files to {val_folder_path}")

    # --- 验证集坐标分布检查 (可选) ---
    if val_indices_in_list:
        val_coords_x = []
        val_coords_y = []
        for idx in val_indices_in_list:
            name = patch_filenames[idx]  # 使用列表中的文件名
            name_without_ext = os.path.splitext(name)[0]
            x, y = parse_coordinates_from_filename(name_without_ext)
            if x is not None:
                val_coords_x.append(x)
                val_coords_y.append(y)

        if val_coords_x:  # 确保有有效坐标
            print(f"\n--- Validation Set Coordinate Check ---")
            print(f"Validation X range: [{min(val_coords_x)}, {max(val_coords_x)}]")
            print(f"Validation Y range: [{min(val_coords_y)}, {max(val_coords_y)}]")


if __name__ == "__main__":
    main()  # 确保调用 main 函数