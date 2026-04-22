import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import pandas as pd

# --- 1. 配置路径 ---
# 本地 OpenMidnight 仓库路径 (请修改为你本地克隆的 openmidnight 父目录路径)
# 注意：是包含 'dinov2' 文件夹的那个目录
openmidnight_path = r"D:\AIPatho\hhy\ESCC\openmidnight\ai-bio-OpenMidnight-main"  # 修改为你的 openmidnight 目录路径
sys.path.insert(0, openmidnight_path)

# 本地权重文件路径 (请修改为你本地的路径)
checkpoint_path = r"teacher_checkpoint_load.pt"  # 修改为你的本地权重路径

# 数据集路径 (请修改为你的训练集和验证集文件夹路径)
train_patches_dir = r"D:\AIPatho\ESCC_aipatho\patch\HYZ15040\train_patches"  # 修改为你的训练集路径
val_patches_dir = r"D:\AIPatho\ESCC_aipatho\patch\HYZ15040\val_patches"  # 修改为你的验证集路径

# 标签 CSV 文件路径 (请修改为你本地的标签文件路径)
labels_csv_path = r"D:\AIPatho\ESCC_aipatho\HYZ15040_ssGSEA_scores.csv"  # 修改为你的标签 CSV 路径

# 特征和标签保存路径 (请修改为你想保存的路径)
output_train_features_path = r"D:\AIPatho\hhy\ESCC\openmidnight\train_features.npy"
output_train_labels_path = r"D:\AIPatho\hhy\ESCC\openmidnight\train_labels.npy"
output_val_features_path = r"D:\AIPatho\hhy\ESCC\openmidnight\val_features.npy"
output_val_labels_path = r"D:\AIPatho\hhy\ESCC\openmidnight\val_labels.npy"

# --- 2. 加载标签数据 ---
print(f"加载标签数据: {labels_csv_path}")
labels_df = pd.read_csv(labels_csv_path)
# 假设 CSV 第一列是 patch 名称 (如 'patch_x4641_y16969')
# 假设 CSV 第 M 列开始是 8 个归一化后的生物指标分数
# 请根据你实际的 CSV 文件结构调整 M 的值
M_index = 12  # M 列的索引 (Python 从 0 开始计数，所以 M=13 对应索引 12)
patch_names_all = labels_df.iloc[:, 0].values.astype(str)
targets_all = labels_df.iloc[:, M_index:M_index + 8].values.astype(np.float32)

# 创建一个字典，将 patch 名称映射到其在 CSV 中的索引和目标值
name_to_data = {name: {'idx': i, 'targets': targets_all[i]} for i, name in enumerate(patch_names_all)}
print(f"从 CSV 加载了 {len(name_to_data)} 个 patch 的标签。")

# --- 3. 加载模型结构 ---
print("加载 OpenMidnight 模型结构...")
try:
    from dinov2.models import vision_transformer as vits

    model = vits.vit_giant2(
        patch_size=14,
        num_register_tokens=4,
        img_size=224,
    )
    print(f"模型结构 (vit_giant2 with 4 reg tokens) 加载成功。")
except Exception as e:
    print(f"加载模型结构失败: {e}")
    exit(1)

# --- 4. 加载本地权重 ---
print(f"加载权重文件: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "pos_embed" in checkpoint:
        pos_embed = checkpoint["pos_embed"]
        model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
        print("已应用 'pos_embed' 到模型。")
    model.load_state_dict(checkpoint, strict=False)
    print("模型权重加载成功。")
except Exception as e:
    print(f"加载权重失败: {e}")
    exit(1)

# --- 5. 设置模型为评估模式并移动到 GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #是否有可用的 GPU
model = model.to(device) # 将模型移动到选定的设备 (GPU 或 CPU)
model.eval()
print(f"模型已设置为评估模式，使用设备: {device}。总参数量: {sum(p.numel() for p in model.parameters()):,}")

# --- 6. 定义图像预处理 ---
# 注意：此处使用 ImageNet 归一化，但 OpenMidnight (基于 DINOv2) 可能使用不同的参数
# **请务必查找并使用 OpenMidnight 或 DINOv2 的官方归一化参数！**
# 以下仅为示例，可能不准确
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 归一化 (可能不适用于 OpenMidnight)
        std=[0.229, 0.224, 0.225]
    )
])


# --- 7. 特征提取函数 ---
def extract_features_from_folder(folder_path, name_to_data_map, output_features_path, output_labels_path):
    """从指定文件夹提取特征和标签"""
    print(f"\n开始处理文件夹: {folder_path}")
    features_list = []
    labels_list = []

    all_files = os.listdir(folder_path)
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    print(f"文件夹中共发现 {len(png_files)} 个 PNG 文件。")

    for i, png_file in enumerate(png_files):
        if i % 100 == 0:  # 每处理100个文件打印一次进度
            print(f"  已处理 {i}/{len(png_files)} 个文件...")

        # 从文件名去掉扩展名，得到 CSV 中对应的名称
        csv_name = os.path.splitext(png_file)[0]

        # 检查是否在标签数据中存在
        if csv_name not in name_to_data_map:
            print(f"    警告: 文件 {png_file} 在标签 CSV 中未找到，跳过。")
            continue

        # 获取对应的标签
        target = name_to_data_map[csv_name]['targets']

        # 构造图像路径
        img_path = os.path.join(folder_path, png_file)

        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)  # 添加批次维度 -> [1, C, H, W]

            # 移动数据到模型所在的设备
            input_tensor = input_tensor.to(device)  # <--- 添加这行

            # 提取特征
            with torch.no_grad():
                raw_output = model(input_tensor)  # raw_output 会在 GPU 上

                # 获取 [CLS] token 特征
                if raw_output.dim() == 3:
                    cls_features = raw_output[:, 0, :]  # [1, embed_dim]
                elif raw_output.dim() == 2:
                    cls_features = raw_output  # [1, embed_dim]
                else:
                    print(f"    警告: 意外的输出形状 {raw_output.shape}，跳过 {png_file}。")
                    continue

                # 转换为 numpy (这会将数据从 GPU 复制回 CPU 内存)
                feature_np = cls_features.cpu().squeeze(0).numpy()  # [embed_dim,]

            # 添加到列表
            features_list.append(feature_np)
            labels_list.append(target)

        except Exception as e:
            print(f"    处理文件 {img_path} 时出错: {e}")
            continue  # 跳过有问题的文件

    print(f"  处理完成。成功提取特征的文件数: {len(features_list)}")

    if features_list:
        # 转换为 numpy array
        features_array = np.stack(features_list)  # [num_samples, embed_dim]
        labels_array = np.stack(labels_list)  # [num_samples, 8]

        print(f"  特征数组形状: {features_array.shape}")
        print(f"  标签数组形状: {labels_array.shape}")

        # 保存特征和标签
        os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

        print(f"  保存特征到: {output_features_path}")
        np.save(output_features_path, features_array)
        print(f"  保存标签到: {output_labels_path}")
        np.save(output_labels_path, labels_array)
        print("  特征和标签保存成功！")
    else:
        print(f"  警告: 没有成功提取到任何特征，未保存文件。")


# --- 8. 执行特征提取 ---
# 提取训练集特征和标签
extract_features_from_folder(
    train_patches_dir,
    name_to_data,
    output_train_features_path,
    output_train_labels_path
)

# 提取验证集特征和标签
extract_features_from_folder(
    val_patches_dir,
    name_to_data,
    output_val_features_path,
    output_val_labels_path
)

print("\n--- 所有特征提取完成 ---")
