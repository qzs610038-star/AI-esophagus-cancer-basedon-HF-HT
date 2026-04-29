"""
EGNv2 UNI2-h 特征数据集适配器
从 UNI2-h 预提取特征缓存（.pt 文件）加载数据，替代 ResNet 实时特征提取
保留原始像素坐标（不做归一化），供 EGN-v2 空间图构建使用
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset


def parse_coordinates(filename):
    """从文件名 patch_x4641_y16969.pt 或 patch_x4641_y16969.png 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class EGNv2UNIDataset(Dataset):
    def __init__(self, feature_cache_dir, patches_dir, labels_csv,
                 target_cols=None, split='train'):
        """
        Args:
            feature_cache_dir: uni2h_cache/{patient}/{split}/ 目录
            patches_dir: 原始patches目录（仅用于交集匹配，可选）
            labels_csv: ssGSEA zscore CSV路径
            split: 'train' or 'val'
        """
        self.feature_cache_dir = feature_cache_dir
        self.patches_dir = patches_dir
        self.split = split

        # 加载标签
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
            target_cols = list(df.columns[1:])
        self.target_cols = target_cols

        # 构建标签映射: patch_stem -> target_values
        # CSV 中的 id 可能是 patch_x4641_y16969.png，需要兼容处理
        self.label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '').replace('.pt', '')
            self.label_map[stem] = row[target_cols].values.astype(np.float32)

        # 扫描缓存目录，建立已有特征的 stem 集合
        cached_stems = set()
        if os.path.isdir(feature_cache_dir):
            for fname in os.listdir(feature_cache_dir):
                if fname.lower().endswith('.pt'):
                    cached_stems.add(fname[:-3])  # 去掉 .pt 后缀

        # 三层交集过滤
        self.samples = []  # (pt_path, raw_x, raw_y, targets)
        all_x, all_y = [], []

        if patches_dir and os.path.isdir(patches_dir):
            # 有 patches_dir，做三层交集：缓存 + patches + 标签
            for fname in sorted(os.listdir(patches_dir)):
                if not fname.lower().endswith('.png'):
                    continue
                stem = fname.replace('.png', '')
                if stem not in self.label_map:
                    continue
                if stem not in cached_stems:
                    continue
                x, y = parse_coordinates(fname)
                if x is None:
                    continue
                pt_path = os.path.join(feature_cache_dir, f"{stem}.pt")
                targets = self.label_map[stem]
                self.samples.append((pt_path, x, y, targets))
                all_x.append(x)
                all_y.append(y)
        else:
            # 无 patches_dir，做两层交集：缓存 + 标签
            for stem in sorted(cached_stems):
                if stem not in self.label_map:
                    continue
                x, y = parse_coordinates(stem)
                if x is None:
                    continue
                pt_path = os.path.join(feature_cache_dir, f"{stem}.pt")
                targets = self.label_map[stem]
                self.samples.append((pt_path, x, y, targets))
                all_x.append(x)
                all_y.append(y)

        print(f"[EGNv2UNIDataset] 加载 {len(self.samples)} 个样本 "
              f"from {feature_cache_dir} (split={split})")
        if all_x:
            print(f"  坐标范围: x=[{min(all_x)}, {max(all_x)}], y=[{min(all_y)}, {max(all_y)}]")
        print(f"  目标列: {target_cols}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, x, y, targets = self.samples[idx]

        # 从 .pt 文件加载特征
        feature = torch.load(pt_path, map_location='cpu', weights_only=True)
        # 处理可能的 dict 格式
        if isinstance(feature, dict) and "feature" in feature:
            feature = feature["feature"]
        # 确保 float32
        feature = feature.float()
        # 如果特征不是 1D，则展平
        if feature.dim() > 1:
            feature = feature.flatten()
        # 确保形状为 (1536,)
        assert feature.shape[0] == 1536, (
            f"特征维度不匹配: 期望 1536, 实际 {feature.shape[0]}, path={pt_path}"
        )

        targets = torch.tensor(targets, dtype=torch.float32)

        # 保留原始像素坐标（与 EGNv2Dataset 保持一致，返回 float32 tensor）
        return (feature,
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                targets)

    @classmethod
    def from_multiple_patients(cls, patient_configs, verbose=True):
        """
        合并多个患者数据集，用于跨患者训练

        Args:
            patient_configs: list of dicts with keys:
                feature_cache_dir, patches_dir, labels_csv, patient_name, split
        Returns:
            merged_dataset: ConcatDataset
            target_cols: list, 目标列名
        """
        datasets = []
        target_cols = None

        for i, config in enumerate(patient_configs):
            feature_cache_dir = config['feature_cache_dir']
            patches_dir = config.get('patches_dir', None)
            labels_csv = config['labels_csv']
            patient_name = config.get('patient_name', f'patient_{i}')
            split = config.get('split', 'train')

            if verbose:
                print(f"\n[MultiPatient-EGNv2-UNI] 加载患者 {patient_name} ({split})...")

            dataset = cls(
                feature_cache_dir=feature_cache_dir,
                patches_dir=patches_dir,
                labels_csv=labels_csv,
                target_cols=target_cols,
                split=split,
            )

            if target_cols is None:
                target_cols = dataset.target_cols

            datasets.append(dataset)

            if verbose:
                print(f"  样本数: {len(dataset)}")

        merged_dataset = ConcatDataset(datasets)

        if verbose:
            total_samples = sum(len(d) for d in datasets)
            print(f"\n[MultiPatient-EGNv2-UNI] 合并完成: {len(datasets)} 个患者, "
                  f"共 {total_samples} 个样本")

        return merged_dataset, target_cols
