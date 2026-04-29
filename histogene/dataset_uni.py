"""
HisToGene UNI2-h 特征数据集适配器（方案A）
从 UNI2-h 预提取特征缓存（.pt 文件）加载数据，替代 PNG 图像加载
坐标处理与标签匹配逻辑与原始 dataset.py 完全一致
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset


def parse_coordinates(filename):
    """从文件名 patch_x4641_y16969.png 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class HisToGeneUNIDataset(Dataset):
    def __init__(self, feature_cache_dir, patches_dir, labels_csv,
                 target_cols=None, n_pos=128, coord_stats=None):
        """
        Args:
            feature_cache_dir: UNI2-h 特征缓存目录（含 .pt 文件）
            patches_dir: PNG 图像目录（仅用于枚举文件名和解析坐标，不加载图像）
            labels_csv: Z-score 标准化后的标签 CSV
            target_cols: 目标列名列表（默认自动检测：除第一列 patch_id 外的所有列）
            n_pos: 位置编码的最大索引
            coord_stats: 坐标统计 dict {'x_min', 'x_max', 'y_min', 'y_max'}（推理时从训练集传入）
        """
        self.feature_cache_dir = feature_cache_dir
        self.patches_dir = patches_dir
        self.n_pos = n_pos

        # 加载标签
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
            # 自动检测：除第一列 patch_id 外的所有列作为目标列
            target_cols = list(df.columns[1:])
        self.target_cols = target_cols

        # 构建标签映射: patch_stem -> target_values
        self.label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '')
            self.label_map[stem] = row[target_cols].values.astype(np.float32)

        # 扫描缓存目录，建立已有特征的 stem 集合
        cached_stems = set()
        for fname in os.listdir(feature_cache_dir):
            if fname.lower().endswith('.pt'):
                cached_stems.add(fname[:-3])  # 去掉 .pt 后缀

        # 扫描图像文件并匹配标签 + 缓存
        self.samples = []  # (stem, x, y, targets)
        all_x, all_y = [], []

        for fname in sorted(os.listdir(patches_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname.replace('.png', '')
            # 仅保留缓存目录和CSV中都有对应项的patch
            if stem not in self.label_map:
                continue
            if stem not in cached_stems:
                continue
            x, y = parse_coordinates(fname)
            if x is None:
                continue
            targets = self.label_map[stem]
            self.samples.append((stem, x, y, targets))
            all_x.append(x)
            all_y.append(y)

        # 坐标统计（用于归一化到 [0, n_pos-1]）
        if coord_stats is not None:
            self.x_min, self.x_max = coord_stats['x_min'], coord_stats['x_max']
            self.y_min, self.y_max = coord_stats['y_min'], coord_stats['y_max']
        else:
            self.x_min = min(all_x) if all_x else 0
            self.x_max = max(all_x) if all_x else 1
            self.y_min = min(all_y) if all_y else 0
            self.y_max = max(all_y) if all_y else 1

        print(f"[HisToGeneUNIDataset] 加载 {len(self.samples)} 个样本 from {patches_dir}")
        print(f"  特征缓存: {feature_cache_dir}")
        print(f"  坐标范围: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]")
        print(f"  目标列: {target_cols}")

    def get_coord_stats(self):
        return {'x_min': self.x_min, 'x_max': self.x_max,
                'y_min': self.y_min, 'y_max': self.y_max}

    def _coord_to_index(self, val, vmin, vmax):
        """将坐标值归一化映射到 [0, n_pos-1]"""
        if vmax == vmin:
            return 0
        normalized = (val - vmin) / (vmax - vmin)
        idx = int(np.clip(normalized * (self.n_pos - 1), 0, self.n_pos - 1))
        return idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, x, y, targets = self.samples[idx]

        # 从 .pt 文件加载特征
        pt_path = os.path.join(self.feature_cache_dir, f"{stem}.pt")
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
            f"特征维度不匹配: 期望 1536, 实际 {feature.shape[0]}, stem={stem}"
        )

        # 坐标映射
        pos_x = self._coord_to_index(x, self.x_min, self.x_max)
        pos_y = self._coord_to_index(y, self.y_min, self.y_max)

        targets = torch.tensor(targets, dtype=torch.float32)

        return (feature,
                torch.tensor(pos_x, dtype=torch.long),
                torch.tensor(pos_y, dtype=torch.long),
                targets)

    @classmethod
    def from_multiple_patients(cls, patient_configs, n_pos=128, verbose=True):
        """
        多患者联合训练：合并多个患者的 Dataset

        Args:
            patient_configs: list of dicts, 每个包含:
                - patches_dir: str, 该患者的 patch 目录
                - labels_csv: str, 该患者的标签 CSV
                - patient_name: str, 患者名称（可选，用于 coord_stats 键名）
                - feature_cache_dir: str, 该患者的 UNI2-h 特征缓存目录
            n_pos: 位置编码最大索引
            verbose: 是否打印详细信息

        Returns:
            merged_dataset: ConcatDataset 合并后的数据集
            coord_stats_dict: dict, 每个患者的坐标统计 {patient_name: {x_min, x_max, y_min, y_max}}
            target_cols: list, 目标列名（所有患者应一致）
        """
        datasets = []
        coord_stats_dict = {}
        target_cols = None

        for i, config in enumerate(patient_configs):
            patches_dir = config['patches_dir']
            labels_csv = config['labels_csv']
            feature_cache_dir = config['feature_cache_dir']
            patient_name = config.get('patient_name', f'patient_{i}')

            if verbose:
                print(f"\n[MultiPatient-UNI] 加载患者 {patient_name}...")

            # 创建该患者的 Dataset（独立坐标归一化）
            dataset = cls(
                feature_cache_dir=feature_cache_dir,
                patches_dir=patches_dir,
                labels_csv=labels_csv,
                target_cols=target_cols,  # 第一个患者为 None 自动检测，后续保持一致
                n_pos=n_pos,
                coord_stats=None,  # 各自独立计算坐标统计
            )

            # 记录该患者的坐标统计
            coord_stats_dict[patient_name] = dataset.get_coord_stats()

            # 保持目标列一致（后续患者使用第一个患者的目标列）
            if target_cols is None:
                target_cols = dataset.target_cols

            datasets.append(dataset)

            if verbose:
                cs = coord_stats_dict[patient_name]
                print(f"  样本数: {len(dataset)}, 坐标范围: x=[{cs['x_min']}, {cs['x_max']}], y=[{cs['y_min']}, {cs['y_max']}]")

        # 合并所有患者的 Dataset
        merged_dataset = ConcatDataset(datasets)

        if verbose:
            total_samples = sum(len(d) for d in datasets)
            print(f"\n[MultiPatient-UNI] 合并完成: {len(datasets)} 个患者, 共 {total_samples} 个样本")

        return merged_dataset, coord_stats_dict, target_cols
