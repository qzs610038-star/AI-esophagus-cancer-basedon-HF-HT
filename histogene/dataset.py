"""
HisToGene 数据集适配器
从 PNG 图像加载数据，解析坐标，匹配标签
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms


def parse_coordinates(filename):
    """从文件名 patch_x4641_y16969.png 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class HisToGeneDataset(Dataset):
    def __init__(self, patches_dir, labels_csv, target_cols=None,
                 n_pos=128, transform=None, coord_stats=None):
        """
        Args:
            patches_dir: PNG 图像目录
            labels_csv: Z-score 标准化后的标签 CSV
            target_cols: 目标列名列表（默认自动检测：除第一列 patch_id 外的所有列）
            n_pos: 位置编码的最大索引
            transform: 图像变换
            coord_stats: 坐标统计 dict {'x_min', 'x_max', 'y_min', 'y_max'}（推理时从训练集传入）
        """
        self.patches_dir = patches_dir
        self.n_pos = n_pos
        self.transform = transform

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

        # 扫描图像文件并匹配标签
        self.samples = []  # (filepath, x, y, targets)
        all_x, all_y = [], []

        for fname in sorted(os.listdir(patches_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname.replace('.png', '')
            if stem not in self.label_map:
                continue
            x, y = parse_coordinates(fname)
            if x is None:
                continue
            filepath = os.path.join(patches_dir, fname)
            targets = self.label_map[stem]
            self.samples.append((filepath, x, y, targets))
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

        print(f"[HisToGeneDataset] 加载 {len(self.samples)} 个样本 from {patches_dir}")
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
        filepath, x, y, targets = self.samples[idx]

        # 加载图像
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 坐标映射
        pos_x = self._coord_to_index(x, self.x_min, self.x_max)
        pos_y = self._coord_to_index(y, self.y_min, self.y_max)

        targets = torch.tensor(targets, dtype=torch.float32)

        return image, torch.tensor(pos_x, dtype=torch.long), \
               torch.tensor(pos_y, dtype=torch.long), targets

    @classmethod
    def from_multiple_patients(cls, patient_configs, n_pos=128, transform=None, verbose=True):
        """
        多患者联合训练：合并多个患者的 Dataset

        Args:
            patient_configs: list of dicts, 每个包含:
                - patches_dir: str, 该患者的 patch 目录
                - labels_csv: str, 该患者的标签 CSV
                - patient_name: str, 患者名称（可选，用于 coord_stats 键名）
            n_pos: 位置编码最大索引
            transform: 图像变换
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
            patient_name = config.get('patient_name', f'patient_{i}')

            if verbose:
                print(f"\n[MultiPatient] 加载患者 {patient_name}...")

            # 创建该患者的 Dataset（独立坐标归一化）
            dataset = cls(
                patches_dir=patches_dir,
                labels_csv=labels_csv,
                target_cols=target_cols,  # 第一个患者为 None 自动检测，后续保持一致
                n_pos=n_pos,
                transform=transform,
                coord_stats=None,  # 各自独立计算坐标统计
            )

            # 记录该患者的坐标统计
            coord_stats_dict[patient_name] = dataset.get_coord_stats()

            # 保持目标列一致（后续患者使用第一个患者的目标列）
            if target_cols is None:
                target_cols = dataset.target_cols

            datasets.append(dataset)

            if verbose:
                print(f"  样本数: {len(dataset)}, 坐标范围: x=[{coord_stats_dict[patient_name]['x_min']}, {coord_stats_dict[patient_name]['x_max']}], y=[{coord_stats_dict[patient_name]['y_min']}, {coord_stats_dict[patient_name]['y_max']}]")

        # 合并所有患者的 Dataset
        merged_dataset = ConcatDataset(datasets)

        if verbose:
            total_samples = sum(len(d) for d in datasets)
            print(f"\n[MultiPatient] 合并完成: {len(datasets)} 个患者, 共 {total_samples} 个样本")

        return merged_dataset, coord_stats_dict, target_cols
