"""
EGN-v1 数据集适配器
从 PNG 图像加载数据，解析坐标，匹配标签
与 EGNv2 数据集类似，保留原始像素坐标（不做 n_pos 映射）
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


class EGNv1Dataset(Dataset):
    """
    EGN-v1 数据集

    与 EGNv2Dataset 类似，但：
    - 不做 n_pos 坐标映射，保留原始像素坐标
    - __getitem__ 返回 (image, raw_x, raw_y, targets)
    """

    def __init__(self, patches_dir, labels_csv, target_cols=None, transform=None):
        """
        Args:
            patches_dir: PNG 图像目录
            labels_csv: Z-score 标准化后的标签 CSV
            target_cols: 目标列名列表（默认自动检测）
            transform: 图像变换
        """
        self.patches_dir = patches_dir
        self.transform = transform

        # 加载标签
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
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

        self._all_x = np.array(all_x) if all_x else np.array([])
        self._all_y = np.array(all_y) if all_y else np.array([])

        print(f"[EGNv1Dataset] 加载 {len(self.samples)} 个样本 from {patches_dir}")
        if all_x:
            print(f"  坐标范围: x=[{min(all_x)}, {max(all_x)}], y=[{min(all_y)}, {max(all_y)}]")
        print(f"  目标列: {target_cols}")

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

        targets = torch.tensor(targets, dtype=torch.float32)

        return image, torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32), targets

    def get_all_coords(self):
        """返回所有样本坐标 (N, 2)"""
        if len(self._all_x) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([self._all_x, self._all_y], axis=1)

    def get_all_targets(self):
        """返回所有标签 (N, 30)"""
        if len(self.samples) == 0:
            return np.zeros((0, len(self.target_cols)), dtype=np.float32)
        return np.array([s[3] for s in self.samples], dtype=np.float32)

    @classmethod
    def from_multiple_patients(cls, patient_configs, transform=None, verbose=True):
        """
        多患者支持 - 合并多个患者的 Dataset

        Args:
            patient_configs: list of dicts, 每个包含:
                - patches_dir: str
                - labels_csv: str
                - patient_name: str (可选)
            transform: 图像变换
            verbose: 是否打印详细信息

        Returns:
            merged_dataset: ConcatDataset
            target_cols: list, 目标列名
        """
        datasets = []
        target_cols = None

        for i, config in enumerate(patient_configs):
            patches_dir = config['patches_dir']
            labels_csv = config['labels_csv']
            patient_name = config.get('patient_name', f'patient_{i}')

            if verbose:
                print(f"\n[MultiPatient] 加载患者 {patient_name}...")

            dataset = cls(
                patches_dir=patches_dir,
                labels_csv=labels_csv,
                target_cols=target_cols,  # 第一个患者为 None 自动检测
                transform=transform,
            )

            if target_cols is None:
                target_cols = dataset.target_cols

            datasets.append(dataset)

            if verbose:
                print(f"  样本数: {len(dataset)}")

        merged_dataset = ConcatDataset(datasets)

        if verbose:
            total_samples = sum(len(d) for d in datasets)
            print(f"\n[MultiPatient] 合并完成: {len(datasets)} 个患者, 共 {total_samples} 个样本")

        return merged_dataset, target_cols
