"""
dataset_online.py — 在线图像加载 Dataset
==============================

加载原始 PNG 病理图像（而非预提取特征缓存），用于在线 backbone 前向传播训练。
坐标解析、标签匹配、多患者合并逻辑与 dataset_uni_tokens.py 完全一致。

用法:
    from uni2h.uni2h_utils import load_uni2h_backbone
    backbone, transform, _ = load_uni2h_backbone(device='cpu')
    ds = OnlinePatchDataset(patches_dir, labels_csv, transform=transform)
    image, pos_x, pos_y, targets = ds[0]  # image: [3, 224, 224]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


# ═══════════════════════════════════════════════════════════════
# 坐标解析（与 dataset_uni_tokens.py 完全一致）
# ═══════════════════════════════════════════════════════════════

def parse_coordinates(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """从文件名 patch_x4641_y16969.png 解析坐标。"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# ═══════════════════════════════════════════════════════════════
# OnlinePatchDataset
# ═══════════════════════════════════════════════════════════════

class OnlinePatchDataset(Dataset):
    """在线图像 Dataset：加载原始 PNG 图像 + 标签。

    不依赖预提取特征缓存。图像在 __getitem__ 时通过 transform 实时处理。
    """

    def __init__(
        self,
        patches_dir: str,
        labels_csv: str,
        transform: Callable,
        target_cols: Optional[List[str]] = None,
        n_pos: int = 128,
        n_targets: int = 30,
        coord_stats: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            patches_dir: PNG 图像目录
            labels_csv: Z-score 标准化后的标签 CSV
            transform: torchvision transform（UNI2-H 官方预处理）
            target_cols: 目标列名（默认自动检测：除第一列 patch_id 外的所有列）
            n_pos: 坐标编码的最大索引
            n_targets: 通路数
            coord_stats: 坐标统计（推理时从训练集传入），None 则从数据自动计算
        """
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.n_pos = n_pos

        # ── 加载标签 ──
        df = pd.read_csv(labels_csv)
        id_col = df.columns[0]
        if target_cols is None:
            target_cols = list(df.columns[1:])
        self.target_cols = target_cols

        # patch_stem → target_values
        self.label_map: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            stem = str(row[id_col])
            if stem.lower().endswith('.png'):
                stem = stem[:-4]
            self.label_map[stem] = row[target_cols].values.astype(np.float32)

        # ── 扫描 patches 目录，二层交集过滤：.png ∩ CSV 标签 ──
        self.samples: List[Tuple[str, int, int, np.ndarray]] = []
        all_x: List[int] = []
        all_y: List[int] = []

        for fname in sorted(os.listdir(patches_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname[:-4]  # 去掉 .png
            if stem not in self.label_map:
                continue
            x, y = parse_coordinates(fname)
            if x is None:
                continue
            targets = self.label_map[stem]
            self.samples.append((stem, x, y, targets))
            all_x.append(x)
            all_y.append(y)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"OnlinePatchDataset: 无有效样本。patches_dir={patches_dir}, "
                f"labels_csv={labels_csv}"
            )

        # ── 坐标统计（与 HisToGeneUNITokensDataset 一致）──
        if coord_stats is not None:
            self.x_min, self.x_max = coord_stats['x_min'], coord_stats['x_max']
            self.y_min, self.y_max = coord_stats['y_min'], coord_stats['y_max']
        else:
            self.x_min = min(all_x) if all_x else 0
            self.x_max = max(all_x) if all_x else 1
            self.y_min = min(all_y) if all_y else 0
            self.y_max = max(all_y) if all_y else 1

        print(f"[OnlinePatchDataset] 加载 {len(self.samples)} 个样本 from {patches_dir}")
        print(f"  坐标范围: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]")
        print(f"  目标列数: {len(target_cols)}")

    def get_coord_stats(self) -> Dict[str, int]:
        return {'x_min': self.x_min, 'x_max': self.x_max,
                'y_min': self.y_min, 'y_max': self.y_max}

    def _coord_to_index(self, val: int, vmin: int, vmax: int) -> int:
        """将原始像素坐标映射到 [0, n_pos-1] 范围内。"""
        if vmax == vmin:
            return 0
        normalized = (val - vmin) / (vmax - vmin)
        return int(np.clip(normalized * (self.n_pos - 1), 0, self.n_pos - 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem, x, y, targets = self.samples[idx]

        # ── 加载并预处理图像 ──
        img_path = self.patches_dir / f"{stem}.png"
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)  # [3, 224, 224]

        pos_x = self._coord_to_index(x, self.x_min, self.x_max)
        pos_y = self._coord_to_index(y, self.y_min, self.y_max)

        return (
            image_tensor,
            torch.tensor(pos_x, dtype=torch.long),
            torch.tensor(pos_y, dtype=torch.long),
            torch.tensor(targets, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════════════
# 多患者合并（与 dataset_uni_tokens.py 接口一致）
# ═══════════════════════════════════════════════════════════════

def from_multiple_patients(
    patient_configs: List[Dict],
    transform: Callable,
    n_pos: int = 128,
    n_targets: int = 30,
    verbose: bool = True,
) -> Tuple[ConcatDataset, Dict[str, Dict[str, int]], List[str]]:
    """多患者联合训练：合并多个患者的 OnlinePatchDataset。

    Args:
        patient_configs: list of dicts:
            - patches_dir: str
            - labels_csv: str
            - patient_name: str (可选)
        transform: UNI2-H 官方预处理 transform
        n_pos: 坐标编码最大索引
        n_targets: 通路数
        verbose: 是否打印详细信息

    Returns:
        merged_dataset: ConcatDataset
        coord_stats_dict: {patient_name: {x_min, x_max, y_min, y_max}}
        target_cols: 目标列名列表
    """
    datasets: List[OnlinePatchDataset] = []
    coord_stats_dict: Dict[str, Dict[str, int]] = {}
    target_cols: Optional[List[str]] = None

    for i, config in enumerate(patient_configs):
        patches_dir = config['patches_dir']
        labels_csv = config['labels_csv']
        patient_name = config.get('patient_name', f'patient_{i}')

        if verbose:
            print(f"\n[MultiPatient-Online] 加载患者 {patient_name} ...")

        dataset = OnlinePatchDataset(
            patches_dir=patches_dir,
            labels_csv=labels_csv,
            transform=transform,
            target_cols=target_cols,
            n_pos=n_pos,
            n_targets=n_targets,
            coord_stats=None,
        )

        coord_stats_dict[patient_name] = dataset.get_coord_stats()

        if target_cols is None:
            target_cols = dataset.target_cols

        datasets.append(dataset)

        if verbose:
            cs = coord_stats_dict[patient_name]
            print(f"  样本数: {len(dataset)}, "
                  f"坐标范围: x=[{cs['x_min']}, {cs['x_max']}], y=[{cs['y_min']}, {cs['y_max']}]")

    merged = ConcatDataset(datasets)

    if verbose:
        total = sum(len(d) for d in datasets)
        print(f"\n[MultiPatient-Online] 合并完成: {len(datasets)} 个患者, 共 {total} 个样本")

    return merged, coord_stats_dict, target_cols
