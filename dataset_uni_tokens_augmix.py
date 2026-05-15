"""
HisToGene UNI2-h Token序列数据集适配器 — AugMix版本
===================================================
在原始 HisToGeneUNITokensDataset 基础上，新增：
  1. 增强特征加载：支持从 uni2h_cache_tokens_aug/ 目录加载增强版token
  2. 随机增强采样：训练时以 aug_sample_prob 概率选择增强变体
  3. 完全向后兼容：当不使用增强特征时，行为与原数据集完全一致

约束：
  - 不修改 dataset_uni_tokens.py
  - 验证集（is_train=False）永远使用原始特征
"""

import os
import random
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


class HisToGeneUNITokensAugMixDataset(Dataset):
    """支持 H&E 增强特征随机采样的 UNI Token 数据集

    训练时以 aug_sample_prob 概率从增强特征目录中随机选取一个变体，
    否则使用原始特征。验证/测试时始终使用原始特征。
    """

    def __init__(self, patches_dir, feature_cache_dir, labels_csv,
                 augmented_cache_dir=None, n_augments=3,
                 aug_sample_prob=0.5, is_train=True,
                 target_cols=None, n_pos=128, n_targets=30,
                 coord_stats=None):
        """
        Args:
            patches_dir: PNG 图像目录（用于坐标解析和交集过滤，不加载图像）
            feature_cache_dir: uni2h_cache_tokens/{patient}/{split}/ 下的 .pt 文件目录
            labels_csv: Z-score 标准化后的标签 CSV
            augmented_cache_dir: 增强特征目录 (uni2h_cache_tokens_aug/{patient}/{split}/)，
                                 若为 None 或目录不存在则不使用增强
            n_augments: 增强变体数量（文件名后缀 _aug1 ~ _augN）
            aug_sample_prob: 训练时选择增强版本的概率 (0~1)
            is_train: 训练模式启用增强采样，验证模式永远使用原始特征
            target_cols: 目标列名列表（默认自动检测）
            n_pos: 位置编码的最大索引
            n_targets: 目标数量（仅在无法自动检测时使用）
            coord_stats: 坐标统计 dict {'x_min', 'x_max', 'y_min', 'y_max'}
        """
        self.feature_cache_dir = feature_cache_dir
        self.patches_dir = patches_dir
        self.n_pos = n_pos
        self.is_train = is_train
        self.n_augments = n_augments
        self.aug_sample_prob = aug_sample_prob

        # 增强目录：仅在训练模式且目录存在时启用
        if augmented_cache_dir and is_train and os.path.isdir(augmented_cache_dir):
            self.augmented_cache_dir = augmented_cache_dir
        else:
            self.augmented_cache_dir = None

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

        # 扫描缓存目录，建立已有特征的 stem 集合
        cached_stems = set()
        for fname in os.listdir(feature_cache_dir):
            if fname.lower().endswith('.pt'):
                cached_stems.add(fname[:-3])  # 去掉 .pt 后缀

        # 三层交集过滤：缓存 .pt ∩ patches .png ∩ CSV标签
        self.samples = []  # (stem, x, y, targets)
        all_x, all_y = [], []

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

        # 统计增强文件数
        aug_count = 0
        if self.augmented_cache_dir:
            aug_count = sum(1 for f in os.listdir(self.augmented_cache_dir)
                           if f.lower().endswith('.pt'))

        mode_str = "训练" if is_train else "验证/测试"
        print(f"[AugMixDataset] 加载 {len(self.samples)} 个样本 ({mode_str}) from {patches_dir}")
        print(f"  原始特征: {feature_cache_dir}")
        if self.augmented_cache_dir:
            print(f"  增强特征: {self.augmented_cache_dir} ({aug_count} files, "
                  f"n_augments={n_augments}, prob={aug_sample_prob})")
        else:
            print(f"  增强特征: 未启用")
        print(f"  坐标范围: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]")
        print(f"  目标列数: {len(target_cols)}")

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

        # 决定加载哪个特征文件
        if (self.is_train
                and self.augmented_cache_dir is not None
                and random.random() < self.aug_sample_prob):
            # 随机选择一个增强变体
            aug_idx = random.randint(1, self.n_augments)
            pt_path = os.path.join(self.augmented_cache_dir, f"{stem}_aug{aug_idx}.pt")
            if not os.path.exists(pt_path):
                # fallback 到原始特征
                pt_path = os.path.join(self.feature_cache_dir, f"{stem}.pt")
        else:
            # 使用原始特征
            pt_path = os.path.join(self.feature_cache_dir, f"{stem}.pt")

        tokens = torch.load(pt_path, map_location='cpu', weights_only=True)
        # 处理可能的 dict 格式
        if isinstance(tokens, dict) and "tokens" in tokens:
            tokens = tokens["tokens"]
        elif isinstance(tokens, dict) and "feature" in tokens:
            tokens = tokens["feature"]
        # 确保 float32
        tokens = tokens.float()
        # 确保形状为 [num_tokens, 1536]（2D tensor）
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # [1536] -> [1, 1536]
        assert tokens.dim() == 2 and tokens.shape[1] == 1536, (
            f"Token特征维度不匹配: 期望 [num_tokens, 1536], 实际 {tokens.shape}, stem={stem}"
        )

        # 坐标映射
        pos_x = self._coord_to_index(x, self.x_min, self.x_max)
        pos_y = self._coord_to_index(y, self.y_min, self.y_max)

        targets = torch.tensor(targets, dtype=torch.float32)

        return (tokens,
                torch.tensor(pos_x, dtype=torch.long),
                torch.tensor(pos_y, dtype=torch.long),
                targets)

    @classmethod
    def from_multiple_patients(cls, patient_configs, n_pos=128, n_targets=30,
                               augmented_cache_dir_map=None,
                               n_augments=3, aug_sample_prob=0.5,
                               is_train=True, verbose=True):
        """
        多患者联合训练：合并多个患者的 Dataset

        Args:
            patient_configs: list of dicts, 每个包含:
                - patches_dir: str
                - labels_csv: str
                - feature_cache_dir: str
                - patient_name: str（可选）
            augmented_cache_dir_map: dict {patient_split_name: aug_cache_dir}，可选
            n_augments: 增强变体数量
            aug_sample_prob: 训练时选择增强版本的概率
            is_train: 是否为训练模式
            verbose: 是否打印详细信息

        Returns:
            merged_dataset, coord_stats_dict, target_cols
        """
        datasets = []
        coord_stats_dict = {}
        target_cols = None

        if augmented_cache_dir_map is None:
            augmented_cache_dir_map = {}

        for i, config in enumerate(patient_configs):
            patches_dir = config['patches_dir']
            labels_csv = config['labels_csv']
            feature_cache_dir = config['feature_cache_dir']
            patient_name = config.get('patient_name', f'patient_{i}')

            # 查找该患者分片的增强目录
            aug_dir = config.get('augmented_cache_dir',
                                 augmented_cache_dir_map.get(patient_name))

            if verbose:
                print(f"\n[MultiPatient-AugMix] 加载患者 {patient_name}...")

            dataset = cls(
                patches_dir=patches_dir,
                feature_cache_dir=feature_cache_dir,
                labels_csv=labels_csv,
                augmented_cache_dir=aug_dir,
                n_augments=n_augments,
                aug_sample_prob=aug_sample_prob,
                is_train=is_train,
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
                      f"坐标范围: x=[{cs['x_min']}, {cs['x_max']}], "
                      f"y=[{cs['y_min']}, {cs['y_max']}]")

        merged_dataset = ConcatDataset(datasets)

        if verbose:
            total_samples = sum(len(d) for d in datasets)
            print(f"\n[MultiPatient-AugMix] 合并完成: {len(datasets)} 个患者, "
                  f"共 {total_samples} 个样本")

        return merged_dataset, coord_stats_dict, target_cols
