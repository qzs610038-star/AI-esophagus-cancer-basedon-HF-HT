"""
图级别数据集 - 返回整个患者的所有patch作为一个图
用于GAT全图训练

每个样本是一个患者的完整图：
    - tokens: List[Tensor], 每个patch的token序列
    - pos_x: [N] LongTensor, 坐标x索引
    - pos_y: [N] LongTensor, 坐标y索引
    - labels: [N, 30] FloatTensor, 通路标签
    - edge_index: [2, E] LongTensor, KNN图的边
    - patient_name: str, 患者名称
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from graph_utils_gat import build_knn_graph


def parse_coordinates(filename: str):
    """从文件名 patch_x4641_y16969.png 解析坐标"""
    match = re.search(r'x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def graph_collate_fn(batch):
    """
    自定义 collate 函数。
    由于每个患者的patch数不同，batch_size=1时直接返回单个样本。
    多个患者时返回list。
    """
    if len(batch) == 1:
        return batch[0]
    return batch


class GraphTokenDataset(Dataset):
    """
    图级别数据集 - 每个样本是一个患者的完整图。

    对每个患者执行三层交集过滤（与dataset_uni_tokens.py一致）：
        cache .pt文件 ∩ PNG图像 ∩ CSV标签

    坐标处理：
        - 解析文件名获取像素坐标
        - 归一化映射到 [0, n_pos-1]

    图构建：
        - 使用KNN图（默认k=6）连接空间邻近的patch
    """

    def __init__(self,
                 patient_dirs: list,
                 csv_paths: list,
                 cache_dirs: list,
                 patient_names: list = None,
                 n_pos: int = 128,
                 k_neighbors: int = 6,
                 split: str = 'train'):
        """
        Args:
            patient_dirs: list of patient patch目录路径（可包含多个split目录）
            csv_paths: list of 对应的zscore CSV路径
            cache_dirs: list of UNI token缓存目录路径
            patient_names: list of 患者名称（可选，用于标识）
            n_pos: 坐标嵌入范围 [0, n_pos-1]
            k_neighbors: KNN图的k值
            split: 'train' or 'val'/'test'（用于打印信息）
        """
        super().__init__()
        self.n_pos = n_pos
        self.k_neighbors = k_neighbors
        self.patients = []
        self.target_cols = None

        if patient_names is None:
            patient_names = [f'patient_{i}' for i in range(len(patient_dirs))]

        assert len(patient_dirs) == len(csv_paths) == len(cache_dirs), \
            "patient_dirs, csv_paths, cache_dirs 长度必须一致"

        for idx, (patch_dir, csv_path, cache_dir, pname) in enumerate(
            zip(patient_dirs, csv_paths, cache_dirs, patient_names)
        ):
            patient_data = self._load_patient(patch_dir, csv_path, cache_dir, pname)
            if patient_data is not None:
                self.patients.append(patient_data)

        print(f"\n[GraphTokenDataset] {split} 集: 加载 {len(self.patients)} 个患者图")
        for p in self.patients:
            print(f"  {p['name']}: {p['n_patches']} patches, "
                  f"{p['edge_index'].shape[1]} edges (k={k_neighbors})")

    def _load_patient(self, patch_dir: str, csv_path: str,
                      cache_dir: str, patient_name: str) -> dict:
        """加载单个患者的完整图数据"""

        # 检查路径
        if not os.path.isdir(patch_dir):
            print(f"[WARNING] patch目录不存在: {patch_dir}")
            return None
        if not os.path.isfile(csv_path):
            print(f"[WARNING] CSV不存在: {csv_path}")
            return None
        if not os.path.isdir(cache_dir):
            print(f"[WARNING] cache目录不存在: {cache_dir}")
            return None

        # 加载标签CSV
        df = pd.read_csv(csv_path)
        id_col = df.columns[0]
        if self.target_cols is None:
            self.target_cols = list(df.columns[1:])

        # 构建标签映射: stem -> target_values
        label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '')
            label_map[stem] = row[self.target_cols].values.astype(np.float32)

        # 扫描缓存目录中的.pt文件
        cached_stems = set()
        for fname in os.listdir(cache_dir):
            if fname.lower().endswith('.pt'):
                cached_stems.add(fname[:-3])

        # 三层交集过滤: cache .pt ∩ patches .png ∩ CSV标签
        samples = []  # (stem, x, y, targets)
        all_x, all_y = [], []

        for fname in sorted(os.listdir(patch_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = fname.replace('.png', '')

            if stem not in label_map:
                continue
            if stem not in cached_stems:
                continue

            x, y = parse_coordinates(fname)
            if x is None:
                continue

            targets = label_map[stem]
            samples.append((stem, x, y, targets))
            all_x.append(x)
            all_y.append(y)

        if len(samples) == 0:
            print(f"[WARNING] 患者 {patient_name} 无有效样本（三层交集为空）")
            return None

        # 坐标归一化映射到 [0, n_pos-1]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        pos_x_list = []
        pos_y_list = []
        coords_for_graph = []

        for stem, x, y, targets in samples:
            px = self._coord_to_index(x, x_min, x_max)
            py = self._coord_to_index(y, y_min, y_max)
            pos_x_list.append(px)
            pos_y_list.append(py)
            coords_for_graph.append([x, y])

        # 构建KNN图
        coords_array = np.array(coords_for_graph, dtype=np.float32)
        edge_index = build_knn_graph(coords_array, k=self.k_neighbors)

        # 准备token路径列表
        token_paths = [os.path.join(cache_dir, f"{s[0]}.pt") for s in samples]

        # 准备标签
        labels = np.stack([s[3] for s in samples], axis=0)

        patient_data = {
            'name': patient_name,
            'n_patches': len(samples),
            'token_paths': token_paths,
            'pos_x': torch.tensor(pos_x_list, dtype=torch.long),
            'pos_y': torch.tensor(pos_y_list, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'edge_index': edge_index,
            'coord_stats': {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max
            },
        }
        return patient_data

    def _coord_to_index(self, val: int, vmin: int, vmax: int) -> int:
        """将坐标值归一化映射到 [0, n_pos-1]"""
        if vmax == vmin:
            return 0
        normalized = (val - vmin) / (vmax - vmin)
        idx = int(np.clip(normalized * (self.n_pos - 1), 0, self.n_pos - 1))
        return idx

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        """
        返回一个患者的完整图数据。

        Returns:
            dict with keys:
                - tokens: List[Tensor], 每个 shape [num_tokens, 1536]
                - pos_x: [N] LongTensor
                - pos_y: [N] LongTensor
                - labels: [N, 30] FloatTensor
                - edge_index: [2, E] LongTensor
                - patient_name: str
        """
        patient = self.patients[idx]

        # 逐个加载token（按需加载，避免内存爆炸）
        all_tokens = []
        for pt_path in patient['token_paths']:
            tokens = torch.load(pt_path, map_location='cpu', weights_only=True)
            # 处理可能的 dict 格式
            if isinstance(tokens, dict) and "tokens" in tokens:
                tokens = tokens["tokens"]
            elif isinstance(tokens, dict) and "feature" in tokens:
                tokens = tokens["feature"]
            # 确保 float32
            tokens = tokens.float()
            # 确保形状为 [num_tokens, 1536]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            all_tokens.append(tokens)

        return {
            'tokens': all_tokens,
            'pos_x': patient['pos_x'],
            'pos_y': patient['pos_y'],
            'labels': patient['labels'],
            'edge_index': patient['edge_index'],
            'patient_name': patient['name'],
        }


class MultiSplitGraphTokenDataset(Dataset):
    """
    多split合并的图数据集。
    将一个患者的 train + val split 合并为单个图。
    用于跨患者训练时，训练患者使用全部数据。
    """

    def __init__(self,
                 patient_configs: list,
                 n_pos: int = 128,
                 k_neighbors: int = 6,
                 split: str = 'train'):
        """
        Args:
            patient_configs: list of dict, 每个包含:
                - patient_name: str
                - patch_dirs: list of str (可含多个split目录)
                - csv_path: str
                - cache_dirs: list of str (与patch_dirs对应)
            n_pos: 坐标嵌入范围
            k_neighbors: KNN图k值
            split: 标识信息
        """
        super().__init__()
        self.n_pos = n_pos
        self.k_neighbors = k_neighbors
        self.patients = []
        self.target_cols = None

        for config in patient_configs:
            patient_data = self._load_merged_patient(config)
            if patient_data is not None:
                self.patients.append(patient_data)

        print(f"\n[MultiSplitGraphTokenDataset] {split} 集: "
              f"加载 {len(self.patients)} 个患者图")
        for p in self.patients:
            print(f"  {p['name']}: {p['n_patches']} patches, "
                  f"{p['edge_index'].shape[1]} edges (k={k_neighbors})")

    def _load_merged_patient(self, config: dict) -> dict:
        """加载并合并一个患者的多个split"""
        patient_name = config['patient_name']
        patch_dirs = config['patch_dirs']
        csv_path = config['csv_path']
        cache_dirs = config['cache_dirs']

        if not os.path.isfile(csv_path):
            print(f"[WARNING] CSV不存在: {csv_path}")
            return None

        # 加载标签CSV
        df = pd.read_csv(csv_path)
        id_col = df.columns[0]
        if self.target_cols is None:
            self.target_cols = list(df.columns[1:])

        label_map = {}
        for _, row in df.iterrows():
            stem = str(row[id_col]).replace('.png', '')
            label_map[stem] = row[self.target_cols].values.astype(np.float32)

        # 合并多个split的样本
        samples = []
        all_x, all_y = [], []

        for patch_dir, cache_dir in zip(patch_dirs, cache_dirs):
            if not os.path.isdir(patch_dir) or not os.path.isdir(cache_dir):
                continue

            cached_stems = set()
            for fname in os.listdir(cache_dir):
                if fname.lower().endswith('.pt'):
                    cached_stems.add(fname[:-3])

            for fname in sorted(os.listdir(patch_dir)):
                if not fname.lower().endswith('.png'):
                    continue
                stem = fname.replace('.png', '')
                if stem not in label_map:
                    continue
                if stem not in cached_stems:
                    continue
                x, y = parse_coordinates(fname)
                if x is None:
                    continue
                targets = label_map[stem]
                # 记录cache_dir用于后续加载
                samples.append((stem, x, y, targets, cache_dir))
                all_x.append(x)
                all_y.append(y)

        if len(samples) == 0:
            print(f"[WARNING] 患者 {patient_name} 无有效样本")
            return None

        # 坐标归一化
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        pos_x_list, pos_y_list = [], []
        coords_for_graph = []

        for stem, x, y, targets, cache_dir in samples:
            px = self._coord_to_index(x, x_min, x_max)
            py = self._coord_to_index(y, y_min, y_max)
            pos_x_list.append(px)
            pos_y_list.append(py)
            coords_for_graph.append([x, y])

        # 构建KNN图
        coords_array = np.array(coords_for_graph, dtype=np.float32)
        edge_index = build_knn_graph(coords_array, k=self.k_neighbors)

        # Token路径
        token_paths = [os.path.join(s[4], f"{s[0]}.pt") for s in samples]
        labels = np.stack([s[3] for s in samples], axis=0)

        return {
            'name': patient_name,
            'n_patches': len(samples),
            'token_paths': token_paths,
            'pos_x': torch.tensor(pos_x_list, dtype=torch.long),
            'pos_y': torch.tensor(pos_y_list, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'edge_index': edge_index,
            'coord_stats': {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max
            },
        }

    def _coord_to_index(self, val: int, vmin: int, vmax: int) -> int:
        if vmax == vmin:
            return 0
        normalized = (val - vmin) / (vmax - vmin)
        return int(np.clip(normalized * (self.n_pos - 1), 0, self.n_pos - 1))

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        all_tokens = []
        for pt_path in patient['token_paths']:
            tokens = torch.load(pt_path, map_location='cpu', weights_only=True)
            if isinstance(tokens, dict) and "tokens" in tokens:
                tokens = tokens["tokens"]
            elif isinstance(tokens, dict) and "feature" in tokens:
                tokens = tokens["feature"]
            tokens = tokens.float()
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            all_tokens.append(tokens)

        return {
            'tokens': all_tokens,
            'pos_x': patient['pos_x'],
            'pos_y': patient['pos_y'],
            'labels': patient['labels'],
            'edge_index': patient['edge_index'],
            'patient_name': patient['name'],
        }
