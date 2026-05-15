"""
图构建工具模块 - P0-3 GAT升级
支持KNN图和半径图两种空间图构建方式

使用 torch_geometric 的高效实现，适用于大规模patch空间图构建。
"""

import numpy as np
import torch
from torch_geometric.nn import knn_graph, radius_graph


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    将原始像素坐标归一化到[0, 1]范围。
    避免KNN被坐标量级影响。

    Args:
        coords: (N, 2) numpy数组，原始像素坐标 (x, y)

    Returns:
        normalized: (N, 2) numpy数组，归一化后的坐标
    """
    if len(coords) == 0:
        return coords.copy()

    coords = coords.astype(np.float64)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    ranges = max_vals - min_vals

    # 防止除零：若某维度所有值相同，归一化为0.5
    ranges[ranges == 0] = 1.0
    normalized = (coords - min_vals) / ranges

    return normalized.astype(np.float32)


def build_knn_graph(coords: np.ndarray, k: int = 6) -> torch.Tensor:
    """
    基于归一化空间坐标构建KNN图。

    Args:
        coords: (N, 2) numpy数组或tensor，表示patch的像素坐标
        k: 每个节点的最近邻数量，默认6

    Returns:
        edge_index: (2, E) LongTensor，双向边（无向图）
    """
    if len(coords) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # 归一化坐标
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.asarray(coords)

    normed = normalize_coords(coords_np)
    pos = torch.from_numpy(normed).float()

    # 自适应k值：不超过节点数-1
    n_nodes = pos.shape[0]
    actual_k = min(k, n_nodes - 1)
    if actual_k < 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # 使用 torch_geometric 的 knn_graph（已自动生成双向边）
    edge_index = knn_graph(pos, k=actual_k, loop=False)

    return edge_index


def build_radius_graph(coords: np.ndarray, radius: float = 0.15) -> torch.Tensor:
    """
    基于归一化空间坐标构建半径图（备选方案）。

    注意：radius 是在归一化坐标空间 [0,1] 上的半径，
    而非原始像素坐标空间。

    Args:
        coords: (N, 2) numpy数组或tensor，表示patch的像素坐标
        radius: 归一化空间中的连接半径，默认0.15

    Returns:
        edge_index: (2, E) LongTensor，双向边（无向图）
    """
    if len(coords) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # 归一化坐标
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.asarray(coords)

    normed = normalize_coords(coords_np)
    pos = torch.from_numpy(normed).float()

    # 使用 torch_geometric 的 radius_graph
    edge_index = radius_graph(pos, r=radius, loop=False)

    return edge_index
