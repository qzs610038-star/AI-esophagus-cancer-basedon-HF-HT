"""
EGN-v1 图构建模块
支持三种图构建方式：KNN图（基于特征相似性）、空间图（基于坐标距离）、混合图
EGN-v1 默认使用 KNN 图（与 EGNv2 的空间半径图不同）
"""

import torch
import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph


def build_knn_graph(features, k=10):
    """
    基于特征相似性构建 KNN 图（EGN-v1 默认图构建方式）

    Args:
        features: (N, D) numpy array，特征向量
        k: int，最近邻数量

    Returns:
        edge_index: (2, num_edges) LongTensor，无向图
    """
    if len(features) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    n_samples = len(features)
    actual_k = min(k, n_samples - 1)

    if actual_k < 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # 使用 kneighbors_graph
    adj = kneighbors_graph(
        features, n_neighbors=actual_k,
        mode='connectivity', metric='euclidean', include_self=False
    )

    # 转换为 COO 格式
    adj_coo = adj.tocoo()

    # 构建双向边（确保无向图）
    rows = adj_coo.row
    cols = adj_coo.col

    edge_index = np.stack([
        np.concatenate([rows, cols]),
        np.concatenate([cols, rows])
    ], axis=0)

    # 去重
    edge_set = set()
    unique_edges = []
    for i in range(edge_index.shape[1]):
        e = (edge_index[0, i], edge_index[1, i])
        if e not in edge_set:
            edge_set.add(e)
            unique_edges.append(e)

    if len(unique_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = np.array(unique_edges).T
    edge_index = torch.from_numpy(edge_index).long()

    print(f"[Graph] 构建 KNN 图: {n_samples} 节点, {edge_index.shape[1]} 条边, k={actual_k}")

    return edge_index


def build_spatial_graph(coords, radius=300):
    """
    基于空间距离构建邻接图（备选方案）

    Args:
        coords: (N, 2) numpy array，像素坐标
        radius: float，距离阈值

    Returns:
        edge_index: (2, num_edges) LongTensor，无向图（双向边）
    """
    if len(coords) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # 使用 radius_neighbors_graph 构建邻接矩阵
    adj = radius_neighbors_graph(
        coords, radius=radius, mode='connectivity',
        metric='euclidean', include_self=False
    )

    # 转换为 COO 格式
    adj_coo = adj.tocoo()

    # 构建双向边
    rows = adj_coo.row
    cols = adj_coo.col

    # 合并正向和反向边（确保无向图）
    edge_index = np.stack([
        np.concatenate([rows, cols]),
        np.concatenate([cols, rows])
    ], axis=0)

    # 去重
    edge_set = set()
    unique_edges = []
    for i in range(edge_index.shape[1]):
        e = (edge_index[0, i], edge_index[1, i])
        if e not in edge_set:
            edge_set.add(e)
            unique_edges.append(e)

    if len(unique_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = np.array(unique_edges).T
    edge_index = torch.from_numpy(edge_index).long()

    print(f"[Graph] 构建空间图: {len(coords)} 节点, {edge_index.shape[1]} 条边, radius={radius}")

    return edge_index


def build_hybrid_graph(features, coords, k=10, radius=300):
    """
    混合图构建：合并 KNN 特征图和空间半径图的边

    Args:
        features: (N, D) numpy array，特征向量
        coords: (N, 2) numpy array，像素坐标
        k: int，KNN 最近邻数量
        radius: float，空间图距离阈值

    Returns:
        edge_index: (2, num_edges) LongTensor，无向图
    """
    knn_edges = build_knn_graph(features, k=k)
    spatial_edges = build_spatial_graph(coords, radius=radius)

    if knn_edges.shape[1] == 0:
        return spatial_edges
    if spatial_edges.shape[1] == 0:
        return knn_edges

    # 合并两张图的边并去重
    all_edges = torch.cat([knn_edges, spatial_edges], dim=1)

    edge_set = set()
    unique_edges = []
    for i in range(all_edges.shape[1]):
        e = (all_edges[0, i].item(), all_edges[1, i].item())
        if e not in edge_set:
            edge_set.add(e)
            unique_edges.append(e)

    if len(unique_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(unique_edges, dtype=torch.long).T

    print(f"[Graph] 构建混合图: {len(features)} 节点, {edge_index.shape[1]} 条边 "
          f"(KNN k={k} + Spatial radius={radius})")

    return edge_index
