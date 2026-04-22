"""
EGN-v1 代表库构建与特征提取模块
负责：ViT 特征提取、代表库构建、完整预处理与缓存
适配 ViT-Large 1024 维特征输出
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from egnv1.model import ViTFeatureExtractor, ExemplarLibrary
from egnv1.graph_builder import build_knn_graph, build_spatial_graph, build_hybrid_graph


def extract_all_features(dataset, feature_extractor, device, batch_size=16):
    """
    提取数据集中所有样本的 ViT 特征

    Args:
        dataset: EGNv1Dataset 实例
        feature_extractor: ViTFeatureExtractor 实例
        device: torch.device
        batch_size: int（ViT 显存大，默认 16）

    Returns:
        features: (N, 1024) Tensor
        coords: (N, 2) Tensor
        targets: (N, 30) Tensor
    """
    feature_extractor.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    all_features = []
    all_coords = []
    all_targets = []

    with torch.no_grad():
        for images, raw_x, raw_y, targets in tqdm(loader, desc="Extracting ViT features"):
            images = images.to(device, non_blocking=True)
            feats = feature_extractor(images)  # (B, 1024)
            all_features.append(feats.cpu())
            all_coords.append(torch.stack([raw_x, raw_y], dim=1))  # (B, 2)
            all_targets.append(targets)

    features = torch.cat(all_features, dim=0)
    coords = torch.cat(all_coords, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"[Features] 提取完成: {features.shape[0]} 样本, {features.shape[1]} 维特征")
    return features, coords, targets


def build_exemplar_library(features, targets, n_exemplars=None, method='full'):
    """
    构建代表库

    Args:
        features: (N, 1024) Tensor
        targets: (N, 30) Tensor
        n_exemplars: int or None，None 表示使用全部样本
        method: 'full' | 'kmeans'
    Returns:
        ExemplarLibrary 对象
    """
    if n_exemplars is not None and n_exemplars > 0 and n_exemplars < len(features):
        if method == 'kmeans':
            from sklearn.cluster import MiniBatchKMeans
            print(f"[Exemplar] 使用 KMeans 聚类到 {n_exemplars} 个代表...")
            kmeans = MiniBatchKMeans(n_clusters=n_exemplars, batch_size=1024, random_state=42)
            labels = kmeans.fit_predict(features.numpy())
            exemplar_features = torch.from_numpy(kmeans.cluster_centers_).float()

            # 每个聚类中心的目标值取该簇样本的均值
            exemplar_targets = torch.zeros(n_exemplars, targets.shape[1])
            for i in range(n_exemplars):
                mask = labels == i
                if mask.any():
                    exemplar_targets[i] = targets[mask].mean(dim=0)
        else:
            # 均匀采样
            print(f"[Exemplar] 均匀采样 {n_exemplars} 个代表...")
            indices = np.linspace(0, len(features) - 1, n_exemplars, dtype=int)
            exemplar_features = features[indices]
            exemplar_targets = targets[indices]
    else:
        # 使用全部样本
        print(f"[Exemplar] 使用全量 {len(features)} 个样本作为代表")
        exemplar_features = features
        exemplar_targets = targets

    lib = ExemplarLibrary(exemplar_features, exemplar_targets)
    print(f"[Exemplar] 代表库构建完成: {len(lib.features)} 个代表")
    return lib


def compute_exemplar_agg_features(features, exemplar_lib, hidden_dim, k=10, device=None, proj_layer=None):
    """
    计算每个节点的 exemplar 聚合特征

    重要实现细节：
    - 确保所有张量在同一设备
    - 不每次创建新 Linear 层（避免梯度图问题）
    - proj_layer 可复用

    Args:
        features: (N, 1024) Tensor，节点特征
        exemplar_lib: ExemplarLibrary 实例
        hidden_dim: int，与模型 hidden_dim 一致
        k: int，最近邻数量
        device: torch.device
        proj_layer: 可选的投影层，如果提供则使用该层进行投影

    Returns:
        agg_features: (N, hidden_dim) Tensor，exemplar 加权平均特征
        proj_layer: 返回使用的投影层（如果是新创建的）
    """
    if device is None:
        device = torch.device('cpu')

    indices, distances = exemplar_lib.retrieve(features, k=k)

    # 距离转换为权重（距离越近权重越大）
    dists_t = torch.from_numpy(distances).float()
    # 避免除零：加小常数
    weights = 1.0 / (dists_t + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化

    # 获取代表的特征并加权平均
    exemplar_feats = exemplar_lib.get_features(indices)  # (N, k, 1024)

    # 加权平均
    weights_expanded = weights.unsqueeze(-1)  # (N, k, 1)
    agg_feats = (exemplar_feats.float() * weights_expanded).sum(dim=1)  # (N, 1024)

    # 投影到 hidden_dim
    agg_feats = agg_feats.to(device)
    if proj_layer is None:
        # 使用简单的线性投影（无参数，直接截断或填充）
        if agg_feats.shape[1] >= hidden_dim:
            agg_feats = agg_feats[:, :hidden_dim]
        else:
            # 填充零到 hidden_dim
            padding = torch.zeros(agg_feats.shape[0], hidden_dim - agg_feats.shape[1], device=device)
            agg_feats = torch.cat([agg_feats, padding], dim=1)
    else:
        agg_feats = proj_layer(agg_feats)

    return agg_feats, proj_layer


def preprocess_and_cache(dataset_name, train_dataset, val_dataset,
                         feature_extractor, device, cache_dir,
                         n_exemplars=None, graph_type='knn',
                         k_neighbors=10, radius=300):
    """
    完整预处理流程，缓存结果到 cache_dir/

    缓存文件：
    - {dataset_name}_train_features.pth  (features, coords, targets)
    - {dataset_name}_val_features.pth
    - {dataset_name}_train_graph.pth     (edge_index)
    - {dataset_name}_val_graph.pth
    - {dataset_name}_exemplars.pth       (ExemplarLibrary)

    Args:
        dataset_name: 数据集名称
        train_dataset: 训练集
        val_dataset: 验证集
        feature_extractor: ViTFeatureExtractor 实例
        device: torch.device
        cache_dir: 缓存目录
        n_exemplars: 代表库大小（None=全量）
        graph_type: 图构建方式 ('knn'/'spatial'/'hybrid')
        k_neighbors: KNN k 值
        radius: 空间图半径
    """
    os.makedirs(cache_dir, exist_ok=True)

    # --- 训练集特征 ---
    train_feat_path = os.path.join(cache_dir, f"{dataset_name}_train_features.pth")
    if os.path.isfile(train_feat_path):
        print(f"[Cache] 加载训练集特征缓存: {train_feat_path}")
        data = torch.load(train_feat_path, weights_only=False)
        train_features, train_coords, train_targets = data['features'], data['coords'], data['targets']
    else:
        print("[Cache] 提取训练集 ViT 特征...")
        train_features, train_coords, train_targets = extract_all_features(
            train_dataset, feature_extractor, device)
        torch.save({
            'features': train_features,
            'coords': train_coords,
            'targets': train_targets,
        }, train_feat_path)
        print(f"[Cache] 训练集特征已缓存: {train_feat_path}")

    # --- 验证集特征 ---
    val_feat_path = os.path.join(cache_dir, f"{dataset_name}_val_features.pth")
    if os.path.isfile(val_feat_path):
        print(f"[Cache] 加载验证集特征缓存: {val_feat_path}")
        data = torch.load(val_feat_path, weights_only=False)
        val_features, val_coords, val_targets = data['features'], data['coords'], data['targets']
    else:
        print("[Cache] 提取验证集 ViT 特征...")
        val_features, val_coords, val_targets = extract_all_features(
            val_dataset, feature_extractor, device)
        torch.save({
            'features': val_features,
            'coords': val_coords,
            'targets': val_targets,
        }, val_feat_path)
        print(f"[Cache] 验证集特征已缓存: {val_feat_path}")

    # --- 训练集图 ---
    train_graph_path = os.path.join(cache_dir, f"{dataset_name}_train_graph.pth")
    if os.path.isfile(train_graph_path):
        print(f"[Cache] 加载训练集图缓存: {train_graph_path}")
        train_edge_index = torch.load(train_graph_path, weights_only=False)['edge_index']
    else:
        print(f"[Cache] 构建训练集图 (type={graph_type})...")
        train_edge_index = _build_graph_by_type(
            graph_type, train_features.numpy(), train_coords.numpy(),
            k=k_neighbors, radius=radius
        )
        torch.save({'edge_index': train_edge_index}, train_graph_path)
        print(f"[Cache] 训练集图已缓存: {train_graph_path}")

    # --- 验证集图 ---
    val_graph_path = os.path.join(cache_dir, f"{dataset_name}_val_graph.pth")
    if os.path.isfile(val_graph_path):
        print(f"[Cache] 加载验证集图缓存: {val_graph_path}")
        val_edge_index = torch.load(val_graph_path, weights_only=False)['edge_index']
    else:
        print(f"[Cache] 构建验证集图 (type={graph_type})...")
        val_edge_index = _build_graph_by_type(
            graph_type, val_features.numpy(), val_coords.numpy(),
            k=k_neighbors, radius=radius
        )
        torch.save({'edge_index': val_edge_index}, val_graph_path)
        print(f"[Cache] 验证集图已缓存: {val_graph_path}")

    # --- 代表库 ---
    exemplar_path = os.path.join(cache_dir, f"{dataset_name}_exemplars.pth")
    if os.path.isfile(exemplar_path):
        print(f"[Cache] 加载代表库缓存: {exemplar_path}")
        exemplar_lib = ExemplarLibrary.load(exemplar_path)
    else:
        print("[Cache] 构建代表库...")
        exemplar_lib = build_exemplar_library(train_features, train_targets,
                                               n_exemplars=n_exemplars)
        exemplar_lib.save(exemplar_path)

    return {
        'train_features': train_features,
        'train_coords': train_coords,
        'train_targets': train_targets,
        'train_edge_index': train_edge_index,
        'val_features': val_features,
        'val_coords': val_coords,
        'val_targets': val_targets,
        'val_edge_index': val_edge_index,
        'exemplar_lib': exemplar_lib,
    }


def _build_graph_by_type(graph_type, features, coords, k=10, radius=300):
    """根据 graph_type 选择图构建方式"""
    if graph_type == 'knn':
        return build_knn_graph(features, k=k)
    elif graph_type == 'spatial':
        return build_spatial_graph(coords, radius=radius)
    elif graph_type == 'hybrid':
        return build_hybrid_graph(features, coords, k=k, radius=radius)
    else:
        print(f"[WARNING] 未知 graph_type={graph_type}，默认使用 knn")
        return build_knn_graph(features, k=k)
