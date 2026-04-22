"""
EGNv2 模型定义 - PFMval 项目
基于 ResNet-50 + GraphSAGE + Exemplar Graph Neural Network 的通路评分预测模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ResNetFeatureExtractor(nn.Module):
    """基于 torchvision ResNet-50 的特征提取器"""

    def __init__(self, freeze_layers=3):
        """
        Args:
            freeze_layers: 冻结前 N 个 layer（1=layer1, 2=layer2, 3=layer3）
        """
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 去掉最后的 avgpool + fc，保留卷积特征
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 自适应池化，将空间维度压为 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 冻结指定层
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n):
        """冻结前 n 个 ResNet layer"""
        layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        freeze_names = layer_names[:n + 2]  # conv1, bn1 + 前 n 个 layer

        for name, param in self.features.named_parameters():
            # 检查参数是否属于需要冻结的层
            should_freeze = any(fn in name for fn in freeze_names)
            if should_freeze:
                param.requires_grad = False

        frozen_count = sum(1 for p in self.features.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.features.parameters())
        print(f"[ResNetFeatureExtractor] 冻结 {frozen_count}/{total_count} 个参数组")

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) 输入图像
        Returns:
            (B, 2048) 特征向量
        """
        feat = self.features(x)       # (B, 2048, H', W')
        feat = self.avgpool(feat)     # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1) # (B, 2048)
        return feat


class ExemplarLibrary:
    """代表库管理 - 存储训练集特征和目标，支持 KNN 检索"""

    def __init__(self, features, targets):
        """
        Args:
            features: Tensor(K, D) 代表特征
            targets:  Tensor(K, 30) 代表目标值
        """
        self.features = features.cpu()
        self.targets = targets.cpu()
        # 预拟合 NearestNeighbors
        self._nn = NearestNeighbors(
            n_neighbors=min(10, len(features)),
            algorithm='auto', metric='euclidean'
        )
        self._nn.fit(self.features.numpy())

    def retrieve(self, query_features, k=10):
        """
        查询最近 k 个代表

        Args:
            query_features: Tensor(N, D) 查询特征
            k: 最近邻数量
        Returns:
            indices: (N, k) 最近邻索引
            distances: (N, k) 最近邻距离
        """
        if k > len(self.features):
            k = len(self.features)
        query_np = query_features.detach().cpu().numpy()
        distances, indices = self._nn.kneighbors(query_np, n_neighbors=k)
        return indices, distances

    def get_features(self, indices):
        """获取指定索引的代表特征"""
        return self.features[indices]

    def get_targets(self, indices):
        """获取指定索引的代表目标值"""
        return self.targets[indices]

    def save(self, path):
        """保存代表库到文件"""
        torch.save({
            'features': self.features,
            'targets': self.targets,
        }, path)
        print(f"[ExemplarLibrary] 已保存到 {path}")

    @classmethod
    def load(cls, path):
        """从文件加载代表库"""
        data = torch.load(path, weights_only=False)
        lib = cls(data['features'], data['targets'])
        print(f"[ExemplarLibrary] 已从 {path} 加载 {len(lib.features)} 个代表")
        return lib


class EGNv2Model(nn.Module):
    """
    完整 EGNv2 模型（图级训练版本）

    输入:
        node_features: (N, in_dim) ResNet 提取的特征
        edge_index: (2, E) 图边
        exemplar_agg_features: (N, hidden_dim) 聚合后的 exemplar 特征（可选）

    输出:
        (N, n_targets) 每个节点的 30 通路预测
    """

    def __init__(self, in_dim=2048, hidden_dim=512, n_targets=30,
                 graph_layers=2, dropout=0.3, k_exemplars=10):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_targets = n_targets
        self.k_exemplars = k_exemplars

        # 1. 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. GraphSAGE 层
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for _ in range(graph_layers):
            self.graph_convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.graph_norms.append(nn.LayerNorm(hidden_dim))

        self.graph_dropout = nn.Dropout(dropout)

        # 3. Exemplar 融合层
        self.exemplar_fuse = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, node_features, edge_index, exemplar_agg_features=None):
        """
        Args:
            node_features: (N, in_dim) 节点特征
            edge_index: (2, E) 边索引
            exemplar_agg_features: (N, hidden_dim) 聚合后的 exemplar 特征（可选）
        Returns:
            (N, n_targets) 预测值
        """
        # 1. 特征投影
        h = self.feature_proj(node_features)  # (N, hidden_dim)

        # 2. 图卷积
        for conv, norm in zip(self.graph_convs, self.graph_norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = h + h_res  # 残差连接
            h = nn.functional.gelu(h)
            h = self.graph_dropout(h)

        # 3. Exemplar 融合（如果提供了 exemplar 特征）
        if exemplar_agg_features is not None:
            h = self.exemplar_fuse(torch.cat([h, exemplar_agg_features], dim=-1))
        # 否则直接使用图卷积输出

        # 4. 回归
        out = self.regressor(h)  # (N, n_targets)
        return out
