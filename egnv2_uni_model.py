"""
EGNv2 + UNI2-h 集成模型定义
基于 egnv2.model.EGNv2Model，去掉 ResNet-50 特征提取器，
直接接收 UNI2-h 预提取的 1536 维特征。
"""

from egnv2.model import EGNv2Model


def create_egnv2_uni_model(hidden_dim=512, output_dim=30, graph_layers=2,
                           dropout=0.1, k_exemplars=10):
    """
    创建使用 UNI2-h 特征的 EGN-v2 模型（in_dim=1536）

    Args:
        hidden_dim: 图卷积隐藏维度（默认 512）
        output_dim: 预测通路数（默认 30）
        graph_layers: GraphSAGE 层数（默认 2）
        dropout: Dropout 比率（默认 0.1）
        k_exemplars: Exemplar KNN 的 k 值（默认 10）

    Returns:
        EGNv2Model 实例，in_dim=1536
    """
    return EGNv2Model(
        in_dim=1536,
        hidden_dim=hidden_dim,
        n_targets=output_dim,
        graph_layers=graph_layers,
        dropout=dropout,
        k_exemplars=k_exemplars,
    )
