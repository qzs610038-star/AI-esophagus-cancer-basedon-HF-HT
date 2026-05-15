"""
HisToGene-UNI Token + GAT 模型
在方案B基础上引入图注意力网络建模空间邻域关系

架构概述：
    UNI2-h tokens [N, num_tokens, 1536]
    → LightweightTokenEncoder → [N, 1536] (逐patch编码)
    → GATConv × 2 (空间邻域聚合, with residual)
    → +坐标嵌入
    → MLP 回归头
    → [N, n_targets]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class LightweightTokenEncoder(nn.Module):
    """将 [B, num_tokens, 1536] 编码为 [B, embed_dim]

    使用投影到较小维度 + Transformer Encoder + 全局平均池化 + 输出投影的轻量设计。
    参数量约 3.2M，控制在合理范围。

    注意：从 model_uni_tokens.py 复制，保持独立不import，确保不修改原有文件。
    """

    def __init__(self, embed_dim: int = 1536, hidden_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        # 输入投影：1536 → hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 输出投影：hidden_dim → embed_dim
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_tokens, 1536]
        Returns:
            encoded: [B, embed_dim]
        """
        x = self.input_proj(tokens)      # [B, num_tokens, hidden_dim]
        x = self.encoder(x)              # [B, num_tokens, hidden_dim]
        x = x.mean(dim=1)               # [B, hidden_dim] 全局平均池化
        x = self.output_proj(x)          # [B, embed_dim]
        x = self.norm(x)
        return x


class HisToGeneUNITokensGAT(nn.Module):
    """
    HisToGene-UNI Token + GAT 模型

    在方案B (Token序列编码) 基础上引入图注意力网络，
    显式建模patch之间的空间邻域关系。

    输入：
        - all_tokens: List[Tensor] 或 Tensor [N, num_tokens, 1536]
        - pos_x: (N,) X坐标索引 (long)
        - pos_y: (N,) Y坐标索引 (long)
        - edge_index: (2, E) 图的边

    输出：
        - predictions: (N, n_targets) 通路评分预测
    """

    def __init__(self,
                 input_dim: int = 1536,
                 hidden_dim: int = 512,
                 gat_hidden: int = 256,
                 gat_heads: int = 4,
                 gat_layers: int = 2,
                 n_pos: int = 128,
                 n_targets: int = 30,
                 mlp_dim: int = 2048,
                 dropout: float = 0.2,
                 num_encoder_heads: int = 8,
                 num_encoder_layers: int = 1):
        """
        Args:
            input_dim: UNI token维度 (1536)
            hidden_dim: Token encoder 隐藏维度
            gat_hidden: GAT每头输出维度
            gat_heads: GAT注意力头数
            gat_layers: GAT层数
            n_pos: 坐标嵌入范围
            n_targets: 通路目标数
            mlp_dim: 回归头隐层维度
            dropout: Dropout率
            num_encoder_heads: Token encoder 注意力头数
            num_encoder_layers: Token encoder 层数
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_targets = n_targets

        # Token编码器
        self.token_encoder = LightweightTokenEncoder(
            embed_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=num_encoder_heads,
            n_layers=num_encoder_layers,
            dropout=dropout
        )

        # GAT层
        # Layer 1: input_dim(1536) → gat_hidden*gat_heads (256*4=1024), concat=True
        # Layer 2: gat_hidden*gat_heads(1024) → gat_hidden*gat_heads(1024), concat=False
        gat_out_dim = gat_hidden * gat_heads  # 1024

        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        for i in range(gat_layers):
            if i == 0:
                # 第一层：从input_dim映射到gat_hidden*gat_heads
                conv = GATConv(
                    in_channels=input_dim,
                    out_channels=gat_hidden,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout
                )
            else:
                # 后续层：维度保持 gat_out_dim → gat_out_dim
                conv = GATConv(
                    in_channels=gat_out_dim,
                    out_channels=gat_out_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout
                )
            self.gat_convs.append(conv)
            self.gat_norms.append(nn.LayerNorm(gat_out_dim))
            self.gat_dropouts.append(nn.Dropout(dropout))

        # 投影层：将input_dim映射到gat_out_dim（用于第一层的residual）
        self.input_proj = nn.Linear(input_dim, gat_out_dim)

        # 坐标嵌入（维度匹配GAT输出）
        self.x_embed = nn.Embedding(n_pos, gat_out_dim)
        self.y_embed = nn.Embedding(n_pos, gat_out_dim)

        # 回归头
        self.head = nn.Sequential(
            nn.LayerNorm(gat_out_dim),
            nn.Linear(gat_out_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_targets)
        )

    def forward(self, all_tokens, pos_x: torch.Tensor,
                pos_y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_tokens: List[Tensor] 每个元素 shape [num_tokens, 1536]
                        或 Tensor [N, num_tokens, 1536]
            pos_x: [N] LongTensor, 坐标x索引 [0, n_pos-1]
            pos_y: [N] LongTensor, 坐标y索引 [0, n_pos-1]
            edge_index: [2, E] LongTensor, 图的边

        Returns:
            predictions: [N, n_targets] 所有patch的通路预测
        """
        device = pos_x.device

        # 1. Token编码: 逐patch编码（节省显存）
        if isinstance(all_tokens, (list, tuple)):
            patch_features = []
            for tokens in all_tokens:
                # tokens: [num_tokens, 1536] → [1, num_tokens, 1536]
                if tokens.dim() == 2:
                    tokens = tokens.unsqueeze(0)
                tokens = tokens.to(device)
                feat = self.token_encoder(tokens)  # [1, 1536]
                patch_features.append(feat)
            patch_features = torch.cat(patch_features, dim=0)  # [N, 1536]
        else:
            # Tensor [N, num_tokens, 1536] - 可以批量处理
            all_tokens = all_tokens.to(device)
            patch_features = self.token_encoder(all_tokens)  # [N, 1536]

        # 2. GAT层: 空间邻域聚合
        x = patch_features
        for i, (conv, norm, drop) in enumerate(
            zip(self.gat_convs, self.gat_norms, self.gat_dropouts)
        ):
            if i == 0:
                # 第一层: input_dim → gat_out_dim, 使用投影做residual
                x_res = self.input_proj(x)  # [N, gat_out_dim]
                x = conv(x, edge_index)     # [N, gat_out_dim]
                x = F.elu(x)
                x = norm(x)
                x = drop(x)
                x = x + x_res  # residual
            else:
                # 后续层: gat_out_dim → gat_out_dim
                x_res = x
                x = conv(x, edge_index)
                x = F.elu(x)
                x = norm(x)
                x = drop(x)
                x = x + x_res  # residual

        # 3. 坐标编码融合
        x = x + self.x_embed(pos_x) + self.y_embed(pos_y)

        # 4. 回归头
        x = self.head(x)  # [N, n_targets]
        return x

    def count_parameters(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> None:
    """打印模型各模块的参数量"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"模型总参数量: {total:,} ({total/1e6:.2f}M)")
    print(f"{'='*50}")

    if hasattr(model, 'token_encoder'):
        enc_params = sum(p.numel() for p in model.token_encoder.parameters() if p.requires_grad)
        print(f"  Token Encoder: {enc_params:,} ({enc_params/1e6:.2f}M)")

    if hasattr(model, 'gat_convs'):
        gat_params = sum(p.numel() for p in model.gat_convs.parameters() if p.requires_grad)
        gat_params += sum(p.numel() for p in model.gat_norms.parameters() if p.requires_grad)
        print(f"  GAT Layers: {gat_params:,} ({gat_params/1e6:.2f}M)")

    if hasattr(model, 'x_embed'):
        embed_params = sum(p.numel() for p in model.x_embed.parameters() if p.requires_grad)
        embed_params += sum(p.numel() for p in model.y_embed.parameters() if p.requires_grad)
        print(f"  坐标嵌入: {embed_params:,} ({embed_params/1e6:.2f}M)")

    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        print(f"  回归头: {head_params:,} ({head_params/1e6:.2f}M)")

    print()


if __name__ == '__main__':
    model = HisToGeneUNITokensGAT()
    count_parameters(model)

    # 测试前向传播
    N = 20  # 模拟20个patch
    num_tokens = 257
    dummy_tokens = [torch.randn(num_tokens, 1536) for _ in range(N)]
    dummy_pos_x = torch.randint(0, 128, (N,))
    dummy_pos_y = torch.randint(0, 128, (N,))

    # 构建简单的KNN图用于测试
    from graph_utils_gat import build_knn_graph
    import numpy as np
    coords = np.random.rand(N, 2) * 1000
    edge_index = build_knn_graph(coords, k=6)

    with torch.no_grad():
        output = model(dummy_tokens, dummy_pos_x, dummy_pos_y, edge_index)
    print(f"  输入: {N} patches, tokens [{num_tokens}, 1536], edge_index {edge_index.shape}")
    print(f"  输出: {output.shape}")
    print("  前向传播测试通过!")
