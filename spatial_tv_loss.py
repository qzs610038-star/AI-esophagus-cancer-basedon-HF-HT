"""
空间 Total Variation (TV) 正则化损失
====================================
惩罚空间相邻 patch 之间预测值的剧烈跳变，利用"相邻组织区域基因表达连续"
这一生物学先验作为平滑约束。

用法:
    tv_loss_fn = SpatialTVLoss(k=6, mode='l1')
    tv = tv_loss_fn(predictions, pos_x, pos_y)
    loss = huber_loss + tv_weight * tv

不依赖 torch_geometric，纯 PyTorch 实现 batch 内 KNN 建图。
"""

import torch
import torch.nn as nn


class SpatialTVLoss(nn.Module):
    """Batch 内空间 Total Variation 平滑损失。

    对 batch 内每个 patch，找到其 k 个空间最近邻，惩罚预测值差异。
    使用 L1 或 L2 范数。

    Args:
        k: KNN 邻居数，默认 6
        mode: 'l1' (|Δpred|) 或 'l2' ((Δpred)²)，默认 'l1'
        min_batch_size: 低于此 batch size 跳过计算（返回零），默认 4
    """

    def __init__(self, k: int = 6, mode: str = "l1", min_batch_size: int = 4):
        super().__init__()
        self.k = k
        self.mode = mode
        self.min_batch_size = min_batch_size

    def forward(self, pred: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """计算 batch 内空间 TV 损失。

        Args:
            pred: (B, n_targets) 模型预测值
            pos_x: (B,) X 坐标索引 (long, 0..n_pos-1)
            pos_y: (B,) Y 坐标索引 (long, 0..n_pos-1)

        Returns:
            scalar TV loss，若 batch 太小则返回零
        """
        B = pred.size(0)
        if B < self.min_batch_size:
            return pred.new_zeros(())

        # 构建空间坐标矩阵 (B, 2)，float 类型
        coords = torch.stack([
            pos_x.float(),
            pos_y.float()
        ], dim=1)  # (B, 2)

        # 计算 pairwise 欧氏距离 (B, B)
        # ||a-b||² = ||a||² + ||b||² - 2abᵀ
        sq_norm = (coords ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        dist_sq = sq_norm + sq_norm.T - 2 * (coords @ coords.T)  # (B, B)
        dist_sq = dist_sq.clamp(min=0)  # 数值稳定性

        # 自适应 k：不超过 batch_size - 1
        actual_k = min(self.k, B - 1)
        if actual_k < 1:
            return pred.new_zeros(())

        # 取每个节点的 k 个最近邻（排除自身 distance≈0）
        # 自身距离为 0，取 top-k+1 后排除自身
        _, indices = dist_sq.topk(actual_k + 1, dim=1, largest=False)  # (B, k+1)

        # 排除自身（通常在第 0 位）
        # 安全做法：构建 mask 排除自身
        self_mask = indices != torch.arange(B, device=indices.device).unsqueeze(1)  # (B, k+1)
        # 取前 k 个非自身的邻居
        neighbor_indices_list = []
        for i in range(B):
            row = indices[i]
            mask = self_mask[i]
            valid = row[mask][:actual_k]
            neighbor_indices_list.append(valid)

        # 构建边索引 (2, E)，E = B * actual_k
        src = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, actual_k).reshape(-1)
        dst = torch.stack(neighbor_indices_list, dim=0).reshape(-1)  # (B * actual_k,)

        # 计算成对预测差异
        diff = pred[src] - pred[dst]  # (E, n_targets)

        if self.mode == "l1":
            loss = diff.abs().mean()
        else:  # l2
            loss = (diff ** 2).mean()

        return loss


class SpatialLaplacianLoss(nn.Module):
    """空间拉普拉斯平滑损失（备选方案）。

    与 TV Loss 类似，但使用 L2 范数并除以邻居数，对 outlier 更敏感。
    适用于预测值范围较小的场景。

    Args:
        k: KNN 邻居数
        min_batch_size: 最低 batch size
    """

    def __init__(self, k: int = 6, min_batch_size: int = 4):
        super().__init__()
        self.k = k
        self.min_batch_size = min_batch_size

    def forward(self, pred: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        B = pred.size(0)
        if B < self.min_batch_size:
            return pred.new_zeros(())

        coords = torch.stack([pos_x.float(), pos_y.float()], dim=1)
        sq_norm = (coords ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.T - 2 * (coords @ coords.T)
        dist_sq = dist_sq.clamp(min=0)

        actual_k = min(self.k, B - 1)
        if actual_k < 1:
            return pred.new_zeros(())

        _, indices = dist_sq.topk(actual_k + 1, dim=1, largest=False)
        self_mask = indices != torch.arange(B, device=indices.device).unsqueeze(1)

        neighbor_indices_list = []
        for i in range(B):
            row = indices[i]
            mask = self_mask[i]
            valid = row[mask][:actual_k]
            neighbor_indices_list.append(valid)

        src = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, actual_k).reshape(-1)
        dst = torch.stack(neighbor_indices_list, dim=0).reshape(-1)

        # 每个节点对其邻居均值的偏差
        pred_neighbors = pred[dst].view(B, actual_k, -1)  # (B, k, n_targets)
        pred_center = pred.unsqueeze(1)                    # (B, 1, n_targets)
        diff = pred_center - pred_neighbors                # (B, k, n_targets)
        loss = (diff ** 2).mean()

        return loss


# ── 便捷工厂函数 ──────────────────────────────────────────────────────────

def create_tv_loss(mode: str = "l1", k: int = 6) -> nn.Module:
    """创建 TV Loss 实例。

    Args:
        mode: 'l1' (推荐), 'l2', 或 'laplacian'
        k: KNN 邻居数

    Returns:
        SpatialTVLoss 或 SpatialLaplacianLoss 实例
    """
    if mode == "laplacian":
        return SpatialLaplacianLoss(k=k)
    return SpatialTVLoss(k=k, mode=mode)
