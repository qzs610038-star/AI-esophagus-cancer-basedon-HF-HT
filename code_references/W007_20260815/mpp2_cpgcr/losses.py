"""W007 四臂的绝对误差与完整候选对比损失。"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def full_candidate_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """仅排除 padding/缺失项；不按患者身份缩小候选集合。"""

    valid = valid_mask.to(dtype=torch.bool)
    if valid.ndim != 1:
        raise ValueError("valid_mask must be one-dimensional")
    return valid[:, None] & valid[None, :]


def patient_balanced_z_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    patients: Sequence[str],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """先按患者求 spot×通路 MSE，再对患者等权平均。"""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share [batch, pathway] shape")
    if len(patients) != prediction.shape[0]:
        raise ValueError("patients must align with prediction rows")
    valid = (
        torch.ones(prediction.shape[0], dtype=torch.bool, device=prediction.device)
        if valid_mask is None
        else valid_mask.to(device=prediction.device, dtype=torch.bool)
    )
    row_mse = (prediction - target).square().mean(dim=1)
    patient_losses = []
    for patient in dict.fromkeys(patients):
        patient_mask = torch.tensor(
            [item == patient for item in patients],
            device=prediction.device,
            dtype=torch.bool,
        ) & valid
        if patient_mask.any():
            patient_losses.append(row_mse[patient_mask].mean())
    if not patient_losses:
        raise ValueError("patient-balanced loss has no valid patient rows")
    return torch.stack(patient_losses).mean()


def _valid_views(
    image_embedding: torch.Tensor,
    pathway_embedding: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image_embedding.shape != pathway_embedding.shape or image_embedding.ndim != 2:
        raise ValueError("image and pathway embeddings must share [batch, dim] shape")
    valid = (
        torch.ones(image_embedding.shape[0], dtype=torch.bool, device=image_embedding.device)
        if valid_mask is None
        else valid_mask.to(device=image_embedding.device, dtype=torch.bool)
    )
    if valid.sum() < 2:
        raise ValueError("contrastive loss requires at least two valid candidates")
    return image_embedding[valid], pathway_embedding[valid], valid


def bidirectional_hard_contrastive_loss(
    image_embedding: torch.Tensor,
    pathway_embedding: torch.Tensor,
    *,
    tau_z: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """同 spot 为唯一正样本、全部其他有效 spot 为候选的双向 InfoNCE。"""

    image, pathway, _ = _valid_views(image_embedding, pathway_embedding, valid_mask)
    image = F.normalize(image, dim=1)
    pathway = F.normalize(pathway, dim=1)
    logits = image @ pathway.T / float(tau_z)
    identity = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, identity)
        + F.cross_entropy(logits.T, identity)
    )


def continuous_soft_targets(
    targets: torch.Tensor,
    *,
    tau_y: float,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分别按两个方向逐行归一化连续通路几何目标。"""

    if targets.ndim != 2:
        raise ValueError("targets must have [batch, pathway] shape")
    valid = (
        torch.ones(targets.shape[0], dtype=torch.bool, device=targets.device)
        if valid_mask is None
        else valid_mask.to(device=targets.device, dtype=torch.bool)
    )
    selected = targets[valid]
    distances = (selected[:, None, :] - selected[None, :, :]).square().mean(dim=2)
    image_to_pathway = F.softmax(-distances / float(tau_y), dim=1)
    pathway_to_image = F.softmax(-distances.T / float(tau_y), dim=1)
    full_ip = targets.new_zeros((targets.shape[0], targets.shape[0]))
    full_pi = targets.new_zeros((targets.shape[0], targets.shape[0]))
    valid_indices = torch.where(valid)[0]
    full_ip[valid_indices[:, None], valid_indices[None, :]] = image_to_pathway
    full_pi[valid_indices[:, None], valid_indices[None, :]] = pathway_to_image
    return full_ip, full_pi


def bidirectional_soft_contrastive_loss(
    image_embedding: torch.Tensor,
    pathway_embedding: torch.Tensor,
    targets: torch.Tensor,
    *,
    tau_z: float,
    tau_y: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """完整候选集合上的双向软目标交叉熵。"""

    image, pathway, valid = _valid_views(image_embedding, pathway_embedding, valid_mask)
    if targets.shape[0] != valid.shape[0]:
        raise ValueError("targets must align with embedding rows")
    image = F.normalize(image, dim=1)
    pathway = F.normalize(pathway, dim=1)
    logits = image @ pathway.T / float(tau_z)
    q_ip_full, q_pi_full = continuous_soft_targets(
        targets,
        tau_y=tau_y,
        valid_mask=valid,
    )
    q_ip = q_ip_full[valid][:, valid]
    q_pi = q_pi_full[valid][:, valid]
    loss_ip = -(q_ip * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_pi = -(q_pi * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_ip + loss_pi)


def fit_train_fold_tau_y(
    targets: torch.Tensor,
    patients: Sequence[str],
    *,
    split_name: str,
    seed: int = 42,
    max_pairs: int = 100_000,
) -> float:
    """仅从训练折、15 个患者对近似等额抽样拟合距离中位数。"""

    if split_name != "train":
        raise ValueError("tau_y may only be fitted from the training split")
    if targets.ndim != 2 or len(patients) != targets.shape[0]:
        raise ValueError("targets and patients are misaligned")
    unique_patients = list(dict.fromkeys(patients))
    patient_pairs = list(combinations(unique_patients, 2))
    if len(patient_pairs) != 15:
        raise ValueError("tau_y contract requires exactly six training patients")
    patient_indices = {
        patient: np.flatnonzero(np.asarray(patients, dtype=object) == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    base_quota, remainder = divmod(max_pairs, len(patient_pairs))
    sampled_distances: list[torch.Tensor] = []
    for pair_index, (left, right) in enumerate(patient_pairs):
        left_indices = patient_indices[left]
        right_indices = patient_indices[right]
        quota = base_quota + (1 if pair_index < remainder else 0)
        total_pairs = len(left_indices) * len(right_indices)
        take = min(quota, total_pairs)
        if take == total_pairs:
            left_take = np.repeat(left_indices, len(right_indices))
            right_take = np.tile(right_indices, len(left_indices))
        else:
            flat = rng.choice(total_pairs, size=take, replace=False)
            left_take = left_indices[flat // len(right_indices)]
            right_take = right_indices[flat % len(right_indices)]
        left_tensor = targets[torch.as_tensor(left_take, device=targets.device)]
        right_tensor = targets[torch.as_tensor(right_take, device=targets.device)]
        sampled_distances.append((left_tensor - right_tensor).square().mean(dim=1).cpu())
    distances = torch.cat(sampled_distances)
    tau_y = float(distances.median().item())
    if not np.isfinite(tau_y) or tau_y <= 0:
        raise ValueError("training-fold tau_y must be finite and positive")
    return tau_y


class ArmLoss(nn.Module):
    """三臂统一入口，差异仅为对比目标形式。"""

    def __init__(self, arm: str, *, contrastive_weight: float, tau_z: float, tau_y: float | None):
        super().__init__()
        if arm not in {"RCC", "HCR", "CPGCR"}:
            raise ValueError("ArmLoss supports RCC, HCR and CPGCR")
        self.arm = arm
        self.contrastive_weight = float(contrastive_weight)
        self.tau_z = float(tau_z)
        self.tau_y = tau_y

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        patients: Sequence[str],
        image_embedding: torch.Tensor,
        pathway_embedding: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        absolute = patient_balanced_z_mse(prediction, target, patients, valid_mask)
        contrastive = absolute.new_zeros(())
        if self.arm == "HCR":
            if pathway_embedding is None:
                raise ValueError("HCR requires pathway embeddings")
            contrastive = bidirectional_hard_contrastive_loss(
                image_embedding,
                pathway_embedding,
                tau_z=self.tau_z,
                valid_mask=valid_mask,
            )
        elif self.arm == "CPGCR":
            if pathway_embedding is None or self.tau_y is None:
                raise ValueError("CPGCR requires pathway embeddings and train-fold tau_y")
            contrastive = bidirectional_soft_contrastive_loss(
                image_embedding,
                pathway_embedding,
                target,
                tau_z=self.tau_z,
                tau_y=self.tau_y,
                valid_mask=valid_mask,
            )
        total = absolute + self.contrastive_weight * contrastive
        return total, {"absolute": absolute, "contrastive": contrastive, "total": total}
