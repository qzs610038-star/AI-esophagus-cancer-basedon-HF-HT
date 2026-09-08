"""Truncated pathway relation support sets and loss. Training and precheck share this module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from errors import ConfigError

FAR_SCOPES = ("all", "same_patient")


@dataclass
class ThresholdResult:
    q10: float
    q50: float
    pair_indices: np.ndarray
    distances: np.ndarray
    n_points: int
    n_pairs: int
    seed: int
    quantile_method: str
    near_quantile: float
    far_quantile: float
    distance_mean: float
    distance_std: float
    distance_min: float
    distance_max: float


@dataclass
class BatchSupport:
    near_indices: list[list[int]]
    far_indices: list[list[int]]
    valid: np.ndarray
    distances: np.ndarray
    n_invalid_no_near: int = 0
    n_invalid_no_far: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid))


def mean_squared_pathway_distance(y_i: np.ndarray, y_j: np.ndarray) -> np.ndarray:
    left = np.asarray(y_i, dtype=np.float64)
    right = np.asarray(y_j, dtype=np.float64)
    return np.mean((left - right) ** 2, axis=-1)


def pairwise_distance_matrix(labels_z: np.ndarray) -> np.ndarray:
    y = np.asarray(labels_z, dtype=np.float64)
    diff = y[:, None, :] - y[None, :, :]
    return np.mean(diff * diff, axis=-1)


def sample_threshold_pairs(n_points: int, n_pairs: int, seed: int) -> np.ndarray:
    if n_points < 2:
        raise ValueError("阈值点对需要至少2个训练点")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    index_i = rng.integers(0, n_points, size=n_pairs, dtype=np.int64)
    offset = rng.integers(0, n_points - 1, size=n_pairs, dtype=np.int64)
    index_j = offset + (offset >= index_i).astype(np.int64)
    return np.stack([index_i, index_j], axis=1)


def compute_distance_thresholds(
    train_labels_z: np.ndarray,
    *,
    n_pairs: int = 200000,
    seed: int = 20260907,
    near_quantile: float = 0.1,
    far_quantile: float = 0.5,
    quantile_method: str = "linear",
    pair_indices: np.ndarray | None = None,
) -> ThresholdResult:
    labels = np.asarray(train_labels_z, dtype=np.float64)
    n_points = int(labels.shape[0])
    if pair_indices is None:
        pair_indices = sample_threshold_pairs(n_points, n_pairs, seed)
    else:
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        n_pairs = int(pair_indices.shape[0])
    distances = mean_squared_pathway_distance(labels[pair_indices[:, 0]], labels[pair_indices[:, 1]])
    q10 = float(np.quantile(distances, near_quantile, method=quantile_method))
    q50 = float(np.quantile(distances, far_quantile, method=quantile_method))
    return ThresholdResult(
        q10=q10,
        q50=q50,
        pair_indices=pair_indices,
        distances=distances,
        n_points=n_points,
        n_pairs=n_pairs,
        seed=int(seed),
        quantile_method=quantile_method,
        near_quantile=float(near_quantile),
        far_quantile=float(far_quantile),
        distance_mean=float(np.mean(distances)),
        distance_std=float(np.std(distances, ddof=1)) if n_pairs > 1 else 0.0,
        distance_min=float(np.min(distances)),
        distance_max=float(np.max(distances)),
    )


def _far_generator(run_seed: int, epoch: int, batch_index: int, center_fixed_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [int(run_seed), int(epoch), int(batch_index), int(center_fixed_index)]
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _sample_far_indices(
    candidate_local: list[int],
    max_far: int,
    *,
    run_seed: int,
    epoch: int,
    batch_index: int,
    center_fixed_index: int,
) -> list[int]:
    if not candidate_local:
        return []
    if len(candidate_local) <= max_far:
        return list(candidate_local)
    rng = _far_generator(run_seed, epoch, batch_index, center_fixed_index)
    chosen = rng.choice(len(candidate_local), size=int(max_far), replace=False)
    return [candidate_local[int(k)] for k in chosen]


def build_batch_support(
    labels_z: np.ndarray,
    *,
    patient_ids: Sequence[str],
    fixed_indices: Sequence[int],
    q10: float,
    q50: float,
    max_near: int = 8,
    max_far: int = 32,
    far_patient_scope: str = "all",
    run_seed: int,
    epoch: int,
    batch_index: int,
) -> BatchSupport:
    if far_patient_scope not in FAR_SCOPES:
        raise ConfigError(f"未实现 relation.far_patient_scope={far_patient_scope!r}")
    y = np.asarray(labels_z, dtype=np.float64)
    n = int(y.shape[0])
    distances = pairwise_distance_matrix(y)
    patients = [str(v) for v in patient_ids]
    fixed = np.asarray(list(fixed_indices), dtype=np.int64)
    if fixed.shape[0] != n or len(patients) != n:
        raise ValueError("labels/patient_ids/fixed_indices 长度必须一致")

    near_indices: list[list[int]] = []
    far_indices: list[list[int]] = []
    valid = np.zeros(n, dtype=bool)
    n_no_near = 0
    n_no_far = 0
    for i in range(n):
        near_cands: list[tuple[float, int, int]] = []
        for j in range(n):
            if j == i:
                continue
            dist = float(distances[i, j])
            if dist <= q10:
                near_cands.append((dist, int(fixed[j]), j))
        near_cands.sort()
        near = [item[2] for item in near_cands[: int(max_near)]]
        near_set = set(near)

        far_cands: list[tuple[int, int]] = []
        for j in range(n):
            if j == i or j in near_set:
                continue
            dist = float(distances[i, j])
            if dist >= q50 and dist > q10:
                if far_patient_scope == "same_patient" and patients[j] != patients[i]:
                    continue
                far_cands.append((int(fixed[j]), j))
        far_cands.sort()
        far_local = [item[1] for item in far_cands]
        far = _sample_far_indices(
            far_local,
            int(max_far),
            run_seed=run_seed,
            epoch=epoch,
            batch_index=batch_index,
            center_fixed_index=int(fixed[i]),
        )
        near_indices.append(near)
        far_indices.append(far)
        if not near:
            n_no_near += 1
        if not far:
            n_no_far += 1
        valid[i] = bool(near) and bool(far)
    return BatchSupport(
        near_indices=near_indices,
        far_indices=far_indices,
        valid=valid,
        distances=distances,
        n_invalid_no_near=n_no_near,
        n_invalid_no_far=n_no_far,
    )


def _connected_zero(z: torch.Tensor) -> torch.Tensor:
    return z.reshape(-1)[0] * 0.0


def teacher_q_from_distances(near_distances: torch.Tensor, tau_y: float) -> torch.Tensor:
    return torch.softmax(-near_distances / float(tau_y), dim=0)


def relation_loss(
    z: torch.Tensor,
    labels_z: torch.Tensor,
    support: BatchSupport,
    *,
    tau_y: float,
    tau_z: float,
) -> tuple[torch.Tensor, dict]:
    """Soft CE over truncated support. Q and labels have no grad; P comes from z and flows to h."""
    if z.ndim != 2:
        raise ValueError("z 必须是 [B, relation_dim]")
    labels = labels_z.detach()
    batch_size = int(z.shape[0])
    terms: list[torch.Tensor] = []
    q_entropies: list[float] = []
    q_norm_entropies: list[float] = []
    n_single_near = 0
    for i in range(batch_size):
        if not bool(support.valid[i]):
            continue
        near = list(support.near_indices[i])
        far = list(support.far_indices[i])
        support_idx = near + far
        y_i = labels[i]
        y_near = labels[near]
        d_near = torch.mean((y_i - y_near) ** 2, dim=-1)
        q_near = teacher_q_from_distances(d_near, tau_y)
        q = z.new_zeros(len(support_idx))
        q[: len(near)] = q_near
        q = q.detach()
        z_support = z[support_idx]
        logits = torch.sum(z_support * z[i], dim=-1) / float(tau_z)
        log_p = torch.log_softmax(logits, dim=0)
        terms.append(-(q * log_p).sum())
        q_np = q_near.detach().cpu().numpy().astype(np.float64)
        entropy = float(-np.sum(q_np * np.log(np.clip(q_np, 1e-12, 1.0))))
        q_entropies.append(entropy)
        if len(near) <= 1:
            n_single_near += 1
        else:
            q_norm_entropies.append(entropy / float(np.log(len(near))))
    stats = {
        "n_centers": batch_size,
        "n_valid": int(support.n_valid),
        "n_invalid": batch_size - int(support.n_valid),
        "n_invalid_no_near": int(support.n_invalid_no_near),
        "n_invalid_no_far": int(support.n_invalid_no_far),
        "n_single_near": n_single_near,
        "sum_ce": 0.0,
        "denominator": int(support.n_valid),
        "mean_q_entropy": float(np.nanmean(q_entropies)) if q_entropies else float("nan"),
        "mean_q_normalized_entropy": float(np.nanmean(q_norm_entropies)) if q_norm_entropies else float("nan"),
    }
    if not terms:
        stats["loss"] = 0.0
        return _connected_zero(z), stats
    stacked = torch.stack(terms)
    loss = stacked.mean()
    stats["sum_ce"] = float(stacked.detach().sum().cpu())
    stats["loss"] = float(loss.detach().cpu())
    return loss, stats


def qp_tables(
    z: torch.Tensor,
    labels_z: torch.Tensor,
    support: BatchSupport,
    *,
    tau_y: float,
    tau_z: float,
) -> list[dict]:
    """Q/P tables using the same teacher_q and softmax path as relation_loss."""
    labels = labels_z.detach()
    rows: list[dict] = []
    for i in range(int(z.shape[0])):
        near = list(support.near_indices[i])
        far = list(support.far_indices[i])
        row = {
            "center": i,
            "valid": bool(support.valid[i]),
            "near": near,
            "far": far,
            "Q": None,
            "P": None,
        }
        if not bool(support.valid[i]):
            rows.append(row)
            continue
        support_idx = near + far
        y_i = labels[i]
        y_near = labels[near]
        d_near = torch.mean((y_i - y_near) ** 2, dim=-1)
        q_near = teacher_q_from_distances(d_near, tau_y)
        q = z.new_zeros(len(support_idx))
        q[: len(near)] = q_near
        logits = torch.sum(z[support_idx] * z[i], dim=-1) / float(tau_z)
        p = torch.softmax(logits, dim=0)
        row["support"] = support_idx
        row["Q"] = q.detach().cpu().tolist()
        row["P"] = p.detach().cpu().tolist()
        rows.append(row)
    return rows


def support_from_config(
    labels_z: np.ndarray,
    *,
    patient_ids: Sequence[str],
    fixed_indices: Sequence[int],
    config: dict,
    thresholds: ThresholdResult,
    run_seed: int,
    epoch: int,
    batch_index: int,
) -> BatchSupport:
    rel = config["relation"]
    return build_batch_support(
        labels_z,
        patient_ids=patient_ids,
        fixed_indices=fixed_indices,
        q10=thresholds.q10,
        q50=thresholds.q50,
        max_near=int(rel["max_near"]),
        max_far=int(rel["max_far"]),
        far_patient_scope=str(rel["far_patient_scope"]),
        run_seed=int(run_seed),
        epoch=int(epoch),
        batch_index=int(batch_index),
    )
