"""Fixed one-hop spatial graph built from frozen UNI2-h morphology features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import OnlineRecord


@dataclass(frozen=True)
class SpatialGraph:
    neighbor_index: np.ndarray
    neighbor_weight: np.ndarray
    neighbor_mask: np.ndarray
    patient_ids: tuple[str, ...]
    degree: np.ndarray

    def spatial_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute sum_j a_ij (h_j - h_i) without learning graph weights."""
        if hidden.ndim != 2 or hidden.shape[0] != self.neighbor_index.shape[0]:
            raise ValueError("hidden must be [graph_points, hidden_dim]")
        index = torch.as_tensor(self.neighbor_index, device=hidden.device, dtype=torch.long)
        mask = torch.as_tensor(self.neighbor_mask, device=hidden.device)
        weights = torch.as_tensor(
            self.neighbor_weight, device=hidden.device, dtype=hidden.dtype
        )
        safe_index = index.clamp_min(0)
        neighbors = hidden[safe_index]
        differences = neighbors - hidden[:, None, :]
        return (differences * weights[..., None] * mask[..., None]).sum(dim=1)


def _normalise(vectors: np.ndarray, epsilon: float) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, epsilon)


def build_spatial_graph(
    records: Sequence[OnlineRecord],
    frozen_features: np.ndarray,
    *,
    native_steps: Mapping[str, float],
    radius_in_native_steps: float = 1.5,
    max_neighbors: int = 8,
    distance_sigma: float = 1.0,
    image_temperature: float = 0.2,
    self_raw_weight: float = 1.0,
    epsilon: float = 1e-12,
) -> SpatialGraph:
    """Build a directed fixed graph; candidates never cross a patient."""
    if not records:
        raise ValueError("cannot build an empty spatial graph")
    features = np.asarray(frozen_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(records):
        raise ValueError("frozen_features must be [len(records), feature_dim]")
    if not np.isfinite(features).all():
        raise ValueError("frozen morphology features contain non-finite values")
    if radius_in_native_steps <= 0 or max_neighbors < 1 or distance_sigma <= 0:
        raise ValueError("invalid fixed graph geometry")
    if image_temperature <= 0 or self_raw_weight <= 0:
        raise ValueError("invalid morphology/self weights")
    normalised = _normalise(features, epsilon)
    neighbor_index = np.full((len(records), max_neighbors), -1, dtype=np.int64)
    neighbor_weight = np.zeros((len(records), max_neighbors), dtype=np.float32)
    neighbor_mask = np.zeros((len(records), max_neighbors), dtype=bool)
    degree = np.zeros(len(records), dtype=np.int64)

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.patient, []).append(index)
    for patient, members in groups.items():
        if patient not in native_steps:
            raise KeyError(f"missing native spatial step for patient={patient}")
        step = float(native_steps[patient])
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"invalid native step for patient={patient}: {step}")
        radius = radius_in_native_steps * step
        coords = np.asarray([(records[index].x, records[index].y) for index in members], dtype=np.float64)
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for local_i, global_i in enumerate(members):
            candidates = [
                (float(distances[local_i, local_j]), records[global_j].identity, global_j)
                for local_j, global_j in enumerate(members)
                if local_j != local_i and distances[local_i, local_j] <= radius
            ]
            candidates.sort(key=lambda item: (item[0], item[1]))
            chosen = candidates[:max_neighbors]
            raw_weights: list[float] = []
            for distance, _, global_j in chosen:
                rho = distance / step
                geometry = np.exp(-(rho * rho) / (2.0 * distance_sigma * distance_sigma))
                cosine = float(normalised[global_i] @ normalised[global_j])
                morphology = np.exp((cosine - 1.0) / image_temperature)
                raw_weights.append(float(geometry * morphology))
            denominator = self_raw_weight + float(np.sum(raw_weights))
            for slot, ((_, _, global_j), raw_weight) in enumerate(zip(chosen, raw_weights)):
                if records[global_i].patient != records[global_j].patient:
                    raise AssertionError("spatial graph crossed a patient boundary")
                neighbor_index[global_i, slot] = global_j
                neighbor_weight[global_i, slot] = raw_weight / denominator
                neighbor_mask[global_i, slot] = True
            degree[global_i] = len(chosen)
    return SpatialGraph(
        neighbor_index=neighbor_index,
        neighbor_weight=neighbor_weight,
        neighbor_mask=neighbor_mask,
        patient_ids=tuple(record.patient for record in records),
        degree=degree,
    )
