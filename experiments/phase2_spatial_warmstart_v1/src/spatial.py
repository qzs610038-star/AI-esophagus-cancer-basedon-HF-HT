"""Fixed one-hop graph for within-patient, within-partition spatial residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SpatialPoint:
    """Identity and native coordinate required to construct the fixed graph."""

    identity: str
    patient: str
    partition: str
    x: float
    y: float


@dataclass(frozen=True)
class SpatialGraph:
    """Padded directed neighbor table with explicit self-normalisation weight."""

    neighbor_index: np.ndarray
    neighbor_weight: np.ndarray
    neighbor_mask: np.ndarray
    self_weight: np.ndarray
    degree: np.ndarray
    point_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    neighbor_distance_native_steps: np.ndarray
    neighbor_cosine_similarity: np.ndarray
    radius_in_native_steps: float
    max_neighbors: int
    distance_sigma: float
    image_temperature: float
    self_raw_weight: float

    def __post_init__(self) -> None:
        count = len(self.point_ids)
        expected_matrix = (count, int(self.max_neighbors))
        for name in (
            "neighbor_index",
            "neighbor_weight",
            "neighbor_mask",
            "neighbor_distance_native_steps",
            "neighbor_cosine_similarity",
        ):
            if np.asarray(getattr(self, name)).shape != expected_matrix:
                raise ValueError(f"{name} must have shape {expected_matrix}")
        if np.asarray(self.self_weight).shape != (count,) or np.asarray(self.degree).shape != (count,):
            raise ValueError("self_weight and degree must have shape [points]")
        if len(self.patient_ids) != count or len(self.partitions) != count:
            raise ValueError("graph identity metadata length mismatch")

        index = np.asarray(self.neighbor_index)
        weights = np.asarray(self.neighbor_weight)
        mask = np.asarray(self.neighbor_mask, dtype=bool)
        self_weights = np.asarray(self.self_weight)
        if not np.isfinite(weights).all() or not np.isfinite(self_weights).all():
            raise ValueError("graph weights contain non-finite values")
        if (weights < 0).any() or (self_weights < 0).any():
            raise ValueError("graph weights must be non-negative")
        if np.any(index[mask] < 0) or np.any(index[mask] >= count):
            raise ValueError("graph contains an out-of-range neighbor index")
        if np.any(index[~mask] != -1) or np.any(weights[~mask] != 0):
            raise ValueError("unused neighbor slots must use index=-1 and weight=0")
        if not np.array_equal(np.asarray(self.degree), mask.sum(axis=1)):
            raise ValueError("degree does not match neighbor_mask")
        if not np.allclose(self_weights + weights.sum(axis=1), 1.0, atol=1e-6, rtol=0):
            raise ValueError("self and neighbor weights must sum to one")
        for source, target in zip(*np.nonzero(mask)):
            neighbor = int(index[source, target])
            if self.patient_ids[source] != self.patient_ids[neighbor]:
                raise ValueError("spatial graph crosses a patient boundary")
            if self.partitions[source] != self.partitions[neighbor]:
                raise ValueError("spatial graph crosses a partition boundary")

    def spatial_delta(self, hidden: Tensor) -> Tensor:
        """Return ``sum_j a_ij (h_j - h_i)`` on the tensor's device."""

        if hidden.ndim != 2 or hidden.shape[0] != len(self.point_ids):
            raise ValueError("hidden must have shape [graph_points, hidden_dim]")
        index = torch.as_tensor(self.neighbor_index, dtype=torch.long, device=hidden.device)
        mask = torch.as_tensor(self.neighbor_mask, dtype=torch.bool, device=hidden.device)
        weight = torch.as_tensor(self.neighbor_weight, dtype=hidden.dtype, device=hidden.device)
        safe_index = index.clamp_min(0)
        difference = hidden[safe_index] - hidden[:, None, :]
        return (difference * weight[..., None] * mask[..., None]).sum(dim=1)

    def smooth_predictions(self, prediction: Tensor, *, beta: float = 1.0) -> Tensor:
        """Apply the fixed diagnostic ``p + beta * sum_j a_ij(p_j-p_i)``."""

        return prediction + float(beta) * self.spatial_delta(prediction)

    def edge_table(self) -> dict[str, np.ndarray]:
        """Export valid directed edges as aligned, serialization-ready columns."""

        sources, slots = np.nonzero(self.neighbor_mask)
        targets = self.neighbor_index[sources, slots].astype(np.int64, copy=False)
        return {
            "source_index": sources.astype(np.int64, copy=False),
            "target_index": targets,
            "source_id": np.asarray([self.point_ids[index] for index in sources], dtype=str),
            "target_id": np.asarray([self.point_ids[index] for index in targets], dtype=str),
            "patient_id": np.asarray([self.patient_ids[index] for index in sources], dtype=str),
            "partition": np.asarray([self.partitions[index] for index in sources], dtype=str),
            "weight": self.neighbor_weight[sources, slots].astype(np.float32, copy=False),
            "source_self_weight": self.self_weight[sources].astype(np.float32, copy=False),
            "distance_native_steps": self.neighbor_distance_native_steps[sources, slots].astype(
                np.float32, copy=False
            ),
            "cosine_similarity": self.neighbor_cosine_similarity[sources, slots].astype(
                np.float32, copy=False
            ),
        }


def _normalise_rows(features: np.ndarray, epsilon: float) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, epsilon)


def build_spatial_graph(
    points: Sequence[SpatialPoint],
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
    """Build a deterministic graph without labels or cross-split candidates."""

    if not points:
        raise ValueError("cannot build an empty spatial graph")
    if radius_in_native_steps <= 0 or max_neighbors < 1 or distance_sigma <= 0:
        raise ValueError("invalid graph geometry parameters")
    if image_temperature <= 0 or self_raw_weight <= 0 or epsilon <= 0:
        raise ValueError("invalid graph weighting parameters")

    features = np.asarray(frozen_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(points):
        raise ValueError("frozen_features must have shape [len(points), feature_dim]")
    if features.shape[1] < 1 or not np.isfinite(features).all():
        raise ValueError("frozen_features must be finite and non-empty")
    normalised = _normalise_rows(features, float(epsilon))

    seen: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], list[int]] = {}
    for index, point in enumerate(points):
        key = (str(point.patient), str(point.partition), str(point.identity))
        if key in seen:
            raise ValueError(f"duplicate point identity within patient/partition: {key}")
        seen.add(key)
        if not np.isfinite([float(point.x), float(point.y)]).all():
            raise ValueError(f"non-finite coordinate for point={point.identity}")
        groups.setdefault((str(point.patient), str(point.partition)), []).append(index)

    shape = (len(points), int(max_neighbors))
    neighbor_index = np.full(shape, -1, dtype=np.int64)
    neighbor_weight = np.zeros(shape, dtype=np.float32)
    neighbor_mask = np.zeros(shape, dtype=bool)
    neighbor_distance = np.zeros(shape, dtype=np.float32)
    neighbor_cosine = np.zeros(shape, dtype=np.float32)
    self_weight = np.ones(len(points), dtype=np.float32)
    degree = np.zeros(len(points), dtype=np.int64)

    for (patient, _partition), members in groups.items():
        if patient not in native_steps:
            raise KeyError(f"missing native spatial step for patient={patient}")
        step = float(native_steps[patient])
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"invalid native spatial step for patient={patient}: {step}")
        radius = float(radius_in_native_steps) * step
        coordinates = np.asarray([(points[i].x, points[i].y) for i in members], dtype=np.float64)
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)

        for local_source, source in enumerate(members):
            candidates: list[tuple[float, str, int]] = []
            for local_target, target in enumerate(members):
                distance = float(distances[local_source, local_target])
                if source != target and distance <= radius:
                    candidates.append((distance, str(points[target].identity), target))
            candidates.sort(key=lambda item: (item[0], item[1]))
            chosen = candidates[: int(max_neighbors)]

            raw_weights: list[float] = []
            cosines: list[float] = []
            for distance, _identity, target in chosen:
                rho = distance / step
                geometry = np.exp(-(rho * rho) / (2.0 * distance_sigma * distance_sigma))
                cosine = float(np.clip(normalised[source] @ normalised[target], -1.0, 1.0))
                morphology = np.exp((cosine - 1.0) / image_temperature)
                raw_weights.append(float(geometry * morphology))
                cosines.append(cosine)

            denominator = float(self_raw_weight + np.sum(raw_weights, dtype=np.float64))
            self_weight[source] = float(self_raw_weight / denominator)
            for slot, ((distance, _identity, target), raw, cosine) in enumerate(
                zip(chosen, raw_weights, cosines)
            ):
                neighbor_index[source, slot] = target
                neighbor_weight[source, slot] = raw / denominator
                neighbor_mask[source, slot] = True
                neighbor_distance[source, slot] = distance / step
                neighbor_cosine[source, slot] = cosine
            degree[source] = len(chosen)

    return SpatialGraph(
        neighbor_index=neighbor_index,
        neighbor_weight=neighbor_weight,
        neighbor_mask=neighbor_mask,
        self_weight=self_weight,
        degree=degree,
        point_ids=tuple(str(point.identity) for point in points),
        patient_ids=tuple(str(point.patient) for point in points),
        partitions=tuple(str(point.partition) for point in points),
        neighbor_distance_native_steps=neighbor_distance,
        neighbor_cosine_similarity=neighbor_cosine,
        radius_in_native_steps=float(radius_in_native_steps),
        max_neighbors=int(max_neighbors),
        distance_sigma=float(distance_sigma),
        image_temperature=float(image_temperature),
        self_raw_weight=float(self_raw_weight),
    )
