"""Pure, fold-safe numerical transforms for the Phase 3 transfer pipeline.

These functions operate only on arrays supplied by the caller.  In particular,
``GalleryResidualCalibrator.fit`` must receive training-fold gallery/targets;
validation and test arrays are transform-only inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # Torch is optional for import, but supported when installed.
    import torch
except ImportError:  # pragma: no cover - exercised only in numpy-only installs
    torch = None


def _is_torch(value: object) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _as_float_matrix(value: object, name: str) -> np.ndarray:
    array = value.detach().cpu().numpy() if _is_torch(value) else np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return array.astype(np.float64, copy=False)


def _restore(array: np.ndarray, template: object) -> object:
    if _is_torch(template):
        return torch.as_tensor(array, device=template.device, dtype=template.dtype)
    return array


def _validate_groups(groups: Sequence[object], n_rows: int, name: str = "groups") -> np.ndarray:
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1 or len(labels) != n_rows:
        raise ValueError(f"{name} must be one label per input row")
    if any(item is None for item in labels):
        raise ValueError(f"{name} cannot contain None")
    return labels


def within_group_percentile_rank(values: object, groups: Sequence[object]) -> object:
    """Rank every pathway independently within each group on the [0, 1] scale."""
    matrix = _as_float_matrix(values, "values")
    labels = _validate_groups(groups, matrix.shape[0])
    result = np.empty_like(matrix)
    for label in dict.fromkeys(labels.tolist()):
        indices = np.flatnonzero(labels == label)
        block = matrix[indices]
        if len(indices) == 1:
            result[indices] = 0.5
            continue
        order = np.argsort(block, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order, np.arange(block.shape[1])] = np.arange(len(indices))[:, None]
        # Average tied ranks, rather than allowing input order to decide ties.
        for column in range(block.shape[1]):
            sorted_values = block[order[:, column], column]
            start = 0
            while start < len(indices):
                stop = start + 1
                while stop < len(indices) and sorted_values[stop] == sorted_values[start]:
                    stop += 1
                if stop - start > 1:
                    ranks[order[start:stop, column], column] = (start + stop - 1) / 2.0
                start = stop
        result[indices] = ranks / (len(indices) - 1)
    return _restore(result, values)


def slice_percentile_rank(values: object, slice_ids: Sequence[object]) -> object:
    """Convenience wrapper for within-slice pathway percentile ranks."""
    return within_group_percentile_rank(values, slice_ids)


def patient_percentile_rank(values: object, patient_ids: Sequence[object]) -> object:
    """Convenience wrapper for within-patient pathway percentile ranks."""
    return within_group_percentile_rank(values, patient_ids)


def spatial_smooth(
    features: object,
    coordinates: object,
    *,
    k: int = 5,
    radius: Optional[float] = None,
    lam: float = 0.5,
    coordinate_permutation: Optional[Sequence[int]] = None,
) -> object:
    """Mix each row with a coordinate-defined kNN or radius-neighbour mean.

    ``coordinate_permutation`` reassigns coordinates without changing features,
    making the coordinate-shuffle negative control explicit and reproducible.
    """
    matrix = _as_float_matrix(features, "features")
    coords = _as_float_matrix(coordinates, "coordinates")
    if coords.shape[0] != matrix.shape[0] or coords.shape[1] < 1:
        raise ValueError("coordinates must have one finite coordinate vector per feature row")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be a positive finite number")
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and in [0, 1]")
    if coordinate_permutation is not None:
        permutation = np.asarray(coordinate_permutation)
        if permutation.ndim != 1 or len(permutation) != len(matrix) or not np.array_equal(np.sort(permutation), np.arange(len(matrix))):
            raise ValueError("coordinate_permutation must be a permutation of row indices")
        coords = coords[permutation]
    if lam == 0.0 or len(matrix) < 2:
        return _restore(matrix.copy(), features)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbour_mean = matrix.copy()
    for row in range(len(matrix)):
        if radius is None:
            neighbours = np.argsort(distances[row], kind="mergesort")[: min(k, len(matrix) - 1)]
        else:
            neighbours = np.flatnonzero(distances[row] <= radius)
        if len(neighbours):
            neighbour_mean[row] = matrix[neighbours].mean(axis=0)
    return _restore((1.0 - lam) * matrix + lam * neighbour_mean, features)


@dataclass
class GalleryResidualCalibrator:
    """Add a per-pathway residual estimated exclusively from a training fold."""

    residual_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, train_gallery: object, train_targets: object) -> "GalleryResidualCalibrator":
        gallery = _as_float_matrix(train_gallery, "train_gallery")
        targets = _as_float_matrix(train_targets, "train_targets")
        if gallery.shape != targets.shape:
            raise ValueError("train_gallery and train_targets must have identical shape")
        if gallery.shape[0] == 0:
            raise ValueError("training fold cannot be empty")
        self.residual_ = (targets - gallery).mean(axis=0)
        return self

    def transform(self, gallery: object) -> object:
        if self.residual_ is None:
            raise RuntimeError("fit on a training fold before transform")
        matrix = _as_float_matrix(gallery, "gallery")
        if matrix.shape[1] != len(self.residual_):
            raise ValueError("gallery pathway dimension differs from fitted training fold")
        return _restore(matrix + self.residual_, gallery)


@dataclass
class KNNGalleryResidualCalibrator:
    """Training-fold kNN gallery residual correction.

    Neighbours are selected in a caller-supplied morphology/reference space;
    only residuals stored during ``fit`` can affect later transforms.
    """

    k: int = 5
    reference_: Optional[np.ndarray] = field(default=None, init=False)
    residual_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, train_reference: object, train_prediction: object,
            train_target: object) -> "KNNGalleryResidualCalibrator":
        reference = _as_float_matrix(train_reference, "train_reference")
        prediction = _as_float_matrix(train_prediction, "train_prediction")
        target = _as_float_matrix(train_target, "train_target")
        if prediction.shape != target.shape or len(reference) != len(prediction):
            raise ValueError("training gallery rows and prediction/target shapes must align")
        if not isinstance(self.k, int) or self.k < 1 or self.k > len(reference):
            raise ValueError("k must be within the training gallery size")
        self.reference_ = reference.copy()
        self.residual_ = (target - prediction).copy()
        return self

    def transform(self, reference: object, prediction: object) -> object:
        if self.reference_ is None or self.residual_ is None:
            raise RuntimeError("fit on a training fold before transform")
        query = _as_float_matrix(reference, "reference")
        base = _as_float_matrix(prediction, "prediction")
        if len(query) != len(base) or query.shape[1] != self.reference_.shape[1] or base.shape[1] != self.residual_.shape[1]:
            raise ValueError("query and prediction dimensions differ from fitted gallery")
        distances = np.linalg.norm(query[:, None, :] - self.reference_[None, :, :], axis=2)
        neighbours = np.argsort(distances, axis=1, kind="mergesort")[:, : self.k]
        correction = self.residual_[neighbours].mean(axis=1)
        return _restore(base + correction, prediction)


def patient_balanced_prototypes(embeddings: object, patient_ids: Sequence[object], high_mask: object) -> Tuple[object, object]:
    """Average patient means, giving each eligible patient equal prototype weight."""
    matrix = _as_float_matrix(embeddings, "embeddings")
    labels = _validate_groups(patient_ids, matrix.shape[0], "patient_ids")
    mask = np.asarray(high_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(matrix):
        raise ValueError("high_mask must be one boolean per embedding")
    high_means, low_means = [], []
    for patient in dict.fromkeys(labels.tolist()):
        rows = labels == patient
        if np.any(rows & mask):
            high_means.append(matrix[rows & mask].mean(axis=0))
        if np.any(rows & ~mask):
            low_means.append(matrix[rows & ~mask].mean(axis=0))
    if not high_means or not low_means:
        raise ValueError("both high and low prototypes require at least one eligible patient")
    return _restore(np.mean(high_means, axis=0), embeddings), _restore(np.mean(low_means, axis=0), embeddings)


def prototype_cosine_contrast(embeddings: object, high_prototype: object, low_prototype: object) -> object:
    """Return cosine(embedding, high) minus cosine(embedding, low)."""
    matrix = _as_float_matrix(embeddings, "embeddings")
    high = np.asarray(high_prototype.detach().cpu().numpy() if _is_torch(high_prototype) else high_prototype, dtype=float)
    low = np.asarray(low_prototype.detach().cpu().numpy() if _is_torch(low_prototype) else low_prototype, dtype=float)
    if high.ndim != 1 or low.ndim != 1 or high.shape != low.shape or high.shape[0] != matrix.shape[1] or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("prototypes must be finite vectors matching embedding dimension")
    def cosine(reference: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(reference)
        if np.any(denominator == 0):
            raise ValueError("cosine contrast is undefined for zero-norm vectors")
        return matrix.dot(reference) / denominator
    return _restore(cosine(high) - cosine(low), embeddings)


def combine_uncertainty(*components: object, weights: Optional[Sequence[float]] = None) -> object:
    """Weighted mean of aligned, finite uncertainty components."""
    if not components:
        raise ValueError("at least one uncertainty component is required")
    arrays = [np.asarray(x.detach().cpu().numpy() if _is_torch(x) else x, dtype=float) for x in components]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("components must have identical finite shapes")
    weight_array = np.ones(len(arrays)) if weights is None else np.asarray(weights, dtype=float)
    if weight_array.shape != (len(arrays),) or not np.isfinite(weight_array).all() or np.any(weight_array < 0) or weight_array.sum() == 0:
        raise ValueError("weights must be finite, non-negative, and sum to a positive value")
    return _restore(np.average(np.stack(arrays), axis=0, weights=weight_array), components[0])


def identity_permutation(n_rows: int, *, like: object = None) -> object:
    """Deterministic identity negative-control permutation."""
    if not isinstance(n_rows, (int, np.integer)) or n_rows < 0:
        raise ValueError("n_rows must be a non-negative integer")
    result = np.arange(n_rows, dtype=np.int64)
    return _restore(result, like) if _is_torch(like) else result


def patch_pathway_shuffle(values: object, *, seed: int) -> object:
    """Independently shuffle patches within every pathway, preserving marginals."""
    matrix = _as_float_matrix(values, "values")
    rng = np.random.default_rng(seed)
    result = matrix.copy()
    for column in range(result.shape[1]):
        result[:, column] = result[rng.permutation(result.shape[0]), column]
    return _restore(result, values)


def shuffle_patch_pathway_pairing(values: object, *, seed: int) -> object:
    """Shuffle whole pathway vectors across patches, preserving 30-D covariance."""
    matrix = _as_float_matrix(values, "values")
    result = matrix[np.random.default_rng(seed).permutation(len(matrix))]
    return _restore(result, values)


def permute_pathway_identity(values: object, permutation: Sequence[int]) -> object:
    """Permute pathway columns while preserving patch rows and numeric values."""
    matrix = _as_float_matrix(values, "values")
    order = np.asarray(permutation)
    if order.shape != (matrix.shape[1],) or not np.array_equal(np.sort(order), np.arange(matrix.shape[1])):
        raise ValueError("permutation must contain every pathway column exactly once")
    return _restore(matrix[:, order], values)


def leave_one_patient_out_prototype_scores(
    embeddings: object,
    patient_ids: Sequence[object],
    high_mask: object,
) -> object:
    """Score every Phase 2 patient using prototypes built from other patients."""
    matrix = _as_float_matrix(embeddings, "embeddings")
    labels = _validate_groups(patient_ids, len(matrix), "patient_ids")
    mask = np.asarray(high_mask, dtype=bool)
    if mask.shape != (len(matrix),):
        raise ValueError("high_mask must be one boolean per embedding")
    result = np.empty(len(matrix), dtype=float)
    unique = list(dict.fromkeys(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("LOPO prototype validation requires at least two patients")
    for patient in unique:
        held_out = labels == patient
        train = ~held_out
        high, low = patient_balanced_prototypes(matrix[train], labels[train], mask[train])
        result[held_out] = prototype_cosine_contrast(matrix[held_out], high, low)
    return _restore(result, embeddings)


def pathway_patient_balanced_prototypes(
    embeddings: object,
    pathway_scores: object,
    patient_ids: Sequence[object],
    *,
    high_quantile: float = 0.8,
    low_quantile: float = 0.2,
) -> Tuple[object, object]:
    """Build high/low morphology prototypes independently for every pathway."""
    matrix = _as_float_matrix(embeddings, "embeddings")
    scores = _as_float_matrix(pathway_scores, "pathway_scores")
    labels = _validate_groups(patient_ids, len(matrix), "patient_ids")
    if len(scores) != len(matrix):
        raise ValueError("embeddings and pathway_scores must share rows")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    high_all, low_all = [], []
    for column in range(scores.shape[1]):
        high_rows, low_rows = [], []
        for patient in dict.fromkeys(labels.tolist()):
            rows = np.flatnonzero(labels == patient)
            patient_scores = scores[rows, column]
            high_cut = np.quantile(patient_scores, high_quantile)
            low_cut = np.quantile(patient_scores, low_quantile)
            high_rows.append(matrix[rows[patient_scores >= high_cut]].mean(axis=0))
            low_rows.append(matrix[rows[patient_scores <= low_cut]].mean(axis=0))
        high_all.append(np.mean(high_rows, axis=0))
        low_all.append(np.mean(low_rows, axis=0))
    return _restore(np.stack(high_all), embeddings), _restore(np.stack(low_all), embeddings)


def pathway_prototype_contrast(embeddings: object, high_prototypes: object,
                               low_prototypes: object) -> object:
    """Return one morphology contrast score per patch and pathway."""
    matrix = _as_float_matrix(embeddings, "embeddings")
    high = _as_float_matrix(high_prototypes, "high_prototypes")
    low = _as_float_matrix(low_prototypes, "low_prototypes")
    if high.shape != low.shape or high.shape[1] != matrix.shape[1]:
        raise ValueError("prototype matrices must share [pathway, embedding_dim]")
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    high_norm = np.linalg.norm(high, axis=1, keepdims=True).T
    low_norm = np.linalg.norm(low, axis=1, keepdims=True).T
    if (matrix_norm == 0).any() or (high_norm == 0).any() or (low_norm == 0).any():
        raise ValueError("cosine contrast is undefined for zero-norm vectors")
    result = matrix @ high.T / (matrix_norm * high_norm) - matrix @ low.T / (matrix_norm * low_norm)
    return _restore(result, embeddings)


def covariance_matched_generic_controls(reference: object, *, n_controls: int, seed: int) -> object:
    """Generate reproducible generic 30-D Gaussian controls matched to reference moments."""
    matrix = _as_float_matrix(reference, "reference")
    if matrix.shape[0] < 2 or matrix.shape[1] > 30:
        raise ValueError("reference must have at least two rows and no more than 30 columns")
    if not isinstance(n_controls, (int, np.integer)) or n_controls < 1:
        raise ValueError("n_controls must be a positive integer")
    dimensions = matrix.shape[1]
    mean = np.zeros(30)
    mean[:dimensions] = matrix.mean(axis=0)
    covariance = np.eye(30) * max(float(np.var(matrix)), 1e-8)
    covariance[:dimensions, :dimensions] = np.atleast_2d(np.cov(matrix, rowvar=False))
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    result = np.random.default_rng(seed).multivariate_normal(mean, covariance, size=n_controls)
    return _restore(result, reference)
