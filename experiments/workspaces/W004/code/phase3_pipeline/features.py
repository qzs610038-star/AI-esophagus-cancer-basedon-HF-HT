"""Fold-aware wiring for Phase 2/3 pathway representations and controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import PatchBag, validate_patch_bag
from .transforms import (
    KNNGalleryResidualCalibrator,
    combine_uncertainty,
    covariance_matched_generic_controls,
    pathway_patient_balanced_prototypes,
    pathway_prototype_contrast,
    leave_one_patient_out_prototype_scores,
    patient_percentile_rank,
    permute_pathway_identity,
    shuffle_patch_pathway_pairing,
    slice_percentile_rank,
    spatial_smooth,
)


@dataclass(frozen=True)
class PathwayViews:
    absolute: np.ndarray
    ordinal: np.ndarray
    prototype: np.ndarray
    uncertainty: np.ndarray


class Phase3FeaturePipeline:
    """Connect fold-fitted Phase 2 evidence to transform-only Phase 3 bags."""

    def __init__(self, *, pathway_dim: int = 30, gallery_k: int = 5) -> None:
        self.pathway_dim = pathway_dim
        self.gallery = KNNGalleryResidualCalibrator(k=gallery_k)
        self.high_prototypes: np.ndarray | None = None
        self.low_prototypes: np.ndarray | None = None
        self.phase2_error: np.ndarray | None = None
        self.lopo_scores: np.ndarray | None = None
        self.lopo_direction: np.ndarray | None = None

    def fit_phase2_training_fold(
        self,
        morphology: np.ndarray,
        predicted_pathway: np.ndarray,
        true_pathway: np.ndarray,
        patient_ids: Sequence[str],
    ) -> "Phase3FeaturePipeline":
        """Fit gallery/prototypes only from the Phase 2 training patient fold."""
        morphology = np.asarray(morphology, dtype=float)
        predicted = np.asarray(predicted_pathway, dtype=float)
        truth = np.asarray(true_pathway, dtype=float)
        if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[1] != self.pathway_dim:
            raise ValueError("Phase 2 pathway matrices must align and match pathway_dim")
        if morphology.ndim != 2 or len(morphology) != len(predicted) or len(patient_ids) != len(predicted):
            raise ValueError("Phase 2 morphology, pathway, and patient rows must align")
        self.gallery.fit(morphology, predicted, truth)
        high, low = pathway_patient_balanced_prototypes(morphology, truth, patient_ids)
        self.high_prototypes = np.asarray(high)
        self.low_prototypes = np.asarray(low)
        self.phase2_error = np.mean(np.abs(truth - predicted), axis=0)
        lopo = np.empty_like(truth, dtype=float)
        direction = np.empty(self.pathway_dim, dtype=float)
        patient_array = np.asarray(patient_ids, dtype=object)
        for column in range(self.pathway_dim):
            high_mask = np.zeros(len(truth), dtype=bool)
            for patient in dict.fromkeys(patient_array.tolist()):
                rows = patient_array == patient
                high_mask[rows] = truth[rows, column] >= np.median(truth[rows, column])
            lopo[:, column] = np.asarray(leave_one_patient_out_prototype_scores(morphology, patient_ids, high_mask))
            high_mean = lopo[high_mask, column].mean()
            low_mean = lopo[~high_mask, column].mean()
            direction[column] = high_mean - low_mean
        self.lopo_scores = lopo
        self.lopo_direction = direction
        return self

    def lopo_stability_report(self) -> dict[str, object]:
        if self.lopo_scores is None or self.lopo_direction is None:
            raise RuntimeError("fit_phase2_training_fold before requesting LOPO evidence")
        return {
            "pathway_count": int(len(self.lopo_direction)),
            "positive_direction_count": int((self.lopo_direction > 0).sum()),
            "direction_delta": self.lopo_direction.copy(),
            "finite": bool(np.isfinite(self.lopo_scores).all()),
            "evidence_scope": "phase2_training_fold_lopo_diagnostic",
        }

    def transform_slide(
        self,
        bag: PatchBag,
        *,
        spatial_lambda: float = 0.0,
        spatial_k: int = 5,
        gallery_correction: bool = False,
    ) -> PathwayViews:
        validate_patch_bag(bag, self.pathway_dim)
        if self.high_prototypes is None or self.low_prototypes is None or self.phase2_error is None:
            raise RuntimeError("fit_phase2_training_fold before Phase 3 transform")
        base_absolute = np.asarray(self.gallery.transform(bag.pathology, bag.pathway)) if gallery_correction else bag.pathway
        absolute = np.asarray(spatial_smooth(base_absolute, bag.coordinates, k=spatial_k, lam=spatial_lambda))
        ordinal = np.asarray(slice_percentile_rank(bag.pathway, [bag.slide_id] * len(bag.pathway)))
        prototype = np.asarray(pathway_prototype_contrast(bag.pathology, self.high_prototypes, self.low_prototypes))
        disagreement = np.abs(absolute - prototype)
        uncertainty = np.asarray(combine_uncertainty(np.broadcast_to(self.phase2_error, absolute.shape), disagreement))
        maximum = np.maximum(uncertainty.max(axis=0, keepdims=True), 1e-8)
        uncertainty = np.clip(uncertainty / maximum, 0, 1)
        return PathwayViews(absolute.astype(np.float32), ordinal.astype(np.float32), prototype.astype(np.float32), uncertainty.astype(np.float32))

    def transform_patient_ordinal(self, bags: Sequence[PatchBag]) -> list[np.ndarray]:
        """Patient-context sensitivity representation; never used implicitly."""
        if not bags:
            raise ValueError("patient context bags are empty")
        case_ids = {bag.case_id for bag in bags}
        if len(case_ids) != 1:
            raise ValueError("patient ordinal context must contain one patient")
        for bag in bags:
            validate_patch_bag(bag, self.pathway_dim)
        values = np.vstack([bag.pathway for bag in bags])
        ranked = np.asarray(patient_percentile_rank(values, [bags[0].case_id] * len(values)))
        outputs, start = [], 0
        for bag in bags:
            outputs.append(ranked[start:start + len(bag.pathway)])
            start += len(bag.pathway)
        return outputs

    def counterfactual(self, absolute: np.ndarray, kind: str, *, seed: int) -> np.ndarray:
        values = np.asarray(absolute)
        if kind == "covariance_matched_30d":
            return np.asarray(covariance_matched_generic_controls(values, n_controls=len(values), seed=seed))
        if kind == "pathway_identity_permutation":
            permutation = np.random.default_rng(seed).permutation(values.shape[1])
            return np.asarray(permute_pathway_identity(values, permutation))
        if kind == "patch_pathway_shuffle":
            return np.asarray(shuffle_patch_pathway_pairing(values, seed=seed))
        raise ValueError(f"unknown counterfactual kind: {kind}")


class ExternalEvaluationGuard:
    """Freeze a fitted feature pipeline before external transformation."""

    def __init__(self, fitted_pipeline: Phase3FeaturePipeline) -> None:
        if fitted_pipeline.high_prototypes is None:
            raise ValueError("external guard requires a fitted internal pipeline")
        self.pipeline = fitted_pipeline

    def transform_only(self, bag: PatchBag, **kwargs) -> PathwayViews:
        return self.pipeline.transform_slide(bag, **kwargs)

    def fit(self, *_args, **_kwargs):
        raise RuntimeError("fitting on an external evaluation cohort is forbidden")
