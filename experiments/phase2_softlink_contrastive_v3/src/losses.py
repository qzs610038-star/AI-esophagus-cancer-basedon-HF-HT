"""Numerically stable, bidirectional soft contrastive losses for Phase2 v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_TEMPERATURE_CANDIDATES = (0.03, 0.1, 0.3, 1.0, 3.0)


def _as_patient_id_list(patient_ids: Sequence[Any] | Tensor) -> list[Any]:
    return patient_ids.detach().cpu().reshape(-1).tolist() if isinstance(patient_ids, Tensor) else list(patient_ids)


def _teacher_representation(
    labels: Tensor,
    patient_ids: Sequence[Any] | Tensor,
    mode: str,
    *,
    patient_means: Mapping[Any, Tensor | Sequence[float]] | None = None,
    centered_representation: Tensor | None = None,
) -> Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, pathways]")
    ids = _as_patient_id_list(patient_ids)
    if mode == "global":
        if len(ids) != labels.shape[0]:
            raise ValueError("patient_ids must have one entry per label row")
        return labels.detach().float()
    if mode == "patient_centered":
        if patient_means is not None and centered_representation is not None:
            raise ValueError("provide either patient_means or centered_representation, not both")
        if centered_representation is not None:
            if centered_representation.shape != labels.shape:
                raise ValueError("centered_representation must have the same shape as labels")
            return centered_representation.detach().to(device=labels.device, dtype=torch.float32)
        if patient_means is None:
            raise ValueError(
                "patient_centered teachers require protocol/fold pre-fitted patient_means "
                "or an explicit centered_representation"
            )
        try:
            means = torch.stack([
                torch.as_tensor(patient_means[patient], device=labels.device, dtype=torch.float32)
                for patient in ids
            ])
        except KeyError as error:
            raise KeyError(f"missing pre-fitted mean for patient {error.args[0]!r}") from error
        if means.shape != labels.shape:
            raise ValueError("each patient mean must match the pathway dimension of labels")
        return labels.detach().float() - means
    raise ValueError("mode must be 'global' or 'patient_centered'")


def _off_diagonal_probabilities(representation: Tensor, tau_y: float) -> tuple[Tensor, Tensor]:
    if tau_y <= 0:
        raise ValueError("tau_y must be positive")
    count = representation.shape[0]
    if count < 2:
        raise ValueError("a contrastive batch needs at least two distinct candidates")
    # Build both directional score matrices independently.  They have equal
    # values for a symmetric squared distance, but are not manufactured by
    # transposing an already-normalised target distribution.
    image_to_pathway_distance = (representation[:, None, :] - representation[None, :, :]).square().mean(dim=-1)
    pathway_to_image_distance = (representation[None, :, :] - representation[:, None, :]).square().mean(dim=-1)
    image_to_pathway_logits = -image_to_pathway_distance / float(tau_y)
    pathway_to_image_logits = -pathway_to_image_distance / float(tau_y)
    diagonal = torch.eye(count, device=representation.device, dtype=torch.bool)
    image_to_pathway_logits = image_to_pathway_logits.masked_fill(diagonal, float("-inf"))
    pathway_to_image_logits = pathway_to_image_logits.masked_fill(diagonal, float("-inf"))
    # FP32 softmax is deliberate: it is the teacher's stable candidate normalisation.
    image_to_pathway = torch.softmax(image_to_pathway_logits.float(), dim=1)
    pathway_to_image = torch.softmax(pathway_to_image_logits.float(), dim=1)
    return image_to_pathway, pathway_to_image


def build_teacher_targets(
    labels: Tensor,
    patient_ids: Sequence[Any] | Tensor,
    mode: str = "global",
    tau_y: float = 0.1,
    rho: float = 0.25,
    *,
    patient_means: Mapping[Any, Tensor | Sequence[float]] | None = None,
    centered_representation: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Create detached I->Y and Y->I target distributions.

    ``q_i2y[i, j]`` is image ``i`` anchored against pathway candidate ``j``.
    ``q_y2i[j, i]`` is pathway ``j`` anchored against image candidate ``i``.
    Both candidate axes are normalised independently, even though symmetric
    label distances can make their values numerically equal after orientation.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    representation = _teacher_representation(
        labels, patient_ids, mode,
        patient_means=patient_means, centered_representation=centered_representation,
    )
    image_to_pathway, pathway_to_image = _off_diagonal_probabilities(representation, tau_y)
    count = representation.shape[0]
    diagonal = torch.eye(count, device=representation.device, dtype=torch.float32)
    q_i2y = rho * diagonal + (1.0 - rho) * image_to_pathway
    # This is a second normalisation with pathway anchors and image candidates;
    # do not replace it with q_i2y.T.
    q_y2i = rho * diagonal + (1.0 - rho) * pathway_to_image
    return q_i2y.detach(), q_y2i.detach()


def bidirectional_soft_contrastive_loss(
    image_embeddings: Tensor,
    teacher_embeddings: Tensor,
    q_i2y: Tensor,
    q_y2i: Tensor,
    tau_z: float = 0.1,
    *,
    return_details: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Mean equal-weight soft cross entropy in both directions.

    The pathway MLP embedding stays differentiable; only label-derived teacher
    target distributions are stopped.  Each directional softmax normalises its
    own candidate axis, not a transposed loss surrogate.
    """
    if tau_z <= 0:
        raise ValueError("tau_z must be positive")
    if image_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 [batch, embedding]")
    if image_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("image and teacher embeddings must have identical shape")
    batch = image_embeddings.shape[0]
    if q_i2y.shape != (batch, batch) or q_y2i.shape != (batch, batch):
        raise ValueError("both teacher targets must have shape [batch, batch]")
    u = F.normalize(image_embeddings.float(), dim=-1)
    v = F.normalize(teacher_embeddings.float(), dim=-1)
    logits_i2y = (u @ v.transpose(0, 1)) / float(tau_z)
    logits_y2i = (v @ u.transpose(0, 1)) / float(tau_z)
    targets_i2y = q_i2y.detach().to(device=logits_i2y.device, dtype=torch.float32)
    targets_y2i = q_y2i.detach().to(device=logits_y2i.device, dtype=torch.float32)
    loss_i2y = -(targets_i2y * F.log_softmax(logits_i2y, dim=1)).sum(dim=1).mean()
    loss_y2i = -(targets_y2i * F.log_softmax(logits_y2i, dim=1)).sum(dim=1).mean()
    loss = 0.5 * (loss_i2y + loss_y2i)
    if return_details:
        return {"loss": loss, "image_to_pathway": loss_i2y, "pathway_to_image": loss_y2i}
    return loss


def _normalised_off_diagonal_entropy(
    labels: Tensor,
    patient_ids: Sequence[Any] | Tensor,
    mode: str,
    tau: float,
    *,
    patient_means: Mapping[Any, Tensor | Sequence[float]] | None = None,
    centered_representation: Tensor | None = None,
) -> float:
    representation = _teacher_representation(
        labels, patient_ids, mode,
        patient_means=patient_means, centered_representation=centered_representation,
    )
    row_probabilities, _ = _off_diagonal_probabilities(representation, tau)
    entropy = -(row_probabilities * row_probabilities.clamp_min(torch.finfo(row_probabilities.dtype).tiny).log()).sum(dim=1)
    return (entropy / torch.log(torch.tensor(float(representation.shape[0] - 1), device=entropy.device))).mean().item()


def _coerce_batches(
    labels_or_batches: Tensor | Sequence[Any] | None,
    patient_ids: Sequence[Any] | Tensor | None,
    batches: Sequence[Any] | None,
) -> list[tuple[Tensor, Sequence[Any] | Tensor, Tensor | None]]:
    raw_batches = batches
    if raw_batches is None:
        if patient_ids is not None:
            if not isinstance(labels_or_batches, Tensor):
                raise TypeError("labels must be a tensor when patient_ids is supplied")
            raw_batches = [(labels_or_batches, patient_ids)]
        else:
            raw_batches = labels_or_batches  # type: ignore[assignment]
    if not isinstance(raw_batches, Sequence):
        raise TypeError("patient-balanced batches must be a sequence")
    coerced: list[tuple[Tensor, Sequence[Any] | Tensor, Tensor | None]] = []
    for batch in raw_batches:
        if isinstance(batch, dict):
            labels, ids = batch["labels"], batch["patient_ids"]
            centered = batch.get("centered_representation")
        else:
            if len(batch) == 2:
                labels, ids = batch
                centered = None
            elif len(batch) == 3:
                labels, ids, centered = batch
            else:
                raise ValueError("each batch must be (labels, patient_ids[, centered_representation])")
        if not isinstance(labels, Tensor):
            raise TypeError("each batch labels value must be a Tensor")
        if centered is not None and not isinstance(centered, Tensor):
            raise TypeError("centered_representation must be a Tensor")
        coerced.append((labels, ids, centered))
    return coerced


def select_teacher_temperature(
    labels_or_batches: Tensor | Sequence[Any] | None = None,
    patient_ids: Sequence[Any] | Tensor | None = None,
    *,
    batches: Sequence[Any] | None = None,
    mode: str = "global",
    candidates: Sequence[float] = DEFAULT_TEMPERATURE_CANDIDATES,
    target_entropy: float = 0.7,
    rho: float = 0.25,
    patient_means: Mapping[Any, Tensor | Sequence[float]] | None = None,
    centered_representations: Sequence[Tensor] | None = None,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Choose tau_y from 20 supplied patient-balanced training batches.

    Candidate ties deliberately favour the larger temperature, as fixed by v4.
    ``rho`` is accepted to keep the selection contract explicit; entropy is
    calculated on the non-diagonal candidate normalisation and is therefore
    independent of the diagonal anchor mass.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    provided = _coerce_batches(labels_or_batches, patient_ids, batches)
    if len(provided) != 20:
        raise ValueError("select_teacher_temperature requires exactly 20 patient-balanced batches")
    if centered_representations is not None and len(centered_representations) != len(provided):
        raise ValueError("centered_representations must contain one tensor per supplied batch")
    if mode == "patient_centered" and patient_means is None and centered_representations is None and any(
        centered is None for _, _, centered in provided
    ):
        raise ValueError(
            "patient_centered temperature selection requires pre-fitted patient_means "
            "or one explicit centered representation per batch"
        )
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError("candidates must contain positive temperatures")
    scores = {
        float(candidate): sum(
            _normalised_off_diagonal_entropy(
                labels,
                ids,
                mode,
                float(candidate),
                patient_means=patient_means,
                centered_representation=(
                    centered_representations[index] if centered_representations is not None else centered
                ),
            )
            for index, (labels, ids, centered) in enumerate(provided)
        )
        / len(provided)
        for candidate in candidates
    }
    chosen = min(scores, key=lambda candidate: (abs(scores[candidate] - target_entropy), -candidate))
    if return_details:
        return {"tau_y": chosen, "mean_normalised_entropy": scores[chosen], "candidate_entropies": scores}
    return chosen
