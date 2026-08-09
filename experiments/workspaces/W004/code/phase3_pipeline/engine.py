"""Small deterministic fold engine used by local tests and later job wrappers.

The engine has no CLI side effects and never discovers data implicitly. A
caller must pass explicit, already split examples. Validation selects the
checkpoint; test examples are evaluated only after the checkpoint is frozen.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SlideExample:
    case_id: str
    slide_id: str
    pathology: Tensor
    pathway: Tensor
    label: int


@dataclass(frozen=True)
class JointSlideExample:
    case_id: str
    slide_id: str
    pathology: Tensor
    pathway: Tensor
    pcr: float
    mpr: float


@dataclass
class FoldTrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    evidence_scope: str
    warnings: list[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_examples(examples: Sequence[SlideExample], name: str) -> None:
    if not examples:
        raise ValueError(f"{name} examples are empty")
    seen: set[str] = set()
    for item in examples:
        if item.slide_id in seen:
            raise ValueError(f"duplicate slide_id in {name}: {item.slide_id}")
        seen.add(item.slide_id)
        if item.label not in (0, 1):
            raise ValueError(f"{name} contains unknown/non-binary label for {item.slide_id}")
        if item.pathology.ndim != 2 or item.pathway.ndim != 2:
            raise ValueError("pathology and pathway must be [patch, feature] matrices")
        if item.pathology.shape[0] != item.pathway.shape[0]:
            raise ValueError(f"unaligned patch counts for {item.slide_id}")
        if not torch.isfinite(item.pathology).all() or not torch.isfinite(item.pathway).all():
            raise ValueError(f"non-finite features for {item.slide_id}")


def _patient_weights(examples: Sequence[SlideExample]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for item in examples:
        counts[item.case_id] = counts.get(item.case_id, 0) + 1
    return {case: 1.0 / count for case, count in counts.items()}


def _forward_one(model: nn.Module, item: SlideExample, device: torch.device) -> Tensor:
    image = item.pathology.to(device).unsqueeze(0)
    pathway = item.pathway.to(device).unsqueeze(0)
    return model(image, pathway).reshape(-1)


@torch.no_grad()
def predict_examples(model: nn.Module, examples: Sequence[SlideExample], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for item in examples:
        logits = _forward_one(model, item, device)
        if logits.numel() != 1:
            raise ValueError("binary fold engine expects exactly one output logit")
        rows.append({
            "case_id": item.case_id,
            "slide_id": item.slide_id,
            "y_true": item.label,
            "y_pred": float(torch.sigmoid(logits[0]).cpu()),
        })
    return pd.DataFrame(rows)


def _loss_over_examples(
    model: nn.Module,
    examples: Sequence[SlideExample],
    criterion: nn.Module,
    device: torch.device,
    patient_weighting: bool,
) -> Tensor:
    weights = _patient_weights(examples) if patient_weighting else {item.case_id: 1.0 for item in examples}
    losses = []
    sample_weights = []
    for item in examples:
        logit = _forward_one(model, item, device)
        target = torch.tensor([float(item.label)], device=device)
        losses.append(criterion(logit, target).reshape(()))
        sample_weights.append(weights[item.case_id])
    weight_tensor = torch.tensor(sample_weights, dtype=losses[0].dtype, device=device)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def train_single_fold(
    model: nn.Module,
    train_examples: Sequence[SlideExample],
    validation_examples: Sequence[SlideExample],
    test_examples: Sequence[SlideExample],
    *,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    patience: int = 5,
    patient_weighting: bool = True,
    patient_overlap_policy: str = "forbid",
    device: str = "cpu",
) -> FoldTrainingResult:
    """Train on train, select on validation, then evaluate frozen test once."""

    for name, values in (("train", train_examples), ("validation", validation_examples), ("test", test_examples)):
        _validate_examples(values, name)
    split_sets = [set(item.slide_id for item in values) for values in (train_examples, validation_examples, test_examples)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError("slide leakage across train/validation/test")
    if patient_overlap_policy not in {"forbid", "allow_with_warning"}:
        raise ValueError("patient_overlap_policy must be 'forbid' or 'allow_with_warning'")
    patient_sets = [set(item.case_id for item in values) for values in (train_examples, validation_examples, test_examples)]
    patient_overlap = set().union(patient_sets[0] & patient_sets[1], patient_sets[0] & patient_sets[2], patient_sets[1] & patient_sets[2])
    warnings: list[str] = []
    evidence_scope = "patient_independent_split"
    if patient_overlap:
        if patient_overlap_policy == "forbid":
            raise ValueError(f"patient overlap across splits requires explicit allow_with_warning: {sorted(patient_overlap)[:10]}")
        warnings.append(f"WARN_PATIENT_NONINDEPENDENCE:{','.join(sorted(patient_overlap))}")
        evidence_scope = "slide_repeated_holdout_patient_nonindependent"
    if len({item.label for item in train_examples}) != 2:
        raise ValueError("training set must contain both classes")
    if epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs, patience, and learning_rate must be positive")
    seed_everything(seed)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = _loss_over_examples(model, train_examples, criterion, torch_device, patient_weighting)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = _loss_over_examples(model, validation_examples, criterion, torch_device, patient_weighting)
        current = float(validation_loss.cpu())
        history.append({"epoch": float(epoch), "train_loss": float(train_loss.detach().cpu()), "validation_loss": current})
        if current < best_loss - 1e-8:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    validation = predict_examples(model, validation_examples, torch_device)
    test = predict_examples(model, test_examples, torch_device)
    return FoldTrainingResult(model, history, validation, test, evidence_scope, warnings)


def annotate_oof(
    predictions: pd.DataFrame,
    *,
    task: str,
    seed: int,
    fold: int,
    model_name: str,
) -> pd.DataFrame:
    table = predictions.copy()
    table["task"] = task
    table["seed"] = seed
    table["fold"] = fold
    table["model"] = model_name
    return table[["case_id", "slide_id", "task", "y_true", "y_pred", "seed", "fold", "model"]]


def annotate_fold_predictions(
    predictions: pd.DataFrame,
    *,
    task: str,
    seed: int,
    fold: int,
    model_name: str,
    prediction_kind: str,
    evidence_scope: str,
    warnings: Sequence[str],
) -> pd.DataFrame:
    """Annotate predictions without conflating repeated holdout with true OOF."""
    if prediction_kind not in {"partition_oof", "repeated_holdout_test"}:
        raise ValueError("prediction_kind must be partition_oof or repeated_holdout_test")
    table = annotate_oof(predictions, task=task, seed=seed, fold=fold, model_name=model_name)
    table["prediction_kind"] = prediction_kind
    table["evidence_scope"] = str(evidence_scope)
    table["warnings"] = "|".join(str(item) for item in warnings) if warnings else "NONE"
    return table


def _validate_joint_examples(examples: Sequence[JointSlideExample], name: str) -> None:
    if not examples:
        raise ValueError(f"{name} examples are empty")
    seen: set[str] = set()
    for item in examples:
        if item.slide_id in seen:
            raise ValueError(f"duplicate slide_id in {name}: {item.slide_id}")
        seen.add(item.slide_id)
        for endpoint_name, target in (("pCR", item.pcr), ("MPR", item.mpr)):
            if not (np.isnan(target) or target in (0, 1)):
                raise ValueError(f"{endpoint_name} must be binary or NaN for {item.slide_id}")
        if item.pcr == 1 and item.mpr == 0:
            raise ValueError(f"endpoint nesting violation for {item.slide_id}")
        if item.pathology.ndim != 2 or item.pathway.ndim != 2 or item.pathology.shape[0] != item.pathway.shape[0]:
            raise ValueError(f"invalid joint feature shape for {item.slide_id}")
        if not torch.isfinite(item.pathology).all() or not torch.isfinite(item.pathway).all():
            raise ValueError(f"non-finite joint features for {item.slide_id}")


def train_joint_single_fold(
    model: nn.Module,
    train_examples: Sequence[JointSlideExample],
    validation_examples: Sequence[JointSlideExample],
    test_examples: Sequence[JointSlideExample],
    *,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> FoldTrainingResult:
    """Train the monotonic joint endpoint model with unknown-label masking."""
    from .models import monotonic_joint_bce_loss

    for name, values in (("train", train_examples), ("validation", validation_examples), ("test", test_examples)):
        _validate_joint_examples(values, name)
    slide_sets = [set(x.slide_id for x in values) for values in (train_examples, validation_examples, test_examples)]
    if slide_sets[0] & slide_sets[1] or slide_sets[0] & slide_sets[2] or slide_sets[1] & slide_sets[2]:
        raise ValueError("slide leakage across joint train/validation/test")
    patient_sets = [set(x.case_id for x in values) for values in (train_examples, validation_examples, test_examples)]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise ValueError("joint patient-level training requires patient-independent splits")
    seed_everything(seed)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def loss_for(items: Sequence[JointSlideExample]) -> Tensor:
        logits, targets, weights = [], [], []
        counts = _patient_weights([SlideExample(x.case_id, x.slide_id, x.pathology, x.pathway, 0) for x in items])
        for item in items:
            logits.append(model(item.pathology.to(torch_device).unsqueeze(0), item.pathway.to(torch_device).unsqueeze(0)).squeeze(0))
            targets.append([item.pcr, item.mpr])
            weights.append(counts[item.case_id])
        return monotonic_joint_bce_loss(torch.stack(logits), torch.tensor(targets, device=torch_device), torch.tensor(weights, device=torch_device))

    best_state, best_loss, stale, history = copy.deepcopy(model.state_dict()), float("inf"), 0, []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_loss = loss_for(train_examples); train_loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): validation_loss = loss_for(validation_examples)
        current = float(validation_loss.cpu())
        history.append({"epoch": float(epoch), "train_loss": float(train_loss.detach().cpu()), "validation_loss": current})
        if current < best_loss - 1e-8:
            best_loss, best_state, stale = current, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state); model.eval()

    def predict(items: Sequence[JointSlideExample]) -> pd.DataFrame:
        rows = []
        with torch.no_grad():
            for item in items:
                probability = torch.sigmoid(model(item.pathology.to(torch_device).unsqueeze(0), item.pathway.to(torch_device).unsqueeze(0)).squeeze(0)).cpu()
                rows.append({"case_id": item.case_id, "slide_id": item.slide_id, "pcr_true": item.pcr, "mpr_true": item.mpr,
                             "pcr_pred": float(probability[0]), "mpr_pred": float(probability[1])})
        return pd.DataFrame(rows)
    return FoldTrainingResult(model, history, predict(validation_examples), predict(test_examples), "patient_independent_joint_endpoints", [])
