"""Data contracts and leakage audits for Phase 3.

``case_id`` is a patient identifier, while ``slide_id`` identifies one slide.
The selected protocol uses slide-level splits, so cross-split patient overlap is
reported explicitly instead of silently treated as patient-grouped evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PatchBag:
    """Aligned patch-level inputs for one slide."""

    case_id: str
    slide_id: str
    pathology: np.ndarray
    pathway: np.ndarray
    coordinates: np.ndarray
    pcr: int | None = None
    mpr: int | None = None


def _finite_matrix(name: str, value: np.ndarray, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix, got {array.shape}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns, got {array.shape[1]}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one patch")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_patch_bag(bag: PatchBag, pathway_dim: int = 30) -> PatchBag:
    """Validate one bag and return it unchanged after all checks pass."""

    if not str(bag.case_id).strip() or not str(bag.slide_id).strip():
        raise ValueError("case_id and slide_id must be non-empty")
    pathology = _finite_matrix("pathology", bag.pathology)
    pathway = _finite_matrix("pathway", bag.pathway, pathway_dim)
    coordinates = _finite_matrix("coordinates", bag.coordinates, 2)
    counts = {pathology.shape[0], pathway.shape[0], coordinates.shape[0]}
    if len(counts) != 1:
        raise ValueError(
            "patch alignment mismatch: pathology, pathway, and coordinates "
            f"have row counts {pathology.shape[0]}, {pathway.shape[0]}, {coordinates.shape[0]}"
        )
    for name, label in (("pcr", bag.pcr), ("mpr", bag.mpr)):
        if label not in (None, 0, 1):
            raise ValueError(f"{name} must be 0, 1, or None (unknown), got {label!r}")
    if bag.pcr == 1 and bag.mpr == 0:
        raise ValueError("endpoint nesting violation: pCR-positive cannot be MPR-negative")
    return bag


def validate_pathway_names(names: Sequence[str], expected_count: int = 30) -> tuple[str, ...]:
    cleaned = tuple(str(name).strip() for name in names)
    if len(cleaned) != expected_count:
        raise ValueError(f"expected {expected_count} pathway names, got {len(cleaned)}")
    if any(not name for name in cleaned):
        raise ValueError("pathway names must be non-empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("pathway names must be unique and order-preserving")
    return cleaned


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    required_columns: Iterable[str] = ("case_id", "slide_id"),
) -> pd.DataFrame:
    """Check identity uniqueness and within-patient label consistency."""

    missing = set(required_columns) - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    table = manifest.copy()
    if table.empty:
        raise ValueError("manifest is empty")
    if table[list(required_columns)].isna().any().any():
        raise ValueError("manifest identity fields contain missing values")
    table["case_id"] = table["case_id"].astype(str)
    table["slide_id"] = table["slide_id"].astype(str)
    mapping_counts = table.groupby("slide_id")["case_id"].nunique()
    bad_slides = mapping_counts[mapping_counts != 1].index.tolist()
    if bad_slides:
        raise ValueError(f"slide_id maps to multiple patients: {bad_slides[:10]}")
    for endpoint in ("pcr", "mpr"):
        if endpoint not in table.columns:
            continue
        known = table[table[endpoint].notna()]
        invalid = sorted(set(known[endpoint].unique()) - {0, 1})
        if invalid:
            raise ValueError(f"{endpoint} contains non-binary known labels: {invalid}")
        conflicts = known.groupby("case_id")[endpoint].nunique()
        conflict_ids = conflicts[conflicts > 1].index.tolist()
        if conflict_ids:
            raise ValueError(f"{endpoint} conflicts within patients: {conflict_ids[:10]}")
    if {"pcr", "mpr"}.issubset(table.columns):
        nested_bad = table[(table["pcr"] == 1) & (table["mpr"] == 0)]
        if not nested_bad.empty:
            raise ValueError(f"pCR/MPR nesting violated for {len(nested_bad)} slides")
    return table


def validate_fold_assignments(
    assignments: pd.DataFrame,
    *,
    split_unit: str,
    expected_slide_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Audit one fold and return leakage/coverage facts.

    For ``split_unit='patient'`` any patient overlap is rejected. For the
    explicitly selected ``slide`` protocol it is returned as review evidence.
    """

    required = {"case_id", "slide_id", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {sorted(missing)}")
    if split_unit not in {"patient", "slide"}:
        raise ValueError("split_unit must be 'patient' or 'slide'")
    table = assignments.copy()
    table["case_id"] = table["case_id"].astype(str)
    table["slide_id"] = table["slide_id"].astype(str)
    unknown = sorted(set(table["split"]) - set(REQUIRED_SPLITS))
    if unknown:
        raise ValueError(f"unknown split labels: {unknown}")
    duplicates = table[table.duplicated("slide_id", keep=False)]["slide_id"].unique().tolist()
    if duplicates:
        raise ValueError(f"slides assigned more than once: {duplicates[:10]}")
    missing_splits = [name for name in REQUIRED_SPLITS if name not in set(table["split"])]
    if missing_splits:
        raise ValueError(f"fold lacks required splits: {missing_splits}")
    expected = None if expected_slide_ids is None else {str(x) for x in expected_slide_ids}
    actual = set(table["slide_id"])
    if expected is not None and expected != actual:
        raise ValueError(
            f"fold coverage mismatch: missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    patients_by_split = {
        name: set(table.loc[table["split"] == name, "case_id"]) for name in REQUIRED_SPLITS
    }
    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(patients_by_split[left] & patients_by_split[right])
        overlaps[f"{left}_{right}"] = overlap
    overlap_count = len(set().union(*map(set, overlaps.values())))
    if split_unit == "patient" and overlap_count:
        raise ValueError(f"patient split leakage detected for {overlap_count} patients")
    return {
        "split_unit": split_unit,
        "slide_count": int(len(table)),
        "patient_count": int(table["case_id"].nunique()),
        "split_slide_counts": {k: int(v) for k, v in table["split"].value_counts().to_dict().items()},
        "patient_overlap": overlaps,
        "patient_overlap_count": overlap_count,
        "patient_overlap_status": "WARN_REVIEW_REQUIRED" if overlap_count else "PASS_NONE",
    }


def assert_train_only_fit(fit_ids: Iterable[str], split_by_id: Mapping[str, str]) -> None:
    """Reject preprocessing fit inputs not belonging to the training fold."""

    leaked = sorted(str(item) for item in fit_ids if split_by_id.get(str(item)) != "train")
    if leaked:
        raise ValueError(f"preprocessing fit attempted outside train split: {leaked[:10]}")
