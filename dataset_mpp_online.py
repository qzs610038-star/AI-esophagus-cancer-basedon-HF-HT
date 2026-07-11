"""Manifest-driven online image datasets for MPP UNI2-h LoRA training.

This module deliberately does not read frozen ``.pt`` features.  Every sample
is resolved by the composite identity ``(mpp_id, patient, patch_stem)`` and is
loaded from the registered MPP image layout::

    <mpp_root>/<mpp_id>/<patient>/patch_images/<patch_stem>.png

Missing, duplicate, or cross-split records are hard failures.  Formal training
must never silently drop samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


TRAIN_PATIENTS = ("HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ")
EXTERNAL_PATIENT = "XZY"
REQUIRED_MANIFEST_COLUMNS = {
    "mpp_id", "patient", "patch_stem", "x", "y", "split", "block_id",
}
IDENTITY_COLUMNS = ["mpp_id", "patient", "patch_stem"]
NON_TARGET_COLUMNS = {
    "x", "y", "spot", "id", "spot_id", "barcode", "index", "patch_id",
    "filename", "patient", "mpp_id", "split", "block_id",
}


@dataclass(frozen=True)
class OnlineMPPSample:
    mpp_id: int
    patient: str
    patch_stem: str
    split: str
    block_id: str
    image_path: Path
    target: torch.Tensor


def _normalise_stem(value: object) -> str:
    return Path(str(value)).stem


def validate_manifest_frame(manifest_df: pd.DataFrame) -> None:
    """Validate schema and one-to-one sample identity before any filtering."""
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest_df.columns))
    if missing:
        raise ValueError(f"split manifest missing columns: {missing}")
    if manifest_df.empty:
        raise ValueError("split manifest is empty")
    if manifest_df[list(REQUIRED_MANIFEST_COLUMNS)].isnull().any().any():
        raise ValueError("split manifest contains null required values")

    duplicate = manifest_df.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicate.any():
        examples = manifest_df.loc[duplicate, IDENTITY_COLUMNS + ["split"]].head(5)
        raise ValueError(
            "duplicate online MPP sample identity: "
            f"{examples.to_dict('records')}"
        )

    split_counts = (
        manifest_df.groupby(IDENTITY_COLUMNS, dropna=False)["split"].nunique()
    )
    if (split_counts > 1).any():
        raise ValueError("online MPP sample appears in more than one split")


def _load_patient_labels(
    labels_csv: str | Path,
    target_cols: Optional[Sequence[str]] = None,
) -> Tuple[dict[str, np.ndarray], List[str]]:
    labels_path = Path(labels_csv)
    if not labels_path.is_file():
        raise FileNotFoundError(f"MPP labels CSV missing: {labels_path}")

    labels = pd.read_csv(labels_path)
    if labels.empty:
        raise ValueError(f"MPP labels CSV is empty: {labels_path}")
    id_col = labels.columns[0]
    stems = labels[id_col].map(_normalise_stem)
    duplicate = stems.duplicated(keep=False)
    if duplicate.any():
        raise ValueError(
            f"duplicate label identity in {labels_path}: "
            f"{stems.loc[duplicate].head(5).tolist()}"
        )

    if target_cols is None:
        numeric = list(labels.select_dtypes(include=["number"]).columns)
        target_cols = [
            column for column in numeric
            if column != id_col and column.lower() not in NON_TARGET_COLUMNS
        ]
    target_cols = list(target_cols)
    if not target_cols:
        raise ValueError(f"no numeric target columns in {labels_path}")
    missing_targets = [column for column in target_cols if column not in labels]
    if missing_targets:
        raise ValueError(f"labels missing target columns: {missing_targets}")
    if labels[target_cols].isnull().any().any():
        raise ValueError(f"labels contain null targets: {labels_path}")

    label_map = {
        stem: row.astype(np.float32, copy=True)
        for stem, row in zip(stems, labels[target_cols].to_numpy())
    }
    return label_map, target_cols


class OnlineManifestMPPDataset(Dataset):
    """Load one patient's fixed manifest split from raw PNG patches."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        mpp_root: str | Path,
        mpp_id: int,
        patient: str,
        split: str,
        labels_csv: str | Path,
        transform: Optional[Callable],
        target_cols: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        validate_manifest_frame(manifest_df)
        if patient == EXTERNAL_PATIENT and split != "external_test":
            raise ValueError("XZY is external_test only")
        if patient != EXTERNAL_PATIENT and split == "external_test":
            raise ValueError("external_test is reserved for XZY")

        self.mpp_id = int(mpp_id)
        self.patient = patient
        self.split = split
        self.transform = transform
        self.mpp_root = Path(mpp_root)

        subset = manifest_df[
            (manifest_df["mpp_id"].astype(int) == self.mpp_id)
            & (manifest_df["patient"].astype(str) == patient)
            & (manifest_df["split"].astype(str) == split)
        ].copy()
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be a positive integer")
            subset = subset.head(limit)
        if subset.empty:
            raise ValueError(f"manifest contains no samples for {mpp_id}/{patient}/{split}")

        label_map, resolved_targets = _load_patient_labels(labels_csv, target_cols)
        self.target_cols = resolved_targets
        expected_stems = subset["patch_stem"].astype(str).map(_normalise_stem)
        missing_labels = sorted(set(expected_stems) - set(label_map))
        extra_labels = sorted(set(label_map) - set(expected_stems))
        if missing_labels or (limit is None and extra_labels):
            raise ValueError(
                f"manifest/label mismatch for {mpp_id}/{patient}/{split}: "
                f"missing_labels={missing_labels[:5]} extra_labels={extra_labels[:5]}"
            )

        image_root = self.mpp_root / str(self.mpp_id) / patient / "patch_images"
        samples: List[OnlineMPPSample] = []
        missing_images: List[str] = []
        for row in subset.itertuples(index=False):
            stem = _normalise_stem(row.patch_stem)
            image_path = image_root / f"{stem}.png"
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            samples.append(OnlineMPPSample(
                mpp_id=self.mpp_id,
                patient=patient,
                patch_stem=stem,
                split=split,
                block_id=str(row.block_id),
                image_path=image_path,
                target=torch.tensor(label_map[stem], dtype=torch.float32),
            ))
        if missing_images:
            raise FileNotFoundError(
                f"missing {len(missing_images)} manifest PNG files; examples={missing_images[:5]}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            value = self.transform(rgb) if self.transform is not None else rgb.copy()
        return value, sample.target.clone()

    def metadata_rows(self) -> List[dict]:
        return [{
            "mpp_id": sample.mpp_id,
            "patient": sample.patient,
            "patch_stem": sample.patch_stem,
            "split": sample.split,
            "block_id": sample.block_id,
            "image_path": str(sample.image_path),
        } for sample in self.samples]


def _patient_label_path(labels_root: Path, split: str, patient: str) -> Path:
    if split == "train":
        return labels_root / "train" / patient / f"{patient}_ssGSEA_zscore.csv"
    if split == "internal_val":
        return labels_root / "val" / patient / f"{patient}_ssGSEA_zscore.csv"
    if split == "external_test":
        return (
            labels_root / "external" / patient
            / f"{patient}_ssGSEA_zscore_by_group_2_train.csv"
        )
    raise ValueError(f"unsupported split: {split}")


def _external_manifest_from_labels(labels_csv: Path, mpp_id: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    if labels.empty:
        raise ValueError(f"external labels are empty: {labels_csv}")
    stems = labels.iloc[:, 0].map(_normalise_stem)
    return pd.DataFrame({
        "mpp_id": mpp_id,
        "patient": EXTERNAL_PATIENT,
        "patch_stem": stems,
        "x": 0,
        "y": 0,
        "split": "external_test",
        "block_id": "external_test",
    })


def build_online_manifest_datasets(
    manifest_df: pd.DataFrame,
    mpp_root: str | Path,
    labels_root: str | Path,
    transform: Callable,
    train_mpp_id: int = 2,
    external_mpp_id: int = 2,
    train_patients: Sequence[str] = TRAIN_PATIENTS,
    limit_per_patient: Optional[int] = None,
) -> Tuple[ConcatDataset, ConcatDataset, OnlineManifestMPPDataset, List[str]]:
    """Build fixed train/internal-val/external datasets without cache fallback."""
    validate_manifest_frame(manifest_df)
    if external_mpp_id != 2:
        raise ValueError("current protocol fixes external_mpp_id=2")
    if EXTERNAL_PATIENT in train_patients:
        raise ValueError("XZY must not appear in train_patients")

    labels_root = Path(labels_root)
    target_cols: Optional[List[str]] = None
    train_sets = []
    val_sets = []
    for patient in train_patients:
        train_set = OnlineManifestMPPDataset(
            manifest_df, mpp_root, train_mpp_id, patient, "train",
            _patient_label_path(labels_root, "train", patient), transform,
            target_cols=target_cols, limit=limit_per_patient,
        )
        if target_cols is None:
            target_cols = train_set.target_cols
        val_set = OnlineManifestMPPDataset(
            manifest_df, mpp_root, train_mpp_id, patient, "internal_val",
            _patient_label_path(labels_root, "internal_val", patient), transform,
            target_cols=target_cols, limit=limit_per_patient,
        )
        train_sets.append(train_set)
        val_sets.append(val_set)

    external_labels = _patient_label_path(labels_root, "external_test", EXTERNAL_PATIENT)
    external_manifest = _external_manifest_from_labels(external_labels, external_mpp_id)
    external_set = OnlineManifestMPPDataset(
        external_manifest, mpp_root, external_mpp_id, EXTERNAL_PATIENT,
        "external_test", external_labels, transform, target_cols=target_cols,
        limit=limit_per_patient,
    )
    assert target_cols is not None
    return ConcatDataset(train_sets), ConcatDataset(val_sets), external_set, target_cols
