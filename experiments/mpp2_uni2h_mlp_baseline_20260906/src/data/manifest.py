"""Package-local manifest dataset. Do not import the repository."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from targets.protocol import TargetAdapter

COORD_RE = re.compile(r"x(\d+)_y(\d+)")
SKIP_COLS = {
    "x", "y", "spot", "id", "spot_id", "barcode", "index", "patch_id", "filename",
}


def _load_feature(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_xy(stem: str):
    match = COORD_RE.search(stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _find_pt(cache_root: Path, mpp_id: int, patient: str, stem: str, flat_cache_root: Optional[Path]) -> Optional[Path]:
    fname = f"{stem}.pt"
    candidates = [
        cache_root / f"MPP{mpp_id}_UNI" / patient,
        cache_root / f"MPP{mpp_id}_UNI" / patient / "train",
        cache_root / f"MPP{mpp_id}_UNI" / patient / "val",
        cache_root / str(mpp_id) / patient,
    ]
    if flat_cache_root:
        candidates.extend([
            flat_cache_root / str(mpp_id) / patient,
            flat_cache_root / f"MPP{mpp_id}_UNI" / patient,
        ])
    for folder in candidates:
        path = folder / fname
        if path.exists():
            return path
    return None


def _load_label_map(labels_csv: Path, target_cols: Sequence[str]) -> dict[str, np.ndarray]:
    table = pd.read_csv(labels_csv)
    first = table.columns[0]
    table["_stem"] = table[first].astype(str).map(lambda value: Path(value).stem)
    missing = [name for name in target_cols if name not in table.columns]
    if missing:
        raise ValueError(f"{labels_csv} is missing target columns: {missing[:5]}")
    return {
        row["_stem"]: row[list(target_cols)].to_numpy(dtype=np.float32)
        for _, row in table.iterrows()
    }


class ManifestMPPDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cache_root: str,
        mpp_id: int,
        patient: str,
        split: str,
        labels_csv: str,
        target_cols: Sequence[str],
        allow_missing: bool = False,
        flat_cache_root: Optional[str] = None,
    ):
        self.patient = patient
        self.split = split
        self.target_cols = list(target_cols)
        self.records: list[dict] = []
        self.samples: list[tuple[Path, torch.Tensor]] = []
        self._empty = False
        self.feat_dim = 1536

        subset = manifest_df[(manifest_df["patient"] == patient) & (manifest_df["split"] == split)].copy()
        if subset.empty:
            if allow_missing:
                self._empty = True
                return
            raise ValueError(f"{patient}/{split}: manifest has no patches")

        label_map = {}
        if labels_csv and Path(labels_csv).exists():
            label_map = _load_label_map(Path(labels_csv), self.target_cols)
        elif not allow_missing:
            raise FileNotFoundError(f"label csv missing: {labels_csv}")

        unmatched = []
        for _, row in subset.iterrows():
            stem = str(row["patch_stem"])
            pt_path = _find_pt(Path(cache_root), mpp_id, patient, stem, Path(flat_cache_root) if flat_cache_root else None)
            if pt_path is None:
                unmatched.append((stem, "no .pt"))
                continue
            if label_map and stem not in label_map:
                unmatched.append((stem, "no label"))
                continue
            target = torch.tensor(label_map[stem], dtype=torch.float32) if label_map else torch.zeros(len(self.target_cols))
            x = int(row["x"]) if "x" in row and pd.notna(row["x"]) else (parse_xy(stem)[0] or -1)
            y = int(row["y"]) if "y" in row and pd.notna(row["y"]) else (parse_xy(stem)[1] or -1)
            self.samples.append((pt_path, target))
            self.records.append({
                "patient_id": patient,
                "patch_id": stem,
                "x": x,
                "y": y,
                "split": split,
            })

        if unmatched and not allow_missing:
            raise ValueError(f"{patient}/{split}: {len(unmatched)} unmatched, e.g. {unmatched[:3]}")
        if not self.samples:
            if allow_missing:
                self._empty = True
                return
            raise ValueError(f"{patient}/{split}: no matched samples")

        feature = _load_feature(self.samples[0][0])
        self.feat_dim = int(feature.shape[1] if feature.ndim == 2 else feature.shape[0])
        print(f"  [DS] {patient}/{split}: {len(self.samples)} samples, feat_dim={self.feat_dim}", flush=True)

    def __len__(self):
        return 0 if self._empty else len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        feature = _load_feature(path)
        if feature.ndim == 2:
            feature = feature[0]
        return feature, target


def collect_records(dataset) -> list[dict]:
    if hasattr(dataset, "records"):
        return list(dataset.records)
    if isinstance(dataset, ConcatDataset):
        rows = []
        for part in dataset.datasets:
            rows.extend(collect_records(part))
        return rows
    raise TypeError(f"cannot collect records from {type(dataset)}")


def _merge_patients(
    manifest_df: pd.DataFrame,
    cache_root: str,
    mpp_id: int,
    patients: Sequence[str],
    split: str,
    target: TargetAdapter,
    labels_root: Path,
    allow_missing: bool,
    flat_cache_root: Optional[str],
    train_mpp_id: int,
) -> ConcatDataset:
    parts = []
    for patient in patients:
        labels_csv = target.label_csv(labels_root, split, patient, train_mpp_id)
        part = ManifestMPPDataset(
            manifest_df=manifest_df,
            cache_root=cache_root,
            mpp_id=mpp_id,
            patient=patient,
            split=split,
            labels_csv=str(labels_csv),
            target_cols=target.spec().names,
            allow_missing=allow_missing,
            flat_cache_root=flat_cache_root,
        )
        if len(part) > 0:
            parts.append(part)
    if not parts:
        raise RuntimeError(f"no usable {split} datasets")
    return ConcatDataset(parts)


def build_external(
    flat_cache_root: str,
    external_mpp_id: int,
    external_patient: str,
    labels_csv: Path,
    target_cols: Sequence[str],
    allow_missing: bool,
) -> ManifestMPPDataset:
    candidates = [
        Path(flat_cache_root) / str(external_mpp_id) / external_patient,
        Path(flat_cache_root) / f"MPP{external_mpp_id}_UNI" / external_patient,
    ]
    cache = next((path for path in candidates if path.exists()), None)
    if cache is None:
        if allow_missing:
            empty = ManifestMPPDataset.__new__(ManifestMPPDataset)
            empty._empty = True
            empty.feat_dim = 1536
            empty.target_cols = list(target_cols)
            empty.samples = []
            empty.records = []
            empty.patient = external_patient
            empty.split = "external"
            return empty
        raise FileNotFoundError(f"external cache missing: {candidates}")

    label_map = _load_label_map(labels_csv, target_cols)
    samples = []
    records = []
    for pt_path in sorted(cache.glob("*.pt")):
        if pt_path.stem not in label_map:
            continue
        x, y = parse_xy(pt_path.stem)
        samples.append((pt_path, torch.tensor(label_map[pt_path.stem], dtype=torch.float32)))
        records.append({
            "patient_id": external_patient,
            "patch_id": pt_path.stem,
            "x": -1 if x is None else x,
            "y": -1 if y is None else y,
            "split": "external",
        })
    if not samples:
        raise ValueError(f"external {external_patient}: no matched samples")
    feature = _load_feature(samples[0][0])
    dataset = ManifestMPPDataset.__new__(ManifestMPPDataset)
    dataset.patient = external_patient
    dataset.split = "external"
    dataset.target_cols = list(target_cols)
    dataset.samples = samples
    dataset.records = records
    dataset._empty = False
    dataset.feat_dim = int(feature.shape[1] if feature.ndim == 2 else feature.shape[0])
    print(f"  [DS] external {external_patient}: {len(samples)} samples, feat_dim={dataset.feat_dim}", flush=True)
    return dataset


def build_manifest_datasets(
    manifest_df: pd.DataFrame,
    cache_root: str,
    flat_cache_root: str,
    labels_root: str,
    target: TargetAdapter,
    train_mpp_id: int,
    external_mpp_id: int,
    external_patient: str,
    train_patients: Sequence[str],
    allow_missing: bool = False,
) -> Tuple[ConcatDataset, ConcatDataset, ManifestMPPDataset]:
    labels = Path(labels_root)
    train_ds = _merge_patients(
        manifest_df, cache_root, train_mpp_id, train_patients, "train",
        target, labels, allow_missing, flat_cache_root, train_mpp_id,
    )
    val_ds = _merge_patients(
        manifest_df, cache_root, train_mpp_id, train_patients, "internal_val",
        target, labels, allow_missing, flat_cache_root, train_mpp_id,
    )
    ext_ds = build_external(
        flat_cache_root=flat_cache_root,
        external_mpp_id=external_mpp_id,
        external_patient=external_patient,
        labels_csv=target.label_csv(labels, "external", external_patient, train_mpp_id),
        target_cols=target.spec().names,
        allow_missing=allow_missing,
    )
    return train_ds, val_ds, ext_ds
