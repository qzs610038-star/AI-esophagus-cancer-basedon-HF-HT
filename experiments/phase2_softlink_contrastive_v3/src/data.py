"""Manifest-driven raw-label and raw-image loading for both protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IDENTITY_COLUMNS = ("mpp_id", "patient", "patch_stem")
REQUIRED_COLUMNS = (*IDENTITY_COLUMNS, "x", "y", "split", "block_id")
NON_TARGET_COLUMNS = {
    "x", "y", "spot", "spot_id", "barcode", "index", "patch_id",
    "filename", "patient", "patient_id", "mpp_id", "split", "block_id",
}


@dataclass(frozen=True)
class OnlineRecord:
    mpp_id: int
    patient: str
    patch_stem: str
    x: int
    y: int
    original_split: str
    block_id: str
    image_path: Path
    raw_target: np.ndarray

    @property
    def identity(self) -> tuple[int, str, str]:
        return self.mpp_id, self.patient, self.patch_stem


@dataclass(frozen=True)
class SampleRequest:
    index: int
    epoch: int = 0
    update: int = 0
    slot: int = 0


def read_split_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"split manifest missing columns: {missing}")
    if frame.empty or frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("split manifest is empty or contains null identities")
    duplicate = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate point identities: "
            f"{frame.loc[duplicate, list(IDENTITY_COLUMNS)].head().to_dict('records')}"
        )
    if "XZY" in set(frame["patient"].astype(str)):
        raise ValueError("XZY must not enter the development/LOPO manifest")
    return frame.reset_index(drop=True)


def _stem(value: object) -> str:
    return Path(str(value)).stem


def load_patient_raw_labels(
    path: str | Path,
    pathway_names: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"raw label file is empty: {path}")
    id_column = frame.columns[0]
    if pathway_names is None:
        pathway_names = tuple(
            column for column in frame.select_dtypes(include=["number"]).columns
            if column != id_column and str(column).lower() not in NON_TARGET_COLUMNS
        )
    else:
        pathway_names = tuple(pathway_names)
    if len(pathway_names) != 30:
        raise ValueError(f"expected 30 pathways, got {len(pathway_names)} in {path}")
    missing = [name for name in pathway_names if name not in frame]
    if missing:
        raise ValueError(f"raw labels missing pathways: {missing}")
    stems = frame[id_column].map(_stem)
    if stems.duplicated().any():
        raise ValueError(f"duplicate label stems in {path}")
    values = frame[list(pathway_names)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite raw labels in {path}")
    return {stem: value.copy() for stem, value in zip(stems, values)}, pathway_names


def materialize_records(
    rows: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    raw_label_template: str,
    image_template: str,
    pathway_names: Sequence[str] | None = None,
) -> tuple[list[OnlineRecord], tuple[str, ...]]:
    records_frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if records_frame.empty:
        raise ValueError("cannot materialize an empty protocol partition")
    labels_by_patient: dict[str, dict[str, np.ndarray]] = {}
    resolved_pathways = tuple(pathway_names) if pathway_names is not None else None
    output: list[OnlineRecord] = []
    for row in records_frame.itertuples(index=False):
        patient = str(row.patient)
        mpp_id = int(row.mpp_id)
        if patient not in labels_by_patient:
            label_path = Path(raw_label_template.format(
                group=mpp_id, mpp_id=mpp_id, patient=patient
            ))
            labels, resolved = load_patient_raw_labels(label_path, resolved_pathways)
            labels_by_patient[patient] = labels
            if resolved_pathways is None:
                resolved_pathways = resolved
        stem = _stem(row.patch_stem)
        if stem not in labels_by_patient[patient]:
            raise KeyError(f"missing raw label for {(mpp_id, patient, stem)}")
        image_path = Path(image_template.format(
            group=mpp_id, mpp_id=mpp_id, patient=patient, patch_stem=stem
        ))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing patch image: {image_path}")
        output.append(OnlineRecord(
            mpp_id=mpp_id,
            patient=patient,
            patch_stem=stem,
            x=int(row.x),
            y=int(row.y),
            original_split=str(row.split),
            block_id=str(row.block_id),
            image_path=image_path,
            raw_target=labels_by_patient[patient][stem],
        ))
    if len({record.identity for record in output}) != len(output):
        raise ValueError("materialized protocol partition has duplicate identities")
    assert resolved_pathways is not None
    return output, tuple(resolved_pathways)


def fit_zscore(records: Sequence[OnlineRecord]) -> dict[str, object]:
    """Fit normalization exclusively from the supplied training records."""
    if not records:
        raise ValueError("z-score fitting requires training records")
    values = np.stack([record.raw_target for record in records]).astype(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("training-only z-score fit has zero/non-finite pathway variance")
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "fit_identities": [record.identity for record in records],
        "fit_patients": sorted({record.patient for record in records}),
    }


def _stable_text_code(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def view_seed(
    protocol: str,
    fold: str,
    seed: int,
    epoch: int,
    update: int,
    slot: int,
) -> int:
    values = (
        _stable_text_code(protocol), _stable_text_code(fold), int(seed),
        int(epoch), int(update), int(slot),
    )
    sequence = np.random.SeedSequence(values)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def build_image_transform(*, augment: bool, image_size: int = 224) -> Callable:
    """Return official normalization with deterministic weak augmentation."""
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("torchvision is required for image preprocessing") from exc
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def transform(image: Image.Image, *, random_seed: int = 0) -> torch.Tensor:
        rng = random.Random(int(random_seed))
        image = F.resize(image, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)
        if augment:
            if rng.random() < 0.5:
                image = F.hflip(image)
            if rng.random() < 0.5:
                image = F.vflip(image)
            image = F.adjust_brightness(image, rng.uniform(0.9, 1.1))
            image = F.adjust_contrast(image, rng.uniform(0.9, 1.1))
            image = F.adjust_saturation(image, rng.uniform(0.9, 1.1))
            image = F.adjust_hue(image, rng.uniform(-0.02, 0.02))
        return F.normalize(F.to_tensor(image), mean, std)

    return transform


class OnlinePatchDataset(Dataset):
    """Dataset accepting ``SampleRequest`` so view randomness is cell-shared."""

    def __init__(
        self,
        records: Sequence[OnlineRecord],
        normalization: Mapping[str, object],
        transform: Callable,
        *,
        protocol: str,
        fold: str,
        seed: int,
        augment: bool,
    ) -> None:
        self.records = list(records)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        self.transform = transform
        self.protocol = protocol
        self.fold = fold
        self.seed = int(seed)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, request: int | SampleRequest) -> dict[str, object]:
        if isinstance(request, int):
            request = SampleRequest(request)
        record = self.records[request.index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
            random_seed = view_seed(
                self.protocol, self.fold, self.seed,
                request.epoch, request.update, request.slot,
            ) if self.augment else 0
            tensor = self.transform(image, random_seed=random_seed)
        target = (record.raw_target - self.mean) / self.std
        return {
            "image": tensor,
            "target": torch.from_numpy(target.astype(np.float32, copy=False)),
            "raw_target": torch.from_numpy(record.raw_target.copy()),
            "patient": record.patient,
            "identity": record.identity,
            "xy": (record.x, record.y),
            "block_id": record.block_id,
        }
