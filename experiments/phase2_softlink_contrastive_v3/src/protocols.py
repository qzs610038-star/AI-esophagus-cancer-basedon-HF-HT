"""Protocol-local splits and fitting artifacts.

This module deliberately works on manifest rows only.  It never looks at XZY
or held-out rows while fitting a transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


_PATIENT_COLUMNS = ("patient", "patient_id")
_TRAIN_NAMES = {"train", "training"}
_VAL_NAMES = {"internal_val", "val", "validation"}


@dataclass(frozen=True)
class ProtocolSplit:
    protocol: str
    fold: str
    train_rows: pd.DataFrame
    internal_val_rows: pd.DataFrame
    held_out_rows: pd.DataFrame


@dataclass(frozen=True)
class ZScoreArtifact:
    mean: np.ndarray
    std: np.ndarray
    fit_identity: tuple[tuple[str, ...], ...]
    ddof: int = 1

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std


@dataclass(frozen=True)
class PatientMeansArtifact:
    patient_means: dict[str, np.ndarray]
    fit_identity: tuple[tuple[str, ...], ...]

    def transform(self, values: np.ndarray, patients: Sequence[str]) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        return np.vstack([data[i] - self.patient_means[str(p)] for i, p in enumerate(patients)])


def _frame(manifest: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = manifest.copy() if isinstance(manifest, pd.DataFrame) else pd.DataFrame(manifest)
    if frame.empty:
        raise ValueError("manifest 不能为空")
    if not any(c in frame.columns for c in _PATIENT_COLUMNS):
        raise ValueError("manifest 必须含 patient 或 patient_id")
    if "split" not in frame.columns:
        raise ValueError("manifest 必须含 split")
    return frame.reset_index(drop=True)


def _patient_column(frame: pd.DataFrame) -> str:
    return next(c for c in _PATIENT_COLUMNS if c in frame.columns)


def _identity(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    columns = [c for c in ("patient", "patient_id", "slide_id", "patch_stem", "spot_id") if c in frame.columns]
    if not columns:
        return tuple((str(i),) for i in frame.index)
    return tuple(tuple(str(v) for v in row) for row in frame[columns].itertuples(index=False, name=None))


def _split_rows(frame: pd.DataFrame, names: set[str]) -> pd.DataFrame:
    return frame[frame["split"].astype(str).str.lower().isin(names)].copy().reset_index(drop=True)


def assert_identity_disjoint(split: ProtocolSplit) -> None:
    """Assert point identities are disjoint (train/val may share a patient)."""
    groups = [set(_identity(rows)) for rows in (split.train_rows, split.internal_val_rows, split.held_out_rows)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("训练、内部验证和留出集的点位身份必须互斥")


def build_original_split(manifest: pd.DataFrame | Iterable[dict[str, Any]]) -> ProtocolSplit:
    """Keep the original train/internal-validation rows; external XZY is absent."""
    frame = _frame(manifest)
    train = _split_rows(frame, _TRAIN_NAMES)
    val = _split_rows(frame, _VAL_NAMES)
    if train.empty or val.empty:
        raise ValueError("原始协议必须同时含 train 和 internal_val")
    result = ProtocolSplit("original", "original", train, val, frame.iloc[0:0].copy())
    assert_identity_disjoint(result)
    return result


def build_lopo6_splits(manifest: pd.DataFrame | Iterable[dict[str, Any]]) -> list[ProtocolSplit]:
    """Leave one of the six in-domain patients out, retaining other rows' labels."""
    frame = _frame(manifest)
    patient_col = _patient_column(frame)
    usable = frame[frame[patient_col].astype(str) != "XZY"].copy()
    patients = sorted(usable[patient_col].astype(str).unique())
    if len(patients) != 6:
        raise ValueError(f"LOPO6 要求恰好 6 名非 XZY 患者，当前={patients}")
    splits: list[ProtocolSplit] = []
    for held in patients:
        held_rows = usable[usable[patient_col].astype(str) == held].copy().reset_index(drop=True)
        other = usable[usable[patient_col].astype(str) != held]
        result = ProtocolSplit(
            "lopo6", held,
            _split_rows(other, _TRAIN_NAMES),
            _split_rows(other, _VAL_NAMES), held_rows,
        )
        train_patients = set(result.train_rows[patient_col].astype(str))
        val_patients = set(result.internal_val_rows[patient_col].astype(str))
        held_patients = set(result.held_out_rows[patient_col].astype(str))
        if held_patients != {held} or held in train_patients or held in val_patients:
            raise AssertionError("LOPO 留出患者泄漏到训练或内部验证")
        if "XZY" in train_patients | val_patients | held_patients:
            raise AssertionError("XZY 不得进入 LOPO")
        assert_identity_disjoint(result)
        splits.append(result)
    return splits


def _array_and_identity(train_rows: pd.DataFrame, values: np.ndarray | None, label_columns: Sequence[str] | None):
    if values is None:
        if not label_columns:
            raise ValueError("values 为 None 时必须提供 label_columns")
        values = train_rows[list(label_columns)].to_numpy(dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(train_rows):
        raise ValueError("拟合值必须是与 train_rows 等长的二维数组")
    if not np.isfinite(arr).all():
        raise ValueError("训练集拟合值含非有限数")
    return arr, _identity(train_rows)


def fit_zscore(train_rows: pd.DataFrame, values: np.ndarray | None = None, *, label_columns: Sequence[str] | None = None, ddof: int = 1) -> ZScoreArtifact:
    """Fit global z-score parameters on this protocol's training rows only."""
    arr, identity = _array_and_identity(train_rows, values, label_columns)
    if arr.shape[0] <= ddof:
        raise ValueError("训练行数不足以拟合 z-score")
    std = arr.std(axis=0, ddof=ddof)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("训练集存在零方差或非有限通路，拒绝拟合")
    return ZScoreArtifact(arr.mean(axis=0), std, identity, ddof)


def fit_patient_means(train_rows: pd.DataFrame, values: np.ndarray | None = None, *, label_columns: Sequence[str] | None = None) -> PatientMeansArtifact:
    """Fit each patient's pathway mean using training rows only."""
    arr, identity = _array_and_identity(train_rows, values, label_columns)
    patient_col = _patient_column(train_rows)
    means: dict[str, np.ndarray] = {}
    patients = train_rows[patient_col].astype(str).to_numpy()
    for patient in sorted(set(patients)):
        means[patient] = arr[patients == patient].mean(axis=0)
    return PatientMeansArtifact(means, identity)
