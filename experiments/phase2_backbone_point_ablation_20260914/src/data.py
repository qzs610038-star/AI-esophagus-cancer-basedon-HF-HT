"""Identity-keyed image, label, and normalization data contracts."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from errors import IdentityMismatchError, NonFiniteDataError


COORDINATE_PATTERN = re.compile(r"(?:^|_)x(-?\d+)_y(-?\d+)(?:$|_)")
COMMON_COLUMNS = (
    "row_index",
    "patient_id",
    "source_group",
    "spot_id",
    "split",
    "x",
    "y",
    "image_path",
    "width_px",
    "height_px",
)


@dataclass(frozen=True)
class IdentityRecord:
    row_index: int
    patient_id: str
    source_group: str
    spot_id: str
    split: str
    x: int
    y: int
    image_path: str
    width_px: int
    height_px: int

    @property
    def identity_key(self) -> str:
        return f"{self.patient_id}|{self.source_group}|{self.spot_id}"

    def as_csv_row(self) -> dict:
        return {name: getattr(self, name) for name in COMMON_COLUMNS}


@dataclass(frozen=True)
class Normalization:
    pathway_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    fit_split: str
    ddof: int
    clip_range: tuple[float, float]
    n_train_samples: int
    source_path: str

    def inverse_transform(self, pred_z: np.ndarray) -> np.ndarray:
        values = np.asarray(pred_z, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.pathway_names):
            raise IdentityMismatchError("逆变换输入必须是 [N,30] 且通路顺序已对齐")
        if not np.isfinite(values).all():
            raise NonFiniteDataError("逆变换输入含非有限值")
        return values * self.std + self.mean


def parse_coordinates(stem: str) -> tuple[int, int]:
    match = COORDINATE_PATTERN.search(str(stem))
    if match is None:
        raise IdentityMismatchError(f"无法从 patch 名解析坐标: {stem!r}")
    return int(match.group(1)), int(match.group(2))


def _image_info(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise IdentityMismatchError(f"图像不存在，禁止静默删行: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return int(image.width), int(image.height)
    except OSError as exc:
        raise IdentityMismatchError(f"图像不可读取: {path}: {exc}") from exc


def _source_group(mpp_id: int, patient_id: str) -> str:
    return f"MPP{int(mpp_id)}_{patient_id}_source01"


def _validate_unique(records: Sequence[IdentityRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.identity_key in seen:
            raise IdentityMismatchError(f"重复身份键: {record.identity_key}")
        seen.add(record.identity_key)


def _validate_counts(records: Sequence[IdentityRecord], expected_counts: dict[str, int]) -> None:
    actual = {
        split: sum(record.split == split for record in records)
        for split in ("train", "internal_val", "external_test")
    }
    if actual != {key: int(value) for key, value in expected_counts.items()}:
        raise IdentityMismatchError(f"共同身份计数不匹配: expected={expected_counts}, actual={actual}")


def _write_common_manifest(path: Path, records: Sequence[IdentityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMMON_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_csv_row())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_common_identity_manifest(
    split_manifest_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    *,
    mpp_id: int,
    external_patient: str,
    expected_counts: dict[str, int],
) -> list[IdentityRecord]:
    """Resolve every image by explicit identity and record its actual pixel size."""

    split_path = Path(split_manifest_path)
    root = Path(image_root)
    try:
        with split_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"mpp_id", "patient", "patch_stem", "x", "y", "split"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise IdentityMismatchError(f"split_manifest 缺少列: {sorted(missing)}")
            internal_rows = list(reader)
    except OSError as exc:
        raise IdentityMismatchError(f"无法读取 split_manifest: {split_path}: {exc}") from exc

    records_without_index: list[dict] = []
    for row in internal_rows:
        patient = str(row["patient"]).strip()
        spot = Path(str(row["patch_stem"]).strip()).stem
        split = str(row["split"]).strip()
        if int(row["mpp_id"]) != int(mpp_id):
            raise IdentityMismatchError(f"split_manifest 含非 MPP{mpp_id} 行")
        if split not in {"train", "internal_val"}:
            raise IdentityMismatchError(f"开发集出现未知 split: {split!r}")
        x, y = int(row["x"]), int(row["y"])
        parsed_x, parsed_y = parse_coordinates(spot)
        if (x, y) != (parsed_x, parsed_y):
            raise IdentityMismatchError(f"坐标与 patch 名不一致: {patient}/{spot}")
        image_path = root / patient / "patch_images" / f"{spot}.png"
        width, height = _image_info(image_path)
        records_without_index.append(
            dict(
                patient_id=patient,
                source_group=_source_group(mpp_id, patient),
                spot_id=spot,
                split=split,
                x=x,
                y=y,
                image_path=str(image_path.resolve()),
                width_px=width,
                height_px=height,
            )
        )

    external_dir = root / external_patient / "patch_images"
    if not external_dir.is_dir():
        raise IdentityMismatchError(f"外部图像目录不存在: {external_dir}")
    external_files = sorted(external_dir.glob("*.png"), key=lambda item: item.stem)
    for image_path in external_files:
        spot = image_path.stem
        x, y = parse_coordinates(spot)
        width, height = _image_info(image_path)
        records_without_index.append(
            dict(
                patient_id=str(external_patient),
                source_group=_source_group(mpp_id, str(external_patient)),
                spot_id=spot,
                split="external_test",
                x=x,
                y=y,
                image_path=str(image_path.resolve()),
                width_px=width,
                height_px=height,
            )
        )

    # Development identities are sorted jointly (not split-by-split), matching
    # the accepted point training table. External identities follow afterward.
    records_without_index.sort(
        key=lambda row: (
            row["split"] == "external_test",
            row["patient_id"],
            row["source_group"],
            row["spot_id"],
        )
    )
    records = [IdentityRecord(index, **row) for index, row in enumerate(records_without_index)]
    _validate_unique(records)
    _validate_counts(records, expected_counts)
    _write_common_manifest(Path(output_path), records)
    return records


def load_common_manifest(
    path: str | Path,
    *,
    expected_counts: dict[str, int],
    require_images: bool = False,
) -> list[IdentityRecord]:
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != COMMON_COLUMNS:
                raise IdentityMismatchError("共同身份清单列或列顺序不符合 schema")
            records = [
                IdentityRecord(
                    int(row["row_index"]),
                    str(row["patient_id"]),
                    str(row["source_group"]),
                    str(row["spot_id"]),
                    str(row["split"]),
                    int(row["x"]),
                    int(row["y"]),
                    str(row["image_path"]),
                    int(row["width_px"]),
                    int(row["height_px"]),
                )
                for row in reader
            ]
    except OSError as exc:
        raise IdentityMismatchError(f"无法读取共同身份清单 {path}: {exc}") from exc
    if [record.row_index for record in records] != list(range(len(records))):
        raise IdentityMismatchError("共同身份清单 row_index 必须连续且保持原顺序")
    _validate_unique(records)
    _validate_counts(records, expected_counts)
    if require_images:
        for record in records:
            _image_info(Path(record.image_path))
    return records


def load_pathway_names(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    names = tuple(str(value) for value in (payload.get("pathway_names") or []))
    if len(names) != 30 or len(set(names)) != 30:
        raise IdentityMismatchError("zscore_manifest 必须登记 30 个唯一通路且保持顺序")
    return names


def load_normalization(path: str | Path, pathway_names: Sequence[str]) -> Normalization:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    pathways = payload.get("pathways") or {}
    means: list[float] = []
    standard_deviations: list[float] = []
    for name in pathway_names:
        if name not in pathways:
            raise IdentityMismatchError(f"标准化参数缺少通路 {name}")
        mean = float(pathways[name]["mean"])
        std = float(pathways[name]["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise NonFiniteDataError(f"通路 {name} 的 mean/std 非法")
        means.append(mean)
        standard_deviations.append(std)
    clip = payload.get("clip_range") or [-100.0, 100.0]
    return Normalization(
        tuple(pathway_names),
        np.asarray(means, dtype=np.float64),
        np.asarray(standard_deviations, dtype=np.float64),
        str(payload.get("fit_split") or "train"),
        int(payload.get("ddof", 1)),
        (float(clip[0]), float(clip[1])),
        int(payload.get("n_train_samples") or 0),
        str(Path(path).resolve()),
    )


def _read_label_map(path: Path, pathway_names: Sequence[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise IdentityMismatchError(f"标签文件不存在: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns:
            raise IdentityMismatchError(f"标签文件无表头: {path}")
        missing = [name for name in pathway_names if name not in columns]
        if missing:
            raise IdentityMismatchError(f"标签文件缺通路列: {missing[:5]}")
        first_column = columns[0]
        mapping: dict[str, np.ndarray] = {}
        for row in reader:
            spot = Path(str(row[first_column])).stem
            if spot in mapping:
                raise IdentityMismatchError(f"标签重复 spot: {path}/{spot}")
            values = np.asarray([float(row[name]) for name in pathway_names], dtype=np.float32)
            if not np.isfinite(values).all():
                raise NonFiniteDataError(f"标签含非有限值: {path}/{spot}")
            mapping[spot] = values
    return mapping


def load_labels_for_rows(
    rows: Sequence[IdentityRecord], labels_root: str | Path, pathway_names: Sequence[str]
) -> np.ndarray:
    """Training/validation-only loader: external rows are rejected before any I/O."""

    if any(row.split == "external_test" for row in rows):
        raise IdentityMismatchError("外部 XZY 行禁止进入服务器训练标签加载器")
    root = Path(labels_root)
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    labels = np.empty((len(rows), len(pathway_names)), dtype=np.float32)
    for index, row in enumerate(rows):
        if row.split not in {"train", "internal_val"}:
            raise IdentityMismatchError(f"标签加载遇到未知 split: {row.split}")
        directory = "train" if row.split == "train" else "val"
        key = (row.split, row.patient_id)
        path = root / directory / row.patient_id / f"{row.patient_id}_ssGSEA_zscore.csv"
        if key not in cache:
            cache[key] = _read_label_map(path, pathway_names)
        if row.spot_id not in cache[key]:
            raise IdentityMismatchError(f"标签与身份未对齐: {row.identity_key}")
        labels[index] = cache[key][row.spot_id]
    return labels


def iter_index_batches(indices: Iterable[int], batch_size: int) -> Iterable[np.ndarray]:
    values = np.asarray(list(indices), dtype=np.int64)
    if batch_size < 1:
        raise ValueError("batch_size 必须为正")
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]
