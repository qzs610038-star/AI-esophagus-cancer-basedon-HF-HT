"""Identity-keyed image manifests for feature preparation only."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from errors import IdentityMismatchError
from transforms import FULL_FOV_PROTOCOL


COORDINATE_PATTERN = re.compile(r"(?:^|_)x(-?\d+)_y(-?\d+)(?:$|_)")
COMMON_COLUMNS = ("row_index", "patient_id", "source_group", "spot_id", "split", "x", "y", "image_path", "width_px", "height_px")
DATASET_SPLITS = {"all": None, "development": {"train", "internal_val"}, "train": {"train"}, "internal_val": {"internal_val"}, "external_test": {"external_test"}}


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
        return {column: getattr(self, column) for column in COMMON_COLUMNS}


def parse_coordinates(stem: str) -> tuple[int, int]:
    match = COORDINATE_PATTERN.search(str(stem))
    if not match:
        raise IdentityMismatchError(f"无法从 patch 名解析坐标: {stem!r}")
    return int(match.group(1)), int(match.group(2))


def _image_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise IdentityMismatchError(f"图像不存在，禁止静默删行: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return int(image.width), int(image.height)
    except OSError as exc:
        raise IdentityMismatchError(f"图像不可读取: {path}: {exc}") from exc


def _group(mpp_id: int, patient: str) -> str:
    return f"MPP{int(mpp_id)}_{patient}_source01"


def _validate(records: Sequence[IdentityRecord], expected_counts: dict[str, int] | None = None) -> None:
    if not records or len({record.identity_key for record in records}) != len(records):
        raise IdentityMismatchError("共同身份清单为空或含重复身份")
    if [record.row_index for record in records] != list(range(len(records))):
        raise IdentityMismatchError("共同身份清单 row_index 必须连续且保持顺序")
    if expected_counts is not None:
        observed = {split: sum(row.split == split for row in records) for split in ("train", "internal_val", "external_test")}
        if observed != {key: int(value) for key, value in expected_counts.items()}:
            raise IdentityMismatchError(f"共同身份计数不匹配: expected={expected_counts}, actual={observed}")


def write_common_manifest(path: str | Path, records: Sequence[IdentityRecord]) -> Path:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMMON_COLUMNS)); writer.writeheader()
        writer.writerows(record.as_csv_row() for record in records)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def build_common_identity_manifest(split_manifest_path: str | Path, image_root: str | Path, output_path: str | Path, *, mpp_id: int, external_patient: str, expected_counts: dict[str, int]) -> list[IdentityRecord]:
    """Resolve all requested image paths by identity and retain their observed size."""
    try:
        with Path(split_manifest_path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"mpp_id", "patient", "patch_stem", "x", "y", "split"}
            if required - set(reader.fieldnames or ()):
                raise IdentityMismatchError(f"split_manifest 缺少列: {sorted(required - set(reader.fieldnames or ())) }")
            development = list(reader)
    except OSError as exc:
        raise IdentityMismatchError(f"无法读取 split_manifest: {exc}") from exc
    root, values = Path(image_root), []
    for row in development:
        patient, stem, split = str(row["patient"]).strip(), Path(str(row["patch_stem"]).strip()).stem, str(row["split"]).strip()
        if int(row["mpp_id"]) != int(mpp_id) or split not in {"train", "internal_val"}:
            raise IdentityMismatchError("开发集 mpp_id 或 split 不符合冻结协议")
        x, y = int(row["x"]), int(row["y"])
        if (x, y) != parse_coordinates(stem):
            raise IdentityMismatchError(f"坐标与 patch 名不一致: {patient}/{stem}")
        image = root / patient / "patch_images" / f"{stem}.png"
        width, height = _image_size(image)
        values.append((patient, _group(mpp_id, patient), stem, split, x, y, str(image.resolve()), width, height))
    directory = root / str(external_patient) / "patch_images"
    if not directory.is_dir():
        raise IdentityMismatchError(f"外部图像目录不存在: {directory}")
    for image in sorted(directory.glob("*.png"), key=lambda value: value.stem):
        x, y = parse_coordinates(image.stem); width, height = _image_size(image)
        values.append((str(external_patient), _group(mpp_id, str(external_patient)), image.stem, "external_test", x, y, str(image.resolve()), width, height))
    values.sort(key=lambda row: (row[3] == "external_test", row[0], row[1], row[2]))
    records = [IdentityRecord(index, *row) for index, row in enumerate(values)]
    _validate(records, expected_counts); write_common_manifest(output_path, records)
    return records


def load_common_manifest(path: str | Path, *, expected_counts: dict[str, int] | None = None, require_images: bool = False) -> list[IdentityRecord]:
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != COMMON_COLUMNS:
                raise IdentityMismatchError("共同身份清单列或列顺序不符合 schema")
            records = [IdentityRecord(int(row["row_index"]), str(row["patient_id"]), str(row["source_group"]), str(row["spot_id"]), str(row["split"]), int(row["x"]), int(row["y"]), str(row["image_path"]), int(row["width_px"]), int(row["height_px"])) for row in reader]
    except OSError as exc:
        raise IdentityMismatchError(f"无法读取共同身份清单 {path}: {exc}") from exc
    _validate(records, expected_counts)
    if require_images:
        for record in records:
            if _image_size(Path(record.image_path)) != (record.width_px, record.height_px):
                raise IdentityMismatchError(f"图像尺寸在建清单后改变: {record.identity_key}")
    return records


def select_dataset_rows(rows: Sequence[IdentityRecord], dataset: str, *, protocol: str | None = None) -> list[IdentityRecord]:
    if dataset not in DATASET_SPLITS:
        raise IdentityMismatchError(f"未知特征数据集 {dataset!r}; 可用={list(DATASET_SPLITS)}")
    allowed = DATASET_SPLITS[dataset]
    selected = [row for row in rows if allowed is None or row.split in allowed]
    if not selected:
        raise IdentityMismatchError(f"数据集 {dataset} 没有任何身份")
    if protocol == FULL_FOV_PROTOCOL:
        non_square = [row for row in selected if row.width_px != row.height_px]
        if non_square:
            example = non_square[0]
            raise IdentityMismatchError(f"全视野协议发现 {len(non_square)} 个非方图；示例={example.identity_key}, size={(example.width_px, example.height_px)}")
    return selected


def common_manifest_path(config: dict) -> Path:
    return Path(config["paths"]["feature_caches_root"]) / config["experiment_id"] / "common_identity_v1" / "common_identity_manifest.csv"


def ensure_common_identity_manifest(config: dict, *, require_images: bool = True) -> tuple[Path, list[IdentityRecord]]:
    path, counts = common_manifest_path(config), config["data"]["expected_counts"]
    if path.is_file():
        return path, load_common_manifest(path, expected_counts=counts, require_images=require_images)
    rows = build_common_identity_manifest(config["inputs"]["split_manifest"], config["paths"]["image_root"], path, mpp_id=int(config["data"]["mpp_id"]), external_patient=str(config["data"]["external_patient"]), expected_counts=counts)
    return path, rows


def identities(rows: Iterable[IdentityRecord]) -> list[str]:
    return [row.identity_key for row in rows]
