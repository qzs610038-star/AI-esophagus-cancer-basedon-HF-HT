"""Identity-keyed point tables, labels, features, and one-shot inverse transform."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from errors import IdentityMismatchError, NonFiniteDataError, SlideMappingMissingError

COORD_RE = re.compile(r"x(\d+)_y(\d+)")
VERIFIED_MAPPING_STATUS = "verified"
EXPECTED_TRAIN = 9472
EXPECTED_VAL = 1078
EXPECTED_PATHWAYS = 30
EXPECTED_INPUT_DIM = 1536


@dataclass(frozen=True, order=True)
class PointIdentity:
    patient_id: str
    slide_id: str
    spot_id: str

    def key(self) -> tuple[str, str, str]:
        return (self.patient_id, self.slide_id, self.spot_id)

    def has_verified_slide(self) -> bool:
        return bool(self.slide_id)


@dataclass(frozen=True)
class Normalization:
    pathway_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    fit_split: str
    ddof: int
    clip_applied_after_transform: bool
    clip_range: tuple[float, float]
    n_train_samples: int
    source_path: str
    inverse_note_zh: str = "pred_raw = pred_z * train_sd + train_mean，只逆变换一次，不对预测再裁剪。"

    def inverse_transform(self, pred_z: np.ndarray) -> np.ndarray:
        values = np.asarray(pred_z, dtype=np.float64)
        if values.shape[-1] != len(self.pathway_names):
            raise ValueError(
                f"inverse_transform 末维必须为 {len(self.pathway_names)}，当前={values.shape}"
            )
        return values * self.std + self.mean


@dataclass
class PointTable:
    identities: list[PointIdentity]
    split: np.ndarray
    x: np.ndarray
    y: np.ndarray
    features: np.ndarray | None = None
    labels_z: np.ndarray | None = None
    feature_paths: list[str | None] = field(default_factory=list)
    slide_status: str = "missing"
    pathway_names: tuple[str, ...] = ()
    source: str = "constructed"

    def __len__(self) -> int:
        return len(self.identities)

    def index_by_key(self) -> dict[tuple[str, str, str], int]:
        mapping = {}
        for i, ident in enumerate(self.identities):
            key = ident.key()
            if key in mapping:
                raise IdentityMismatchError(f"重复身份键 {key}")
            mapping[key] = i
        return mapping

    def join_key_without_slide(self, index: int) -> tuple[str, str]:
        ident = self.identities[index]
        return (ident.patient_id, ident.spot_id)

    def split_mask(self, split: str) -> np.ndarray:
        return self.split == split

    def subset(self, indices: Sequence[int]) -> "PointTable":
        idx = np.asarray(list(indices), dtype=np.int64)
        return PointTable(
            identities=[self.identities[i] for i in idx],
            split=self.split[idx],
            x=self.x[idx],
            y=self.y[idx],
            features=None if self.features is None else self.features[idx],
            labels_z=None if self.labels_z is None else self.labels_z[idx],
            feature_paths=[self.feature_paths[i] for i in idx] if self.feature_paths else [],
            slide_status=self.slide_status,
            pathway_names=self.pathway_names,
            source=self.source,
        )

    def require_finite(self) -> None:
        for name, array in (("features", self.features), ("labels_z", self.labels_z)):
            if array is None:
                continue
            if not np.isfinite(array).all():
                bad = int(np.size(array) - np.isfinite(array).sum())
                raise NonFiniteDataError(f"{name} 含 {bad} 个非有限值")


def parse_xy(stem: str) -> tuple[int | None, int | None]:
    match = COORD_RE.search(str(stem))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_pathway_names(zscore_manifest_path: str | Path) -> list[str]:
    payload = load_json(Path(zscore_manifest_path))
    names = payload.get("pathway_names")
    if not names or len(names) != EXPECTED_PATHWAYS:
        raise IdentityMismatchError(
            f"{zscore_manifest_path} 的 pathway_names 必须为{EXPECTED_PATHWAYS}列，当前={None if not names else len(names)}"
        )
    return [str(name) for name in names]


def load_normalization(params_path: str | Path, pathway_names: Sequence[str]) -> Normalization:
    payload = load_json(Path(params_path))
    pathways = payload.get("pathways") or {}
    mean = np.empty(len(pathway_names), dtype=np.float64)
    std = np.empty(len(pathway_names), dtype=np.float64)
    for i, name in enumerate(pathway_names):
        if name not in pathways:
            raise IdentityMismatchError(f"标准化参数缺少通路 {name}")
        mean[i] = float(pathways[name]["mean"])
        std[i] = float(pathways[name]["std"])
        if not np.isfinite(mean[i]) or not np.isfinite(std[i]) or std[i] <= 0:
            raise NonFiniteDataError(f"通路 {name} 的 mean/std 非法")
    clip = payload.get("clip_range") or [-100.0, 100.0]
    return Normalization(
        pathway_names=tuple(pathway_names),
        mean=mean,
        std=std,
        fit_split=str(payload.get("fit_split") or "train"),
        ddof=int(payload.get("ddof", 1)),
        clip_applied_after_transform=bool(payload.get("clip_applied_after_transform", True)),
        clip_range=(float(clip[0]), float(clip[1])),
        n_train_samples=int(payload.get("n_train_samples") or 0),
        source_path=str(Path(params_path)),
    )


def inverse_transform(pred_z: np.ndarray, normalization: Normalization) -> np.ndarray:
    return normalization.inverse_transform(pred_z)


def load_slide_mapping(path: str | Path) -> tuple[dict[str, str], str]:
    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"patient_id", "slide_id", "status"}
    missing = required - set(table.columns)
    if missing:
        raise IdentityMismatchError(f"slide_mapping 缺少列 {sorted(missing)}")
    verified: dict[str, str] = {}
    for _, row in table.iterrows():
        patient = str(row["patient_id"]).strip()
        slide = str(row["slide_id"]).strip()
        status = str(row["status"]).strip().lower()
        if status == VERIFIED_MAPPING_STATUS:
            if not slide:
                raise IdentityMismatchError(f"patient={patient} 标记 verified 但 slide_id 为空")
            if patient in verified and verified[patient] != slide:
                raise IdentityMismatchError(f"patient={patient} 存在冲突 slide_id")
            verified[patient] = slide
    status = VERIFIED_MAPPING_STATUS if verified else "missing"
    return verified, status


def load_slide_geometry(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    needed = {"patient_id", "slide_id", "s", "coordinate_unit", "patch_coverage_size", "source", "status"}
    missing = needed - set(table.columns)
    if missing:
        raise IdentityMismatchError(f"slide_geometry 缺少列 {sorted(missing)}")
    return table


def geometry_by_slide(geometry: pd.DataFrame) -> dict[str, dict]:
    bound: dict[str, dict] = {}
    for _, row in geometry.iterrows():
        slide = "" if pd.isna(row["slide_id"]) else str(row["slide_id"]).strip()
        if not slide:
            continue
        bound[slide] = {
            "s": float(row["s"]),
            "coordinate_unit": str(row["coordinate_unit"]),
            "patch_coverage_size": None if pd.isna(row["patch_coverage_size"]) else float(row["patch_coverage_size"]),
            "source": str(row["source"]),
            "status": str(row["status"]),
        }
    return bound


def load_split_manifest(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    needed = {"patient", "patch_stem", "x", "y", "split"}
    missing = needed - set(table.columns)
    if missing:
        raise IdentityMismatchError(f"split_manifest 缺少列 {sorted(missing)}")
    if "slide_id" in table.columns:
        raise IdentityMismatchError(
            "split_manifest 出现 slide_id 列，但本包必须以独立 slide_mapping 的 verified 行作为切片身份来源"
        )
    return table


def _sort_indices(identities: Sequence[PointIdentity]) -> list[int]:
    return sorted(range(len(identities)), key=lambda i: identities[i].key())


def make_point_table(
    *,
    patient_ids: Sequence[str],
    spot_ids: Sequence[str],
    splits: Sequence[str],
    x: Sequence[float],
    y: Sequence[float],
    slide_ids: Sequence[str] | None = None,
    features: np.ndarray | None = None,
    labels_z: np.ndarray | None = None,
    feature_paths: Sequence[str | None] | None = None,
    slide_status: str = "missing",
    pathway_names: Sequence[str] = (),
    source: str = "constructed",
    sort_identities: bool = True,
) -> PointTable:
    n = len(patient_ids)
    if not (len(spot_ids) == len(splits) == len(x) == len(y) == n):
        raise IdentityMismatchError("构造点表时身份/坐标长度不一致")
    slides = list(slide_ids) if slide_ids is not None else [""] * n
    if len(slides) != n:
        raise IdentityMismatchError("slide_ids 长度与点数不一致")
    identities = [
        PointIdentity(patient_id=str(patient_ids[i]), slide_id=str(slides[i] or ""), spot_id=str(spot_ids[i]))
        for i in range(n)
    ]
    paths = list(feature_paths) if feature_paths is not None else [None] * n
    if features is not None:
        features = np.asarray(features, dtype=np.float32)
        if int(features.shape[0]) != n:
            raise IdentityMismatchError("features 行数与点数不一致")
    if labels_z is not None:
        labels_z = np.asarray(labels_z, dtype=np.float32)
        if int(labels_z.shape[0]) != n:
            raise IdentityMismatchError("labels 行数与点数不一致")
    table = PointTable(
        identities=identities,
        split=np.asarray(list(splits), dtype=object),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        features=None if features is None else np.asarray(features, dtype=np.float32),
        labels_z=None if labels_z is None else np.asarray(labels_z, dtype=np.float32),
        feature_paths=list(paths),
        slide_status=slide_status,
        pathway_names=tuple(pathway_names),
        source=source,
    )
    if sort_identities:
        table = table.subset(_sort_indices(table.identities))
    seen: set[tuple[str, str, str]] = set()
    for ident in table.identities:
        key = ident.key()
        if key in seen:
            raise IdentityMismatchError(f"重复身份键 {key}")
        seen.add(key)
    table.require_finite()
    return table


def point_table_from_manifest(
    manifest: pd.DataFrame,
    mapping: dict[str, str],
    mapping_status: str,
) -> PointTable:
    slides = [mapping.get(str(row["patient"]), "") for _, row in manifest.iterrows()]
    status = VERIFIED_MAPPING_STATUS if mapping_status == VERIFIED_MAPPING_STATUS and all(slides) else "missing"
    return make_point_table(
        patient_ids=[str(v) for v in manifest["patient"].tolist()],
        spot_ids=[str(v) for v in manifest["patch_stem"].tolist()],
        splits=[str(v) for v in manifest["split"].tolist()],
        x=manifest["x"].tolist(),
        y=manifest["y"].tolist(),
        slide_ids=slides,
        slide_status=status,
        source="split_manifest",
    )


def count_split(table: PointTable, split: str) -> int:
    return int((table.split == split).sum())


def assert_mpp2_counts(table: PointTable) -> None:
    n_train = count_split(table, "train")
    n_val = count_split(table, "internal_val")
    if n_train != EXPECTED_TRAIN or n_val != EXPECTED_VAL:
        raise IdentityMismatchError(
            f"划分计数必须为 train={EXPECTED_TRAIN}, internal_val={EXPECTED_VAL}，当前 train={n_train}, internal_val={n_val}"
        )


def label_csv_path(labels_root: Path, split: str, patient: str, train_mpp_id: int = 2) -> Path:
    if split == "train":
        return Path(labels_root) / "train" / patient / f"{patient}_ssGSEA_zscore.csv"
    if split == "internal_val":
        return Path(labels_root) / "val" / patient / f"{patient}_ssGSEA_zscore.csv"
    if split == "external":
        return (
            Path(labels_root)
            / "external"
            / patient
            / f"{patient}_ssGSEA_zscore_by_group_{train_mpp_id}_train.csv"
        )
    raise IdentityMismatchError(f"未知 split: {split}")


def load_label_map(csv_path: Path, pathway_names: Sequence[str]) -> dict[str, np.ndarray]:
    table = pd.read_csv(csv_path)
    first = table.columns[0]
    missing = [name for name in pathway_names if name not in table.columns]
    if missing:
        raise IdentityMismatchError(f"{csv_path} 缺少通路列 {missing[:8]}")
    mapping: dict[str, np.ndarray] = {}
    for _, row in table.iterrows():
        stem = Path(str(row[first])).stem
        if stem in mapping:
            raise IdentityMismatchError(f"{csv_path} 重复 spot {stem}")
        mapping[stem] = row[list(pathway_names)].to_numpy(dtype=np.float32)
    return mapping


def attach_labels(
    table: PointTable,
    labels_root: str | Path,
    pathway_names: Sequence[str],
    *,
    train_mpp_id: int = 2,
) -> PointTable:
    root = Path(labels_root)
    if not root.exists():
        raise FileNotFoundError(
            "标签根不存在，且本包不会回退仓库内旧标签。"
            f" 配置路径={root}。本机若无此目录，需在服务器现场使用活动修复 v003。"
        )
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    labels = np.empty((len(table), len(pathway_names)), dtype=np.float32)
    for i, ident in enumerate(table.identities):
        split = str(table.split[i])
        key = (split, ident.patient_id)
        if key not in cache:
            path = label_csv_path(root, split, ident.patient_id, train_mpp_id)
            if not path.is_file():
                raise IdentityMismatchError(f"缺少标签文件 {path}")
            cache[key] = load_label_map(path, pathway_names)
        patient_map = cache[key]
        if ident.spot_id not in patient_map:
            raise IdentityMismatchError(
                f"标签与划分未按身份对齐: patient={ident.patient_id} split={split} spot={ident.spot_id} 不在 {label_csv_path(root, split, ident.patient_id, train_mpp_id)}"
            )
        labels[i] = patient_map[ident.spot_id]
    out = table.subset(range(len(table)))
    out.labels_z = labels
    out.pathway_names = tuple(pathway_names)
    out.require_finite()
    return out


def load_feature_vector(path: Path, expected_dim: int = EXPECTED_INPUT_DIM) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 2:
        array = array[0]
    elif array.ndim != 1:
        raise IdentityMismatchError(f"{path} 特征维数非法: shape={array.shape}")
    if int(array.shape[0]) != int(expected_dim):
        raise IdentityMismatchError(f"{path} 期待{expected_dim}维，实际={array.shape[0]}")
    if not np.isfinite(array).all():
        raise NonFiniteDataError(f"{path} 特征含非有限值")
    return np.asarray(array, dtype=np.float32)


def find_feature_path(
    stem: str,
    patient: str,
    *,
    mpp_id: int,
    partner_cache: str | Path | None,
    flat_cache: str | Path | None,
) -> Path | None:
    fname = f"{stem}.pt"
    candidates: list[Path] = []
    if partner_cache:
        root = Path(partner_cache)
        candidates.extend(
            [
                root / f"MPP{mpp_id}_UNI" / patient / fname,
                root / f"MPP{mpp_id}_UNI" / patient / "train" / fname,
                root / f"MPP{mpp_id}_UNI" / patient / "val" / fname,
                root / str(mpp_id) / patient / fname,
            ]
        )
    if flat_cache:
        root = Path(flat_cache)
        candidates.extend(
            [
                root / str(mpp_id) / patient / fname,
                root / f"MPP{mpp_id}_UNI" / patient / fname,
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_feature_source_manifest(path: str | Path) -> dict:
    return load_json(Path(path))


def attach_features(
    table: PointTable,
    *,
    feature_manifest: dict,
    expected_dim: int = EXPECTED_INPUT_DIM,
) -> PointTable:
    partner = feature_manifest.get("partner_cache_root")
    flat = feature_manifest.get("flat_cache_root")
    mpp_id = int(feature_manifest.get("mpp_id") or 2)
    vectors = np.empty((len(table), expected_dim), dtype=np.float32)
    paths: list[str | None] = []
    missing: list[tuple[str, str]] = []
    for i, ident in enumerate(table.identities):
        found = find_feature_path(
            ident.spot_id,
            ident.patient_id,
            mpp_id=mpp_id,
            partner_cache=partner,
            flat_cache=flat,
        )
        if found is None:
            missing.append((ident.patient_id, ident.spot_id))
            paths.append(None)
            continue
        vectors[i] = load_feature_vector(found, expected_dim=expected_dim)
        paths.append(str(found))
    if missing:
        example = missing[:3]
        raise FileNotFoundError(
            f"缺少 {len(missing)} 个特征文件，例如 {example}。partner={partner} flat={flat}。"
            "不得用标签代替特征，服务器现场仍需验证缓存。"
        )
    out = table.subset(range(len(table)))
    out.features = vectors
    out.feature_paths = paths
    out.require_finite()
    return out


def require_verified_slides(table: PointTable, *, context: str) -> None:
    missing = [ident for ident in table.identities if not ident.has_verified_slide()]
    if table.slide_status != VERIFIED_MAPPING_STATUS or missing:
        n = len(missing)
        example = missing[0].key() if missing else ("?", "?", "?")
        raise SlideMappingMissingError(
            "SLIDE_MAPPING_UNVERIFIED: 当前没有可审计的 slide_id 映射，"
            f"不能把 patient_id 当作 slide_id。受阻操作={context}。"
            f" 缺失点数={n}/{len(table)}，示例身份={example}。"
            " point/relation 可不依赖图；spatial/joint/precheck图必须在映射 verified 后才能运行。"
        )


def build_unlabeled_point_table_from_feature_directory(
    feature_dir: str | Path,
    *,
    patient_id: str,
    split: str = "external",
    slide_id: str | None = None,
    slide_status: str = "missing",
    expected_dim: int = EXPECTED_INPUT_DIM,
    load_vectors: bool = False,
    source_note: str = "unlabeled_feature_directory",
) -> PointTable:
    directory = Path(feature_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"完整特征目录不存在: {directory}。该接口不使用标签构图，也不伪造点表。"
            "外部 XZY 点表在本机未核实，服务器现场仍需验证。"
        )
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"{directory} 中没有 .pt 特征文件")
    slide = str(slide_id or "")
    patient_ids = []
    spot_ids = []
    splits = []
    xs = []
    ys = []
    slides = []
    paths = []
    vectors = []
    for path in files:
        stem = path.stem
        x, y = parse_xy(stem)
        if x is None or y is None:
            raise IdentityMismatchError(f"无法从 {path.name} 解析 x/y")
        patient_ids.append(patient_id)
        spot_ids.append(stem)
        splits.append(split)
        xs.append(x)
        ys.append(y)
        slides.append(slide)
        paths.append(str(path))
        if load_vectors:
            vectors.append(load_feature_vector(path, expected_dim=expected_dim))
    features = np.stack(vectors, axis=0) if load_vectors else None
    status = slide_status if slide else "missing"
    return make_point_table(
        patient_ids=patient_ids,
        spot_ids=spot_ids,
        splits=splits,
        x=xs,
        y=ys,
        slide_ids=slides,
        features=features,
        feature_paths=paths,
        slide_status=status,
        source=source_note,
    )


def iter_index_batches(
    indices: Sequence[int],
    batch_size: int,
    *,
    keep_last: bool = True,
) -> Iterable[tuple[int, np.ndarray]]:
    order = np.asarray(list(indices), dtype=np.int64)
    start = 0
    batch_index = 0
    n = int(order.shape[0])
    while start < n:
        end = min(start + batch_size, n)
        chunk = order[start:end]
        if (end - start) < batch_size and not keep_last and start > 0:
            break
        yield batch_index, chunk
        batch_index += 1
        start = end


def load_split_point_table(config: dict) -> PointTable:
    data = config["data"]
    manifest = load_split_manifest(data["split_manifest_file"])
    mapping, mapping_status = load_slide_mapping(data["slide_mapping_file"])
    table = point_table_from_manifest(manifest, mapping, mapping_status)
    assert_mpp2_counts(table)
    return table


def load_normalization_from_config(config: dict) -> Normalization:
    names = load_pathway_names(config["data"]["zscore_manifest_file"])
    return load_normalization(config["data"]["normalization_file"], names)
