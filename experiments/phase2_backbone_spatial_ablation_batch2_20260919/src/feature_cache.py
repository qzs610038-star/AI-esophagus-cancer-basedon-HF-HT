"""Atomic selected-feature cache with COMPLETE + metadata.json reuse contract."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from errors import IdentityMismatchError, NonFiniteDataError


FEATURE_FILENAME = "features.npy"
METADATA_FILENAME = "metadata.json"
IDENTITY_FILENAME = "identities.jsonl"
COMPLETE_NAME = "COMPLETE"
IDENTITY_SCHEMA = "patient_id|source_group|spot_id"


class FeatureCacheWriter:
    def __init__(self, final_dir: str | Path, *, n_rows: int, feature_dim: int, identities: Sequence[str]) -> None:
        self.final_dir = Path(final_dir).resolve()
        self.parent = self.final_dir.parent
        if self.final_dir.exists():
            raise FileExistsError(f"缓存版本目录已存在，禁止覆盖: {self.final_dir}")
        if n_rows < 1 or feature_dim < 1 or len(identities) != n_rows or len(set(identities)) != len(identities):
            raise IdentityMismatchError("缓存行数、维数或身份清单非法")
        self.parent.mkdir(parents=True, exist_ok=True)
        self.staging = self.parent / f".{self.final_dir.name}.staging-{uuid.uuid4().hex}"
        self.staging.mkdir(exist_ok=False)
        self.n_rows = int(n_rows)
        self.feature_dim = int(feature_dim)
        self.identities = list(identities)
        self.written = 0
        self._closed = False
        self.features = np.lib.format.open_memmap(
            self.staging / FEATURE_FILENAME, mode="w+", dtype=np.float32, shape=(self.n_rows, self.feature_dim)
        )

    def write_rows(self, start: int, features: np.ndarray, mean_patch: np.ndarray | None = None) -> None:
        if mean_patch is not None:
            raise IdentityMismatchError("本批主缓存不保存 mean_patch / 未使用 token")
        if self._closed or int(start) != self.written:
            raise IdentityMismatchError("缓存必须按共同身份清单连续写入")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_dim or self.written + values.shape[0] > self.n_rows:
            raise IdentityMismatchError("特征写入形状或边界非法")
        if not np.isfinite(values).all():
            raise NonFiniteDataError("待写入特征含 NaN/Inf")
        self.features[self.written:self.written + len(values)] = values
        self.written += len(values)

    def finalize(self, metadata: Mapping[str, object]) -> Path:
        if self._closed or self.written != self.n_rows:
            raise IdentityMismatchError(f"缓存尚未写满: {self.written}/{self.n_rows}")
        self.features.flush()
        del self.features
        with (self.staging / IDENTITY_FILENAME).open("w", encoding="utf-8") as handle:
            for index, identity in enumerate(self.identities):
                handle.write(json.dumps({"row_index": index, "identity_key": identity}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema_version": "batch2-spatial-v1",
            "status": "complete",
            "n_rows": self.n_rows,
            "feature_dim": self.feature_dim,
            "dtype": "float32",
            "identity_schema": IDENTITY_SCHEMA,
            "features": {"selected": {"path": FEATURE_FILENAME, "role": "primary_training_input"}},
            **dict(metadata),
        }
        (self.staging / METADATA_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self.staging / COMPLETE_NAME).write_text("complete\n", encoding="ascii")
        os.replace(self.staging, self.final_dir)
        self._closed = True
        return self.final_dir

    def abort(self) -> None:
        if self._closed or not self.staging.exists():
            return
        if self.staging.parent != self.parent or not self.staging.name.startswith(f".{self.final_dir.name}.staging-"):
            raise RuntimeError("拒绝清理未验证的缓存暂存目录")
        del self.features
        shutil.rmtree(self.staging)
        self._closed = True


def _identities(path: Path) -> list[str]:
    result: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("row_index", -1)) != index or not row.get("identity_key"):
                raise IdentityMismatchError("缓存 identities.jsonl 不连续或含空身份")
            result.append(str(row["identity_key"]))
    return result


def _require_contract(metadata: Mapping[str, object], expected_metadata: Mapping[str, object] | None) -> None:
    for key, expected in (expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise IdentityMismatchError(f"缓存元数据不匹配: {key}; expected={expected!r}, actual={metadata.get(key)!r}")


def load_feature_cache(
    cache_dir: str | Path,
    *,
    expected_identities: Sequence[str],
    expected_dim: int,
    expected_metadata: Mapping[str, object] | None = None,
) -> dict:
    root = Path(cache_dir).resolve()
    if not (root / COMPLETE_NAME).is_file():
        raise IdentityMismatchError(f"特征缓存不完整或缺 COMPLETE: {root}")
    try:
        metadata = json.loads((root / METADATA_FILENAME).read_text(encoding="utf-8"))
        identities = _identities(root / IDENTITY_FILENAME)
        features = np.load(root / FEATURE_FILENAME, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IdentityMismatchError(f"无法读取特征缓存 {root}: {exc}") from exc
    if metadata.get("status") != "complete" or metadata.get("dtype") != "float32":
        raise IdentityMismatchError("缓存完成状态或精度不匹配")
    if metadata.get("identity_schema") != IDENTITY_SCHEMA:
        raise IdentityMismatchError("缓存身份协议不匹配")
    if identities != list(expected_identities) or len(set(identities)) != len(identities):
        raise IdentityMismatchError("缓存身份或顺序与请求数据集不一致")
    if features.shape != (len(identities), int(expected_dim)) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise IdentityMismatchError("特征缓存形状、类型或数值不符合协议")
    if metadata.get("features", {}).get("selected", {}).get("role") != "primary_training_input":
        raise IdentityMismatchError("缓存没有登记选定特征为训练输入")
    _require_contract(metadata, expected_metadata)
    return {
        "cls": features,
        "features": features,
        "identities": identities,
        "metadata": metadata,
        "cache_dir": str(root),
    }
