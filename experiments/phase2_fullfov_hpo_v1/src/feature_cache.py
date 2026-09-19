"""Atomic, versioned CLS cache storage with an explicit reuse contract."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from errors import IdentityMismatchError, NonFiniteDataError


class FeatureCacheWriter:
    def __init__(self, final_dir: str | Path, *, n_rows: int, feature_dim: int, identities: Sequence[str], include_mean_patch: bool = False) -> None:
        self.final_dir, self.parent = Path(final_dir).resolve(), Path(final_dir).resolve().parent
        if self.final_dir.exists():
            raise FileExistsError(f"缓存版本目录已存在，禁止覆盖: {self.final_dir}")
        if n_rows < 1 or feature_dim < 1 or len(identities) != n_rows or len(set(identities)) != len(identities):
            raise IdentityMismatchError("缓存行数、维数或身份清单非法")
        self.parent.mkdir(parents=True, exist_ok=True)
        self.staging = self.parent / f".{self.final_dir.name}.staging-{uuid.uuid4().hex}"
        self.staging.mkdir(exist_ok=False)
        self.n_rows, self.feature_dim, self.identities, self.include_mean_patch, self.written, self._closed = int(n_rows), int(feature_dim), list(identities), bool(include_mean_patch), 0, False
        self.cls = np.lib.format.open_memmap(self.staging / "cls.npy", mode="w+", dtype=np.float32, shape=(self.n_rows, self.feature_dim))
        self.mean_patch = np.lib.format.open_memmap(self.staging / "mean_patch.npy", mode="w+", dtype=np.float32, shape=(self.n_rows, self.feature_dim)) if self.include_mean_patch else None

    def write_rows(self, start: int, cls: np.ndarray, mean_patch: np.ndarray | None = None) -> None:
        if self._closed or int(start) != self.written:
            raise IdentityMismatchError("缓存必须按共同身份清单连续写入")
        cls_values = np.asarray(cls, dtype=np.float32)
        if cls_values.ndim != 2 or cls_values.shape[1] != self.feature_dim or self.written + cls_values.shape[0] > self.n_rows:
            raise IdentityMismatchError("CLS 写入形状或边界非法")
        if not np.isfinite(cls_values).all():
            raise NonFiniteDataError("待写入 CLS 含 NaN/Inf")
        if self.include_mean_patch:
            if mean_patch is None:
                raise IdentityMismatchError("缓存协议要求 mean_patch，但提取器没有返回")
            values = np.asarray(mean_patch, dtype=np.float32)
            if values.shape != cls_values.shape or not np.isfinite(values).all():
                raise IdentityMismatchError("mean_patch 形状或数值非法")
            assert self.mean_patch is not None
            self.mean_patch[self.written:self.written + len(values)] = values
        self.cls[self.written:self.written + len(cls_values)] = cls_values
        self.written += len(cls_values)

    def finalize(self, metadata: Mapping[str, object]) -> Path:
        if self._closed or self.written != self.n_rows:
            raise IdentityMismatchError(f"缓存尚未写满: {self.written}/{self.n_rows}")
        self.cls.flush()
        if self.mean_patch is not None:
            self.mean_patch.flush()
        del self.cls
        if self.mean_patch is not None:
            del self.mean_patch
        with (self.staging / "identities.jsonl").open("w", encoding="utf-8") as handle:
            for index, identity in enumerate(self.identities):
                handle.write(json.dumps({"row_index": index, "identity_key": identity}, ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        payload = {"schema_version": "2.0", "status": "complete", "n_rows": self.n_rows, "feature_dim": self.feature_dim, "dtype": "float32", "identity_schema": "patient_id|source_group|spot_id", "features": {"cls": {"path": "cls.npy", "role": "primary_training_input"}}, **dict(metadata)}
        if self.include_mean_patch:
            payload["features"]["mean_patch"] = {"path": "mean_patch.npy", "role": "auxiliary_not_training_input"}
        (self.staging / "cache_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.staging / "COMPLETE").write_text("complete\n", encoding="ascii")
        os.replace(self.staging, self.final_dir)
        self._closed = True
        return self.final_dir

    def abort(self) -> None:
        if self._closed or not self.staging.exists():
            return
        if self.staging.parent != self.parent or not self.staging.name.startswith(f".{self.final_dir.name}.staging-"):
            raise RuntimeError("拒绝清理未验证的缓存暂存目录")
        del self.cls
        if self.mean_patch is not None:
            del self.mean_patch
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


def load_feature_cache(cache_dir: str | Path, *, expected_identities: Sequence[str], expected_dim: int, expected_metadata: Mapping[str, object] | None = None, require_mean_patch: bool = False) -> dict:
    root = Path(cache_dir).resolve()
    if not (root / "COMPLETE").is_file():
        raise IdentityMismatchError(f"特征缓存不完整或缺 COMPLETE: {root}")
    try:
        metadata = json.loads((root / "cache_meta.json").read_text(encoding="utf-8"))
        identities, cls = _identities(root / "identities.jsonl"), np.load(root / "cls.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IdentityMismatchError(f"无法读取特征缓存 {root}: {exc}") from exc
    if metadata.get("status") != "complete" or metadata.get("dtype") != "float32" or metadata.get("identity_schema") != "patient_id|source_group|spot_id":
        raise IdentityMismatchError("缓存完成状态、精度或身份协议不匹配")
    if identities != list(expected_identities) or len(set(identities)) != len(identities):
        raise IdentityMismatchError("缓存身份或顺序与请求数据集不一致")
    if cls.shape != (len(identities), int(expected_dim)) or cls.dtype != np.float32 or not np.isfinite(cls).all():
        raise IdentityMismatchError("CLS 缓存形状、类型或数值不符合协议")
    if metadata.get("features", {}).get("cls", {}).get("role") != "primary_training_input":
        raise IdentityMismatchError("缓存没有登记 CLS 为训练输入")
    _require_contract(metadata, expected_metadata)
    result = {"cls": cls, "identities": identities, "metadata": metadata, "cache_dir": str(root)}
    mean_info = metadata.get("features", {}).get("mean_patch")
    if require_mean_patch and not mean_info:
        raise IdentityMismatchError("调用方要求 mean_patch，但缓存没有此可选辅助特征")
    if mean_info:
        mean = np.load(root / str(mean_info.get("path")), mmap_mode="r", allow_pickle=False)
        if mean.shape != cls.shape or mean.dtype != np.float32 or not np.isfinite(mean).all():
            raise IdentityMismatchError("mean_patch 缓存形状、类型或数值不符合协议")
        result["mean_patch"] = mean
    return result
