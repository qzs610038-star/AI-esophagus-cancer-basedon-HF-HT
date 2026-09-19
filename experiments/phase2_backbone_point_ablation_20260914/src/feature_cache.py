"""Versioned aggregate feature cache with a final atomic commit marker."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from errors import IdentityMismatchError, NonFiniteDataError


class FeatureCacheWriter:
    def __init__(
        self,
        final_dir: str | Path,
        *,
        n_rows: int,
        feature_dim: int,
        identities: Sequence[str],
    ) -> None:
        self.final_dir = Path(final_dir).resolve()
        self.parent = self.final_dir.parent
        if self.final_dir.exists():
            raise FileExistsError(f"缓存版本目录已存在，禁止覆盖: {self.final_dir}")
        if n_rows < 1 or feature_dim < 1 or len(identities) != int(n_rows):
            raise IdentityMismatchError("缓存行数、维数或身份数非法")
        if len(set(identities)) != len(identities):
            raise IdentityMismatchError("缓存身份存在重复")
        self.parent.mkdir(parents=True, exist_ok=True)
        self.staging = self.parent / f".{self.final_dir.name}.staging-{uuid.uuid4().hex}"
        self.staging.mkdir(parents=False, exist_ok=False)
        self.n_rows = int(n_rows)
        self.feature_dim = int(feature_dim)
        self.identities = list(identities)
        self.written = 0
        self._closed = False
        self.cls = np.lib.format.open_memmap(
            self.staging / "cls.npy", mode="w+", dtype=np.float32, shape=(self.n_rows, self.feature_dim)
        )
        self.mean_patch = np.lib.format.open_memmap(
            self.staging / "mean_patch.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.n_rows, self.feature_dim),
        )

    def write_rows(self, start: int, cls: np.ndarray, mean_patch: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("缓存 writer 已关闭")
        if int(start) != self.written:
            raise IdentityMismatchError("缓存必须按共同身份清单连续写入")
        cls_values = np.asarray(cls)
        mean_values = np.asarray(mean_patch)
        if cls_values.dtype != np.float32 or mean_values.dtype != np.float32:
            raise IdentityMismatchError("缓存输入必须已经是 float32")
        if cls_values.shape != mean_values.shape or cls_values.ndim != 2:
            raise IdentityMismatchError("CLS 与 mean-patch 必须是同形二维矩阵")
        if cls_values.shape[1] != self.feature_dim:
            raise IdentityMismatchError("缓存特征维数与模型登记不一致")
        end = int(start) + int(cls_values.shape[0])
        if end > self.n_rows:
            raise IdentityMismatchError("缓存写入越界")
        if not np.isfinite(cls_values).all() or not np.isfinite(mean_values).all():
            raise NonFiniteDataError("待写入特征含 NaN/Inf")
        self.cls[start:end] = cls_values
        self.mean_patch[start:end] = mean_values
        self.written = end

    def finalize(self, metadata: dict) -> Path:
        if self._closed:
            raise RuntimeError("缓存 writer 已关闭")
        if self.written != self.n_rows:
            raise IdentityMismatchError(
                f"缓存尚未写满: {self.written}/{self.n_rows}，不会生成 COMPLETE"
            )
        self.cls.flush()
        self.mean_patch.flush()
        del self.cls
        del self.mean_patch
        with (self.staging / "identities.jsonl").open("w", encoding="utf-8") as handle:
            for index, identity in enumerate(self.identities):
                handle.write(json.dumps({"row_index": index, "identity_key": identity}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema_version": "1.0",
            "status": "complete",
            "n_rows": self.n_rows,
            "feature_dim": self.feature_dim,
            "dtype": "float32",
            "features": {
                "cls": {"path": "cls.npy", "role": "primary_training_input"},
                "mean_patch": {"path": "mean_patch.npy", "role": "auxiliary_not_training_input"},
            },
            **dict(metadata),
        }
        (self.staging / "cache_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self.staging / "COMPLETE").write_text("complete\n", encoding="ascii")
        os.replace(self.staging, self.final_dir)
        self._closed = True
        return self.final_dir

    def abort(self) -> None:
        """Remove only this writer's uniquely named staging directory."""

        if self._closed or not self.staging.exists():
            return
        staging = self.staging.resolve()
        parent = self.parent.resolve()
        if staging.parent != parent or not staging.name.startswith(f".{self.final_dir.name}.staging-"):
            raise RuntimeError("拒绝清理未验证的缓存路径")
        del self.cls
        del self.mean_patch
        shutil.rmtree(staging)
        self._closed = True


def _load_identities(path: Path) -> list[str]:
    identities: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("row_index", -1)) != expected_index:
                raise IdentityMismatchError("特征缓存身份 row_index 不连续")
            identities.append(str(row.get("identity_key")))
    return identities


def load_feature_cache(
    cache_dir: str | Path,
    *,
    expected_identities: Sequence[str],
    expected_dim: int,
) -> dict:
    root = Path(cache_dir).resolve()
    if not (root / "COMPLETE").is_file():
        raise IdentityMismatchError(f"特征缓存不完整或缺 COMPLETE: {root}")
    try:
        metadata = json.loads((root / "cache_meta.json").read_text(encoding="utf-8"))
        identities = _load_identities(root / "identities.jsonl")
        cls = np.load(root / "cls.npy", mmap_mode="r", allow_pickle=False)
        mean_patch = np.load(root / "mean_patch.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IdentityMismatchError(f"无法读取特征缓存 {root}: {exc}") from exc
    if identities != list(expected_identities):
        raise IdentityMismatchError("特征缓存身份或行顺序与共同清单不一致")
    expected_shape = (len(identities), int(expected_dim))
    if cls.shape != expected_shape or mean_patch.shape != expected_shape:
        raise IdentityMismatchError(
            f"特征缓存形状错误: cls={cls.shape}, mean_patch={mean_patch.shape}, expected={expected_shape}"
        )
    if cls.dtype != np.float32 or mean_patch.dtype != np.float32:
        raise IdentityMismatchError("特征缓存必须为 float32")
    if not np.isfinite(cls).all() or not np.isfinite(mean_patch).all():
        raise NonFiniteDataError("特征缓存含 NaN/Inf")
    if metadata.get("features", {}).get("cls", {}).get("role") != "primary_training_input":
        raise IdentityMismatchError("缓存没有明确登记 CLS 为唯一训练输入")
    # Deliberately do not return mean_patch to the training caller.
    return {
        "cls": cls,
        "identities": identities,
        "metadata": metadata,
        "cache_dir": str(root),
    }
