"""Load the accepted full-FOV cache and construct leakage-safe arm datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data import Normalization, PointTable, attach_labels, load_normalization, load_pathway_names, load_slide_mapping, make_point_table
from errors import IdentityMismatchError
from feature_cache import load_feature_cache
from input_data import identities, load_common_manifest
from sampling import arm_indices
from transforms import describe_transform


@dataclass
class RuntimeData:
    full: PointTable
    raw_labels: np.ndarray
    dense_normalization: Normalization
    common_manifest_path: str
    feature_cache_path: str


@dataclass(frozen=True)
class ArmNormalization:
    pathway_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    ddof: int
    n_train_samples: int
    fit_arm: str

    def to_z(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=np.float64) - self.mean) / self.std

    def to_raw(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def as_dict(self) -> dict:
        return {
            "pathway_names": list(self.pathway_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "ddof": self.ddof,
            "n_train_samples": self.n_train_samples,
            "fit_arm": self.fit_arm,
            "fit_split": "train_only_retained_points",
        }


def check_server_inputs(config: dict) -> dict:
    checks = []
    for name, path_value, kind in (
        ("accepted_common_manifest", config["paths"]["accepted_common_manifest"], "file"),
        ("accepted_feature_cache", config["paths"]["accepted_feature_cache"], "directory"),
        ("frozen_seed45_checkpoint", config["paths"]["frozen_seed45_checkpoint"], "file"),
        ("labels_root", config["paths"]["labels_root"], "directory"),
        ("runs_root", config["paths"]["runs_root"], "directory"),
        ("weights_root", config["paths"]["weights_root"], "directory"),
    ):
        path = Path(path_value)
        exists = path.is_file() if kind == "file" else path.is_dir()
        checks.append({"name": name, "path": str(path), "kind": kind, "exists": bool(exists), "status": "PASS" if exists else "WARN"})
    local_inputs = []
    for name, path_value in config["inputs"].items():
        path = Path(path_value)
        local_inputs.append({"name": name, "path": str(path), "exists": path.is_file(), "status": "PASS" if path.is_file() else "FAIL"})
    return {
        "server_checks": checks,
        "local_contract_checks": local_inputs,
        "server_ready": all(item["exists"] for item in checks),
        "local_contract_ready": all(item["exists"] for item in local_inputs),
        "note": "本地缺少服务器绝对路径时记WARN；formal/smoke真实数据运行仍会严格失败，不静默降级。",
    }


def load_runtime_data(config: dict) -> RuntimeData:
    counts = config["data"]["expected_counts"]
    records = load_common_manifest(config["paths"]["accepted_common_manifest"], expected_counts=counts, require_images=False)
    cache_contract = config["feature_cache_contract"]
    cache = load_feature_cache(
        config["paths"]["accepted_feature_cache"],
        expected_identities=identities(records),
        expected_dim=int(config["data"]["input_dim"]),
        expected_metadata={
            "experiment_id": cache_contract["source_experiment_id"],
            "protocol_version": cache_contract["protocol_version"],
            "model": cache_contract["model"],
            "dataset": cache_contract["dataset"],
            "preprocessing": describe_transform(config["data"]["full_fov_protocol"]),
        },
    )
    mapping, status = load_slide_mapping(config["inputs"]["slide_mapping"])
    slides = [mapping.get(row.patient_id, "") for row in records]
    if any(not value for value in slides):
        raise IdentityMismatchError("共同身份清单中存在未验证 slide_id")
    pathway_names = load_pathway_names(config["inputs"]["zscore_manifest"])
    table = make_point_table(
        patient_ids=[row.patient_id for row in records],
        spot_ids=[row.spot_id for row in records],
        splits=["external" if row.split == "external_test" else row.split for row in records],
        x=[row.x for row in records],
        y=[row.y for row in records],
        slide_ids=slides,
        features=np.asarray(cache["cls"], dtype=np.float32),
        slide_status=status,
        pathway_names=pathway_names,
        source="accepted_phase2_fullfov_hpo_v1_cache",
        sort_identities=True,
    )
    table = attach_labels(table, config["paths"]["labels_root"], pathway_names, train_mpp_id=int(config["data"]["mpp_id"]))
    table.split = np.asarray(["external_test" if value == "external" else value for value in table.split], dtype=object)
    observed = {split: int(np.sum(table.split == split)) for split in counts}
    if observed != {key: int(value) for key, value in counts.items()}:
        raise IdentityMismatchError(f"完整点表计数不匹配: {observed}")
    dense_norm = load_normalization(config["inputs"]["dense_zscore_params"], pathway_names)
    raw = dense_norm.inverse_transform(table.labels_z)
    if not np.isfinite(raw).all():
        raise IdentityMismatchError("逆变换后的真实标签含非有限值")
    return RuntimeData(table, raw, dense_norm, config["paths"]["accepted_common_manifest"], config["paths"]["accepted_feature_cache"])


def fit_arm_normalization(runtime: RuntimeData, selected_train_indices: Sequence[int], arm: str) -> ArmNormalization:
    indices = np.asarray(selected_train_indices, dtype=np.int64)
    if np.any(runtime.full.split[indices] != "train"):
        raise IdentityMismatchError("训练标准器只能使用本臂保留的训练点")
    values = np.asarray(runtime.raw_labels[indices], dtype=np.float64)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    if values.shape[0] < 2 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise IdentityMismatchError("本臂训练标准器不可计算")
    return ArmNormalization(tuple(runtime.full.pathway_names), mean, std, 1, int(len(indices)), arm)


def subset_with_targets(runtime: RuntimeData, indices: Sequence[int], normalization: ArmNormalization) -> PointTable:
    idx = np.asarray(indices, dtype=np.int64)
    table = runtime.full.subset(idx)
    table.labels_z = normalization.to_z(runtime.raw_labels[idx]).astype(np.float32)
    table.pathway_names = tuple(runtime.full.pathway_names)
    table.require_finite()
    return table


def make_arm_tables(runtime: RuntimeData, arm: str, seed: int, config: dict) -> tuple[PointTable, PointTable, PointTable, ArmNormalization, np.ndarray]:
    train_indices = arm_indices(runtime.full, arm, seed, config)
    normalization = fit_arm_normalization(runtime, train_indices, arm)
    val_indices = np.flatnonzero(runtime.full.split == "internal_val")
    external_indices = np.flatnonzero(runtime.full.split == "external_test")
    return (
        subset_with_targets(runtime, train_indices, normalization),
        subset_with_targets(runtime, val_indices, normalization),
        subset_with_targets(runtime, external_indices, normalization),
        normalization,
        train_indices,
    )


def predictions_to_dense_z(pred_arm_z: np.ndarray, arm_norm: ArmNormalization, dense_norm: Normalization) -> tuple[np.ndarray, np.ndarray]:
    raw = arm_norm.to_raw(pred_arm_z)
    dense_z = (raw - dense_norm.mean) / dense_norm.std
    return np.asarray(dense_z, dtype=np.float64), np.asarray(raw, dtype=np.float64)
