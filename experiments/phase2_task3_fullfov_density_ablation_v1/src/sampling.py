"""Deterministic train-only sampling for the three task-3 arms."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from data import PointTable
from errors import IdentityMismatchError

ARMS = ("dense", "stride_2x2", "random_equal")


def _train_by_patient(table: PointTable) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, ident in enumerate(table.identities):
        if str(table.split[index]) == "train":
            groups[ident.patient_id].append(index)
    return {patient: np.asarray(indices, dtype=np.int64) for patient, indices in sorted(groups.items())}


def stride_indices(table: PointTable, phase: tuple[int, int] = (0, 0)) -> np.ndarray:
    kept: list[int] = []
    px, py = int(phase[0]), int(phase[1])
    for _, indices in _train_by_patient(table).items():
        xs = sorted({float(table.x[i]) for i in indices})
        ys = sorted({float(table.y[i]) for i in indices})
        xr, yr = {value: rank for rank, value in enumerate(xs)}, {value: rank for rank, value in enumerate(ys)}
        kept.extend(int(i) for i in indices if xr[float(table.x[i])] % 2 == px and yr[float(table.y[i])] % 2 == py)
    return np.asarray(sorted(kept), dtype=np.int64)


def arm_indices(table: PointTable, arm: str, seed: int, config: dict) -> np.ndarray:
    if arm not in ARMS:
        raise IdentityMismatchError(f"未知采样臂 {arm!r}")
    groups = _train_by_patient(table)
    dense = np.concatenate(list(groups.values()))
    if arm == "dense":
        selected = np.asarray(sorted(dense.tolist()), dtype=np.int64)
    else:
        stride = stride_indices(table, tuple(config["sampling"]["stride_phase"]))
        if arm == "stride_2x2":
            selected = stride
        else:
            target_counts: dict[str, int] = defaultdict(int)
            for i in stride:
                target_counts[table.identities[int(i)].patient_id] += 1
            chosen: list[int] = []
            offset = int(config["randomness"]["random_sampling_offset"])
            for patient_index, (patient, candidates) in enumerate(groups.items()):
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), offset, patient_index]))
                draw = rng.choice(candidates, size=target_counts[patient], replace=False)
                chosen.extend(int(i) for i in draw)
            selected = np.asarray(sorted(chosen), dtype=np.int64)
    expected = int(config["sampling"]["expected_dense"] if arm == "dense" else config["sampling"]["expected_sparse"])
    if len(selected) != expected or len(set(selected.tolist())) != expected:
        raise IdentityMismatchError(f"{arm} 采样数量必须为 {expected}，当前={len(selected)}")
    return selected


def counts_by_patient(table: PointTable, indices: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for index in indices:
        counts[table.identities[int(index)].patient_id] += 1
    return dict(sorted(counts.items()))


def assert_random_equal_contract(table: PointTable, config: dict, seeds=(45, 46, 47)) -> None:
    stride = arm_indices(table, "stride_2x2", int(seeds[0]), config)
    expected = counts_by_patient(table, stride)
    masks = []
    for seed in seeds:
        random_indices = arm_indices(table, "random_equal", int(seed), config)
        if counts_by_patient(table, random_indices) != expected:
            raise IdentityMismatchError("随机等量臂未逐患者匹配规则隔点数量")
        masks.append(tuple(random_indices.tolist()))
    if len(set(masks)) != len(masks):
        raise IdentityMismatchError("随机等量掩码没有随种子变化")

