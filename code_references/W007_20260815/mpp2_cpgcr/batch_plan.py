"""六患者等权、每患者五 spot 的可复放 batch plan。"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from torch.utils.data import Sampler


@dataclass(frozen=True)
class SampleIdentity:
    index: int
    patient: str
    spot_id: str


@dataclass(frozen=True)
class PlannedBatch:
    epoch: int
    batch: int
    indices: tuple[int, ...]


class _PatientCycle:
    def __init__(self, indices: Sequence[int], *, seed: int):
        if not indices:
            raise ValueError("patient queue must not be empty")
        self.indices = list(indices)
        self.rng = random.Random(seed)
        self.queue: list[int] = []

    def take(self, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            if not self.queue:
                self.queue = list(self.indices)
                self.rng.shuffle(self.queue)
            needed = count - len(selected)
            selected.extend(self.queue[:needed])
            del self.queue[:needed]
        return selected


def build_patient_balanced_plan(
    samples: Sequence[SampleIdentity],
    *,
    patients: Sequence[str],
    spots_per_patient: int,
    epochs: int,
    seed: int,
) -> list[PlannedBatch]:
    """生成所有 epoch 的固定批次；最大患者决定每 epoch 批次数。"""

    expected_patients = list(patients)
    if len(expected_patients) != 6 or len(set(expected_patients)) != 6:
        raise ValueError("batch plan requires six distinct patients")
    grouped = {
        patient: [item.index for item in samples if item.patient == patient]
        for patient in expected_patients
    }
    missing = [patient for patient, indices in grouped.items() if not indices]
    if missing:
        raise ValueError(f"batch plan patients have no training samples: {missing}")
    batches_per_epoch = math.ceil(max(map(len, grouped.values())) / spots_per_patient)
    plan: list[PlannedBatch] = []
    for epoch in range(1, epochs + 1):
        cycles = {
            patient: _PatientCycle(
                grouped[patient],
                seed=seed + epoch * 10_007 + patient_index * 1_000_003,
            )
            for patient_index, patient in enumerate(expected_patients)
        }
        for batch_index in range(1, batches_per_epoch + 1):
            indices: list[int] = []
            for patient in expected_patients:
                indices.extend(cycles[patient].take(spots_per_patient))
            plan.append(PlannedBatch(epoch, batch_index, tuple(indices)))
    return plan


def write_batch_plan_csv(
    path: str | Path,
    plan: Sequence[PlannedBatch],
    samples: Sequence[SampleIdentity],
) -> Path:
    """按清单和文件大小治理保存可复放样本顺序，不计算普通日志哈希。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    identities = {item.index: item for item in samples}
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "batch", "position", "dataset_index", "patient", "spot_id"],
        )
        writer.writeheader()
        for planned in plan:
            for position, index in enumerate(planned.indices):
                identity = identities[index]
                writer.writerow({
                    "epoch": planned.epoch,
                    "batch": planned.batch,
                    "position": position,
                    "dataset_index": index,
                    "patient": identity.patient,
                    "spot_id": identity.spot_id,
                })
    return destination


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, plan: Sequence[PlannedBatch], *, epoch: int):
        self.batches = [list(item.indices) for item in plan if item.epoch == epoch]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)
