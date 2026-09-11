"""Deterministic training schedule primitives; no model/UNI weights are required."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence
import numpy as np


def augmentation_seed(protocol: str, fold: str, seed: int, epoch: int, update: int, slot: int) -> int:
    """Stable stream identifier without process-randomised Python hashes."""
    def text_entropy(text: str) -> int:
        value = 0
        for byte in str(text).encode("utf-8"):
            value = (value * 257 + byte + 1) & 0xFFFFFFFF
        return value
    sequence = np.random.SeedSequence([
        text_entropy(protocol), text_entropy(fold), int(seed), int(epoch), int(update), int(slot),
    ])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class PatientBalancedBatchSampler:
    """Patient-balanced batches with no duplicate index inside any batch."""
    def __init__(self, patient_ids: Sequence[str], batch_size: int, *, protocol="original", fold="original", seed=42):
        self.patient_ids = np.asarray(patient_ids, dtype=object).astype(str)
        self.batch_size, self.protocol, self.fold, self.seed = int(batch_size), str(protocol), str(fold), int(seed)
        if self.batch_size < 1 or not len(self.patient_ids): raise ValueError("空数据或非法 batch_size")
        self.by_patient = {p: np.where(self.patient_ids == p)[0] for p in sorted(set(self.patient_ids))}
        maximum_patient_slots = ceil(self.batch_size / len(self.by_patient))
        too_small = {p: len(indices) for p, indices in self.by_patient.items() if len(indices) < maximum_patient_slots}
        if too_small:
            raise ValueError(
                f"患者点数不足以组成批内无重复的 patient-balanced batch: "
                f"每位患者最多需 {maximum_patient_slots} 点, 实际不足 {too_small}"
            )

    def batches(self, epoch: int) -> list[np.ndarray]:
        n_updates = ceil(len(self.patient_ids) / self.batch_size)
        rng = np.random.default_rng(augmentation_seed(self.protocol,self.fold,self.seed,epoch,0,0))
        orders = {p: rng.permutation(idx) for p,idx in self.by_patient.items()}
        cursors = {p: 0 for p in self.by_patient}; patients=list(self.by_patient)
        out=[]
        for update in range(n_updates):
            batch=[]
            start=(update*self.batch_size) % len(patients)
            for slot in range(self.batch_size):
                p=patients[(start+slot)%len(patients)]; order=orders[p]
                if cursors[p] >= len(order):
                    refreshed = rng.permutation(self.by_patient[p])
                    # When a patient's permutation wraps inside a batch, move
                    # indices already used by that batch to the end of the new
                    # cycle.  Nothing is discarded; reuse is merely delayed.
                    already_used = np.isin(refreshed, np.asarray(batch, dtype=np.int64))
                    orders[p] = np.concatenate((refreshed[~already_used], refreshed[already_used]))
                    cursors[p]=0; order=orders[p]
                candidate=int(order[cursors[p]]); cursors[p]+=1
                if candidate in batch: raise AssertionError("批内出现重复点")
                batch.append(candidate)
            out.append(np.asarray(batch,dtype=np.int64))
        return out


def select_learning_rate(pilots: Sequence[Mapping]) -> dict:
    """Epoch-5 rule: larger PCC, then smaller zMSE, then smaller LR."""
    rows=[dict(x) for x in pilots]
    if not rows: raise ValueError("没有 lr 试点")
    for row in rows:
        if int(row.get("epoch",5)) != 5: raise ValueError("lr 试点必须使用第5轮")
        row["pcc"]=float(row.get("patient_macro_pathway_pcc",row.get("pcc"))); row["zMSE"]=float(row.get("zMSE",row.get("z_mse"))); row["lr"]=float(row["lr"])
    return min(rows,key=lambda x:(-x["pcc"],x["zMSE"],x["lr"]))


def select_best_epoch(history: Sequence[Mapping]) -> dict:
    rows=[dict(x) for x in history]
    if not rows: raise ValueError("没有内部验证历史")
    for row in rows:
        row["epoch"]=int(row["epoch"]); row["pcc"]=float(row.get("patient_macro_pathway_pcc",row.get("pcc"))); row["zMSE"]=float(row.get("zMSE",row.get("z_mse")))
    return min(rows,key=lambda x:(-x["pcc"],x["zMSE"],x["epoch"]))


@dataclass(frozen=True)
class Stage2Task:
    cell: str
    head: str
    h_mode: str
    endpoint: int = 5
    def key(self) -> str: return f"{self.cell}/{self.h_mode}/{self.head}/e{self.endpoint}"


MAIN_CELLS=("frozen_regression","frozen_centered_contrastive","r8_regression","r8_global_contrastive","r8_centered_contrastive","r2_regression")

def stage2_task_matrix(best_epochs: Mapping[str,int] | None = None) -> list[Stage2Task]:
    tasks=[Stage2Task(cell,head,"inherit") for cell in MAIN_CELLS for head in ("point","spatial")]
    tasks += [Stage2Task(cell,head,"reset") for cell in ("r8_regression","r8_global_contrastive","r8_centered_contrastive") for head in ("point","spatial")]
    tasks += [Stage2Task(cell,head,"continue") for cell in ("r8_regression","r8_centered_contrastive") for head in ("point","spatial")]
    if best_epochs:
        tasks += [Stage2Task(cell,head,"inherit",int(epoch)) for cell,epoch in best_epochs.items() if cell in MAIN_CELLS and int(epoch)!=5 for head in ("point","spatial")]
    return tasks


def selected_lr_for_seed(seed: int, selected_lr_seed42: float) -> float:
    """Seeds 43/44 are not independently tuned: they reuse protocol/fold seed-42 LR."""
    if int(seed) not in (42,43,44): raise ValueError("方案仅定义 42/43/44 三个种子")
    return float(selected_lr_seed42)


def contrastive_lambda(epoch: int, update: int = 0, updates_per_epoch: int = 1, *, maximum: float = 0.05) -> float:
    """First epoch ramps 0→maximum by update; later epochs are fixed."""
    if int(epoch) < 1 or int(update) < 0 or int(updates_per_epoch) < 1:
        raise ValueError("epoch/update/updates_per_epoch 非法")
    if int(epoch) > 1:
        return float(maximum)
    if int(updates_per_epoch) == 1:
        return 0.0
    return float(maximum) * int(update) / (int(updates_per_epoch) - 1)


def _metric_row(epoch: int, metrics: Mapping) -> dict:
    row = dict(metrics)
    row["epoch"] = int(epoch)
    row["pcc"] = float(row.get("patient_macro_pathway_pcc", row.get("pcc")))
    row["zMSE"] = float(row.get("zMSE", row.get("z_mse")))
    return row


def _capture_state(model):
    if model is None:
        return None
    state_dict = getattr(model, "state_dict", None)
    return deepcopy(state_dict()) if callable(state_dict) else deepcopy(model)


def run_training_loop(
    train_step,
    validate,
    *,
    model=None,
    state_getter=None,
    phase: str = "stage2",
    max_epochs: int | None = None,
    updates_per_epoch: int = 1,
    checkpoint_callback=None,
    patience: int = 10,
    early_stop_start_epoch: int = 16,
    min_delta: float = 1e-4,
    selection_start_epoch: int | None = None,
) -> dict:
    """Execute a callback-driven loop suitable for CPU/tensor smoke tests.

    ``train_step(epoch, update, lambda_value)`` performs one update and
    ``validate(epoch)`` returns internal-only PCC/zMSE.  The function never
    constructs a UNI model or opens an input file.
    """
    defaults = {"warmup": 60, "stage1": 5, "stage2": 60}
    if phase not in defaults:
        raise ValueError("phase 必须为 warmup、stage1 或 stage2")
    epochs = int(defaults[phase] if max_epochs is None else max_epochs)
    if phase == "stage1" and epochs != 5:
        raise ValueError("阶段一固定为 5 轮")
    if epochs < 1 or int(updates_per_epoch) < 1:
        raise ValueError("轮数或每轮更新数非法")
    selection_start = int(
        (1 if phase == "stage1" else 6)
        if selection_start_epoch is None else selection_start_epoch
    )
    if selection_start < 1 or selection_start > epochs:
        raise ValueError("selection_start_epoch 必须落在训练轮次内")
    history: list[dict] = []
    best: dict | None = None
    formal_state = None
    no_improvement = 0
    early_stop_best_pcc = -np.inf
    stopped_early = False
    last_state = None
    def capture_state():
        return deepcopy(state_getter()) if state_getter is not None else _capture_state(model)
    for epoch in range(1, epochs + 1):
        updates = []
        for update in range(int(updates_per_epoch)):
            lam = contrastive_lambda(epoch, update, updates_per_epoch) if phase == "stage1" else 0.0
            updates.append(train_step(epoch, update, lam))
        row = _metric_row(epoch, validate(epoch))
        row["lambda_first_update"] = contrastive_lambda(epoch, 0, updates_per_epoch) if phase == "stage1" else 0.0
        row["lambda_last_update"] = contrastive_lambda(epoch, updates_per_epoch - 1, updates_per_epoch) if phase == "stage1" else 0.0
        row["train_updates"] = updates
        history.append(row)
        epoch_state = capture_state()
        if checkpoint_callback is not None:
            checkpoint_callback("epoch", epoch, epoch_state, row)
        improved = False
        if epoch >= selection_start:
            candidate = select_best_epoch(history[selection_start - 1 :])
            improved = best is None or candidate["epoch"] == epoch
            if improved:
                best = candidate
                formal_state = epoch_state
                if checkpoint_callback is not None:
                    checkpoint_callback("formal", epoch, formal_state, candidate)
        pcc_improved_for_patience = row["pcc"] > early_stop_best_pcc + float(min_delta)
        if epoch >= selection_start and pcc_improved_for_patience:
            early_stop_best_pcc = row["pcc"]
        if phase != "stage1" and epoch >= int(early_stop_start_epoch):
            if pcc_improved_for_patience:
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= int(patience):
                stopped_early = True
                last_state = capture_state()
                break
        last_state = capture_state()
    last_epoch = history[-1]["epoch"]
    if checkpoint_callback is not None:
        checkpoint_callback("last", last_epoch, last_state, history[-1])
    return {
        "phase": phase, "history": history, "formal": best,
        "formal_state": formal_state, "last": {"epoch": last_epoch, "state": last_state},
        "stopped_early": stopped_early, "epochs_completed": last_epoch,
        "selection_start_epoch": selection_start,
    }


def run_common_warmup(train_step, validate, **kwargs) -> dict:
    """Common H/C warm-up, capped at 60 epochs."""
    return run_training_loop(train_step, validate, phase="warmup", max_epochs=min(int(kwargs.pop("max_epochs", 60)), 60), **kwargs)


def run_stage1(train_step, validate, **kwargs) -> dict:
    """Five online adaptation epochs with the prescribed first-epoch ramp."""
    return run_training_loop(train_step, validate, phase="stage1", max_epochs=5, **kwargs)


def run_stage2(train_step, validate, **kwargs) -> dict:
    """Cache-head fitting, capped at 60 epochs with internal-only selection."""
    return run_training_loop(train_step, validate, phase="stage2", max_epochs=min(int(kwargs.pop("max_epochs", 60)), 60), **kwargs)
