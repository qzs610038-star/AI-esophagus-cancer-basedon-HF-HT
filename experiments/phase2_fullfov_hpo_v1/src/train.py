"""Fresh H/C/B training for point, spatial, and spatial-no-B controls."""
from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn

from data import PointTable, iter_index_batches, load_normalization, load_split_point_table
from errors import ConfigError, UnsupportedAlgorithmError
from graph import SpatialGraph, build_split_graph, gather_neighbors
from model import ARM_FLAGS, SoftlinkModel, assert_spatial_zero_initialized, build_model, make_center_dropout_mask
from run_io import append_event, write_json
from selection import CheckpointChoice, EarlyStopState, patient_macro_metrics, update_checkpoint_choice, update_early_stop

SUPPORTED_ARMS = ("point", "spatial", "no_b", "spatial_no_b")
DEFAULT_OFFSETS = {"model_seed_offset": 0, "center_seed_offset": 100000, "dropout_seed_offset": 200000}

def uses_spatial(arm: str) -> bool:
    if arm not in SUPPORTED_ARMS:
        raise ConfigError(f"只支持 {SUPPORTED_ARMS}，当前={arm!r}")
    return arm in {"spatial", "no_b", "spatial_no_b"}

def _seed(config: dict, seed: int, name: str) -> int:
    return int((config.get("randomness") or {}).get(name, DEFAULT_OFFSETS[name])) + int(seed)

def _cpu_build(config: dict, arm: str, seed: int) -> SoftlinkModel:
    previous = torch.get_rng_state()
    try:
        torch.manual_seed(_seed(config, seed, "model_seed_offset"))
        model = build_model(config, arm).cpu()
        assert_spatial_zero_initialized(model)
        return model
    finally:
        torch.set_rng_state(previous)

def make_center_rng(config: dict, seed: int) -> np.random.Generator:
    return np.random.default_rng(_seed(config, seed, "center_seed_offset"))

def make_dropout_generator(config: dict, seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed(config, seed, "dropout_seed_offset"))
    return gen

def next_center_order(rng: np.random.Generator, n_points: int) -> np.ndarray:
    return rng.permutation(int(n_points)).astype(np.int64, copy=False)

def epoch_center_order(n_points: int, *, seed: int, epoch: int, config: dict) -> np.ndarray:
    if epoch < 1:
        raise ValueError("epoch 从 1 开始")
    rng = make_center_rng(config, seed)
    result = None
    for _ in range(epoch):
        result = next_center_order(rng, n_points)
    return result

def iter_center_batches(order: Sequence[int], config: dict):
    if not bool(config["training"].get("keep_last_batch", True)):
        raise ConfigError("必须保留最后一个不完整中心批")
    return iter_index_batches(order, int(config["training"]["batch_size"]), keep_last=True)

def apply_precision_guards(config: dict) -> None:
    training = config["training"]
    if training.get("precision", "float32") != "float32" or training.get("amp", False) or training.get("tf32", False):
        raise UnsupportedAlgorithmError("本包训练固定 FP32，amp/tf32 必须关闭")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def resolve_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

def _optimizer_groups(model: SoftlinkModel, training: dict) -> list[dict]:
    lr = float(training["learning_rate"])
    multiplier = float(training.get("b_lr_multiplier", 1.0))
    if multiplier <= 0:
        raise ConfigError("b_lr_multiplier 必须为正")
    hc = list(model.shared.parameters()) + list(model.point_head.parameters())
    groups = [{"params": hc, "lr": lr, "group_name": "H_C"}]
    if model.use_spatial:
        groups.append({"params": list(model.spatial_head.parameters()), "lr": lr * multiplier, "group_name": "B"})
    return groups

def build_optimizer(model: SoftlinkModel, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    name = str(training.get("optimizer", "AdamW"))
    if name not in {"Adam", "AdamW"}:
        raise UnsupportedAlgorithmError(f"optimizer 只能是 Adam 或 AdamW，当前={name}")
    wd = float(training.get("weight_decay", 0.0))
    if name == "Adam" and wd != 0.0:
        raise ConfigError("Adam 搜索臂的 weight_decay 必须为 0，避免混淆 L2 与 decoupled decay")
    cls = torch.optim.Adam if name == "Adam" else torch.optim.AdamW
    return cls(_optimizer_groups(model, training), lr=float(training["learning_rate"]), weight_decay=wd,
               betas=tuple(training.get("betas", (0.9, 0.999))), eps=float(training.get("optimizer_epsilon", 1e-8)))

def optimizer_effective_config(optimizer: torch.optim.Optimizer) -> list[dict]:
    return [{"group_name": g.get("group_name"), "lr": float(g["lr"]), "weight_decay": float(g["weight_decay"]),
             "betas": list(g["betas"]), "eps": float(g["eps"]), "n_parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]

def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict, *, n_train: int):
    training = config["training"]
    schedule = str(training.get("lr_schedule", "constant"))
    if schedule == "constant":
        return None
    if schedule not in {"warmup_cosine", "warmup5_cosine"}:
        raise ConfigError("lr_schedule 只能为 constant 或 warmup_cosine")
    steps_per_epoch = math.ceil(int(n_train) / int(training["batch_size"]))
    warmup_steps = int(training.get("warmup_epochs", 5)) * steps_per_epoch
    total_steps = int(training["max_epochs"]) * steps_per_epoch
    ratio = float(training.get("cosine_min_lr_ratio", 0.01))
    if not 0 < ratio <= 1 or total_steps <= 0:
        raise ConfigError("余弦日程参数非法")
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)
    scheduler._phase2_total_steps = total_steps
    scheduler._phase2_warmup_steps = warmup_steps
    return scheduler

@dataclass
class TrainingState:
    arm: str; seed: int; model: SoftlinkModel; optimizer: torch.optim.Optimizer
    center_rng: np.random.Generator; dropout_gen: torch.Generator; scheduler: object | None
    initial_state_dict: dict

def prepare_training_state(config: dict, arm: str, seed: int, n_train: int, *, device: str | torch.device | None = "cpu") -> TrainingState:
    model = _cpu_build(config, arm, seed).to(torch.device(device or "cpu")).float()
    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, config, n_train=n_train)
    return TrainingState(arm, int(seed), model, optimizer, make_center_rng(config, seed), make_dropout_generator(config, seed), scheduler,
                         {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

def regression_mse(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("回归预测和标签必须为同形 [batch, pathway]")
    numerator = ((pred.float() - target.float()) ** 2).sum()
    denom = torch.tensor(float(pred.numel()), device=pred.device)
    return numerator / denom, numerator, denom

def _forward(model: SoftlinkModel, table: PointTable, indices: Sequence[int], graph: SpatialGraph | None, dropout_gen: torch.Generator | None, train: bool, device: torch.device):
    features = torch.as_tensor(table.features, dtype=torch.float32, device=device)
    idx = np.asarray(indices, dtype=np.int64)
    kwargs = {}
    if uses_spatial("spatial") and graph is not None:
        batch = gather_neighbors(graph, [table.identities[int(i)] for i in idx])
        kwargs = {"neighbor_features": torch.as_tensor(batch.neighbor_features(table), dtype=torch.float32, device=device),
                  "neighbor_weights": torch.as_tensor(batch.neighbor_weights, dtype=torch.float32, device=device),
                  "neighbor_mask": torch.as_tensor(batch.neighbor_mask, device=device)}
    mask = None
    if train and model.dropout_p:
        mask = make_center_dropout_mask((len(idx), model.hidden_dim), model.dropout_p, dropout_gen, device="cpu", dtype=torch.float32).to(device)
    return model(features.index_select(0, torch.as_tensor(idx, device=device)), center_dropout_mask=mask, **kwargs)

def forward_batch(model: SoftlinkModel, table: PointTable, indices: Sequence[int], *, graph: SpatialGraph | None, dropout_gen: torch.Generator | None = None, apply_dropout: bool = False, device: str | torch.device = "cpu", features_t=None):
    """Compatibility inference/training entrypoint used by the prediction module."""
    del features_t
    return _forward(model, table, indices, graph, dropout_gen, apply_dropout, torch.device(device))

def evaluate_split(model: SoftlinkModel, table: PointTable, config: dict, *, graph: SpatialGraph | None = None, device: str | torch.device = "cpu"):
    if table.labels_z is None or table.features is None:
        raise ConfigError("验证需要标签和缓存特征")
    dev = torch.device(device); model.eval(); blocks = []
    with torch.no_grad():
        for _, idx in iter_center_batches(np.arange(len(table)), config):
            blocks.append(_forward(model, table, idx, graph, None, False, dev)["y_hat"].detach().cpu().numpy())
    pred = np.concatenate(blocks, axis=0)
    metrics = patient_macro_metrics(pred, table.labels_z, [i.patient_id for i in table.identities], pathway_names=table.pathway_names or None)
    return metrics, pred, np.asarray(table.labels_z, dtype=np.float32)

def _split(table: PointTable, value: str) -> PointTable:
    return table.subset(np.flatnonzero(table.split == value))

def _load_table(config: dict, table: PointTable | None) -> PointTable:
    out = table if table is not None else load_split_point_table(config)
    if out.features is None or out.labels_z is None:
        raise ConfigError("训练点表必须包含缓存 CLS 与 z 分数标签")
    if out.features.shape[1] != int(config["data"]["input_dim"]):
        raise ConfigError("缓存维度与 data.input_dim 不一致")
    if out.labels_z.shape[1] != int(config["data"]["output_dim"]):
        raise ConfigError("标签通路数与 data.output_dim 不一致")
    out.require_finite(); return out

def _checkpoint(path: Path, model: SoftlinkModel, optimizer: torch.optim.Optimizer, *, epoch: int, choice: CheckpointChoice, config: dict, arm: str, seed: int, kind: str):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                "selection": choice.as_dict(), "kind": kind, "config": config, "arm": arm, "seed": seed,
                "input_dim": int(model.input_dim), "hidden_dim": int(model.hidden_dim), "output_dim": int(model.output_dim)}, path)

def reject_resume(resume: bool = False, checkpoint_path: str | Path | None = None) -> None:
    if resume or checkpoint_path is not None:
        raise ConfigError("本包记录重试尝试但不支持完整续训；必须新建运行目录")

def train_arm(config: dict, arm: str, seed: int, run_dir: str | Path, *, checkpoint_dir: str | Path, point_table: PointTable | None = None, device: str | torch.device | None = None, resume: bool = False, checkpoint_path: str | Path | None = None, geometry_table=None) -> dict:
    """Train a fresh point/spatial/no_b arm and return only internal selection data."""
    reject_resume(resume, checkpoint_path); apply_precision_guards(config)
    if arm not in SUPPORTED_ARMS: raise ConfigError(f"arm={arm!r} 未实现")
    run, weights = Path(run_dir), Path(checkpoint_dir)
    run.mkdir(parents=True, exist_ok=True); (run / "raw").mkdir(exist_ok=True); weights.mkdir(parents=True, exist_ok=False)
    table = _load_table(config, point_table); train, val = _split(table, "train"), _split(table, "internal_val")
    if not len(train) or not len(val): raise ConfigError("必须有 train 和 internal_val")
    graph_train = graph_val = None
    if uses_spatial(arm):
        graph_train = build_split_graph(train, "train", config, geometry_table=geometry_table)
        graph_val = build_split_graph(val, "internal_val", config, geometry_table=geometry_table)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = prepare_training_state(config, arm, seed, len(train), device=dev)
    write_json(run / "raw" / "optimizer_effective.json", {"optimizer": type(state.optimizer).__name__, "groups": optimizer_effective_config(state.optimizer), "scheduler": str(config["training"].get("lr_schedule", "constant"))})
    torch.save(state.initial_state_dict, run / "raw" / "initial_weights.pt")
    sel = config.get("selection", {}); formal_start = int(sel.get("formal_start_epoch", 6)); tolerance = float(sel.get("checkpoint_tolerance", 1e-6))
    early = EarlyStopState(); best = CheckpointChoice(kind="formal"); history=[]; total_steps=0
    append_event(run / "events.jsonl", stage="train_start", arm=arm, seed=int(seed), device=str(dev))
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        state.model.train(); order = next_center_order(state.center_rng, len(train)); sse=denom=0.0
        for _, batch in iter_center_batches(order, config):
            state.optimizer.zero_grad(set_to_none=True)
            out = _forward(state.model, train, batch, graph_train, state.dropout_gen, True, dev)
            target = torch.as_tensor(train.labels_z[batch], dtype=torch.float32, device=dev)
            loss, num, den = regression_mse(out["y_hat"], target); loss.backward(); state.optimizer.step()
            if state.scheduler is not None: state.scheduler.step()
            sse += float(num.detach().cpu()); denom += float(den.detach().cpu()); total_steps += 1
        metrics, pred, target = evaluate_split(state.model, val, config, graph=graph_val, device=dev)
        score, tie = float(metrics.patient_macro_pathway_pcc), float(metrics.patient_macro_z_mse_selection)
        if epoch == formal_start - 1:
            warmup_choice = CheckpointChoice(epoch=epoch, score=score, mse=tie, kind="warmup")
            _checkpoint(weights / "warmup.pt", state.model, state.optimizer, epoch=epoch,
                        choice=warmup_choice, config=config, arm=arm, seed=seed, kind="warmup")
        if epoch >= formal_start:
            update = update_checkpoint_choice(best, epoch=epoch, score=score, mse=tie, lambda_value=0.0, tolerance=tolerance, kind="formal")
            if update is not best:
                best = update; _checkpoint(weights / "formal_best.pt", state.model, state.optimizer, epoch=epoch, choice=best, config=config, arm=arm, seed=seed, kind="formal")
                prediction_record = {
                    "pred_z": pred, "target_z": target,
                    "patient_id": np.asarray([i.patient_id for i in val.identities], dtype=str),
                    "slide_id": np.asarray([i.slide_id for i in val.identities], dtype=str),
                    "spot_id": np.asarray([i.spot_id for i in val.identities], dtype=str),
                    "x": np.asarray(val.x, dtype=np.float64), "y": np.asarray(val.y, dtype=np.float64),
                    "pathway_names": np.asarray(val.pathway_names, dtype=str),
                    "split": np.asarray("internal_val"), "formal_epoch": np.asarray(epoch),
                    "selection_metric": np.asarray("patient_macro_pathway_pcc"),
                }
                if (config.get("inputs") or {}).get("normalization"):
                    norm = load_normalization(config["inputs"]["normalization"], val.pathway_names)
                    prediction_record.update(
                        mean=norm.mean, std=norm.std,
                        pred_raw=pred * norm.std + norm.mean,
                        target_available=np.asarray(True),
                        checkpoint_metadata=np.asarray({
                            "task_id": config.get("task_id"), "stage": config.get("task_stage"),
                            "model": config.get("model_name"), "protocol": config.get("input_protocol"),
                            "arm": arm, "seed": int(seed), "recipe": config.get("task_recipe"),
                            "split": "internal_val", "kind": "formal", "epoch": int(epoch),
                            "checkpoint": str((weights / "formal_best.pt").resolve()),
                            "input_dim": int(config["data"]["input_dim"]),
                            "hidden_dim": int(config["model"]["hidden_dim"]),
                        }, dtype=object),
                    )
                np.savez_compressed(run / "raw" / "internal_best.npz", **prediction_record)
        early = update_early_stop(early, epoch=epoch, score=score, formal_start_epoch=formal_start,
                                  count_start_epoch=int(sel.get("early_stop_count_start_epoch", 16)), min_delta=float(sel.get("early_stop_min_delta", 1e-4)), patience=int(sel.get("early_stop_patience", 10)))
        _checkpoint(weights / "last.pt", state.model, state.optimizer, epoch=epoch, choice=best, config=config, arm=arm, seed=seed, kind="last")
        history.append({"epoch": epoch, "train_z_mse": sse/denom, "patient_macro_pathway_pcc": score, "patient_macro_z_mse_selection": tie,
                        "pooled_z_mse": float(metrics.pooled_z_mse), "formal_best_epoch": best.epoch, "early_stop_count": early.count,
                        "early_stop_reason": early.reason, "lrs": [float(g["lr"]) for g in state.optimizer.param_groups]})
        if early.stopped: break
    with (run / "raw" / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    write_json(run / "raw" / "history.json", history)
    result = {"arm": arm, "seed": int(seed), "status": "completed" if best.epoch is not None else "no_formal_endpoint", "epochs_completed": len(history), "n_optimizer_steps": total_steps,
              "formal_endpoint": None if best.epoch is None else best.as_dict(), "formal_checkpoint": str(weights / "formal_best.pt") if best.epoch is not None else None,
              "warmup_checkpoint": str(weights / "warmup.pt") if formal_start > 1 and (weights / "warmup.pt").is_file() else None,
              "last_checkpoint": str(weights / "last.pt"), "early_stop": early.as_dict(), "resume_supported": False,
              "optimizer_effective_path": "raw/optimizer_effective.json", "selection_metric": "patient_macro_pathway_pcc", "tie_metric": "patient_macro_z_mse_selection"}
    write_json(run / "raw" / "train_result.json", result); write_json(run / "metrics.json", result)
    append_event(run / "events.jsonl", stage="train_end", arm=arm, seed=int(seed), status=result["status"])
    return result
