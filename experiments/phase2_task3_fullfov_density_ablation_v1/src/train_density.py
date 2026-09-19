"""Fixed-update task-3 density ablation training and frozen-baseline evaluation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from data_runtime import ArmNormalization, RuntimeData, make_arm_tables, predictions_to_dense_z, subset_with_targets
from graph import build_split_graph, gather_neighbors, graph_stats
from metrics_export import save_evaluation
from model import assert_spatial_zero_initialized, build_model, make_center_dropout_mask
from sampling import counts_by_patient
from selection import CheckpointChoice, patient_macro_metrics, update_checkpoint_choice


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_forward(model, table, indices: Sequence[int], graph, device: torch.device, dropout_gen=None, apply_dropout=False):
    idx = np.asarray(indices, dtype=np.int64)
    center = torch.as_tensor(table.features[idx], dtype=torch.float32, device=device)
    kwargs = {}
    if graph is not None:
        neighbors = gather_neighbors(graph, [table.identities[int(i)] for i in idx])
        kwargs = {
            "neighbor_features": torch.as_tensor(neighbors.neighbor_features(table), dtype=torch.float32, device=device),
            "neighbor_weights": torch.as_tensor(neighbors.neighbor_weights, dtype=torch.float32, device=device),
            "neighbor_mask": torch.as_tensor(neighbors.neighbor_mask, dtype=torch.bool, device=device),
        }
    mask = None
    if apply_dropout and model.dropout_p:
        mask = make_center_dropout_mask((len(idx), model.hidden_dim), model.dropout_p, dropout_gen, device="cpu", dtype=torch.float32).to(device)
    return model(center, center_dropout_mask=mask, **kwargs)["y_hat"]


def predict_table(model, table, graph, config: dict, device: torch.device) -> np.ndarray:
    model.eval(); blocks = []; batch_size = int(config["optimizer"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(table), batch_size):
            indices = np.arange(start, min(start + batch_size, len(table)), dtype=np.int64)
            blocks.append(_batch_forward(model, table, indices, graph, device).detach().cpu().numpy())
    return np.concatenate(blocks, axis=0)


def _selection_eval(model, val, val_graph, arm_norm: ArmNormalization, runtime: RuntimeData, config: dict, device: torch.device) -> tuple[dict, np.ndarray]:
    pred_arm_z = predict_table(model, val, val_graph, config, device)
    pred_dense_z, _ = predictions_to_dense_z(pred_arm_z, arm_norm, runtime.dense_normalization)
    true_raw = arm_norm.to_raw(val.labels_z)
    true_dense_z = (true_raw - runtime.dense_normalization.mean) / runtime.dense_normalization.std
    metrics = patient_macro_metrics(pred_dense_z, true_dense_z, [identity.patient_id for identity in val.identities], pathway_names=val.pathway_names, require_patient_count=6)
    return metrics.as_dict(), pred_dense_z


def _save_checkpoint(path: Path, model, optimizer, *, choice: CheckpointChoice, selector: str, arm: str, seed: int, update: int, epoch: int, config: dict, arm_norm: ArmNormalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "selection": choice.as_dict(),
        "selector": selector,
        "arm": arm,
        "seed": int(seed),
        "update": int(update),
        "epoch": int(epoch),
        "config": config,
        "arm_normalization": arm_norm.as_dict(),
        "scratch_training": True,
        "initialization_checkpoint": None,
    }, path)


def _load_model_checkpoint(path: Path, config: dict, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = build_model(config, "spatial")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device), payload


def check_frozen_reference_compatibility(config: dict) -> dict:
    """Strictly load the frozen reference on CPU before any training starts."""
    checkpoint = Path(config["paths"]["frozen_seed45_checkpoint"])
    if not checkpoint.is_file():
        return {
            "name": "frozen_checkpoint_structure",
            "path": str(checkpoint),
            "exists": False,
            "status": "WARN",
            "detail": "checkpoint文件不存在，结构兼容性未核验",
        }
    try:
        model, payload = _load_model_checkpoint(checkpoint, config, torch.device("cpu"))
        result = {
            "name": "frozen_checkpoint_structure",
            "path": str(checkpoint),
            "exists": True,
            "status": "PASS",
            "detail": "strict=True加载通过",
            "spatial_bias": model.spatial_head.bias is not None,
            "checkpoint_kind": payload.get("kind"),
            "checkpoint_arm": payload.get("arm"),
            "checkpoint_seed": payload.get("seed"),
        }
        del model, payload
        return result
    except Exception as exc:
        return {
            "name": "frozen_checkpoint_structure",
            "path": str(checkpoint),
            "exists": True,
            "status": "FAIL",
            "detail": f"strict=True加载失败: {type(exc).__name__}: {exc}",
        }


def train_arm(config: dict, runtime: RuntimeData, arm: str, seed: int, run_dir: str | Path, weights_dir: str | Path, *, mode: str = "formal", requested_device: str | None = None) -> dict:
    if mode not in {"formal", "smoke"}:
        raise ValueError("mode 必须为 formal 或 smoke")
    cfg = copy.deepcopy(config)
    if mode == "smoke":
        cfg["training"]["max_updates"] = 2
        cfg["selection"]["primary"].update({"every_updates": 1, "opportunities": 2})
        cfg["selection"]["sensitivity"].update({"epochs": 1, "opportunities": 1})
    run, weights = Path(run_dir), Path(weights_dir)
    run.mkdir(parents=True, exist_ok=False); weights.mkdir(parents=True, exist_ok=False)
    train, val, external, arm_norm, train_indices = make_arm_tables(runtime, arm, int(seed), cfg)
    geometry = pd.read_csv(cfg["inputs"]["slide_geometry"])
    graph_train = build_split_graph(train, "train", cfg, geometry_table=geometry)
    graph_val = build_split_graph(val, "internal_val", cfg, geometry_table=geometry)
    dev = _device(requested_device)
    torch.manual_seed(int(seed) + int(cfg["randomness"]["model_init_offset"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + int(cfg["randomness"]["model_init_offset"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(cfg, "spatial").to(dev)
    assert_spatial_zero_initialized(model)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(initial_state, weights / "initial_weights.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optimizer"]["learning_rate"]), weight_decay=float(cfg["optimizer"]["weight_decay"]))
    batch_rng = np.random.default_rng(int(seed) + int(cfg["randomness"]["batch_order_offset"]))
    dropout_gen = torch.Generator(device="cpu").manual_seed(int(seed) + int(cfg["randomness"]["dropout_offset"]))
    primary_cfg, sensitivity_cfg = cfg["selection"]["primary"], cfg["selection"]["sensitivity"]
    primary_choice, sensitivity_choice = CheckpointChoice(kind="equal_updates"), CheckpointChoice(kind="equal_50_epochs")
    primary_records, sensitivity_records, train_history = [], [], []
    update, epoch = 0, 0
    batch_size, max_updates = int(cfg["optimizer"]["batch_size"]), int(cfg["training"]["max_updates"])
    while update < max_updates:
        epoch += 1; model.train(); order = batch_rng.permutation(len(train)); epoch_losses = []
        for start in range(0, len(order), batch_size):
            if update >= max_updates:
                break
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = _batch_forward(model, train, batch, graph_train, dev, dropout_gen, True)
            target = torch.as_tensor(train.labels_z[batch], dtype=torch.float32, device=dev)
            loss = torch.mean((prediction - target) ** 2)
            loss.backward(); optimizer.step(); update += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if update % int(primary_cfg["every_updates"]) == 0:
                metrics, _ = _selection_eval(model, val, graph_val, arm_norm, runtime, cfg, dev)
                current = update_checkpoint_choice(primary_choice, epoch=update, score=float(metrics["patient_macro_pathway_pcc"]), mse=float(metrics["patient_macro_z_mse_selection"]), lambda_value=0.0, kind="equal_updates")
                record = {"opportunity": len(primary_records) + 1, "update": update, "epoch": epoch, "metrics": metrics, "selected": current is not primary_choice}
                primary_records.append(record)
                if current is not primary_choice:
                    primary_choice = current
                    _save_checkpoint(weights / primary_cfg["checkpoint_name"], model, optimizer, choice=current, selector="equal_updates", arm=arm, seed=seed, update=update, epoch=epoch, config=cfg, arm_norm=arm_norm)
        train_history.append({"epoch": epoch, "last_update": update, "mean_train_mse": float(np.mean(epoch_losses)) if epoch_losses else None})
        if epoch <= int(sensitivity_cfg["epochs"]):
            metrics, _ = _selection_eval(model, val, graph_val, arm_norm, runtime, cfg, dev)
            current = update_checkpoint_choice(sensitivity_choice, epoch=epoch, score=float(metrics["patient_macro_pathway_pcc"]), mse=float(metrics["patient_macro_z_mse_selection"]), lambda_value=0.0, kind="equal_50_epochs")
            record = {"opportunity": len(sensitivity_records) + 1, "update": update, "epoch": epoch, "metrics": metrics, "selected": current is not sensitivity_choice}
            sensitivity_records.append(record)
            if current is not sensitivity_choice:
                sensitivity_choice = current
                _save_checkpoint(weights / sensitivity_cfg["checkpoint_name"], model, optimizer, choice=current, selector="equal_50_epochs", arm=arm, seed=seed, update=update, epoch=epoch, config=cfg, arm_norm=arm_norm)
    expected_primary, expected_sensitivity = int(primary_cfg["opportunities"]), int(sensitivity_cfg["opportunities"])
    if len(primary_records) != expected_primary or len(sensitivity_records) != expected_sensitivity:
        raise RuntimeError(f"选择机会数量错误: primary={len(primary_records)}/{expected_primary}, sensitivity={len(sensitivity_records)}/{expected_sensitivity}")
    checkpoints = {"equal_updates": weights / primary_cfg["checkpoint_name"], "equal_50_epochs": weights / sensitivity_cfg["checkpoint_name"]}
    if not all(path.is_file() for path in checkpoints.values()):
        raise RuntimeError("两个独立选择器的checkpoint尚未全部冻结")
    evaluations = {}
    if mode == "formal":
        graph_external = build_split_graph(external, "external_test", cfg, geometry_table=geometry)
        for selector, checkpoint in checkpoints.items():
            selected_model, payload = _load_model_checkpoint(checkpoint, cfg, dev)
            evaluations[selector] = {}
            for split, table, graph in (("internal_val", val, graph_val), ("external_test", external, graph_external)):
                pred_arm_z = predict_table(selected_model, table, graph, cfg, dev)
                pred_dense_z, pred_raw = predictions_to_dense_z(pred_arm_z, arm_norm, runtime.dense_normalization)
                true_raw = arm_norm.to_raw(table.labels_z)
                true_dense_z = (true_raw - runtime.dense_normalization.mean) / runtime.dense_normalization.std
                evaluations[selector][split] = save_evaluation(run / "evaluation" / selector, split, table, pred_dense_z, true_dense_z, pred_raw, true_raw)
    sampling = {"arm": arm, "seed": int(seed), "n_train": len(train), "counts_by_patient": counts_by_patient(runtime.full, train_indices), "random_stream_independent_from_model_and_dropout": True}
    _write_json(run / "sampling_manifest.json", sampling)
    _write_json(run / "arm_normalization.json", arm_norm.as_dict())
    _write_json(run / "selection_history.json", {"equal_updates": primary_records, "equal_50_epochs": sensitivity_records})
    _write_json(run / "train_history.json", train_history)
    _write_json(run / "graph_stats.json", {"train": graph_stats(graph_train, train), "internal_val": graph_stats(graph_val, val)})
    summary = {
        "status": "complete" if mode == "formal" else "smoke_complete_non_evidence",
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "scratch_training": True,
        "initialization_checkpoint": None,
        "fixed_updates": update,
        "early_stopping": False,
        "selection_opportunities": {"equal_updates": len(primary_records), "equal_50_epochs": len(sensitivity_records)},
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "evaluations": evaluations,
    }
    _write_json(run / "run_summary.json", summary)
    return summary


def evaluate_frozen_reference(config: dict, runtime: RuntimeData, output_dir: str | Path, *, requested_device: str | None = None) -> dict:
    checkpoint = Path(config["paths"]["frozen_seed45_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"冻结seed45 checkpoint不存在: {checkpoint}")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    dev = _device(requested_device)
    model, payload = _load_model_checkpoint(checkpoint, config, dev)
    geometry = pd.read_csv(config["inputs"]["slide_geometry"])
    dense_norm = ArmNormalization(tuple(runtime.full.pathway_names), runtime.dense_normalization.mean, runtime.dense_normalization.std, runtime.dense_normalization.ddof, runtime.dense_normalization.n_train_samples, "accepted_dense_reference")
    evaluations = {}
    for split in ("internal_val", "external_test"):
        indices = np.flatnonzero(runtime.full.split == split)
        table = subset_with_targets(runtime, indices, dense_norm)
        graph = build_split_graph(table, split, config, geometry_table=geometry)
        pred_z = predict_table(model, table, graph, config, dev)
        pred_dense_z, pred_raw = predictions_to_dense_z(pred_z, dense_norm, runtime.dense_normalization)
        true_raw = dense_norm.to_raw(table.labels_z)
        true_dense_z = (true_raw - runtime.dense_normalization.mean) / runtime.dense_normalization.std
        evaluations[split] = save_evaluation(output / "evaluation", split, table, pred_dense_z, true_dense_z, pred_raw, true_raw)
    summary = {"role": "frozen_reference_evaluation_only", "checkpoint": str(checkpoint), "used_for_initialization": False, "trained_in_this_package": False, "evaluations": evaluations}
    _write_json(output / "reference_summary.json", summary)
    return summary
