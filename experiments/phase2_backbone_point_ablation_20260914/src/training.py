"""Frozen point-training core for UNI and Virchow2 CLS features."""

from __future__ import annotations

import copy
import csv
import os
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from data import IdentityRecord, iter_index_batches
from errors import ConfigError, NonFiniteDataError
from model import (
    PointRegressor,
    build_paired_regressor,
    make_dropout_generator,
    next_dropout_mask,
)
from run_io import append_event, utc_now, write_json
from selection import (
    CheckpointChoice,
    EarlyStopState,
    patient_macro_metrics,
    update_checkpoint_choice,
    update_early_stop,
)


def make_center_rng(seed: int, *, offset: int = 100_000) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed) + int(offset)))


def next_center_order(rng: np.random.Generator, n_points: int) -> np.ndarray:
    if int(n_points) < 1:
        raise ValueError("n_points 必须为正整数")
    return rng.permutation(int(n_points)).astype(np.int64, copy=False)


def regression_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target 形状不一致: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 2 or prediction.numel() == 0:
        raise ValueError("MSE 需要非空二维 [N, pathways] 张量")
    diff = prediction.float() - target.float()
    numerator = torch.sum(diff * diff)
    denominator = torch.tensor(
        float(prediction.shape[0] * prediction.shape[1]),
        dtype=numerator.dtype,
        device=numerator.device,
    )
    return numerator / denominator, numerator, denominator


def apply_precision_guards(config: dict) -> None:
    training = config["training"]
    if training.get("precision") != "float32":
        raise ConfigError("训练只实现 float32")
    if training.get("amp") is not False:
        raise ConfigError("训练 AMP 必须关闭")
    if training.get("tf32") is not False:
        raise ConfigError("训练 TF32 必须关闭")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.AdamW:
    training = config["training"]
    if training.get("optimizer") != "AdamW":
        raise ConfigError("本包只实现 AdamW")
    apply_precision_guards(config)
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["optimizer_epsilon"]),
    )


def _validate_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    rows: Sequence[IdentityRecord],
    *,
    input_dim: int,
    output_dim: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features)
    target_values = np.asarray(targets)
    if feature_values.dtype != np.float32 or target_values.dtype != np.float32:
        raise ConfigError("训练特征和标签必须是 float32")
    if feature_values.shape != (len(rows), int(input_dim)):
        raise ConfigError(
            f"{split} 特征应为 {(len(rows), input_dim)}，实际={feature_values.shape}"
        )
    if target_values.shape != (len(rows), int(output_dim)):
        raise ConfigError(
            f"{split} 标签应为 {(len(rows), output_dim)}，实际={target_values.shape}"
        )
    if any(row.split != split for row in rows):
        raise ConfigError(f"{split} rows 含其他 split")
    if len({row.identity_key for row in rows}) != len(rows):
        raise ConfigError(f"{split} rows 含重复身份")
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise NonFiniteDataError(f"{split} 特征或标签含 NaN/Inf")
    return feature_values, target_values


def evaluate_model(
    model: PointRegressor,
    features: torch.Tensor,
    targets: np.ndarray,
    rows: Sequence[IdentityRecord],
    pathway_names: Sequence[str],
    *,
    batch_size: int,
    penalty: float,
) -> tuple[object, np.ndarray]:
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for indices in iter_index_batches(range(features.shape[0]), int(batch_size)):
                index_tensor = torch.as_tensor(indices, device=features.device, dtype=torch.long)
                predictions.append(
                    model(features.index_select(0, index_tensor), dropout_mask=None)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
    finally:
        model.train(was_training)
    pred_z = np.concatenate(predictions, axis=0)
    metrics = patient_macro_metrics(
        pred_z,
        np.asarray(targets, dtype=np.float64),
        [row.patient_id for row in rows],
        pathway_names=pathway_names,
        constant_prediction_selection_penalty=float(penalty),
    )
    if not metrics.computable:
        raise ConfigError(f"内部选择指标不可计算: {metrics.incomputable_reason}")
    return metrics, pred_z


def _checkpoint_payload(
    *,
    model: PointRegressor,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    seed: int,
    epoch: int,
    kind: str,
    choice: CheckpointChoice,
    center_rng: np.random.Generator,
    dropout_generator: torch.Generator,
    config: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "protocol_version": config.get("protocol_version"),
        "experiment_id": config.get("experiment_id"),
        "model_name": model_name,
        "arm": "point",
        "seed": int(seed),
        "epoch": int(epoch),
        "kind": kind,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "output_dim": model.output_dim,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "selection": choice.as_dict(),
        "rng": {
            "center_rng": copy.deepcopy(center_rng.bit_generator.state),
            "dropout_generator": dropout_generator.get_state().detach().cpu(),
            "model_seed": int(seed) + int(config["randomness"]["model_seed_offset"]),
            "center_seed": int(seed) + int(config["randomness"]["center_seed_offset"]),
            "dropout_seed": int(seed) + int(config["randomness"]["dropout_seed_offset"]),
        },
        "resume_supported": False,
    }


def _save_checkpoint(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _save_internal_predictions(
    path: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    rows: Sequence[IdentityRecord],
    pathway_names: Sequence[str],
    model_name: str,
    seed: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_z=np.asarray(prediction, dtype=np.float64),
        target_z=np.asarray(target, dtype=np.float64),
        patients=np.asarray([row.patient_id for row in rows], dtype=str),
        source_groups=np.asarray([row.source_group for row in rows], dtype=str),
        spots=np.asarray([row.spot_id for row in rows], dtype=str),
        x=np.asarray([row.x for row in rows], dtype=np.int64),
        y=np.asarray([row.y for row in rows], dtype=np.int64),
        pathways=np.asarray(pathway_names, dtype=str),
        epoch=np.asarray(int(epoch)),
        arm=np.asarray("point"),
        model=np.asarray(model_name),
        seed=np.asarray(int(seed)),
    )


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def train_point_arrays(
    config: dict,
    *,
    model_name: str,
    seed: int,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_rows: Sequence[IdentityRecord],
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_rows: Sequence[IdentityRecord],
    pathway_names: Sequence[str],
    output_state: Mapping[str, torch.Tensor],
    output_state_source: dict | None = None,
    run_dir: str | Path,
    checkpoint_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict:
    """Fresh-train one candidate. It has no resume or external-test interface."""

    if model_name not in {"uni", "virchow2"}:
        raise ConfigError("训练器只允许 uni 或 virchow2；UNI2-h 只能复用历史参照")
    if int(seed) not in {42, 43, 44}:
        raise ConfigError("正式训练 seed 只允许 42、43、44")
    expected_dim = {"uni": 1024, "virchow2": 1280}[model_name]
    output_dim = int(config["model"]["output_dim"])
    train_x, train_y = _validate_arrays(
        train_features,
        train_targets,
        train_rows,
        input_dim=expected_dim,
        output_dim=output_dim,
        split="train",
    )
    val_x, val_y = _validate_arrays(
        val_features,
        val_targets,
        val_rows,
        input_dim=expected_dim,
        output_dim=output_dim,
        split="internal_val",
    )
    if len(pathway_names) != output_dim:
        raise ConfigError("训练通路顺序必须恰为 30 列")
    apply_precision_guards(config)
    threads = int(config["training"]["cpu_threads"])
    if threads > 0:
        torch.set_num_threads(threads)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("请求 CUDA 但当前不可用")
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    run_path = Path(run_dir).resolve()
    weights_path = Path(checkpoint_dir).resolve()
    if run_path == weights_path or run_path in weights_path.parents or weights_path in run_path.parents:
        raise ConfigError("运行输出与权重目录必须分离，且不能互相嵌套")
    if run_path.exists() or weights_path.exists():
        raise FileExistsError("训练单元目录已存在；本协议不 resume、不覆盖")
    (run_path / "raw").mkdir(parents=True)
    weights_path.mkdir(parents=True)

    started_at = utc_now()
    started = time.perf_counter()
    model = build_paired_regressor(
        expected_dim,
        seed=int(seed),
        output_state=output_state,
        hidden_dim=int(config["model"]["hidden_dim"]),
        output_dim=output_dim,
        dropout_p=float(config["model"]["dropout"]),
        model_seed_offset=int(config["randomness"]["model_seed_offset"]),
    ).to(torch_device, dtype=torch.float32)
    optimizer = build_optimizer(model, config)
    center_rng = make_center_rng(int(seed), offset=int(config["randomness"]["center_seed_offset"]))
    dropout_generator = make_dropout_generator(
        int(seed), offset=int(config["randomness"]["dropout_seed_offset"])
    )
    train_features_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=torch_device)
    val_features_tensor = torch.as_tensor(val_x, dtype=torch.float32, device=torch_device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(initial_state, run_path / "raw" / "initial_weights.pt")
    write_json(
        run_path / "raw" / "paired_initialization.json",
        {
            "model_seed": int(seed) + int(config["randomness"]["model_seed_offset"]),
            "projection": "fresh_native_dimension_initialization",
            "copied_parameters": ["point_head.weight -> readout.weight", "point_head.bias -> readout.bias"],
            "forbidden_parameters_not_copied": ["shared.*", "relation_head.*", "spatial_head.*", "trained_checkpoint.*"],
            "source": output_state_source,
        },
    )

    selection_config = config["selection"]
    formal_start = int(selection_config["formal_start_epoch"])
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    warmup_choice = CheckpointChoice(kind="warmup")
    formal_choice = CheckpointChoice(kind="formal")
    early_state = EarlyStopState()
    history: list[dict] = []
    center_orders: dict[str, np.ndarray] = {}
    optimizer_steps = 0
    append_event(run_path / "events.jsonl", stage="train_start", model=model_name, seed=seed, device=str(torch_device))

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = next_center_order(center_rng, len(train_rows))
        center_orders[f"epoch_{epoch}"] = order
        squared_error_sum = 0.0
        mse_denominator = 0.0
        step_losses: list[float] = []
        for indices in iter_index_batches(order, batch_size):
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=torch_device)
            batch_x = train_features_tensor.index_select(0, index_tensor)
            batch_y = torch.as_tensor(train_y[indices], dtype=torch.float32, device=torch_device)
            mask = next_dropout_mask(
                dropout_generator,
                len(indices),
                model.hidden_dim,
                model.dropout_p,
                dtype=torch.float32,
            ).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss, numerator, denominator = regression_mse(model(batch_x, mask), batch_y)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            squared_error_sum += float(numerator.detach().cpu())
            mse_denominator += float(denominator.detach().cpu())
            step_losses.append(float(loss.detach().cpu()))

        metrics, prediction = evaluate_model(
            model,
            val_features_tensor,
            val_y,
            val_rows,
            pathway_names,
            batch_size=batch_size,
            penalty=float(selection_config["constant_prediction_selection_penalty"]),
        )
        score = float(metrics.patient_macro_pathway_pcc)
        tie_mse = float(metrics.patient_macro_z_mse_selection)

        if epoch < formal_start:
            updated = update_checkpoint_choice(
                warmup_choice,
                epoch=epoch,
                score=score,
                mse=tie_mse,
                tolerance=float(selection_config["checkpoint_tolerance"]),
                kind="warmup",
            )
            if updated.epoch != warmup_choice.epoch:
                warmup_choice = updated
                _save_checkpoint(
                    weights_path / "warmup_best.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        model_name=model_name,
                        seed=seed,
                        epoch=epoch,
                        kind="warmup",
                        choice=warmup_choice,
                        center_rng=center_rng,
                        dropout_generator=dropout_generator,
                        config=config,
                    ),
                )
        if epoch >= formal_start:
            updated = update_checkpoint_choice(
                formal_choice,
                epoch=epoch,
                score=score,
                mse=tie_mse,
                tolerance=float(selection_config["checkpoint_tolerance"]),
                kind="formal",
            )
            if updated.epoch != formal_choice.epoch:
                formal_choice = updated
                _save_checkpoint(
                    weights_path / "formal_best.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        model_name=model_name,
                        seed=seed,
                        epoch=epoch,
                        kind="formal",
                        choice=formal_choice,
                        center_rng=center_rng,
                        dropout_generator=dropout_generator,
                        config=config,
                    ),
                )
                _save_internal_predictions(
                    run_path / "raw" / "internal_best.npz",
                    prediction=prediction,
                    target=val_y,
                    rows=val_rows,
                    pathway_names=pathway_names,
                    model_name=model_name,
                    seed=seed,
                    epoch=epoch,
                )

        early_state = update_early_stop(
            early_state,
            epoch=epoch,
            score=score,
            formal_start_epoch=formal_start,
            count_start_epoch=int(selection_config["early_stop_count_start_epoch"]),
            min_delta=float(selection_config["early_stop_min_delta"]),
            patience=int(selection_config["early_stop_patience"]),
        )
        choice_for_last = formal_choice if formal_choice.epoch is not None else warmup_choice
        _save_checkpoint(
            weights_path / "last.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                model_name=model_name,
                seed=seed,
                epoch=epoch,
                kind="last",
                choice=choice_for_last,
                center_rng=center_rng,
                dropout_generator=dropout_generator,
                config=config,
            ),
        )
        history.append(
            {
                "epoch": epoch,
                "n_steps": int(np.ceil(len(train_rows) / batch_size)),
                "mse_numerator": squared_error_sum,
                "mse_denominator": mse_denominator,
                "mse": squared_error_sum / mse_denominator,
                "mse_step_mean": float(np.mean(step_losses)),
                "patient_macro_pathway_pcc": metrics.patient_macro_pathway_pcc,
                "patient_macro_pathway_pcc_measured": metrics.patient_macro_pathway_pcc_measured,
                "patient_macro_z_mse_selection": metrics.patient_macro_z_mse_selection,
                "pooled_z_mse": metrics.pooled_z_mse,
                "n_constant_prediction_penalized": metrics.n_constant_prediction_penalized,
                "warmup_best_epoch": warmup_choice.epoch,
                "formal_best_epoch": formal_choice.epoch,
                "early_stop_reference": early_state.reference,
                "early_stop_count": early_state.count,
                "early_stop_stopped": early_state.stopped,
            }
        )
        _write_history(run_path / "raw" / "history.csv", history)
        write_json(run_path / "raw" / "history.json", {"epochs": history})
        np.savez_compressed(
            run_path / "raw" / "center_orders.npz",
            **center_orders,
            batch_size=np.asarray(batch_size),
            keep_last_batch=np.asarray(True),
        )
        if early_state.stopped:
            break

    status = "completed" if formal_choice.epoch is not None else "no_formal_endpoint"
    elapsed = float(time.perf_counter() - started)
    peak = (
        int(torch.cuda.max_memory_allocated(torch_device))
        if torch_device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    result = {
        "status": status,
        "exit_code": 0 if status == "completed" else 1,
        "started_at": started_at,
        "ended_at": utc_now(),
        "model": model_name,
        "arm": "point",
        "seed": int(seed),
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "n_optimizer_steps": optimizer_steps,
        "formal_endpoint": formal_choice.as_dict() if formal_choice.epoch is not None else None,
        "warmup_best": warmup_choice.as_dict(),
        "early_stop": early_state.as_dict(),
        "formal_checkpoint": str((weights_path / "formal_best.pt").resolve()) if formal_choice.epoch is not None else None,
        "warmup_checkpoint": str((weights_path / "warmup_best.pt").resolve()) if warmup_choice.epoch is not None else None,
        "last_checkpoint": str((weights_path / "last.pt").resolve()),
        "run_dir": str(run_path),
        "weight_directory": str(weights_path),
        "resume_used": False,
        "resume_supported": False,
        "resource_usage": {
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak,
            "device": str(torch_device),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "external_used_for_selection": False,
        "paired_initialization": {
            "copied": ["historical point_head.weight", "historical point_head.bias"],
            "source": output_state_source,
        },
    }
    write_json(run_path / "raw" / "train_result.json", result)
    write_json(
        run_path / "metrics.json",
        {
            "selection_source": "internal_val_only",
            "external_used_for_selection": False,
            "formal": result["formal_endpoint"],
            "warmup": result["warmup_best"],
            "early_stop": result["early_stop"],
        },
    )
    append_event(
        run_path / "events.jsonl",
        stage="train_end",
        model=model_name,
        seed=seed,
        status=status,
        formal_epoch=formal_choice.epoch,
    )
    return result
