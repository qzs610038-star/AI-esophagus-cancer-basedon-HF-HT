"""Paired four-arm training core (v2.1).

Public functions in this module are the contract for later predict/scheduler
code. This file does not start formal training by itself.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from config import SUPPORTED_ARMS
from data import (
    PointTable,
    attach_features,
    attach_labels,
    iter_index_batches,
    load_feature_source_manifest,
    load_pathway_names,
    load_split_point_table,
)
from errors import ConfigError, UnsupportedAlgorithmError
from graph import SpatialGraph, build_split_graph, edges_as_records, gather_neighbors
from model import (
    ARM_FLAGS,
    SoftlinkModel,
    apply_center_dropout_mask,
    assert_spatial_zero_initialized,
    build_model,
    make_center_dropout_mask,
)
from relations import ThresholdResult, compute_distance_thresholds, qp_tables, relation_loss, support_from_config
from run_io import append_event, package_payload, utc_now, write_json
from selection import (
    CheckpointChoice,
    EarlyStopState,
    patient_macro_metrics,
    relation_lambda,
    update_checkpoint_choice,
    update_early_stop,
)

TEMPLATE_LINEAR_ORDER = ("shared", "point_head", "relation_head", "spatial_head")
DEFAULT_OFFSETS = {
    "model_seed_offset": 0,
    "center_seed_offset": 100000,
    "dropout_seed_offset": 200000,
    "diagnostic_seed_offset": 300000,
}


def uses_relation(arm: str) -> bool:
    if arm not in ARM_FLAGS:
        raise ConfigError(f"未知实验臂 {arm!r}")
    return bool(ARM_FLAGS[arm][0])


def uses_spatial(arm: str) -> bool:
    if arm not in ARM_FLAGS:
        raise ConfigError(f"未知实验臂 {arm!r}")
    return bool(ARM_FLAGS[arm][1])


def randomness_offset(config: dict, name: str, default: int | None = None) -> int:
    if default is None:
        default = DEFAULT_OFFSETS[name]
    return int((config.get("randomness") or {}).get(name, default))


def stream_seed(config: dict, seed: int, offset_name: str, default: int | None = None) -> int:
    return randomness_offset(config, offset_name, default) + int(seed)


def lambda_for_epoch(epoch: int, config: dict) -> float:
    training = config["training"]
    return relation_lambda(
        int(epoch),
        warmup_epochs=int(training.get("relation_warmup_epochs", 5)),
        ramp_epochs=int(training.get("relation_ramp_epochs", 10)),
        lambda_max=float(training.get("relation_lambda_max", 0.05)),
    )


def linear_module_order(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, module in model.named_children() if isinstance(module, nn.Linear))


def assert_template_linear_order(model: nn.Module) -> None:
    order = linear_module_order(model)
    if order != TEMPLATE_LINEAR_ORDER:
        raise ConfigError(
            f"完整模板 Linear 创建顺序必须为 {TEMPLATE_LINEAR_ORDER}（W_h/W_o/W_r/B），当前={order}"
        )


def _cpu_param_context():
    device_cm = getattr(torch, "device", None)
    if device_cm is None:
        return nullcontext()
    try:
        return torch.device("cpu")
    except Exception:
        return nullcontext()


def build_full_template(config: dict, seed: int) -> SoftlinkModel:
    """CPU joint template: build_model joint order W_h/W_o/W_r/B, then zero B."""
    offset = randomness_offset(config, "model_seed_offset", 0)
    cpu_state = torch.get_rng_state()
    try:
        with _cpu_param_context():
            torch.manual_seed(int(offset) + int(seed))
            template = build_model(config, "joint")
        template.cpu()
        if template.spatial_head is None:
            raise ConfigError("完整模板必须包含空间头 B")
        nn.init.zeros_(template.spatial_head.weight)
        if template.spatial_head.bias is not None:
            nn.init.zeros_(template.spatial_head.bias)
        assert_template_linear_order(template)
        assert_spatial_zero_initialized(template)
        return template
    finally:
        torch.set_rng_state(cpu_state)


def copy_public_modules(source: SoftlinkModel, dest: SoftlinkModel) -> SoftlinkModel:
    """Copy shared W_h/W_o always; W_r onto relation arms; B onto spatial arms."""
    dest.shared.load_state_dict(source.shared.state_dict())
    dest.point_head.load_state_dict(source.point_head.state_dict())
    if dest.relation_head is not None:
        if source.relation_head is None:
            raise ConfigError("源模板缺少关系头 W_r，无法复制到关系臂")
        dest.relation_head.load_state_dict(source.relation_head.state_dict())
    if dest.spatial_head is not None:
        if source.spatial_head is None:
            raise ConfigError("源模板缺少空间头 B，无法复制到空间臂")
        dest.spatial_head.load_state_dict(source.spatial_head.state_dict())
    dest.cpu()
    return dest


def build_arm_from_template(config: dict, arm: str, template: SoftlinkModel) -> SoftlinkModel:
    cpu_state = torch.get_rng_state()
    try:
        model = build_model(config, arm)
    finally:
        torch.set_rng_state(cpu_state)
    return copy_public_modules(template, model)


def make_center_rng(config: dict, seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(config, seed, "center_seed_offset", 100000)))


def next_center_order(rng: np.random.Generator, n_points: int) -> np.ndarray:
    return rng.permutation(int(n_points)).astype(np.int64, copy=False)


def epoch_center_order(n_points: int, *, seed: int, epoch: int, config: dict) -> np.ndarray:
    if int(epoch) < 1:
        raise ValueError("epoch 从 1 开始")
    rng = make_center_rng(config, seed)
    order = np.empty(0, dtype=np.int64)
    for _ in range(int(epoch)):
        order = next_center_order(rng, n_points)
    return order


def make_dropout_generator(config: dict, seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stream_seed(config, seed, "dropout_seed_offset", 200000))
    return generator


def next_center_dropout_mask(
    generator: torch.Generator,
    n_centers: int,
    hidden_dim: int,
    dropout_p: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return make_center_dropout_mask(
        (int(n_centers), int(hidden_dim)),
        float(dropout_p),
        generator=generator,
        device="cpu",
        dtype=dtype,
    )


def sample_diagnostic_centers(n_points: int, config: dict, seed: int) -> np.ndarray:
    n = int(n_points)
    if n < 1:
        raise ValueError("诊断抽样需要至少 1 个训练点")
    k = min(int(config["training"]["batch_size"]), n)
    rng = np.random.Generator(np.random.PCG64(stream_seed(config, seed, "diagnostic_seed_offset", 300000)))
    return rng.permutation(n).astype(np.int64, copy=False)[:k]


def diagnostic_relation_seed(config: dict, seed: int) -> int:
    return stream_seed(config, seed, "diagnostic_seed_offset", 300000)


def regression_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_MSE = sum of squared errors / (n_centers * output_dim). Returns loss, numerator, denominator."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target 形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 2:
        raise ValueError("MSE 需要 [n_centers, output_dim]")
    n_centers, output_dim = int(pred.shape[0]), int(pred.shape[1])
    if n_centers < 1 or output_dim < 1:
        raise ValueError("MSE 分母需要正的中心数与 output_dim")
    diff = pred.float() - target.float()
    numerator = (diff * diff).sum()
    denominator = torch.tensor(float(n_centers * output_dim), dtype=numerator.dtype, device=numerator.device)
    return numerator / denominator, numerator, denominator


def iter_center_batches(order: Sequence[int], config: dict) -> Iterable[tuple[int, np.ndarray]]:
    training = config["training"]
    if training.get("keep_last_batch") is not True:
        raise ConfigError("training.keep_last_batch 必须为 true")
    return iter_index_batches(order, int(training["batch_size"]), keep_last=True)


def capture_stream_states(center_rng: np.random.Generator, dropout_gen: torch.Generator) -> dict:
    return {
        "center": copy.deepcopy(center_rng.bit_generator.state),
        "dropout": dropout_gen.get_state().detach().cpu().clone(),
    }


def stream_states_equal(left: dict, right: dict) -> bool:
    center_ok = left["center"] == right["center"]
    dropout_ok = bool(torch.equal(left["dropout"], right["dropout"]))
    return center_ok and dropout_ok


def resolve_formal_endpoint(
    formal: CheckpointChoice,
    *,
    last: CheckpointChoice | None = None,
    warmup: CheckpointChoice | None = None,
) -> CheckpointChoice | None:
    del last, warmup
    if formal.epoch is None or formal.score is None or not np.isfinite(float(formal.score)):
        return None
    return formal


def null_relation_head_grads_if_warmup(model: SoftlinkModel, epoch: int, warmup_epochs: int) -> None:
    if int(epoch) <= int(warmup_epochs) and model.relation_head is not None:
        for param in model.relation_head.parameters():
            param.grad = None


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.AdamW:
    training = config["training"]
    if training.get("optimizer", "AdamW") != "AdamW":
        raise UnsupportedAlgorithmError(f"只实现 AdamW，当前={training.get('optimizer')!r}")
    if training.get("precision", "float32") != "float32":
        raise UnsupportedAlgorithmError("只实现 float32")
    if training.get("amp") not in (False, None):
        raise UnsupportedAlgorithmError("不实现 AMP")
    if training.get("tf32") not in (False, None):
        raise UnsupportedAlgorithmError("不实现 TF32")
    return torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        betas=tuple(training.get("betas", (0.9, 0.999))),
        eps=float(training.get("optimizer_epsilon", 1e-8)),
    )


def apply_precision_guards(config: dict) -> None:
    training = config["training"]
    if training.get("amp") is not False:
        raise UnsupportedAlgorithmError("training.amp 必须为 false；本包不实现 AMP")
    if training.get("tf32") is not False:
        raise UnsupportedAlgorithmError("training.tf32 必须为 false；本包不实现 TF32")
    if training.get("precision") != "float32":
        raise UnsupportedAlgorithmError("training.precision 必须为 float32")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_spatial_graphs(
    arm: str,
    table: PointTable,
    config: dict,
    *,
    geometry_table=None,
) -> dict[str, SpatialGraph | None]:
    """Spatial/joint go through graph.build_split_graph; missing mapping fails there."""
    if not uses_spatial(arm):
        return {"train": None, "internal_val": None}
    return {
        "train": build_split_graph(table, "train", config, geometry_table=geometry_table),
        "internal_val": build_split_graph(table, "internal_val", config, geometry_table=geometry_table),
    }


def reject_resume(resume: bool = False, checkpoint_path: str | Path | None = None) -> None:
    if resume or checkpoint_path is not None:
        raise ConfigError("默认从头训练，不 resume、不查找 latest checkpoint")


@dataclass
class TrainingState:
    arm: str
    seed: int
    template: SoftlinkModel
    model: SoftlinkModel
    optimizer: torch.optim.AdamW
    center_rng: np.random.Generator
    dropout_gen: torch.Generator
    diagnostic_indices: np.ndarray
    initial_state_dict: dict
    scheduler: None = None
    extra: dict = field(default_factory=dict)


def prepare_training_state(
    config: dict,
    arm: str,
    seed: int,
    n_train: int,
    *,
    device: str | torch.device | None = "cpu",
) -> TrainingState:
    if arm not in SUPPORTED_ARMS:
        raise ConfigError(f"未知实验臂 {arm!r}，允许 {list(SUPPORTED_ARMS)}")
    device_t = torch.device(device or "cpu")
    template = build_full_template(config, seed)
    model = build_arm_from_template(config, arm, template)
    model.to(device_t)
    model.float()
    optimizer = build_optimizer(model, config)
    return TrainingState(
        arm=arm,
        seed=int(seed),
        template=template,
        model=model,
        optimizer=optimizer,
        center_rng=make_center_rng(config, seed),
        dropout_gen=make_dropout_generator(config, seed),
        diagnostic_indices=sample_diagnostic_centers(n_train, config, seed),
        initial_state_dict={key: value.detach().cpu().clone() for key, value in template.state_dict().items()},
        scheduler=None,
    )


def _features_tensor(table: PointTable, device: torch.device) -> torch.Tensor:
    if table.features is None:
        raise ConfigError("前向需要特征")
    return torch.as_tensor(table.features, dtype=torch.float32, device=device)


def forward_batch(
    model: SoftlinkModel,
    table: PointTable,
    indices: Sequence[int],
    *,
    graph: SpatialGraph | None,
    dropout_gen: torch.Generator | None,
    apply_dropout: bool,
    device: torch.device,
    features_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    idx_list = [int(i) for i in np.asarray(indices).tolist()]
    if features_t is None:
        features_t = _features_tensor(table, device)
    center_index = torch.as_tensor(idx_list, dtype=torch.long, device=device)
    center = features_t.index_select(0, center_index)
    kwargs: dict[str, torch.Tensor | None] = {}
    if model.use_spatial:
        if graph is None:
            raise ConfigError("空间臂必须先构建完整图再按批取邻居，不能绕过")
        gathered = gather_neighbors(graph, [table.identities[i] for i in idx_list])
        neighbor_np = gathered.neighbor_features(table)
        kwargs["neighbor_features"] = (
            None if neighbor_np is None else torch.as_tensor(neighbor_np, dtype=torch.float32, device=device)
        )
        kwargs["neighbor_weights"] = torch.as_tensor(gathered.neighbor_weights, dtype=torch.float32, device=device)
        kwargs["neighbor_mask"] = torch.as_tensor(gathered.neighbor_mask, device=device)
    mask = None
    if apply_dropout and float(model.dropout_p) > 0:
        if dropout_gen is None:
            raise ConfigError("训练中心 Dropout 必须使用独立 CPU torch.Generator")
        mask = next_center_dropout_mask(
            dropout_gen,
            len(idx_list),
            model.hidden_dim,
            model.dropout_p,
            dtype=center.dtype,
        ).to(device)
    return model(center, center_dropout_mask=mask, **kwargs)


def _param_grad_norm(param: torch.Tensor | None) -> float:
    if param is None or param.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(param.grad.detach().float()).cpu())


def _combined_norm(weight_norm: float, bias_norm: float) -> float:
    return float(math.sqrt(weight_norm * weight_norm + bias_norm * bias_norm))


def lambda_weighted_ratio(lambda_value: float, rel_norm: float, mse_norm: float) -> dict:
    if mse_norm == 0.0:
        return {
            "value": None,
            "undefined": True,
            "reason": "mse_grad_norm_zero",
            "lambda": float(lambda_value),
            "rel_norm": float(rel_norm),
            "mse_norm": float(mse_norm),
        }
    return {
        "value": float(lambda_value) * float(rel_norm) / float(mse_norm),
        "undefined": False,
        "reason": None,
        "lambda": float(lambda_value),
        "rel_norm": float(rel_norm),
        "mse_norm": float(mse_norm),
    }


def _identity_rows(table: PointTable, indices: Sequence[int]) -> list[dict]:
    rows = []
    for index in indices:
        ident = table.identities[int(index)]
        rows.append(
            {
                "train_index": int(index),
                "patient_id": ident.patient_id,
                "slide_id": ident.slide_id,
                "spot_id": ident.spot_id,
            }
        )
    return rows


def run_fixed_diagnostics(
    *,
    model: SoftlinkModel,
    config: dict,
    seed: int,
    epoch: int,
    train_table: PointTable,
    diagnostic_indices: np.ndarray,
    thresholds: ThresholdResult | None,
    train_graph: SpatialGraph | None,
    center_rng: np.random.Generator,
    dropout_gen: torch.Generator,
    optimizer: torch.optim.Optimizer | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = "cpu",
    features_t: torch.Tensor | None = None,
) -> dict:
    """Eval, no dropout, separate autograd for MSE/relation on shared W/b. Never step."""
    del optimizer
    before = capture_stream_states(center_rng, dropout_gen)
    device_t = torch.device(device or next(model.parameters()).device)
    was_training = model.training
    indices = np.asarray(diagnostic_indices, dtype=np.int64)
    labels = torch.as_tensor(train_table.labels_z[indices], dtype=torch.float32, device=device_t)
    lam = lambda_for_epoch(int(epoch), config)
    rel_cfg = config["relation"]
    model.eval()
    model.zero_grad(set_to_none=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "seed": int(seed),
        "arm_use_relation": bool(model.use_relation),
        "arm_use_spatial": bool(model.use_spatial),
        "lambda": float(lam),
        "indices": indices.tolist(),
        "identities": _identity_rows(train_table, indices.tolist()),
        "optimizer_stepped": False,
        "dropout_applied": False,
        "support_epoch": int((config.get("randomness") or {}).get("diagnostic_relation_epoch", 0)),
        "support_batch": int((config.get("randomness") or {}).get("diagnostic_relation_batch", 0)),
        "diagnostic_relation_seed": diagnostic_relation_seed(config, seed),
    }
    try:
        out = forward_batch(
            model,
            train_table,
            indices,
            graph=train_graph,
            dropout_gen=dropout_gen,
            apply_dropout=False,
            device=device_t,
            features_t=features_t,
        )
        mse_loss, mse_num, mse_den = regression_mse(out["y_hat"], labels)
        mse_loss.backward()
        mse_w = _param_grad_norm(model.shared.weight)
        mse_b = _param_grad_norm(model.shared.bias)
        payload["mse"] = {
            "loss": float(mse_loss.detach().cpu()),
            "numerator": float(mse_num.detach().cpu()),
            "denominator": float(mse_den.detach().cpu()),
            "shared_weight_grad_norm": mse_w,
            "shared_bias_grad_norm": mse_b,
            "shared_combined_grad_norm": _combined_norm(mse_w, mse_b),
        }
        model.zero_grad(set_to_none=True)

        support = None
        rel_stats = None
        rel_valid = False
        rel_w = 0.0
        rel_b = 0.0
        qp = []
        if model.use_relation:
            if thresholds is None:
                raise ConfigError("关系诊断需要训练阈值")
            randomness = config.get("randomness") or {}
            support = support_from_config(
                train_table.labels_z[indices],
                patient_ids=[train_table.identities[int(i)].patient_id for i in indices.tolist()],
                fixed_indices=[int(i) for i in indices.tolist()],
                config=config,
                thresholds=thresholds,
                run_seed=diagnostic_relation_seed(config, seed),
                epoch=int(randomness.get("diagnostic_relation_epoch", 0)),
                batch_index=int(randomness.get("diagnostic_relation_batch", 0)),
            )
            rel_valid = int(support.n_valid) > 0
            out_rel = forward_batch(
                model,
                train_table,
                indices,
                graph=train_graph,
                dropout_gen=dropout_gen,
                apply_dropout=False,
                device=device_t,
                features_t=features_t,
            )
            rel_loss, rel_stats = relation_loss(
                out_rel["z"],
                labels,
                support,
                tau_y=float(rel_cfg["tau_y"]),
                tau_z=float(rel_cfg["tau_z"]),
            )
            if rel_valid:
                rel_loss.backward()
                rel_w = _param_grad_norm(model.shared.weight)
                rel_b = _param_grad_norm(model.shared.bias)
            qp = qp_tables(
                out_rel["z"],
                labels,
                support,
                tau_y=float(rel_cfg["tau_y"]),
                tau_z=float(rel_cfg["tau_z"]),
            )
            out = out_rel
        mse_combined = _combined_norm(mse_w, mse_b)
        rel_combined = _combined_norm(rel_w, rel_b)
        payload["relation"] = {
            "valid": rel_valid,
            "n_valid": 0 if support is None else int(support.n_valid),
            "stats": rel_stats,
            "shared_weight_grad_norm": rel_w,
            "shared_bias_grad_norm": rel_b,
            "shared_combined_grad_norm": rel_combined,
            "invalid_reason": None if rel_valid or not model.use_relation else "no_valid_relation_centers",
        }
        payload["lambda_weighted_ratio"] = {
            "shared_weight": lambda_weighted_ratio(lam, rel_w, mse_w),
            "shared_bias": lambda_weighted_ratio(lam, rel_b, mse_b),
            "shared_combined": lambda_weighted_ratio(lam, rel_combined, mse_combined),
        }
        payload["near_indices"] = None if support is None else support.near_indices
        payload["far_indices"] = None if support is None else support.far_indices
        payload["Q"] = [row.get("Q") for row in qp]
        payload["P"] = [row.get("P") for row in qp]
        payload["support_index"] = [row.get("support") for row in qp]
        arrays = {
            "indices": indices,
            "h_center": out["h_center"].detach().cpu().numpy(),
            "y_point": out["y_point"].detach().cpu().numpy(),
            "delta": out["delta"].detach().cpu().numpy(),
            "y_hat": out["y_hat"].detach().cpu().numpy(),
        }
        if out["z"] is not None:
            arrays["z"] = out["z"].detach().cpu().numpy()
        payload["array_keys"] = sorted(arrays)
        if output_dir is not None:
            diag_dir = Path(output_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            np.savez(diag_dir / f"epoch_{int(epoch)}.npz", **arrays)
            json_payload = copy.deepcopy(payload)
            write_json(diag_dir / f"epoch_{int(epoch)}.json", json_payload)
        return payload
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        after = capture_stream_states(center_rng, dropout_gen)
        if not stream_states_equal(before, after):
            raise RuntimeError("固定诊断改变了正式中心顺序或 Dropout 生成器状态")


def evaluate_split(
    model: SoftlinkModel,
    table: PointTable,
    config: dict,
    *,
    graph: SpatialGraph | None,
    device: torch.device,
    features_t: torch.Tensor | None = None,
):
    was_training = model.training
    model.eval()
    try:
        n = len(table)
        order = np.arange(n, dtype=np.int64)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for _, batch_pos in iter_center_batches(order, config):
                out = forward_batch(
                    model,
                    table,
                    batch_pos,
                    graph=graph,
                    dropout_gen=None,
                    apply_dropout=False,
                    device=device,
                    features_t=features_t,
                )
                preds.append(out["y_hat"].detach().cpu().numpy().astype(np.float64, copy=False))
        pred = np.concatenate(preds, axis=0) if preds else np.empty((0, model.output_dim), dtype=np.float64)
        target = np.asarray(table.labels_z, dtype=np.float64)
        patients = [ident.patient_id for ident in table.identities]
        names = list(table.pathway_names) if table.pathway_names else None
        penalty = float((config.get("selection") or {}).get("constant_prediction_selection_penalty", -1.0))
        metrics = patient_macro_metrics(
            pred,
            target,
            patients,
            pathway_names=names,
            constant_prediction_selection_penalty=penalty,
        )
        return metrics, pred, target
    finally:
        model.train(was_training)


def _checkpoint_payload(
    *,
    model: SoftlinkModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lambda_value: float,
    arm: str,
    seed: int,
    kind: str,
    choice: CheckpointChoice | None,
    center_rng: np.random.Generator,
    dropout_gen: torch.Generator,
    config: dict,
) -> dict:
    package = {}
    try:
        package = package_payload(Path(config.get("_package_root")) if config.get("_package_root") else None)
    except Exception:
        package = {}
    return {
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "lambda_value": float(lambda_value),
        "kind": kind,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": {
            "center_rng": copy.deepcopy(center_rng.bit_generator.state),
            "dropout_generator": dropout_gen.get_state().detach().cpu().clone(),
            "model_seed_offset": randomness_offset(config, "model_seed_offset", 0),
            "center_seed_offset": randomness_offset(config, "center_seed_offset", 100000),
            "dropout_seed_offset": randomness_offset(config, "dropout_seed_offset", 200000),
            "diagnostic_seed_offset": randomness_offset(config, "diagnostic_seed_offset", 300000),
            "note_zh": "独立随机流状态入账，不用哈希派生种子。",
        },
        "selection": None if choice is None else choice.as_dict(),
        "config_version": config.get("config_version"),
        "plan_version": config.get("plan_version"),
        "code_version": package.get("code_version"),
        "resume_supported": False,
    }


def save_checkpoint(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def save_internal_best_npz(
    path: Path,
    *,
    pred_z: np.ndarray,
    target_z: np.ndarray,
    table: PointTable,
    epoch: int,
    arm: str,
    seed: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pred_z=np.asarray(pred_z, dtype=np.float64),
        target_z=np.asarray(target_z, dtype=np.float64),
        patient_id=np.asarray([ident.patient_id for ident in table.identities], dtype=object),
        slide_id=np.asarray([ident.slide_id for ident in table.identities], dtype=object),
        spot_id=np.asarray([ident.spot_id for ident in table.identities], dtype=object),
        x=np.asarray(table.x, dtype=np.float64),
        y=np.asarray(table.y, dtype=np.float64),
        pathway_names=np.asarray(table.pathway_names, dtype=object),
        indices=np.arange(len(table), dtype=np.int64),
        epoch=np.asarray(int(epoch)),
        arm=np.asarray(arm),
        seed=np.asarray(int(seed)),
    )
    return path


def load_point_table_for_training(config: dict, point_table: PointTable | None = None) -> PointTable:
    table = point_table if point_table is not None else load_split_point_table(config)
    if table.labels_z is None:
        names = load_pathway_names(config["data"]["zscore_manifest_file"])
        table = attach_labels(
            table,
            config["data"]["labels_root"],
            names,
            train_mpp_id=int(config["data"].get("mpp_id") or 2),
        )
    if table.features is None:
        manifest = load_feature_source_manifest(config["data"]["feature_source_manifest"])
        table = attach_features(table, feature_manifest=manifest, expected_dim=int(config["data"]["input_dim"]))
    table.require_finite()
    if table.labels_z is None or table.features is None:
        raise ConfigError("训练需要特征和标准化标签")
    return table


def _split_tables(table: PointTable) -> tuple[PointTable, PointTable]:
    train = table.subset(np.where(table.split == "train")[0].tolist())
    val = table.subset(np.where(table.split == "internal_val")[0].tolist())
    if len(train) < 1:
        raise ConfigError("训练集为空")
    if len(val) < 1:
        raise ConfigError("内部验证集为空")
    return train, val


def _compute_thresholds(train_table: PointTable, config: dict) -> ThresholdResult:
    rel = config["relation"]
    return compute_distance_thresholds(
        train_table.labels_z,
        n_pairs=int(rel["threshold_pairs"]),
        seed=int(rel["threshold_seed"]),
        near_quantile=float(rel["near_quantile"]),
        far_quantile=float(rel["far_quantile"]),
        quantile_method=str(rel["quantile_method"]),
    )


def _write_threshold_refs(run_dir: Path, thresholds: ThresholdResult | None) -> dict:
    if thresholds is None:
        return {"used": False}
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pairs_path = raw / "threshold_pairs.npz"
    np.savez(pairs_path, pair_indices=thresholds.pair_indices, distances=thresholds.distances)
    meta = {
        "used": True,
        "q10": thresholds.q10,
        "q50": thresholds.q50,
        "seed": thresholds.seed,
        "n_pairs": thresholds.n_pairs,
        "quantile_method": thresholds.quantile_method,
        "path": pairs_path.as_posix(),
    }
    write_json(raw / "relations_train.json", meta)
    return meta


def _write_graph_refs(run_dir: Path, graphs: dict[str, SpatialGraph | None]) -> dict:
    refs: dict[str, str | None] = {}
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split, graph in graphs.items():
        if graph is None:
            refs[split] = None
            continue
        path = raw / f"graph_edges_{split}.csv"
        records = edges_as_records(graph)
        if records:
            keys = list(records[0].keys())
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
        else:
            path.write_text("", encoding="utf-8")
        refs[split] = path.as_posix()
    write_json(raw / "graph_refs.json", refs)
    return refs


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_arm(
    config: dict,
    arm: str,
    seed: int,
    run_dir: str | Path,
    *,
    checkpoint_dir: str | Path,
    point_table: PointTable | None = None,
    device: str | torch.device | None = None,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
    geometry_table=None,
) -> dict:
    """Train one arm/seed. Default is a fresh run: no resume, no latest lookup.

    Later scheduler/predict should read the separately registered checkpoint_dir,
    plus raw/history.csv, raw/center_orders.npz, raw/internal_best.npz (if formal).
    """
    reject_resume(resume, checkpoint_path)
    if arm not in SUPPORTED_ARMS:
        raise ConfigError(f"未知实验臂 {arm!r}，允许 {list(SUPPORTED_ARMS)}")
    apply_precision_guards(config)
    threads = int(config["training"].get("cpu_threads") or 0)
    if threads > 0:
        torch.set_num_threads(threads)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "raw").mkdir(exist_ok=True)
    (run_path / "raw" / "diagnostics").mkdir(exist_ok=True)
    checkpoint_path_root = Path(checkpoint_dir).resolve()
    checkpoint_path_root.mkdir(parents=True, exist_ok=False)

    device_t = resolve_device(device)
    table = load_point_table_for_training(config, point_table)
    train_table, val_table = _split_tables(table)
    graphs = load_spatial_graphs(arm, table, config, geometry_table=geometry_table)
    if uses_spatial(arm):
        train_graph = build_split_graph(train_table, "train", config, geometry_table=geometry_table)
        val_graph = build_split_graph(val_table, "internal_val", config, geometry_table=geometry_table)
        graphs = {"train": train_graph, "internal_val": val_graph}
    else:
        train_graph = None
        val_graph = None

    thresholds = _compute_thresholds(train_table, config) if uses_relation(arm) else None
    threshold_ref = _write_threshold_refs(run_path, thresholds)
    graph_ref = _write_graph_refs(run_path, graphs)

    state = prepare_training_state(config, arm, seed, len(train_table), device=device_t)
    model = state.model
    optimizer = state.optimizer
    features_train = _features_tensor(train_table, device_t)
    features_val = _features_tensor(val_table, device_t)
    torch.save(state.initial_state_dict, run_path / "raw" / "initial_weights.pt")
    np.savez(run_path / "raw" / "diagnostic_centers.npz", indices=state.diagnostic_indices)
    write_json(
        run_path / "raw" / "diagnostic_centers.json",
        {
            "indices": state.diagnostic_indices.tolist(),
            "identities": _identity_rows(train_table, state.diagnostic_indices.tolist()),
            "seed": diagnostic_relation_seed(config, seed),
            "note_zh": "第6/15轮使用同一诊断中心与 epoch0/batch0 支持集。",
        },
    )

    selection_cfg = config.get("selection") or {}
    warmup_epochs = int(config["training"].get("relation_warmup_epochs", 5))
    max_epochs = int(config["training"]["max_epochs"])
    formal_start = int(selection_cfg.get("formal_start_epoch", 6))
    tolerance = float(selection_cfg.get("checkpoint_tolerance", 1e-6))
    diag_epochs = {int(v) for v in (config.get("diagnostics") or {}).get("training_epochs") or [6, 15]}
    comparison_epochs = {int(v) for v in (config.get("analysis") or {}).get("same_epoch_comparisons") or [15, 25]}
    warmup_choice = CheckpointChoice(kind="warmup")
    formal_choice = CheckpointChoice(kind="formal")
    early_state = EarlyStopState()
    history_rows: list[dict] = []
    center_orders: dict[str, np.ndarray] = {}
    last_pred = None
    last_target = None
    last_metrics = None
    n_steps_total = 0

    append_event(run_path / "events.jsonl", stage="train_start", arm=arm, seed=int(seed), device=str(device_t))

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = next_center_order(state.center_rng, len(train_table))
        center_orders[f"epoch_{epoch}"] = order
        lam = lambda_for_epoch(epoch, config)
        sse = 0.0
        mse_den = 0.0
        ce_sum = 0.0
        rel_den = 0.0
        step_mse: list[float] = []
        step_rel: list[float] = []
        step_total: list[float] = []
        n_steps = 0

        for batch_index, batch_pos in iter_center_batches(order, config):
            optimizer.zero_grad(set_to_none=True)
            out = forward_batch(
                model,
                train_table,
                batch_pos,
                graph=train_graph,
                dropout_gen=state.dropout_gen,
                apply_dropout=True,
                device=device_t,
                features_t=features_train,
            )
            y = torch.as_tensor(train_table.labels_z[batch_pos], dtype=torch.float32, device=device_t)
            mse_loss, mse_num, mse_denom = regression_mse(out["y_hat"], y)
            rel_term = mse_loss.new_zeros(())
            rel_stats = {"sum_ce": 0.0, "denominator": 0, "loss": 0.0, "n_valid": 0}
            if model.use_relation and lam > 0:
                if thresholds is None:
                    raise ConfigError("关系臂需要阈值")
                support = support_from_config(
                    train_table.labels_z[batch_pos],
                    patient_ids=[train_table.identities[int(i)].patient_id for i in batch_pos.tolist()],
                    fixed_indices=[int(i) for i in batch_pos.tolist()],
                    config=config,
                    thresholds=thresholds,
                    run_seed=int(seed),
                    epoch=int(epoch),
                    batch_index=int(batch_index),
                )
                rel_term, rel_stats = relation_loss(
                    out["z"],
                    y,
                    support,
                    tau_y=float(config["relation"]["tau_y"]),
                    tau_z=float(config["relation"]["tau_z"]),
                )
            total = mse_loss + float(lam) * rel_term
            total.backward()
            null_relation_head_grads_if_warmup(model, epoch, warmup_epochs)
            optimizer.step()
            n_steps += 1
            n_steps_total += 1
            sse += float(mse_num.detach().cpu())
            mse_den += float(mse_denom.detach().cpu())
            ce_sum += float(rel_stats.get("sum_ce") or 0.0)
            rel_den += float(rel_stats.get("denominator") or 0.0)
            step_mse.append(float(mse_loss.detach().cpu()))
            step_rel.append(float(rel_stats.get("loss") or 0.0))
            step_total.append(float(total.detach().cpu()))

        val_metrics, pred_z, target_z = evaluate_split(
            model,
            val_table,
            config,
            graph=val_graph,
            device=device_t,
            features_t=features_val,
        )
        last_pred, last_target, last_metrics = pred_z, target_z, val_metrics
        score = float(val_metrics.patient_macro_pathway_pcc)
        tie_mse = float(val_metrics.patient_macro_z_mse_selection)

        if epoch <= warmup_epochs:
            updated = update_checkpoint_choice(
                warmup_choice,
                epoch=epoch,
                score=score,
                mse=tie_mse,
                lambda_value=lam,
                tolerance=tolerance,
                kind="warmup",
            )
            if updated is not warmup_choice:
                warmup_choice = updated
                save_checkpoint(
                    checkpoint_path_root / "warmup_best.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        lambda_value=lam,
                        arm=arm,
                        seed=seed,
                        kind="warmup",
                        choice=warmup_choice,
                        center_rng=state.center_rng,
                        dropout_gen=state.dropout_gen,
                        config=config,
                    ),
                )
        if epoch >= formal_start:
            updated = update_checkpoint_choice(
                formal_choice,
                epoch=epoch,
                score=score,
                mse=tie_mse,
                lambda_value=lam,
                tolerance=tolerance,
                kind="formal",
            )
            if updated is not formal_choice:
                formal_choice = updated
                save_checkpoint(
                    checkpoint_path_root / "formal_best.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        lambda_value=lam,
                        arm=arm,
                        seed=seed,
                        kind="formal",
                        choice=formal_choice,
                        center_rng=state.center_rng,
                        dropout_gen=state.dropout_gen,
                        config=config,
                    ),
                )
                save_internal_best_npz(
                    run_path / "raw" / "internal_best.npz",
                    pred_z=pred_z,
                    target_z=target_z,
                    table=val_table,
                    epoch=epoch,
                    arm=arm,
                    seed=seed,
                )

        if epoch in comparison_epochs:
            save_internal_best_npz(
                run_path / "raw" / "epoch_predictions" / f"epoch_{epoch}.npz",
                pred_z=pred_z,
                target_z=target_z,
                table=val_table,
                epoch=epoch,
                arm=arm,
                seed=seed,
            )

        early_state = update_early_stop(
            early_state,
            epoch=epoch,
            score=score,
            formal_start_epoch=formal_start,
            count_start_epoch=int(selection_cfg.get("early_stop_count_start_epoch", 16)),
            min_delta=float(selection_cfg.get("early_stop_min_delta", 1e-4)),
            patience=int(selection_cfg.get("early_stop_patience", 10)),
        )

        save_checkpoint(
            checkpoint_path_root / "last.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                lambda_value=lam,
                arm=arm,
                seed=seed,
                kind="last",
                choice=formal_choice if formal_choice.epoch is not None else warmup_choice,
                center_rng=state.center_rng,
                dropout_gen=state.dropout_gen,
                config=config,
            ),
        )

        if epoch in diag_epochs:
            run_fixed_diagnostics(
                model=model,
                config=config,
                seed=seed,
                epoch=epoch,
                train_table=train_table,
                diagnostic_indices=state.diagnostic_indices,
                thresholds=thresholds,
                train_graph=train_graph,
                center_rng=state.center_rng,
                dropout_gen=state.dropout_gen,
                optimizer=optimizer,
                output_dir=run_path / "raw" / "diagnostics",
                device=device_t,
                features_t=features_train,
            )

        mse_epoch = sse / mse_den if mse_den else float("nan")
        rel_epoch = ce_sum / rel_den if rel_den else 0.0
        history_rows.append(
            {
                "epoch": epoch,
                "lambda": lam,
                "n_steps": n_steps,
                "mse_numerator": sse,
                "mse_denominator": mse_den,
                "mse": mse_epoch,
                "relation_numerator": ce_sum,
                "relation_denominator": rel_den,
                "relation_mean": rel_epoch,
                "relation_weighted": lam * rel_epoch,
                "total_step_mean": float(np.mean(step_total)) if step_total else float("nan"),
                "mse_step_mean": float(np.mean(step_mse)) if step_mse else float("nan"),
                "relation_step_mean": float(np.mean(step_rel)) if step_rel else float("nan"),
                "patient_macro_pathway_pcc": val_metrics.patient_macro_pathway_pcc,
                "patient_macro_pathway_pcc_measured": val_metrics.patient_macro_pathway_pcc_measured,
                "patient_macro_z_mse_selection": val_metrics.patient_macro_z_mse_selection,
                "pooled_z_mse": val_metrics.pooled_z_mse,
                "computable": val_metrics.computable,
                "warmup_best_epoch": warmup_choice.epoch,
                "formal_best_epoch": formal_choice.epoch,
                "formal_reason": formal_choice.reason,
                "early_stop_reference": early_state.reference,
                "early_stop_count": early_state.count,
                "early_stop_reason": early_state.reason,
                "early_stop_stopped": early_state.stopped,
            }
        )
        _write_history(run_path / "raw" / "history.csv", history_rows)
        write_json(run_path / "raw" / "history.json", history_rows)
        np.savez(
            run_path / "raw" / "center_orders.npz",
            **center_orders,
            batch_size=np.asarray(int(config["training"]["batch_size"])),
            keep_last_batch=np.asarray(True),
        )
        if early_state.stopped:
            break

    formal_endpoint = resolve_formal_endpoint(formal_choice, last=CheckpointChoice(kind="last"), warmup=warmup_choice)
    result = {
        "arm": arm,
        "seed": int(seed),
        "status": "completed" if formal_endpoint is not None else "no_formal_endpoint",
        "run_dir": str(run_path),
        "device": str(device_t),
        "epochs_completed": history_rows[-1]["epoch"] if history_rows else 0,
        "n_optimizer_steps": n_steps_total,
        "resume_used": False,
        "resume_supported": False,
        "formal_endpoint": None if formal_endpoint is None else formal_endpoint.as_dict(),
        "warmup_best": warmup_choice.as_dict(),
        "early_stop": early_state.as_dict(),
        "checkpoint_directory": str(checkpoint_path_root),
        "formal_checkpoint": str(checkpoint_path_root / "formal_best.pt") if formal_endpoint is not None else None,
        "warmup_checkpoint": str(checkpoint_path_root / "warmup_best.pt") if warmup_choice.epoch is not None else None,
        "last_checkpoint": str(checkpoint_path_root / "last.pt"),
        "history_path": "raw/history.csv",
        "center_orders_path": "raw/center_orders.npz",
        "internal_best_path": "raw/internal_best.npz" if formal_endpoint is not None else None,
        "threshold_ref": threshold_ref,
        "graph_ref": graph_ref,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "note_zh": "无正式终点时不回退 last、warmup 或外部指标。本结果需登记后才能称为已确认。",
        "last_val": None if last_metrics is None else {
            "patient_macro_pathway_pcc": last_metrics.patient_macro_pathway_pcc,
            "patient_macro_z_mse_selection": last_metrics.patient_macro_z_mse_selection,
            "pooled_z_mse": last_metrics.pooled_z_mse,
            "computable": last_metrics.computable,
        },
    }
    write_json(run_path / "raw" / "train_result.json", result)
    write_json(
        run_path / "metrics.json",
        {
            "arm": arm,
            "seed": int(seed),
            "formal_available": formal_endpoint is not None,
            "selection_metric": selection_cfg.get("metric", "patient_macro_pathway_pcc"),
            "tie_metric": selection_cfg.get("tie_metric", "patient_macro_z_mse_selection"),
            "formal": None if formal_endpoint is None else formal_endpoint.as_dict(),
            "warmup": warmup_choice.as_dict(),
            "early_stop": early_state.as_dict(),
            "resume_used": False,
        },
    )
    append_event(
        run_path / "events.jsonl",
        stage="train_end",
        arm=arm,
        seed=int(seed),
        status=result["status"],
        formal_epoch=None if formal_endpoint is None else formal_endpoint.epoch,
    )
    del last_pred, last_target
    return result
