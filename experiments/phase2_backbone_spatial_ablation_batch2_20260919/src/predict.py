"""Prediction from a formal checkpoint and an identity-aligned feature table."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from config import SUPPORTED_ARMS
from data import PointTable
from errors import ConfigError
from graph import build_split_graph
from model import build_model
from train import forward_batch, iter_center_batches, resolve_device


def load_formal_model(
    config: dict, checkpoint_path: str | Path, *, arm: str, seed: int, device=None
) -> tuple[torch.nn.Module, dict]:
    if arm not in SUPPORTED_ARMS:
        raise ConfigError(f"未知实验臂 {arm!r}")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    selection = payload.get("selection") or {}
    if payload.get("kind") != "formal" or selection.get("kind") != "formal":
        raise ConfigError("外部与导出预测只允许正式选中 checkpoint")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ConfigError("checkpoint 的实验臂或种子不匹配")
    if int(payload.get("input_dim", -1)) != int(config["data"]["input_dim"]):
        raise ConfigError("checkpoint 输入维数与编码器不匹配")
    if int(payload.get("hidden_dim", -1)) != int(config["model"]["hidden_dim"]):
        raise ConfigError("checkpoint 隐藏维数与冻结配置不匹配")
    device_t = resolve_device(device)
    model = build_model(config, arm).to(device_t)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(path.resolve()), "kind": "formal", "arm": arm,
        "seed": int(seed), "epoch": int(selection["epoch"]),
        "input_dim": int(payload["input_dim"]), "hidden_dim": int(payload["hidden_dim"]),
    }


def predict_table(
    config: dict, *, arm: str, model: torch.nn.Module,
    table: PointTable, split: str = "external_test", device=None
) -> np.ndarray:
    """Run label-free inference on one complete identity group and split."""
    if table.labels_z is not None:
        raise ConfigError("预测接口不接收外部标签；评分数据在预测后独立接入")
    if table.features is None or len(table) == 0:
        raise ConfigError("预测需要非空冻结图像特征")
    if set(table.split.tolist()) != {split}:
        raise ConfigError("预测表必须恰好属于声明的数据集合")
    device_t = resolve_device(device)
    graph = build_split_graph(table, split, config) if arm == "spatial" else None
    features_t = torch.as_tensor(table.features, dtype=torch.float32, device=device_t)
    rows = []
    model.eval()
    with torch.inference_mode():
        for _, indices in iter_center_batches(np.arange(len(table)), config):
            output = forward_batch(
                model, table, indices, graph=graph, dropout_gen=None,
                apply_dropout=False, device=device_t, features_t=features_t,
            )
            rows.append(output["y_hat"].detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


def save_predictions(
    path: str | Path, *, pred_z: np.ndarray, mean: Sequence[float], std: Sequence[float],
    pathway_names: Sequence[str], table: PointTable, checkpoint_metadata: dict,
    target_z: np.ndarray | None = None,
) -> Path:
    pred_z = np.asarray(pred_z, dtype=np.float64)
    mean_v = np.asarray(mean, dtype=np.float64)
    std_v = np.asarray(std, dtype=np.float64)
    if pred_z.shape != (len(table), len(pathway_names)) or mean_v.shape != std_v.shape or mean_v.size != pred_z.shape[1]:
        raise ConfigError("预测、身份、通路或标准化参数维数不一致")
    if not np.isfinite(pred_z).all():
        raise ConfigError("预测存在非有限值")
    target = np.empty((0, pred_z.shape[1]), dtype=np.float64) if target_z is None else np.asarray(target_z, dtype=np.float64)
    if target_z is not None and (target.shape != pred_z.shape or not np.isfinite(target).all()):
        raise ConfigError("独立评价真值与预测不对齐")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pred_z=pred_z,
        pred_raw=pred_z * std_v + mean_v,
        target_z=target,
        target_available=np.asarray(target_z is not None),
        patient_id=np.asarray([ident.patient_id for ident in table.identities], dtype=str),
        slide_id=np.asarray([ident.slide_id for ident in table.identities], dtype=str),
        spot_id=np.asarray([ident.spot_id for ident in table.identities], dtype=str),
        x=np.asarray(table.x, dtype=np.float64), y=np.asarray(table.y, dtype=np.float64),
        pathway_names=np.asarray(pathway_names, dtype=str),
        mean=mean_v, std=std_v,
        inverse_transform_count=np.asarray(1),
        checkpoint_metadata=np.asarray(checkpoint_metadata, dtype=object),
    )
    return output
