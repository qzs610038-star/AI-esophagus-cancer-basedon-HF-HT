"""Frozen formal-checkpoint prediction. Labels are intentionally absent."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from config import SUPPORTED_ARMS
from data import PointTable, load_feature_vector, load_normalization_from_config, make_point_table
from errors import ConfigError
from graph import build_split_graph
from model import build_model
from train import forward_batch, iter_center_batches, resolve_device


def load_formal_model(
    config: dict,
    checkpoint_path: str | Path,
    *,
    arm: str,
    seed: int,
    device: str | torch.device | None = None,
):
    """Load only a formally selected endpoint; warmup/last are rejected."""
    if arm not in SUPPORTED_ARMS:
        raise ConfigError(f"未知实验臂 {arm!r}")
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    selection = payload.get("selection") or {}
    if payload.get("kind") != "formal" or selection.get("kind") != "formal":
        raise ConfigError(f"外部预测只允许 formal checkpoint，当前={payload.get('kind')!r}")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ConfigError("checkpoint 的 arm/seed 与冻结端点清单不一致")
    device_t = resolve_device(device)
    model = build_model(config, arm).to(device_t)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        "checkpoint": str(path.resolve()),
        "checkpoint_kind": "formal",
        "arm": arm,
        "seed": int(seed),
        "epoch": int(selection["epoch"]),
    }
    return model, metadata


def load_external_point_table(config: dict) -> PointTable:
    """Load the explicit XZY point table. No label columns are read."""
    value = config["data"].get("external_point_table")
    if not value:
        raise ConfigError("data.external_point_table 未核实；不能开始 XZY 外部预测")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"外部点表不存在: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = ("patient_id", "slide_id", "spot_id", "x", "y", "feature_path")
    missing = [name for name in required if not rows or name not in rows[0]]
    if missing:
        raise ConfigError(f"外部点表缺列: {missing}")
    if any(not str(row["slide_id"]).strip() for row in rows):
        raise ConfigError("外部点表 slide_id 为空；不得用 patient_id 顶替")
    feature_paths = []
    features = []
    for row in rows:
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        feature_paths.append(str(feature_path))
        features.append(load_feature_vector(feature_path, expected_dim=int(config["data"]["input_dim"])))
    return make_point_table(
        patient_ids=[row["patient_id"] for row in rows],
        slide_ids=[row["slide_id"] for row in rows],
        spot_ids=[row["spot_id"] for row in rows],
        splits=["external"] * len(rows),
        x=[float(row["x"]) for row in rows],
        y=[float(row["y"]) for row in rows],
        features=np.stack(features, axis=0),
        feature_paths=feature_paths,
        slide_status="verified",
        sort_identities=True,
        source=str(path),
    )


def predict_table(
    config: dict,
    *,
    arm: str,
    model,
    table: PointTable,
    device: str | torch.device | None = None,
) -> np.ndarray:
    if table.labels_z is not None:
        raise ConfigError("外部预测接口不接受标签")
    if table.features is None:
        raise ConfigError("外部预测需要冻结图像特征")
    device_t = resolve_device(device)
    graph = build_split_graph(table, "external", config) if arm in ("spatial", "joint") else None
    features_t = torch.as_tensor(table.features, dtype=torch.float32, device=device_t)
    predictions = []
    with torch.no_grad():
        for _, indices in iter_center_batches(np.arange(len(table)), config):
            output = forward_batch(
                model,
                table,
                indices,
                graph=graph,
                dropout_gen=None,
                apply_dropout=False,
                device=device_t,
                features_t=features_t,
            )
            predictions.append(output["y_hat"].detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float64, copy=False)


def save_predictions(
    path: str | Path,
    *,
    pred_z: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    pathway_names: Sequence[str],
    patient_ids: Sequence[str],
    slide_ids: Sequence[str],
    spot_ids: Sequence[str],
    x: Sequence[float],
    y: Sequence[float],
    checkpoint_metadata: dict,
) -> Path:
    """Invert training normalization exactly once and store no target array."""
    pred_z = np.asarray(pred_z, dtype=np.float64)
    mean_v = np.asarray(mean, dtype=np.float64)
    std_v = np.asarray(std, dtype=np.float64)
    if pred_z.shape[1] != len(pathway_names) or mean_v.shape != std_v.shape or mean_v.size != pred_z.shape[1]:
        raise ConfigError("预测、通路名和标准化参数维数不一致")
    pred_raw = pred_z * std_v + mean_v
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        pred_z=pred_z,
        pred_raw=pred_raw,
        pathway_names=np.asarray(pathway_names, dtype=object),
        patient_id=np.asarray(patient_ids, dtype=object),
        slide_id=np.asarray(slide_ids, dtype=object),
        spot_id=np.asarray(spot_ids, dtype=object),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        inverse_transform_count=np.asarray(1),
        checkpoint_metadata=np.asarray(checkpoint_metadata, dtype=object),
    )
    return output


def predict_endpoint(config: dict, endpoint: dict, output_dir: str | Path, *, device=None) -> dict:
    table = load_external_point_table(config)
    arm, seed = str(endpoint["arm"]), int(endpoint["seed"])
    model, metadata = load_formal_model(config, endpoint["checkpoint"], arm=arm, seed=seed, device=device)
    pred_z = predict_table(config, arm=arm, model=model, table=table, device=device)
    normalization = load_normalization_from_config(config)
    path = Path(output_dir) / "raw" / "external_predictions.npz"
    save_predictions(
        path,
        pred_z=pred_z,
        mean=normalization.mean,
        std=normalization.std,
        pathway_names=normalization.pathway_names,
        patient_ids=[item.patient_id for item in table.identities],
        slide_ids=[item.slide_id for item in table.identities],
        spot_ids=[item.spot_id for item in table.identities],
        x=table.x,
        y=table.y,
        checkpoint_metadata=metadata,
    )
    return {"status": "completed", "arm": arm, "seed": seed, "prediction_path": str(path)}
