"""Formal-checkpoint-only external prediction; this module never accepts labels."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from data import IdentityRecord, Normalization, iter_index_batches
from errors import ConfigError, NonFiniteDataError
from model import PointRegressor


def _load_formal_model(
    checkpoint_path: str | Path,
    *,
    model_name: str,
    seed: int,
    input_dim: int,
    config: dict,
    device: torch.device,
) -> PointRegressor:
    path = Path(checkpoint_path).resolve()
    if path.name != "formal_best.pt" or not path.is_file():
        raise ConfigError("XZY 只允许使用已锁定的 formal_best.pt")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConfigError(f"无法安全读取正式检查点: {exc}") from exc
    expected = {
        "kind": "formal",
        "model_name": model_name,
        "arm": "point",
        "seed": int(seed),
        "input_dim": int(input_dim),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "output_dim": int(config["model"]["output_dim"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ConfigError(f"正式检查点 {key} 不匹配: {payload.get(key)!r} != {value!r}")
    model = PointRegressor(
        input_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        output_dim=int(config["model"]["output_dim"]),
        dropout_p=float(config["model"]["dropout"]),
    )
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ConfigError(f"正式检查点模型状态严格加载失败: {exc}") from exc
    return model.to(device=device, dtype=torch.float32).eval()


def predict_external_arrays(
    features: np.ndarray,
    rows: Sequence[IdentityRecord],
    config: dict,
    model_name: str,
    seed: int,
    checkpoint_path: str | Path,
    output_path: str | Path,
    normalization: Normalization,
    device: str | torch.device | None = None,
) -> dict:
    """Predict XZY only after a formal endpoint is supplied; no target API exists."""

    if model_name not in {"uni", "virchow2"}:
        raise ConfigError("外部预测只允许新候选模型")
    if any(row.split != "external_test" or row.patient_id != config["data"]["external_patient"] for row in rows):
        raise ConfigError("外部预测 rows 必须全部属于 XZY external_test")
    expected_count = int(config["data"]["expected_counts"]["external_test"])
    if len(rows) != expected_count:
        raise ConfigError(f"XZY 行数应为 {expected_count}，实际={len(rows)}")
    input_dim = {"uni": 1024, "virchow2": 1280}[model_name]
    values = np.asarray(features)
    if values.dtype != np.float32 or values.shape != (len(rows), input_dim):
        raise ConfigError(f"XZY 特征应为 float32 {(len(rows), input_dim)}，实际={values.dtype}/{values.shape}")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("XZY 特征含 NaN/Inf")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_formal_model(
        checkpoint_path,
        model_name=model_name,
        seed=seed,
        input_dim=input_dim,
        config=config,
        device=torch_device,
    )
    feature_tensor = torch.as_tensor(values, dtype=torch.float32, device=torch_device)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for indices in iter_index_batches(range(len(rows)), int(config["training"]["batch_size"])):
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=torch_device)
            predictions.append(
                model(feature_tensor.index_select(0, index_tensor), dropout_mask=None)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
    pred_z = np.concatenate(predictions, axis=0)
    pred_raw = normalization.inverse_transform(pred_z)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        pred_z=pred_z,
        pred_raw=pred_raw,
        patients=np.asarray([row.patient_id for row in rows], dtype=str),
        source_groups=np.asarray([row.source_group for row in rows], dtype=str),
        spots=np.asarray([row.spot_id for row in rows], dtype=str),
        x=np.asarray([row.x for row in rows], dtype=np.int64),
        y=np.asarray([row.y for row in rows], dtype=np.int64),
        pathways=np.asarray(normalization.pathway_names, dtype=str),
        checkpoint=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_kind=np.asarray("formal"),
        arm=np.asarray("point"),
        model=np.asarray(model_name),
        seed=np.asarray(int(seed)),
    )
    return {
        "status": "completed",
        "model": model_name,
        "seed": int(seed),
        "n_predictions": len(rows),
        "prediction_path": str(destination.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_kind": "formal",
        "contains_target_z": False,
    }
