from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.protocol import Predictor
from targets.protocol import TargetSpec


def write_prediction_table(
    path: Path,
    records: list[dict],
    labels: np.ndarray,
    preds: np.ndarray,
    spec: TargetSpec,
    model_id: str,
    sampling_id: str,
    code_version: str,
    checkpoint_source: str,
) -> Path:
    if len(records) != len(labels) or len(records) != len(preds):
        raise ValueError("records, labels and predictions must have the same length")
    frame = pd.DataFrame(records)
    for index, name in enumerate(spec.names):
        frame[f"true_{name}"] = labels[:, index]
        frame[f"pred_{name}"] = preds[:, index]
    frame["model_id"] = model_id
    frame["target_id"] = spec.target_id
    frame["sampling_id"] = sampling_id
    frame["code_version"] = code_version
    frame["checkpoint_source"] = checkpoint_source
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_target_spec(path: Path, spec: TargetSpec) -> None:
    path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def predictor_meta(predictor: Predictor) -> dict:
    return {
        "model_id": predictor.model_id,
        "in_dim": predictor.in_dim,
        "out_dim": predictor.out_dim,
    }
