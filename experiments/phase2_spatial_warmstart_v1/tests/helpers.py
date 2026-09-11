from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_data_package(
    root: Path,
    *,
    feature_dim: int = 3,
    output_dim: int = 2,
    include_train_targets: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create the smallest valid train/internal-validation cache contract."""
    package_dir = root / "package"
    inputs = package_dir / "inputs" / "mpp2"
    inputs.mkdir(parents=True)

    manifest_rows = [
        {
            "mpp_id": "2", "patient": "P2", "patch_stem": "patch_x20_y20",
            "x": "20", "y": "20", "split": "train", "block_id": "P2:0:0",
        },
        {
            "mpp_id": "2", "patient": "P1", "patch_stem": "patch_x10_y10",
            "x": "10", "y": "10", "split": "train", "block_id": "P1:0:0",
        },
        {
            "mpp_id": "2", "patient": "P1", "patch_stem": "patch_x30_y40",
            "x": "30", "y": "40", "split": "internal_val", "block_id": "P1:1:1",
        },
    ]
    with (inputs / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    _write_json(
        inputs / "split_info.json",
        {
            "per_patient_main_stride": {
                "P1": {"main_dx": 10, "main_dy": 10},
                "P2": {"main_dx": 20, "main_dy": 20},
            }
        },
    )
    _write_json(
        inputs / "zscore_manifest.json",
        {
            "pathway_names": [f"pathway_{index}" for index in range(output_dim)],
            "ddof": 1,
            "fit_split": "train",
            "excluded_splits": ["internal_val", "external_test"],
        },
    )

    prediction_arrays: dict[str, np.ndarray] = {
        "cls_train": np.arange(2 * feature_dim, dtype=np.float32).reshape(2, feature_dim),
        "pred_train": np.zeros((2, output_dim), dtype=np.float32),
        "patient_train": np.asarray(["P2", "P1"]),
        "cls_val": np.arange(feature_dim, dtype=np.float32).reshape(1, feature_dim) + 10,
        "pred_val": np.zeros((1, output_dim), dtype=np.float32),
        "target_val": np.full((1, output_dim), 1.5, dtype=np.float32),
        "patient_val": np.asarray(["P1"]),
    }
    if include_train_targets:
        prediction_arrays["target_train"] = np.arange(
            2 * output_dim, dtype=np.float32
        ).reshape(2, output_dim)

    graph_arrays = {
        "cls_train": prediction_arrays["cls_train"].copy(),
        "patient_train": np.asarray(["P2", "P1"]),
        "patch_stem_train": np.asarray(["patch_x20_y20", "patch_x10_y10"]),
        "cls_val": prediction_arrays["cls_val"].copy(),
        "patient_val": np.asarray(["P1"]),
        "patch_stem_val": np.asarray(["patch_x30_y40"]),
    }

    feature_cache = root / "stage1_predictions.npz"
    graph_cache = root / "frozen_uni_graph.npz"
    np.savez(feature_cache, **prediction_arrays)
    np.savez(graph_cache, **graph_arrays)
    checkpoint = root / "epoch_5.pt"
    hidden_dim = 2
    torch.save({
        "shared.linear.weight": torch.zeros(hidden_dim, feature_dim),
        "shared.linear.bias": torch.zeros(hidden_dim),
        "regression.linear.weight": torch.zeros(output_dim, hidden_dim),
        "regression.linear.bias": torch.zeros(output_dim),
    }, checkpoint)
    config = {
        "inputs": {
            "feature_cache": str(feature_cache),
            "graph_feature_cache": str(graph_cache),
            "stage1_checkpoint": str(checkpoint),
            "xzy_source": str(root / "must_not_be_read" / "external_labels.csv"),
        },
        "parameters": {
            "input_dim": feature_dim, "hidden_dim": hidden_dim,
            "output_dim": output_dim, "dropout": 0.3,
        },
    }
    return config, prediction_arrays, graph_arrays


def replace_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(Path(path), **arrays)
