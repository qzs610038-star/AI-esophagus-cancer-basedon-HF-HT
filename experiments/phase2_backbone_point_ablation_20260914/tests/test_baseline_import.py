from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline_import import (
    extract_point_output_state,
    load_reference_entry,
    validate_reference_identity_alignment,
    validate_reference_files,
)
from data import IdentityRecord
from errors import ReferenceMismatchError


def _reference_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "historical"
    train_raw = root / "train_42_point" / "raw"
    external_raw = root / "external_42_point" / "raw"
    checkpoint_dir = root / "train_42_point" / "checkpoints"
    train_raw.mkdir(parents=True)
    external_raw.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "shared.weight": torch.zeros(256, 1536),
            "shared.bias": torch.zeros(256),
            "point_head.weight": torch.arange(30 * 256, dtype=torch.float32).reshape(30, 256),
            "point_head.bias": torch.arange(30, dtype=torch.float32),
            "relation_head.weight": torch.ones(30, 256),
        },
        train_raw / "initial_weights.pt",
    )
    center_rng = np.random.Generator(np.random.PCG64(42 + 100_000))
    center_orders = {
        f"epoch_{epoch}": center_rng.permutation(2).astype(np.int64)
        for epoch in range(1, 8)
    }
    np.savez_compressed(
        train_raw / "center_orders.npz",
        **center_orders,
        batch_size=np.asarray(256),
        keep_last_batch=np.asarray(True),
    )
    np.savez_compressed(
        train_raw / "internal_best.npz",
        pred_z=np.zeros((2, 30), np.float32),
        target_z=np.ones((2, 30), np.float32),
        patients=np.asarray(["A", "B"]),
        source_groups=np.asarray(["MPP2_A_source01", "MPP2_B_source01"]),
        spots=np.asarray(["patch_x0_y0", "patch_x1_y1"]),
        pathways=np.asarray([f"p{i}" for i in range(30)]),
        epoch=np.asarray(7),
        arm=np.asarray("point"),
        seed=np.asarray(42),
    )
    np.savez_compressed(
        external_raw / "external_predictions.npz",
        pred_z=np.zeros((1, 30), np.float32),
        pred_raw=np.zeros((1, 30), np.float32),
        patients=np.asarray(["XZY"]),
        source_groups=np.asarray(["MPP2_XZY_source01"]),
        spots=np.asarray(["patch_x2_y2"]),
        pathways=np.asarray([f"p{i}" for i in range(30)]),
        arm=np.asarray("point"),
        seed=np.asarray(42),
    )
    torch.save(
        {
            "arm": "point",
            "seed": 42,
            "epoch": 7,
            "kind": "formal",
            "model_state_dict": {
                "shared.weight": torch.zeros(256, 1536),
                "shared.bias": torch.zeros(256),
                "point_head.weight": torch.zeros(30, 256),
                "point_head.bias": torch.zeros(30),
            },
        },
        checkpoint_dir / "formal_best.pt",
    )
    manifest = {
        "schema_version": "1.0",
        "status": "historical_matched_reference_reused",
        "model": "uni2h",
        "arm": "point",
        "source_experiment": "phase2_softlink_local_v2",
        "source_batch": "20260908_005325_810_8d306c10",
        "source_date": "2026-09-08",
        "contract": {
            "split_id": "MPP2/group_2",
            "label_version": "barcode-repair-v003",
            "input_dim": 1536,
            "hidden_dim": 256,
            "output_dim": 30,
            "dropout": 0.3,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "batch_size": 256,
            "keep_last_batch": True,
            "max_epochs": 60,
            "scheduler": "none",
            "precision": "float32",
            "amp": False,
            "tf32": False,
            "num_workers": 0,
            "cpu_threads": 8,
            "selection_metric": "patient_macro_pathway_pcc",
            "tie_metric": "patient_macro_z_mse_selection",
            "formal_start_epoch": 6,
            "checkpoint_tolerance": 1e-6,
            "early_stop_count_start_epoch": 16,
            "early_stop_min_delta": 1e-4,
            "early_stop_patience": 10,
            "constant_prediction_selection_penalty": -1.0,
            "model_seed_offset": 0,
            "center_seed_offset": 100000,
            "dropout_seed_offset": 200000,
        },
        "seeds": {
            "42": {
                "train_run": "train_42_point",
                "external_run": "external_42_point",
                "formal_epoch": 7,
                "internal_predictions": "train_42_point/raw/internal_best.npz",
                "external_predictions": "external_42_point/raw/external_predictions.npz",
                "initial_weights": "train_42_point/raw/initial_weights.pt",
                "center_orders": "train_42_point/raw/center_orders.npz",
                "formal_checkpoint_registered": "train_42_point/checkpoints/formal_best.pt",
            }
        },
    }
    manifest_path = tmp_path / "reference.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, root


def test_reference_requires_exact_contract_and_formal_point_seed(tmp_path: Path):
    manifest_path, root = _reference_tree(tmp_path)
    entry = load_reference_entry(manifest_path, 42)
    validated = validate_reference_files(
        entry, root, train_count=2, internal_count=2, external_count=1
    )
    assert validated["formal_epoch"] == 7
    assert validated["status"] == "historical_matched_reference_reused"
    rows = [
        IdentityRecord(0, "A", "MPP2_A_source01", "patch_x0_y0", "internal_val", 0, 0, "a", 224, 224),
        IdentityRecord(1, "B", "MPP2_B_source01", "patch_x1_y1", "internal_val", 1, 1, "b", 224, 224),
        IdentityRecord(2, "XZY", "MPP2_XZY_source01", "patch_x2_y2", "external_test", 2, 2, "c", 224, 224),
    ]
    validate_reference_identity_alignment(validated, rows, [f"p{i}" for i in range(30)])


def test_reference_external_prediction_must_not_contain_target(tmp_path: Path):
    manifest_path, root = _reference_tree(tmp_path)
    entry = load_reference_entry(manifest_path, 42)
    np.savez_compressed(
        root / "external_42_point" / "raw" / "external_predictions.npz",
        pred_z=np.zeros((1, 30), np.float32),
        target_z=np.zeros((1, 30), np.float32),
        arm=np.asarray("point"),
        seed=np.asarray(42),
    )
    with pytest.raises(ReferenceMismatchError, match="target_z"):
        validate_reference_files(
            entry, root, train_count=2, internal_count=2, external_count=1
        )


def test_only_historical_point_output_layer_is_extracted(tmp_path: Path):
    manifest_path, root = _reference_tree(tmp_path)
    entry = load_reference_entry(manifest_path, 42)
    state = extract_point_output_state(root / entry["initial_weights"])
    assert set(state) == {"weight", "bias"}
    assert state["weight"].shape == (30, 256)
    assert torch.all(state["bias"] == torch.arange(30))


def test_reference_manifest_rejects_wrong_arm(tmp_path: Path):
    manifest_path, _ = _reference_tree(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["arm"] = "spatial"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceMismatchError, match="point"):
        load_reference_entry(manifest_path, 42)


def test_reference_manifest_rejects_a_different_historical_batch(tmp_path: Path):
    manifest_path, _ = _reference_tree(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_batch"] = "another_batch"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceMismatchError, match="source_batch"):
        load_reference_entry(manifest_path, 42)


def test_reference_rejects_a_registered_checkpoint_that_is_not_the_formal_endpoint(tmp_path: Path):
    manifest_path, root = _reference_tree(tmp_path)
    entry = load_reference_entry(manifest_path, 42)
    checkpoint = root / entry["formal_checkpoint_registered"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["kind"] = "last"
    torch.save(payload, checkpoint)
    with pytest.raises(ReferenceMismatchError, match="formal checkpoint"):
        validate_reference_files(entry, root, internal_count=2, external_count=1, train_count=2)


def test_historical_output_initialization_requires_the_frozen_30_by_256_shape(tmp_path: Path):
    path = tmp_path / "bad_initial.pt"
    torch.save(
        {
            "point_head.weight": torch.zeros(30, 3),
            "point_head.bias": torch.zeros(30),
        },
        path,
    )
    with pytest.raises(ReferenceMismatchError, match="30, 256"):
        extract_point_output_state(path)
