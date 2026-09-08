from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from errors import ConfigError
from helpers import package_config, tiny_model_config
from model import build_model
from predict import load_formal_model, save_predictions
from run_experiments import execute_training_batch, plan_training_tasks, validate_endpoint_manifest
from run_io import read_json


def test_plan_order_and_scope():
    config = package_config()
    tasks = plan_training_tasks(config, scope="all")
    assert [(task.arm, task.seed) for task in tasks] == [
        (arm, seed) for seed in (42, 43, 44) for arm in ("point", "relation", "spatial", "joint")
    ]
    assert [(task.arm, task.seed) for task in plan_training_tasks(config, scope="spatial")] == [
        ("spatial", 42), ("spatial", 43), ("spatial", 44)
    ]
    assert [(task.arm, task.seed) for task in plan_training_tasks(config, scope="seed", seed=43)] == [
        (arm, 43) for arm in ("point", "relation", "spatial", "joint")
    ]


def test_batch_freezes_snapshot_and_marks_remaining_not_run(tmp_path):
    config = package_config()
    observed = []

    def fake_precheck(snapshot, output_dir, **_):
        assert snapshot["model"]["hidden_dim"] == 256
        config["model"]["hidden_dim"] = 999
        return {"status": "ok", "errors": [], "warnings": []}

    def fake_train(snapshot, arm, seed, run_dir, **_):
        observed.append((arm, seed, snapshot["model"]["hidden_dim"]))
        if arm == "relation":
            raise RuntimeError("planned test failure")
        return {"status": "completed", "formal_endpoint": {"epoch": 6, "score": 0.1, "mse": 1.0}}

    result = execute_training_batch(
        config,
        runs_root=tmp_path,
        scope="seed",
        seed=42,
        artifact_kind="test_artifact",
        precheck_runner=fake_precheck,
        train_runner=fake_train,
    )
    assert result["exit_code"] == 1
    assert observed == [("point", 42, 256), ("relation", 42, 256)]
    manifest = read_json(Path(result["batch_dir"]) / "batch_manifest.json")
    assert [item["status"] for item in manifest["tasks"]] == ["succeeded", "failed", "not_run", "not_run"]
    assert read_json(Path(result["batch_dir"]) / "config_snapshot.json")["model"]["hidden_dim"] == 256
    assert (Path(result["batch_dir"]) / "source_snapshot" / "src" / "run_experiments.py").is_file()


def test_full_batch_registers_external_weight_locations_without_embedding_weights(tmp_path):
    config = package_config()
    runs_root = tmp_path / "runs"
    weights_root = tmp_path / "weights"

    def fake_precheck(_snapshot, _output_dir, **_):
        return {"status": "ok", "errors": [], "warnings": []}

    def fake_train(_snapshot, arm, seed, run_dir, *, checkpoint_dir, **_):
        checkpoint_dir.mkdir(parents=True)
        paths = {}
        for name in ("warmup_best.pt", "formal_best.pt", "last.pt"):
            path = checkpoint_dir / name
            path.write_bytes(f"{arm}-{seed}-{name}".encode())
            paths[name] = str(path.resolve())
        return {
            "status": "completed",
            "formal_endpoint": {"epoch": 6, "score": 0.1, "mse": 1.0},
            "formal_checkpoint": paths["formal_best.pt"],
            "warmup_checkpoint": paths["warmup_best.pt"],
            "last_checkpoint": paths["last.pt"],
        }

    result = execute_training_batch(
        config,
        runs_root=runs_root,
        weights_root=weights_root,
        scope="all",
        artifact_kind="test_artifact",
        precheck_runner=fake_precheck,
        train_runner=fake_train,
    )
    assert result["exit_code"] == 0
    batch = Path(result["batch_dir"])
    registry = read_json(batch / "model_weights.json")
    assert registry["weights_root"] == str(weights_root.resolve())
    assert Path(registry["weight_batch_directory"]) == weights_root.resolve() / config["experiment_id"] / batch.name
    assert len(registry["entries"]) == 12
    assert all(Path(item["formal_checkpoint"]).is_file() for item in registry["entries"])
    assert registry["code_version"] == "v2.1.2"
    assert all(item["formal_selection"]["epoch"] == 6 for item in registry["entries"])
    assert all({file["kind"] for file in item["files"]} == {"warmup", "formal", "last"} for item in registry["entries"])
    assert all(file["size_bytes"] > 0 for item in registry["entries"] for file in item["files"])
    assert not list(batch.rglob("*.pt"))
    endpoints = read_json(batch / "formal_endpoints.json")["endpoints"]
    assert {item["checkpoint"] for item in endpoints} == {
        item["formal_checkpoint"] for item in registry["entries"]
    }


def test_endpoint_manifest_rejects_warmup_and_mismatched_metadata(tmp_path):
    config = package_config()
    checkpoint = tmp_path / "formal.pt"
    checkpoint.write_bytes(b"placeholder")
    endpoint = {
        "plan_version": config["plan_version"],
        "config_version": config["config_version"],
        "comparison_name": config["comparison_name"],
        "run_kind": "formal",
        "split_id": config["data"]["split_id"],
        "endpoints": [
            {"arm": arm, "seed": seed, "checkpoint_kind": "formal", "checkpoint": str(checkpoint)}
            for seed in config["training"]["seeds"]
            for arm in config["training"]["arms"]
        ],
    }
    assert len(validate_endpoint_manifest(endpoint, config)) == 12
    bad = copy.deepcopy(endpoint)
    bad["endpoints"][0]["checkpoint_kind"] = "warmup"
    with pytest.raises(ConfigError, match="formal"):
        validate_endpoint_manifest(bad, config)
    bad = copy.deepcopy(endpoint)
    bad["comparison_name"] = "other"
    with pytest.raises(ConfigError, match="comparison_name"):
        validate_endpoint_manifest(bad, config)


def test_formal_prediction_is_label_free_and_inverse_transformed_once(tmp_path):
    config = tiny_model_config(input_dim=8, output_dim=5)
    model = build_model(config, "point")
    checkpoint = tmp_path / "formal.pt"
    torch.save(
        {
            "kind": "formal",
            "arm": "point",
            "seed": 42,
            "model_state_dict": model.state_dict(),
            "selection": {"kind": "formal", "epoch": 6},
        },
        checkpoint,
    )
    loaded, metadata = load_formal_model(config, checkpoint, arm="point", seed=42, device="cpu")
    features = np.zeros((2, 8), dtype=np.float32)
    with torch.no_grad():
        expected_z = loaded(torch.from_numpy(features))["y_hat"].numpy()
    output = tmp_path / "predictions.npz"
    save_predictions(
        output,
        pred_z=expected_z,
        mean=np.arange(5, dtype=float),
        std=np.full(5, 2.0),
        pathway_names=[f"p{i}" for i in range(5)],
        patient_ids=["XZY", "XZY"],
        slide_ids=["s", "s"],
        spot_ids=["a", "b"],
        x=[0, 1],
        y=[0, 1],
        checkpoint_metadata=metadata,
    )
    saved = np.load(output, allow_pickle=True)
    np.testing.assert_allclose(saved["pred_raw"], expected_z * 2.0 + np.arange(5))
    assert "target_z" not in saved.files
    assert saved["inverse_transform_count"].item() == 1
