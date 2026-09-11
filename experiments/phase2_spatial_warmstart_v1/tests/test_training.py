from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from src.data import DataBundle, PartitionData
from src.model import SpatialWarmstartModel
from src.training import arm_policy, configure_arm, train_arm
from src.analyze_local import run_fixed_smoothing


def _case(tmp_path: Path):
    torch.manual_seed(7)
    source = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2, dropout=0.3)
    source.eval()
    checkpoint = tmp_path / "source.pt"
    torch.save({
        "shared.linear.weight": source.shared.linear.weight.detach().clone(),
        "shared.linear.bias": source.shared.linear.bias.detach().clone(),
        "regression.linear.weight": source.regression.linear.weight.detach().clone(),
        "regression.linear.bias": source.regression.linear.bias.detach().clone(),
    }, checkpoint)
    features = np.asarray([
        [0.1, 0.2, 0.3], [0.3, 0.2, 0.4], [0.8, 0.1, 0.2], [0.2, 0.7, 0.3],
    ], dtype=np.float32)
    graph_features = features.copy()
    patient = np.asarray(["P", "P", "P", "P"])
    spots = np.asarray(["a", "b", "c", "d"])
    coords = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    with torch.no_grad():
        pred = source(torch.as_tensor(features), use_spatial=False).prediction.numpy()
    targets = pred + np.asarray([[.2, -.1], [-.1, .2], [.3, -.2], [-.2, .1]], dtype=np.float32)
    part = PartitionData(
        name="train", features=features, graph_features=graph_features,
        patient_ids=patient, spot_ids=spots, coordinates=coords,
        native_steps=np.ones(4, dtype=np.float32), targets=targets,
        source_predictions=pred,
    )
    val = PartitionData(**{**part.__dict__, "name": "internal_val"})
    bundle = DataBundle(
        partitions={"train": part, "internal_val": val}, pathway_names=("a", "b"),
        normalization={"scale": "z", "parameters_embedded": False}, sources={"kind": "synthetic"},
    )
    arms = [
        {"id": "point_continue", "train_H": True, "train_C": True, "train_B": False, "lr_HC": 3e-5, "lr_B": None},
        {"id": "spatial_residual_only", "train_H": False, "train_C": False, "train_B": True, "lr_HC": None, "lr_B": 3e-4},
        {"id": "spatial_joint", "train_H": True, "train_C": True, "train_B": True, "lr_HC": 3e-5, "lr_B": 3e-4},
    ]
    config = {
        "experiment_id": "synthetic", "plan_version": "test", "batch_id": "test",
        "inputs": {"stage1_checkpoint": str(checkpoint)},
        "parameters": {
            "arms": arms, "primary_seed": 42, "input_dim": 3, "hidden_dim": 2,
            "output_dim": 2, "dropout": .3, "weight_decay": 1e-4,
            "optimizer": "AdamW", "lr_schedule": "constant",
            "batch_mode": "full_batch", "loss": "all_point_pathway_mean_zMSE",
            "extend_at_1000_if_not_early_stopped": True,
            "backbone_trainable": False, "lora_enabled": False, "contrastive_enabled": False,
            "initial_update_budget": 2, "hard_max_updates": 4,
            "validation_every_updates": 1, "min_updates_before_early_stop": 4,
            "patience_validation_checks": 99, "min_delta_patient_macro_pcc": 1e-4,
            "initial_prediction_max_abs_tolerance": 1e-6,
            "graph": {"radius_in_native_steps": 1.5, "max_neighbors": 8,
                      "distance_sigma": 1.0, "image_temperature": .2, "self_raw_weight": 1.0},
        },
    }
    return config, bundle


def test_arm_freezing_contract(tmp_path: Path) -> None:
    config, _ = _case(tmp_path)
    for arm, expected in {
        "point_continue": (True, True, False),
        "spatial_residual_only": (False, False, True),
        "spatial_joint": (True, True, True),
    }.items():
        model = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2)
        configure_arm(model, arm_policy(config, arm))
        actual = (
            all(p.requires_grad for p in model.shared.parameters()),
            all(p.requires_grad for p in model.regression.parameters()),
            model.B.requires_grad,
        )
        assert actual == expected


def test_three_arms_complete_synthetic_update_contract(tmp_path: Path) -> None:
    config, bundle = _case(tmp_path)
    for arm in ("spatial_residual_only", "point_continue", "spatial_joint"):
        summary = train_arm(
            config, bundle, arm, tmp_path / f"run_{arm}", tmp_path / f"weights_{arm}", device="cpu"
        )
        assert summary["last_update"] == 4
        assert summary["stop_reason"] == "budget_exhausted"
        assert summary["external_evaluation"] == "not_run_by_training"
        assert Path(summary["warmup"]).is_file()
        assert Path(summary["formal"]).is_file()
        assert Path(summary["last"]).is_file()
        assert Path(summary["model_bundle"]).is_file()


def test_resume_restores_optimizer_rng_and_matches_uninterrupted(tmp_path: Path) -> None:
    config, bundle = _case(tmp_path)
    arm = "spatial_joint"
    continuous = train_arm(
        config, bundle, arm, tmp_path / "continuous_run", tmp_path / "continuous_weights", device="cpu"
    )
    interrupted = train_arm(
        config, bundle, arm, tmp_path / "parent_run", tmp_path / "parent_weights",
        device="cpu", stop_after_updates=2,
    )
    assert interrupted["stop_reason"] == "interrupted"
    resumed = train_arm(
        config, bundle, arm, tmp_path / "child_run", tmp_path / "child_weights",
        device="cpu", resume_checkpoint=interrupted["last"],
    )
    left = torch.load(continuous["last"], map_location="cpu", weights_only=False)
    right = torch.load(resumed["last"], map_location="cpu", weights_only=False)
    assert left["update"] == right["update"] == 4
    assert left["best"] == right["best"]
    for key in left["model_state"]:
        torch.testing.assert_close(left["model_state"][key], right["model_state"][key], rtol=0, atol=0)


def test_local_fixed_smoothing_uses_saved_step0_and_graph(tmp_path: Path) -> None:
    config, bundle = _case(tmp_path)
    run_dir = tmp_path / "returned_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    train_arm(
        config, bundle, "spatial_residual_only", run_dir, tmp_path / "returned_weights", device="cpu"
    )
    report = run_fixed_smoothing(run_dir)
    assert report["beta"] == 1.0
    assert report["used_for_selection"] is False
    assert (run_dir / "analysis" / "fixed_smoothing.json").is_file()
