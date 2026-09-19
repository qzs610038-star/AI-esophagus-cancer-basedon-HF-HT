from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config, model_config, spatial_head_parameter_count
from data import point_table_from_cache_arrays
from graph import build_split_graph, gather_neighbors
from model import build_model, count_parameters, gelu_exact
from selection import CheckpointChoice, EarlyStopState, update_checkpoint_choice, update_early_stop
from train import build_optimizer, optimizer_effective_config, prepare_training_state, train_arm


def _cfg(input_dim: int = 4):
    return {
        "data": {"input_dim": input_dim, "output_dim": 2},
        "model": {
            "hidden_dim": 3, "dropout": 0.0, "shared_bias": True, "readout_bias": True,
            "spatial_bias": False, "activation": "gelu_exact", "spatial_initialization": "zeros",
            "normalization_epsilon": 1e-12,
        },
        "graph": {
            "hops": 1, "directed": True, "radius_in_native_steps": 2.0, "max_neighbors": 12,
            "distance_sigma": 0.5, "use_image_similarity": True, "image_temperature": 0.7716396069760084,
            "self_raw_weight": 0.5,
        },
        "training": {
            "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0, "b_lr_multiplier": 1.0,
            "lr_schedule": "constant", "batch_size": 2, "keep_last_batch": True, "max_epochs": 2,
            "precision": "float32", "amp": False, "tf32": False, "betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
        },
        "selection": {
            "formal_start_epoch": 1, "early_stop_count_start_epoch": 41, "early_stop_patience": 20,
            "early_stop_min_delta": 1e-4, "checkpoint_tolerance": 1e-6,
        },
        "randomness": {"model_seed_offset": 0, "center_seed_offset": 100000, "dropout_seed_offset": 200000},
    }


def _table(features=None):
    rng = np.random.default_rng(3)
    n = 8
    feats = rng.normal(size=(n, 4)).astype("float32") if features is None else features
    return point_table_from_cache_arrays(
        features=feats,
        labels_z=rng.normal(size=(n, 2)).astype("float32"),
        patient_ids=["a"] * 4 + ["b"] * 4,
        slide_ids=["s1"] * 4 + ["s2"] * 4,
        spot_ids=[f"p{i}" for i in range(n)],
        splits=["train"] * 4 + ["internal_val"] * 4,
        x=[0, 1, 0, 1, 0, 1, 0, 1],
        y=[0, 0, 1, 1, 0, 0, 1, 1],
        pathway_names=["x", "y"],
    )


def _geometry():
    return pd.DataFrame({
        "patient_id": ["a", "b"], "slide_id": ["s1", "s2"], "s": [1.0, 1.0],
        "coordinate_unit": ["grid", "grid"], "patch_coverage_size": [1.0, 1.0],
        "source": ["test", "test"], "status": ["verified", "verified"],
    })


def test_same_dim_initial_state_dicts_match_across_model_names():
    left = prepare_training_state(_cfg(1536), "spatial", 45, 8, device="cpu")
    right = prepare_training_state(_cfg(1536), "spatial", 45, 8, device="cpu")
    assert left.initial_state_dict.keys() == right.initial_state_dict.keys()
    for key in left.initial_state_dict:
        torch.testing.assert_close(left.initial_state_dict[key], right.initial_state_dict[key], atol=0, rtol=0)
    other_dim = prepare_training_state(_cfg(1024), "spatial", 45, 8, device="cpu")
    assert tuple(other_dim.model.shared.weight.shape) != tuple(left.model.shared.weight.shape)
    frozen = load_config()
    h0 = prepare_training_state(model_config(frozen, "hoptimus0"), "spatial", 45, 8, device="cpu")
    uni2h = prepare_training_state(model_config(frozen, "uni2h"), "spatial", 45, 8, device="cpu")
    h1 = prepare_training_state(model_config(frozen, "hoptimus1"), "spatial", 45, 8, device="cpu")
    phikon = prepare_training_state(model_config(frozen, "phikonv2"), "spatial", 45, 8, device="cpu")
    uni = prepare_training_state(model_config(frozen, "uni"), "spatial", 45, 8, device="cpu")
    for key in h0.initial_state_dict:
        torch.testing.assert_close(h0.initial_state_dict[key], uni2h.initial_state_dict[key], atol=0, rtol=0)
        torch.testing.assert_close(h1.initial_state_dict[key], uni2h.initial_state_dict[key], atol=0, rtol=0)
    for key in phikon.initial_state_dict:
        torch.testing.assert_close(phikon.initial_state_dict[key], uni.initial_state_dict[key], atol=0, rtol=0)
    assert tuple(h0.model.shared.weight.shape) != tuple(phikon.model.shared.weight.shape)


def test_b_is_zero_no_bias_and_gelu_is_exact():
    model = build_model(_cfg(), "spatial")
    assert model.spatial_head.bias is None
    assert torch.count_nonzero(model.spatial_head.weight) == 0
    x = torch.tensor([[0.5, -0.5, 1.0]])
    torch.testing.assert_close(gelu_exact(x), torch.nn.functional.gelu(x, approximate="none"))
    assert spatial_head_parameter_count(1536) == 408862
    assert spatial_head_parameter_count(1024) == 277790
    counted_cfg = {
        "data": {"input_dim": 1536, "output_dim": 30},
        "model": {"hidden_dim": 256, "dropout": 0.3, "shared_bias": True, "readout_bias": True, "spatial_bias": False},
    }
    assert count_parameters(build_model(counted_cfg, "spatial"))["total"] == 408862


def test_hcb_gradients_and_optimizer_excludes_encoder():
    config = _cfg()
    model = build_model(config, "spatial")
    center = torch.randn(5, 4)
    neighbors = torch.randn(5, 2, 4)
    weights = torch.full((5, 2), 0.5)
    target = torch.randn(5, 2)
    loss = (model(center, neighbors, weights)["y_hat"] - target).square().mean()
    loss.backward()
    for parameter in (model.shared.weight, model.point_head.weight, model.spatial_head.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    optimizer = build_optimizer(model, config)
    names = [g["group_name"] for g in optimizer.param_groups]
    assert names == ["H_C", "B"]
    effective = optimizer_effective_config(optimizer)
    assert effective[0]["lr"] == 1e-4 and effective[0]["weight_decay"] == 0.0
    assert effective[1]["lr"] == 1e-4
    assert set(name for name, _ in model.named_parameters()) == {
        "shared.weight", "shared.bias", "point_head.weight", "point_head.bias", "spatial_head.weight",
    }


def test_isolated_point_falls_back_to_center_prediction():
    model = build_model(_cfg(), "spatial")
    center = torch.randn(1, 4)
    out = model(center, None, None)
    torch.testing.assert_close(out["y_hat"], out["y_point"])
    assert torch.count_nonzero(out["delta"]) == 0
    isolated = point_table_from_cache_arrays(
        features=np.ones((1, 4), dtype="float32"),
        labels_z=np.zeros((1, 2), dtype="float32"),
        patient_ids=["a"],
        slide_ids=["s1"],
        spot_ids=["p0"],
        splits=["train"],
        x=[0],
        y=[0],
        pathway_names=["x", "y"],
    )
    graph = build_split_graph(isolated, "train", _cfg(), geometry_table=_geometry())
    assert graph.node(isolated.identities[0]).degree == 0


def test_graph_is_same_slide_and_uses_current_features():
    table = _table()
    graph = build_split_graph(table, "train", _cfg(), geometry_table=_geometry())
    for ident in graph.identities:
        node = graph.node(ident)
        for edge in node.edges:
            assert edge.neighbor.slide_id == ident.slide_id
    other = _table(features=np.zeros((8, 4), dtype=np.float32))
    other.features[0] = 1
    other.features[1] = -1
    graph_a = build_split_graph(table, "train", _cfg(), geometry_table=_geometry())
    graph_b = build_split_graph(other, "train", _cfg(), geometry_table=_geometry())
    edge_a = graph_a.node(table.identities[0]).edges[0].cosine
    edge_b = graph_b.node(other.identities[0]).edges[0].cosine
    assert edge_a != edge_b


def test_selection_windows_and_synthetic_training(tmp_path: Path):
    search = EarlyStopState()
    for epoch in range(1, 41):
        search = update_early_stop(search, epoch=epoch, score=0.1, formal_start_epoch=1, count_start_epoch=41, patience=20)
    assert search.count == 0 and not search.stopped
    search = update_early_stop(search, epoch=41, score=0.1, formal_start_epoch=1, count_start_epoch=41, patience=20)
    assert search.count == 1
    choice = CheckpointChoice(kind="formal")
    choice = update_checkpoint_choice(choice, epoch=1, score=0.3, mse=0.5, lambda_value=0.0, kind="formal")
    tied = update_checkpoint_choice(choice, epoch=2, score=0.3 + 1e-7, mse=0.4, lambda_value=0.0, tolerance=1e-6, kind="formal")
    assert tied.epoch == 2
    result = train_arm(_cfg(), "spatial", 45, tmp_path / "run", checkpoint_dir=tmp_path / "weights", point_table=_table(), device="cpu", geometry_table=_geometry())
    assert result["status"] == "completed"
    assert (tmp_path / "run" / "raw" / "optimizer_effective.json").is_file()
    checkpoint = torch.load(tmp_path / "weights" / "formal_best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["kind"] == "formal" and checkpoint["selection"]["kind"] == "formal"


def test_package_config_frozen_spatial11_values():
    cfg = load_config()
    assert cfg["training"]["optimizer"] == "Adam"
    assert cfg["training"]["learning_rate"] == 1e-4
    assert cfg["selection"]["early_stop_count_start_epoch"] == 41
    hopt = model_config(cfg, "hoptimus0")
    phikon = model_config(cfg, "phikonv2")
    assert hopt["data"]["input_dim"] == 1536
    assert phikon["data"]["input_dim"] == 1024
    assert hopt["normalization_profile"] == "hoptimus_rgb_v1"
    assert phikon["normalization_profile"] == "imagenet_rgb_v1"
    assert "search" not in cfg
