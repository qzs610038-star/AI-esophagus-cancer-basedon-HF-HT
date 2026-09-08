from __future__ import annotations

import inspect

import numpy as np
import torch

from helpers import tiny_model_config, verified_points
from graph import build_split_graph, gather_neighbors
from model import build_model, count_parameters, forward_parameter_names, predict


def test_b_zero_init_and_empty_or_nonzero_neighbors():
    cfg = tiny_model_config(hidden=17, relation=11, input_dim=8, output_dim=5)
    model = build_model(cfg, "spatial")
    assert model.hidden_dim == 17
    assert model.relation_dim == 11
    assert model.spatial_head is not None
    assert float(model.spatial_head.weight.detach().abs().sum()) == 0.0
    x = torch.randn(3, 8)
    neighbors = torch.randn(3, 2, 8)
    weights = torch.ones(3, 2) / 3.0
    mask = torch.ones(3, 2)
    out = model(x, neighbor_features=neighbors, neighbor_weights=weights, neighbor_mask=mask)
    assert torch.allclose(out["delta"], torch.zeros_like(out["delta"]))
    assert torch.allclose(out["y_hat"], out["y_point"])
    assert out["z"] is None
    isolated = model(x, neighbor_features=torch.zeros(3, 0, 8), neighbor_weights=torch.zeros(3, 0))
    assert torch.allclose(isolated["u"], torch.zeros(3, 17))


def test_non256_shapes_and_four_arms():
    cfg = tiny_model_config(hidden=17, relation=11, input_dim=8, output_dim=5)
    x = torch.randn(2, 8)
    for arm in ("point", "relation", "spatial", "joint"):
        model = build_model(cfg, arm)
        out = model(x)
        assert out["y_hat"].shape == (2, 5)
        assert out["h_center"].shape == (2, 17)
        if arm in ("relation", "joint"):
            assert out["z"].shape == (2, 11)
            assert torch.allclose(out["z"].norm(dim=-1), torch.ones(2), atol=1e-5)
        else:
            assert out["z"] is None
        if arm in ("spatial", "joint"):
            assert model.spatial_head is not None
        else:
            assert model.spatial_head is None


def test_predict_interface_has_no_labels():
    names = set(forward_parameter_names())
    forbidden = {"label", "labels", "y", "target", "targets", "labels_z", "pathway"}
    assert not (names & forbidden)
    sig = inspect.signature(predict)
    assert not ({p for p in sig.parameters} & forbidden)
    cfg = tiny_model_config()
    model = build_model(cfg, "point")
    y_hat = predict(model, torch.randn(4, 8))
    assert y_hat.shape == (4, 5)


def test_eval_prediction_matches_after_reordering_neighbors():
    cfg = tiny_model_config(hidden=17, relation=11, input_dim=4, output_dim=3)
    cfg["graph"]["radius_in_native_steps"] = 2.0
    rng = np.random.default_rng(1)
    features = rng.normal(size=(4, 4)).astype(np.float32)
    table, geo = verified_points(
        patients=["P"] * 4,
        slides=["S"] * 4,
        spots=[f"p{i}" for i in range(4)],
        splits=["train"] * 4,
        x=[0.0, 1.0, 2.0, 3.0],
        y=[0.0, 0.0, 0.0, 0.0],
        features=features,
        s_by_slide={"S": 1.0},
    )
    graph = build_split_graph(table, "train", cfg, geometry_table=geo)
    model = build_model(cfg, "spatial")
    model.eval()
    idents = list(table.identities)
    feat = torch.as_tensor(table.features, dtype=torch.float32)

    def run(order):
        batch = gather_neighbors(graph, [idents[i] for i in order])
        neighbors = torch.as_tensor(batch.neighbor_features(table), dtype=torch.float32)
        weights = torch.as_tensor(batch.neighbor_weights, dtype=torch.float32)
        mask = torch.as_tensor(batch.neighbor_mask)
        centers = feat[torch.as_tensor(batch.center_index)]
        return predict(model, centers, neighbors, weights, mask)

    a = run([0, 1, 2, 3])
    b = run([3, 2, 1, 0])
    torch.testing.assert_close(a, torch.flip(b, dims=[0]), rtol=1e-5, atol=1e-5)


def test_dropout_mask_only_affects_point_head():
    cfg = tiny_model_config(hidden=8, relation=4, input_dim=6, output_dim=3, dropout=0.5)
    model = build_model(cfg, "joint")
    x = torch.randn(2, 6)
    mask = torch.zeros(2, 8)
    mask[:, :4] = 2.0
    out_masked = model(x, center_dropout_mask=mask)
    out_plain = model(x)
    assert not torch.allclose(out_masked["y_point"], out_plain["y_point"])
    assert torch.allclose(out_masked["h_center"], out_plain["h_center"])
    assert torch.allclose(out_masked["z"], out_plain["z"])


def test_parameter_count_matches_spec_for_256():
    cfg = tiny_model_config(hidden=256, relation=256, input_dim=1536, output_dim=30)
    point = count_parameters(build_model(cfg, "point"))["total"]
    relation = count_parameters(build_model(cfg, "relation"))["total"]
    spatial = count_parameters(build_model(cfg, "spatial"))["total"]
    joint = count_parameters(build_model(cfg, "joint"))["total"]
    assert point == 401182
    assert relation == 466718
    assert spatial == 408862
    assert joint == 474398
