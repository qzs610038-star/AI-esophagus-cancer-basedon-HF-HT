from __future__ import annotations

import numpy as np
import pytest

from errors import SlideMappingMissingError
from graph import build_split_graph, gather_neighbors
from helpers import package_config, tiny_model_config, verified_points
from data import load_split_point_table


def test_graph_same_slide_and_split_only():
    features = np.eye(5, 8, dtype=np.float32)
    table, geo = verified_points(
        patients=["P", "P", "P", "P", "P"],
        slides=["S1", "S1", "S2", "S1", "S1"],
        spots=["c", "n", "other_slide", "val", "far"],
        splits=["train", "train", "train", "internal_val", "train"],
        x=[0.0, 1.0, 0.2, 0.2, 10.0],
        y=[0.0, 0.0, 0.0, 0.0, 10.0],
        features=features,
        s_by_slide={"S1": 1.0, "S2": 1.0},
    )
    cfg = tiny_model_config()
    cfg["graph"]["max_neighbors"] = 8
    cfg["graph"]["radius_in_native_steps"] = 1.5
    graph = build_split_graph(table, "train", cfg, geometry_table=geo)
    center = next(ident for ident in table.identities if ident.spot_id == "c")
    names = [edge.neighbor.spot_id for edge in graph.node(center).edges]
    assert names == ["n"]
    assert "other_slide" not in names
    assert "val" not in names
    assert "far" not in names


def test_graph_tie_break_by_identity():
    features = np.ones((3, 4), dtype=np.float32)
    table, geo = verified_points(
        patients=["P", "P", "P"],
        slides=["S", "S", "S"],
        spots=["center", "b_spot", "a_spot"],
        splits=["train", "train", "train"],
        x=[0.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0],
        features=features,
        s_by_slide={"S": 1.0},
    )
    cfg = tiny_model_config()
    cfg["graph"]["max_neighbors"] = 1
    cfg["graph"]["radius_in_native_steps"] = 1.5
    graph = build_split_graph(table, "train", cfg, geometry_table=geo)
    center = next(ident for ident in table.identities if ident.spot_id == "center")
    chosen = graph.node(center).edges[0].neighbor.spot_id
    assert chosen == "a_spot"


def test_empty_neighborhood_u_and_a_self():
    features = np.ones((1, 4), dtype=np.float32)
    table, geo = verified_points(
        patients=["P"],
        slides=["S"],
        spots=["only"],
        splits=["train"],
        x=[0.0],
        y=[0.0],
        features=features,
        s_by_slide={"S": 1.0},
    )
    graph = build_split_graph(table, "train", tiny_model_config(), geometry_table=geo)
    node = graph.node(table.identities[0])
    assert node.degree == 0
    assert node.a_self == pytest.approx(1.0)
    batch = gather_neighbors(graph, [table.identities[0]])
    assert batch.k_max == 0
    assert batch.neighbor_weights.shape == (1, 0)


def test_batch_gather_is_order_invariant():
    rng = np.random.default_rng(0)
    n = 5
    features = rng.normal(size=(n, 6)).astype(np.float32)
    table, geo = verified_points(
        patients=["P"] * n,
        slides=["S"] * n,
        spots=[f"s{i}" for i in range(n)],
        splits=["train"] * n,
        x=list(range(n)),
        y=[0.0] * n,
        features=features,
        s_by_slide={"S": 1.0},
    )
    cfg = tiny_model_config()
    cfg["graph"]["radius_in_native_steps"] = 2.0
    graph = build_split_graph(table, "train", cfg, geometry_table=geo)
    idents = list(table.identities)
    a = gather_neighbors(graph, idents)
    b = gather_neighbors(graph, list(reversed(idents)))
    for i, ident in enumerate(idents):
        j = len(idents) - 1 - i
        np.testing.assert_allclose(a.neighbor_weights[i], b.neighbor_weights[j])
        np.testing.assert_array_equal(a.neighbor_index[i], b.neighbor_index[j])
        assert a.a_self[i] == pytest.approx(b.a_self[j])


def test_use_image_similarity_false_drops_morphology():
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    table, geo = verified_points(
        patients=["P", "P", "P"],
        slides=["S", "S", "S"],
        spots=["c", "same", "diff"],
        splits=["train"] * 3,
        x=[0.0, 1.0, 1.0],
        y=[0.0, 0.0, 0.0],
        features=features,
        s_by_slide={"S": 1.0},
    )
    cfg = tiny_model_config()
    cfg["graph"]["use_image_similarity"] = False
    graph = build_split_graph(table, "train", cfg, geometry_table=geo)
    center = next(ident for ident in table.identities if ident.spot_id == "c")
    b_values = [edge.b for edge in graph.node(center).edges]
    assert b_values[0] == pytest.approx(b_values[1])


def test_missing_mapping_blocks_real_split_graph():
    config = package_config()
    table = load_split_point_table(config)
    with pytest.raises(SlideMappingMissingError, match="SLIDE_MAPPING_UNVERIFIED"):
        build_split_graph(table, "train", config)
