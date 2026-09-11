from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.metrics import (
    SelectionCandidate,
    compute_regression_metrics,
    select_best_candidate,
)
from src.model import SpatialWarmstartModel, load_stage1_hc
from src.spatial import SpatialPoint, build_spatial_graph


def test_model_has_exact_h_c_b_contract_and_zero_b_preserves_point_prediction() -> None:
    model = SpatialWarmstartModel(input_dim=1536, hidden_dim=256, output_dim=30, dropout=0.3)

    assert sum(parameter.numel() for parameter in model.shared.parameters()) == 393_472
    assert sum(parameter.numel() for parameter in model.regression.parameters()) == 7_710
    assert model.B.shape == (30, 256)
    assert model.B.numel() == 7_680
    assert torch.count_nonzero(model.B).item() == 0

    model.eval()
    features = torch.randn(4, 1536)
    spatial_delta = torch.randn(4, 256)
    point = model(features, use_spatial=False)
    spatial = model(features, spatial_delta=spatial_delta, use_spatial=True)

    torch.testing.assert_close(spatial.prediction, point.prediction, rtol=0, atol=0)
    torch.testing.assert_close(spatial.correction, torch.zeros_like(point.prediction))


def test_model_adds_b_times_spatial_delta_to_point_prediction() -> None:
    model = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2, dropout=0.0)
    with torch.no_grad():
        model.B.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))
    model.eval()

    point = model(torch.tensor([[1.0, 2.0, 3.0]]), use_spatial=False)
    spatial = model(
        torch.tensor([[1.0, 2.0, 3.0]]),
        spatial_delta=torch.tensor([[3.0, 4.0]]),
        use_spatial=True,
    )

    torch.testing.assert_close(spatial.correction, torch.tensor([[6.0, -4.0]]))
    torch.testing.assert_close(spatial.prediction, point.prediction + torch.tensor([[6.0, -4.0]]))


def test_explicit_dropout_generator_or_mask_supports_paired_random_streams() -> None:
    model = SpatialWarmstartModel(input_dim=3, hidden_dim=4, output_dim=2, dropout=0.5)
    model.train()
    features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first = model(features, use_spatial=False, dropout_generator=first_generator)
    second = model(features, use_spatial=False, dropout_generator=second_generator)
    torch.testing.assert_close(first.point_prediction, second.point_prediction, rtol=0, atol=0)

    keep_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]], dtype=torch.bool
    )
    masked = model(features, use_spatial=False, dropout_mask=keep_mask)
    expected = model.regression.linear(masked.hidden * keep_mask * 2.0)
    torch.testing.assert_close(masked.point_prediction, expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="either dropout_generator or dropout_mask"):
        model(
            features,
            use_spatial=False,
            dropout_generator=torch.Generator(),
            dropout_mask=keep_mask,
        )


def test_stage1_loader_strictly_loads_h_c_and_only_allows_teacher_keys() -> None:
    model = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2, dropout=0.0)
    source = {
        "shared.linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "shared.linear.bias": torch.tensor([1.0, 2.0]),
        "regression.linear.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "regression.linear.bias": torch.tensor([3.0, 4.0]),
        "teacher.input.weight": torch.ones(2, 2),
    }

    report = load_stage1_hc(model, source)

    assert report.loaded_keys == (
        "shared.linear.weight",
        "shared.linear.bias",
        "regression.linear.weight",
        "regression.linear.bias",
    )
    assert report.ignored_keys == ("teacher.input.weight",)
    torch.testing.assert_close(model.shared.linear.weight, source["shared.linear.weight"])
    torch.testing.assert_close(model.regression.linear.bias, source["regression.linear.bias"])
    assert torch.count_nonzero(model.B).item() == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.pop("regression.linear.bias"), "missing"),
        (lambda state: state.__setitem__("other.weight", torch.ones(1)), "unexpected"),
        (
            lambda state: state.__setitem__("shared.linear.weight", torch.ones(9, 9)),
            "shape",
        ),
    ],
)
def test_stage1_loader_rejects_incomplete_unexpected_or_wrong_shape_state(
    mutation, message: str
) -> None:
    model = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2, dropout=0.0)
    source = {
        "shared.linear.weight": torch.ones(2, 3),
        "shared.linear.bias": torch.ones(2),
        "regression.linear.weight": torch.ones(2, 2),
        "regression.linear.bias": torch.ones(2),
    }
    mutation(source)

    with pytest.raises((KeyError, ValueError), match=message):
        load_stage1_hc(model, source)


def test_spatial_graph_stays_within_patient_and_partition_and_exports_edges() -> None:
    points = [
        SpatialPoint("a", "P1", "train", 0.0, 0.0),
        SpatialPoint("b", "P1", "train", 1.0, 0.0),
        SpatialPoint("c", "P1", "internal_val", 0.0, 0.0),
        SpatialPoint("d", "P2", "train", 0.0, 0.0),
    ]
    graph = build_spatial_graph(
        points,
        np.ones((4, 2), dtype=np.float32),
        native_steps={"P1": 1.0, "P2": 1.0},
    )

    assert graph.degree.tolist() == [1, 1, 0, 0]
    np.testing.assert_allclose(
        graph.self_weight + graph.neighbor_weight.sum(axis=1), np.ones(4), atol=1e-7
    )
    expected_neighbor_weight = math.exp(-0.5) / (1.0 + math.exp(-0.5))
    assert graph.neighbor_weight[0, 0] == pytest.approx(expected_neighbor_weight)
    assert graph.self_weight[2] == pytest.approx(1.0)

    edge_table = graph.edge_table()
    assert edge_table["source_id"].tolist() == ["a", "b"]
    assert edge_table["target_id"].tolist() == ["b", "a"]
    assert edge_table["patient_id"].tolist() == ["P1", "P1"]
    assert edge_table["partition"].tolist() == ["train", "train"]

    delta = graph.spatial_delta(torch.tensor([[1.0], [3.0], [10.0], [20.0]]))
    torch.testing.assert_close(
        delta,
        torch.tensor(
            [
                [2.0 * expected_neighbor_weight],
                [-2.0 * expected_neighbor_weight],
                [0.0],
                [0.0],
            ]
        ),
    )


def test_metrics_report_both_pcc_definitions_invalid_count_and_zmse() -> None:
    prediction = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 5.0], [2.0, 5.0]], dtype=np.float64
    )
    target = np.array(
        [[0.0, 0.0], [2.0, 1.0], [2.0, 1.0], [0.0, 2.0]], dtype=np.float64
    )
    metrics = compute_regression_metrics(prediction, target, ["A", "A", "B", "B"])

    assert metrics["patient_macro_pathway_pcc"] == pytest.approx(-1.0 / 3.0)
    assert metrics["valid_patient_pathway_count"] == 3
    assert metrics["invalid_patient_pathway_count"] == 1
    assert metrics["flattened_pooled_pcc"] == pytest.approx(
        np.corrcoef(prediction.ravel(), target.ravel())[0, 1]
    )
    assert metrics["pooled_pcc"] == metrics["flattened_pooled_pcc"]
    assert metrics["zMSE"] == pytest.approx(np.mean((prediction - target) ** 2))


def test_candidate_selection_uses_pcc_then_zmse_then_earlier_update() -> None:
    candidates = [
        SelectionCandidate(update=20, patient_macro_pathway_pcc=0.5, zMSE=0.3),
        SelectionCandidate(update=10, patient_macro_pathway_pcc=0.5, zMSE=0.2),
        SelectionCandidate(update=0, patient_macro_pathway_pcc=0.5, zMSE=0.2),
        SelectionCandidate(update=30, patient_macro_pathway_pcc=0.49, zMSE=0.1),
    ]

    assert select_best_candidate(candidates).update == 0


def test_architecture_tie_uses_declared_complexity_order() -> None:
    candidates = [
        SelectionCandidate(10, 0.5, 0.2, arm="spatial_joint"),
        SelectionCandidate(10, 0.5, 0.2, arm="spatial_residual_only"),
        SelectionCandidate(10, 0.5, 0.2, arm="point_continue"),
    ]

    assert select_best_candidate(candidates, architecture_tiebreak=True).arm == "point_continue"
