from __future__ import annotations

import inspect

import numpy as np
import torch

from model import (
    PointRegressor,
    build_paired_regressor,
    make_dropout_generator,
    next_dropout_mask,
    parameter_count,
    predict,
)
from training import make_center_rng, next_center_order, regression_mse


def test_three_native_dimensions_share_only_the_registered_output_initialization():
    output_state = {
        "weight": torch.linspace(-0.2, 0.2, 30 * 256).reshape(30, 256),
        "bias": torch.linspace(-0.1, 0.1, 30),
    }
    expected_counts = {1536: 401_182, 1024: 270_110, 1280: 335_646}

    models = {
        dim: build_paired_regressor(dim, seed=42, output_state=output_state)
        for dim in expected_counts
    }

    for dim, model in models.items():
        assert model.input_dim == dim
        assert model.hidden_dim == 256
        assert model.output_dim == 30
        assert parameter_count(model) == expected_counts[dim]
        torch.testing.assert_close(model.readout.weight, output_state["weight"])
        torch.testing.assert_close(model.readout.bias, output_state["bias"])

    assert models[1536].projection.weight.shape != models[1024].projection.weight.shape


def test_prediction_interface_has_no_label_or_external_selection_input():
    forbidden = {"label", "labels", "target", "targets", "labels_z", "pathway", "external"}
    assert not (set(inspect.signature(PointRegressor.forward).parameters) & forbidden)
    assert not (set(inspect.signature(predict).parameters) & forbidden)

    model = PointRegressor(8, hidden_dim=5, output_dim=3, dropout_p=0.3)
    result = predict(model, torch.ones(4, 8))
    assert result.shape == (4, 3)


def test_center_order_and_dropout_are_reproducible_separate_streams():
    order_a = next_center_order(make_center_rng(42), 11)
    order_b = next_center_order(make_center_rng(42), 11)
    assert order_a.tolist() == order_b.tolist()
    assert sorted(order_a.tolist()) == list(range(11))

    mask_a = next_dropout_mask(make_dropout_generator(42), 4, 7, 0.3)
    mask_b = next_dropout_mask(make_dropout_generator(42), 4, 7, 0.3)
    torch.testing.assert_close(mask_a, mask_b, rtol=0, atol=0)
    unique = mask_a.unique()
    assert bool(torch.all((unique == 0) | torch.isclose(unique, torch.tensor(1.0 / 0.7))))


def test_seed42_center_stream_matches_registered_historical_prefixes():
    rng = make_center_rng(42)
    epoch_1 = next_center_order(rng, 9472)
    epoch_2 = next_center_order(rng, 9472)
    assert epoch_1[:16].tolist() == [
        5486, 2988, 6860, 3502, 757, 3197, 9165, 8877,
        2420, 268, 5698, 7345, 6438, 4308, 8762, 6583,
    ]
    assert epoch_2[:16].tolist() == [
        2332, 6334, 5394, 2805, 3264, 9381, 5274, 917,
        1729, 5865, 2664, 3174, 1731, 4183, 2120, 5721,
    ]


def test_mse_contract_is_mean_over_points_and_pathways():
    prediction = torch.tensor([[1.0, 2.0]])
    target = torch.zeros_like(prediction)
    loss, numerator, denominator = regression_mse(prediction, target)
    assert loss.item() == 2.5
    assert numerator.item() == 5.0
    assert denominator.item() == 2.0


def test_explicit_mask_matches_the_point_head_equation():
    torch.manual_seed(9)
    model = PointRegressor(4, hidden_dim=3, output_dim=2, dropout_p=0.3)
    features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[0.0, 1.0 / 0.7, 1.0 / 0.7]])

    actual = model(features, dropout_mask=mask)
    hidden = torch.nn.functional.gelu(model.projection(features), approximate="none")
    expected = model.readout(hidden * mask)
    torch.testing.assert_close(actual, expected)
