import math

import pytest
import torch
import torch.nn as nn

from train_mpp_uni2h_mlp import build_loss_criterion, resolve_loss_metadata


def test_mse_default_is_cpu_compatible_and_matches_historical_mean_reduction():
    predictions = torch.tensor([[0.0, 3.0], [2.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float32)

    criterion = build_loss_criterion("mse", 1.0)

    assert isinstance(criterion, nn.MSELoss)
    assert criterion(predictions, targets).item() == pytest.approx(
        nn.MSELoss()(predictions, targets).item()
    )
    assert resolve_loss_metadata("mse", 1.0) == {
        "loss_type": "mse",
        "delta": None,
        "reduction": "mean",
    }


def test_huber_delta_one_is_cpu_compatible_unweighted_mean_reduction():
    predictions = torch.tensor([[0.0, 3.0], [2.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float32)

    criterion = build_loss_criterion("huber", 1.0)

    assert isinstance(criterion, nn.HuberLoss)
    assert criterion.reduction == "mean"
    assert criterion.delta == pytest.approx(1.0)
    assert criterion(predictions, targets).item() == pytest.approx(1.5)
    assert resolve_loss_metadata("huber", 1.0) == {
        "loss_type": "huber",
        "delta": 1.0,
        "reduction": "mean",
    }


@pytest.mark.parametrize("delta", [0.0, -0.1, math.inf, math.nan])
def test_huber_parameter_validation_rejects_nonpositive_or_nonfinite_delta(delta):
    with pytest.raises(ValueError, match="finite positive"):
        build_loss_criterion("huber", delta)


def test_loss_parameter_validation_rejects_unknown_loss_type():
    with pytest.raises(ValueError, match="loss_type"):
        resolve_loss_metadata("weighted_huber", 1.0)
