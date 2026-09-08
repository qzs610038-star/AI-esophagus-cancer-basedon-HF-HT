from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import make_point_table
from helpers import package_config, tiny_model_config
from train import (
    build_arm_from_template,
    build_full_template,
    epoch_center_order,
    make_dropout_generator,
    regression_mse,
    train_arm,
)


def test_training_random_streams_are_paired_without_sharing_state():
    config = tiny_model_config(hidden=17, relation=11, input_dim=8, output_dim=5)
    first = [epoch_center_order(7, seed=42, epoch=e, config=config) for e in (1, 2)]
    second = [epoch_center_order(7, seed=42, epoch=e, config=config) for e in (1, 2)]
    assert [row.tolist() for row in first] == [row.tolist() for row in second]
    assert all(sorted(row.tolist()) == list(range(7)) for row in first)

    mask_a = torch.rand((4, 5), generator=make_dropout_generator(config, 42))
    mask_b = torch.rand((4, 5), generator=make_dropout_generator(config, 42))
    assert torch.equal(mask_a, mask_b)


def test_mse_contract_reports_unambiguous_sum_and_denominator():
    prediction = torch.tensor([[1.0, 2.0]])
    target = torch.zeros_like(prediction)
    loss, numerator, denominator = regression_mse(prediction, target)
    assert loss.item() == 2.5
    assert numerator.item() == 5.0
    assert denominator.item() == 2


def test_paired_models_share_common_initialization_and_zero_b():
    config = tiny_model_config(hidden=17, relation=11, input_dim=8, output_dim=5)
    template = build_full_template(config, seed=42)
    point = build_arm_from_template(config, "point", template)
    joint = build_arm_from_template(config, "joint", template)
    np.testing.assert_allclose(
        point.shared.weight.detach().numpy(), joint.shared.weight.detach().numpy()
    )
    np.testing.assert_allclose(
        point.point_head.weight.detach().numpy(), joint.point_head.weight.detach().numpy()
    )
    assert torch.count_nonzero(joint.spatial_head.weight).item() == 0


def test_tiny_point_training_writes_formal_endpoint_and_epoch_predictions(tmp_path):
    config = package_config()
    config["data"].update(input_dim=4, output_dim=2)
    config["model"].update(hidden_dim=5, relation_dim=3, dropout=0.0)
    config["training"].update(
        batch_size=2,
        max_epochs=2,
        relation_warmup_epochs=0,
        relation_ramp_epochs=1,
        cpu_threads=1,
    )
    config["selection"].update(
        formal_start_epoch=1,
        early_stop_count_start_epoch=3,
        early_stop_patience=2,
    )
    config["diagnostics"]["training_epochs"] = [1]
    config["analysis"]["same_epoch_comparisons"] = [1, 2]
    rng = np.random.default_rng(7)
    table = make_point_table(
        patient_ids=["p1"] * 4 + ["p2"] * 4,
        slide_ids=["s1"] * 4 + ["s2"] * 4,
        spot_ids=[f"spot{i}" for i in range(8)],
        splits=["train"] * 4 + ["internal_val"] * 4,
        x=np.arange(8),
        y=np.arange(8),
        features=rng.normal(size=(8, 4)).astype(np.float32),
        labels_z=np.array(
            [[0, 0], [1, 2], [2, 1], [3, 3], [0, 1], [1, 0], [2, 3], [3, 2]],
            dtype=np.float32,
        ),
        slide_status="verified",
        pathway_names=["a", "b"],
        sort_identities=False,
    )
    weight_dir = tmp_path / "separate_weights"
    result = train_arm(
        config,
        "point",
        42,
        tmp_path,
        checkpoint_dir=weight_dir,
        point_table=table,
        device="cpu",
    )
    assert result["status"] == "completed"
    assert (weight_dir / "formal_best.pt").is_file()
    assert Path(result["formal_checkpoint"]) == (weight_dir / "formal_best.pt").resolve()
    assert not (tmp_path / "checkpoints").exists()
    assert (tmp_path / "raw" / "internal_best.npz").is_file()
    assert (tmp_path / "raw" / "epoch_predictions" / "epoch_1.npz").is_file()
    assert (tmp_path / "raw" / "epoch_predictions" / "epoch_2.npz").is_file()
