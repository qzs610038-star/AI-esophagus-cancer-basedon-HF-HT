from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import torch

from config import load_config
from data import IdentityRecord
from model import PointRegressor, make_dropout_generator, next_dropout_mask
from predict import predict_external_arrays
from training import build_optimizer, regression_mse, train_point_arrays


def test_optimizer_contract_and_one_update_match_direct_adamw():
    torch.manual_seed(3)
    first = PointRegressor(4, hidden_dim=3, output_dim=2, dropout_p=0.3)
    second = copy.deepcopy(first)
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    target = torch.ones(2, 2)
    mask = next_dropout_mask(make_dropout_generator(42), 2, 3, 0.3)
    config = {"training": {"optimizer": "AdamW", "precision": "float32", "amp": False, "tf32": False, "learning_rate": 3e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999], "optimizer_epsilon": 1e-8}}

    actual_optimizer = build_optimizer(first, config)
    expected_optimizer = torch.optim.AdamW(
        second.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
    )
    actual_loss, _, _ = regression_mse(first(features, mask), target)
    expected_loss = torch.mean((second(features, mask) - target) ** 2)
    actual_loss.backward()
    expected_loss.backward()
    actual_optimizer.step()
    expected_optimizer.step()
    for actual, expected in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _rows(split: str) -> list[IdentityRecord]:
    return [
        IdentityRecord(i, patient, f"MPP2_{patient}_source01", f"patch_x{i}_y{i}", split, i, i, "unused.png", 224, 224)
        for i, patient in enumerate(("A", "A", "B", "B"))
    ]


def test_tiny_training_writes_formal_checkpoint_outside_run_tree(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")
    # Shorten only this synthetic unit; production entry validates shipped values.
    config = copy.deepcopy(config)
    config["training"]["batch_size"] = 2
    config["training"]["max_epochs"] = 6
    config["training"]["cpu_threads"] = 1
    features = np.arange(4 * 1024, dtype=np.float32).reshape(4, 1024) / 4096
    labels = np.stack(
        [np.linspace(0, 1, 30), np.linspace(1, 2, 30), np.linspace(0, 2, 30), np.linspace(2, 4, 30)]
    ).astype(np.float32)
    output_state = {
        "weight": torch.zeros(30, 256),
        "bias": torch.linspace(-0.1, 0.1, 30),
    }
    run_dir = tmp_path / "runs" / "train_42_uni"
    checkpoint_dir = tmp_path / "weights" / "uni_seed42"
    result = train_point_arrays(
        config,
        model_name="uni",
        seed=42,
        train_features=features,
        train_targets=labels,
        train_rows=_rows("train"),
        val_features=features,
        val_targets=labels,
        val_rows=_rows("internal_val"),
        pathway_names=[f"p{i}" for i in range(30)],
        output_state=output_state,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )
    assert result["status"] == "completed"
    assert result["started_at"].endswith("Z")
    assert result["ended_at"].endswith("Z")
    assert result["exit_code"] == 0
    assert result["formal_endpoint"]["epoch"] == 6
    assert (checkpoint_dir / "formal_best.pt").is_file()
    assert (checkpoint_dir / "last.pt").is_file()
    assert not (run_dir / "checkpoints").exists()
    with np.load(run_dir / "raw" / "internal_best.npz", allow_pickle=False) as archive:
        assert archive["pred_z"].shape == (4, 30)
        assert int(archive["seed"]) == 42


def test_external_prediction_api_cannot_accept_labels_or_selection_scores():
    forbidden = {"label", "labels", "target", "targets", "score", "metric"}
    assert not (set(inspect.signature(predict_external_arrays).parameters) & forbidden)
