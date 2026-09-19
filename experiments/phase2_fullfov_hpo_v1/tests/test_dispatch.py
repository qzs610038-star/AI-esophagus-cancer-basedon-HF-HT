"""Small integration checks for the explicit batch coordinator."""

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import stage_runtime
import train
from config import load_config
from dispatch import FULL_PROTOCOL, LEGACY_PROTOCOL, TaskSpec, plan_final_tasks, plan_paired_tasks, planned_external_keys
from model import build_model
from selection import CheckpointChoice, EarlyStopState, update_checkpoint_choice, update_early_stop
from search import build_point_search_plan, record_trial_result
from train import build_optimizer
from transforms import build_transform


def test_four_edge_markers_are_visible_only_in_full_view():
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[:9, :, 0] = 255
    pixels[-9:, :, 1] = 255
    pixels[:, :9, 2] = 255
    pixels[:, -9:, :2] = 255
    image = Image.fromarray(pixels, "RGB")
    full = np.asarray(build_transform(FULL_PROTOCOL).prepare_image(image))
    legacy = np.asarray(build_transform(LEGACY_PROTOCOL).prepare_image(image))
    probes = ((3, 112), (220, 112), (112, 3), (112, 220))
    assert all(full[y, x].max() > 180 for y, x in probes)
    assert all(legacy[y, x].max() < 50 for y, x in probes)


def test_h_c_b_receive_gradients_on_spatial_batch():
    config = {"data": {"input_dim": 4, "output_dim": 2},
              "model": {"hidden_dim": 3, "dropout": 0.0, "shared_bias": True,
                        "readout_bias": True, "spatial_bias": False}}
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
    optimizer_config = {"training": {"optimizer": "AdamW", "learning_rate": .001,
                                     "weight_decay": .0001, "b_lr_multiplier": 3.0}}
    assert [g["group_name"] for g in build_optimizer(model, optimizer_config).param_groups] == ["H_C", "B"]
    for arm in ("point", "no_b"):
        control = build_model(config, arm)
        assert [g["group_name"] for g in build_optimizer(control, optimizer_config).param_groups] == ["H_C"]
        assert not any(p.requires_grad for p in control.spatial_head.parameters())


def test_paired_and_search_selection_windows():
    paired = EarlyStopState()
    for epoch in range(1, 16):
        paired = update_early_stop(paired, epoch=epoch, score=0.2,
                                   formal_start_epoch=6, count_start_epoch=16,
                                   patience=10)
    assert paired.count == 0 and not paired.stopped
    paired = update_early_stop(paired, epoch=16, score=0.2,
                               formal_start_epoch=6, count_start_epoch=16, patience=10)
    assert paired.count == 1
    search = EarlyStopState()
    for epoch in range(1, 41):
        search = update_early_stop(search, epoch=epoch, score=0.1,
                                   formal_start_epoch=1, count_start_epoch=41,
                                   patience=20)
    assert search.count == 0
    search = update_early_stop(search, epoch=41, score=0.1,
                               formal_start_epoch=1, count_start_epoch=41, patience=20)
    assert search.count == 1
    choice = CheckpointChoice(kind="formal")
    choice = update_checkpoint_choice(choice, epoch=6, score=.3, mse=.5,
                                      lambda_value=0., kind="formal")
    tied = update_checkpoint_choice(choice, epoch=7, score=.3 + 1e-7, mse=.4,
                                    lambda_value=0., tolerance=1e-6, kind="formal")
    assert tied.epoch == 7


def test_retry_creates_new_attempt_and_preserves_failure(tmp_path, monkeypatch):
    config = load_config()
    task = TaskSpec("paired-view", "uni2h", LEGACY_PROTOCOL, "point", 42, "historical")
    monkeypatch.setattr(stage_runtime, "_development_table", lambda *args, **kwargs: (object(), "fake-cache", []))
    calls = []
    def fake_train(cfg, arm, seed, run_dir, *, checkpoint_dir, point_table, device):
        calls.append(Path(run_dir))
        Path(checkpoint_dir).mkdir(parents=True)
        if len(calls) == 1:
            raise RuntimeError("synthetic failure")
        return {"status": "completed", "formal_endpoint": {"score": .2, "mse": .3},
                "formal_checkpoint": str(Path(checkpoint_dir) / "formal_best.pt"),
                "last_checkpoint": str(Path(checkpoint_dir) / "last.pt")}
    monkeypatch.setattr(train, "train_arm", fake_train)
    run = tmp_path / "batch" / "paired-view_run"
    weights = tmp_path / "weights"
    first = stage_runtime._run_task(config, task, run_dir=run, weights_dir=weights,
                                    device="cpu", table_cache={})
    second = stage_runtime._run_task(config, task, run_dir=run, weights_dir=weights,
                                     device="cpu", table_cache={})
    third = stage_runtime._run_task(config, task, run_dir=run, weights_dir=weights,
                                    device="cpu", table_cache={})
    assert [first["status"], second["status"], third["status"]] == ["failed", "completed", "reused_complete"]
    assert calls[0] != calls[1] and calls[0].is_dir() and calls[1].is_dir()
    record = stage_runtime._read(stage_runtime._task_path(run.parent, task.task_id))
    assert [a["status"] for a in record["attempts"]] == ["failed", "completed"]


def test_planned_training_and_external_counts():
    assert len(plan_paired_tasks("paired-view")) == 12
    assert len(plan_paired_tasks("paired-recipe")) == 12
    assert len(plan_final_tasks()) == 27
    assert len(set(planned_external_keys())) == 51


def test_failed_search_slot_cannot_be_frozen(tmp_path):
    config = load_config()
    plan = build_point_search_plan(config)
    for index, candidate in enumerate(plan.candidates):
        record_trial_result(candidate, seed=42,
                            status="failed" if index == 0 else "completed",
                            patient_macro_pathway_pcc=None if index == 0 else float(index) / 100,
                            patient_macro_z_mse_selection=None if index == 0 else 1.0)
    stage_runtime._save_plan(tmp_path / "search_point.json", plan)
    assert not stage_runtime._stage_completed(tmp_path, "point")
    with pytest.raises(Exception, match="搜索初搜及复核未完成"):
        stage_runtime.freeze_search(config, run_dir=tmp_path / "freeze_run")


def test_external_gate_checks_all_formal_tasks_before_labels(tmp_path, monkeypatch):
    config = load_config()
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "frozen_selection.json").write_text(json.dumps({"point": {}, "spatial": {}}), encoding="utf-8")
    monkeypatch.setattr(stage_runtime, "_cache_table", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("input accessed too early")))
    with pytest.raises(Exception, match="缺少训练任务记录"):
        stage_runtime.evaluate_external(config, run_dir=batch / "external-eval_run", device="cpu")
