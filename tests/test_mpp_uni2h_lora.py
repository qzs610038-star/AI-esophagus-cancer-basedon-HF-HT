import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from dataset_mpp_online import OnlineManifestMPPDataset, validate_manifest_frame
from model_mpp_uni2h_lora import (
    OnlineMPPModel,
    build_paired_optimizer,
    configure_p0_lora,
    load_accepted_mpp_head,
)
from train_mpp_uni2h_mlp import MPPMLPHead
from train_mpp_uni2h_lora import evaluate, train_one_epoch
from scripts.check_mpp_online_cache_parity import (
    extract_online_cls_feature,
    resolve_unique_cache_path,
)


TARGETS = [f"pathway_{index}" for index in range(30)]


class DummyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, value):
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        return self.proj((q + k + v) / 3.0)


class DummyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = DummyAttention(dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 2)
        self.mlp.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, value):
        return value + self.attn(value)


class DummyBackbone(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.input = nn.Linear(3 * 4 * 4, dim)
        self.blocks = nn.ModuleList([DummyBlock(dim) for _ in range(24)])

    def forward(self, images):
        value = self.input(images.flatten(1))
        for block in self.blocks:
            value = block(value)
        return value


class TokenBackbone(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.projection = nn.Linear(3 * 4 * 4, dim)

    def forward_features(self, images):
        cls = self.projection(images.flatten(1))
        register = torch.zeros_like(cls)
        return torch.stack([cls, register], dim=1)


def _manifest(rows):
    return pd.DataFrame(rows, columns=[
        "mpp_id", "patient", "patch_stem", "x", "y", "split", "block_id",
    ])


def _write_patient_sample(root: Path, patient: str, stem: str, value: float):
    image_dir = root / "2" / patient / "patch_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_dir / f"{stem}.png")
    labels = pd.DataFrame({"barcode": [stem]})
    for target in TARGETS:
        labels[target] = value
    labels_path = root / f"{patient}_labels.csv"
    labels.to_csv(labels_path, index=False)
    return labels_path


def test_online_dataset_scopes_same_patch_stem_by_patient(tmp_path):
    stem = "patch_x1_y2"
    labels_a = _write_patient_sample(tmp_path, "P1", stem, 1.0)
    labels_b = _write_patient_sample(tmp_path, "P2", stem, 2.0)
    manifest = _manifest([
        [2, "P1", stem, 1, 2, "train", "P1:0:0"],
        [2, "P2", stem, 1, 2, "train", "P2:0:0"],
    ])

    transform = lambda image: torch.tensor(list(image.getdata())[0]).float()
    dataset_a = OnlineManifestMPPDataset(
        manifest, tmp_path, 2, "P1", "train", labels_a, transform,
    )
    dataset_b = OnlineManifestMPPDataset(
        manifest, tmp_path, 2, "P2", "train", labels_b, transform,
    )

    assert dataset_a[0][1].tolist() == [1.0] * 30
    assert dataset_b[0][1].tolist() == [2.0] * 30
    assert dataset_a.metadata_rows()[0]["patient"] == "P1"
    assert dataset_b.metadata_rows()[0]["patient"] == "P2"


def test_manifest_duplicate_composite_identity_is_a_hard_failure():
    manifest = _manifest([
        [2, "P1", "same", 1, 2, "train", "P1:0:0"],
        [2, "P1", "same", 1, 2, "internal_val", "P1:0:1"],
    ])
    with pytest.raises(ValueError, match="duplicate online MPP sample identity"):
        validate_manifest_frame(manifest)


def test_online_dataset_missing_png_is_a_hard_failure(tmp_path):
    stem = "patch_x1_y2"
    labels = pd.DataFrame({"barcode": [stem]})
    for target in TARGETS:
        labels[target] = 1.0
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    manifest = _manifest([[2, "P1", stem, 1, 2, "train", "P1:0:0"]])

    with pytest.raises(FileNotFoundError, match="manifest PNG"):
        OnlineManifestMPPDataset(
            manifest, tmp_path, 2, "P1", "train", labels_path, None,
        )


def test_p0_lora_is_zero_delta_and_optimizer_contains_only_head_and_lora():
    torch.manual_seed(7)
    plain_backbone = DummyBackbone()
    lora_backbone = copy.deepcopy(plain_backbone)
    images = torch.randn(2, 3, 4, 4)
    expected = plain_backbone(images)

    lora_parameter_count = configure_p0_lora(lora_backbone)
    actual = lora_backbone(images)
    assert lora_parameter_count > 0
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)

    model = OnlineMPPModel(
        lora_backbone, feature_dim=4, hidden_dim=8, output_dim=30, dropout=0.0,
    )
    optimizer = build_paired_optimizer(model, mode="lora")
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert optimizer_ids == trainable_ids
    assert all(
        not parameter.requires_grad
        for name, parameter in model.backbone.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    )

    loss = model(images).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )


def test_online_model_loads_accepted_mpp_head_checkpoint(tmp_path):
    torch.manual_seed(11)
    source_head = MPPMLPHead(in_dim=4, hidden=8, out_dim=30, dropout=0.0)
    checkpoint_path = tmp_path / "best_checkpoint.pth"
    torch.save({"epoch": 15, "model_state_dict": source_head.state_dict()}, checkpoint_path)

    backbone = DummyBackbone(dim=4)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    model = OnlineMPPModel(
        backbone, feature_dim=4, hidden_dim=8, output_dim=30, dropout=0.0,
    )
    metadata = load_accepted_mpp_head(model, checkpoint_path)

    assert metadata["epoch"] == 15
    for expected, actual in zip(source_head.parameters(), model.head.parameters()):
        assert torch.equal(expected, actual)


def test_cpu_train_and_eval_loop_runs_with_amp_disabled():
    torch.manual_seed(13)
    backbone = DummyBackbone(dim=4)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    model = OnlineMPPModel(
        backbone, feature_dim=4, hidden_dim=8, output_dim=30, dropout=0.0,
    )
    optimizer = build_paired_optimizer(model, mode="frozen")
    images = torch.randn(4, 3, 4, 4)
    targets = torch.randn(4, 30)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    train_loss, train_metrics = train_one_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"), scaler,
        amp_enabled=False, grad_accum_steps=2, gradient_clip=1.0,
    )
    val_loss, val_metrics, predictions, labels = evaluate(
        model, loader, criterion, torch.device("cpu"), amp_enabled=False,
    )

    assert train_loss > 0
    assert val_loss > 0
    assert set(train_metrics) >= {"pcc", "mae", "r2"}
    assert set(val_metrics) >= {"pcc", "mae", "r2"}
    assert predictions.shape == labels.shape == (4, 30)


def test_lora_training_help_renders_without_loading_uni2h():
    project_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "train_mpp_uni2h_lora.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "--manifest_labels_root" in completed.stdout
    assert "--head_checkpoint_sha256" in completed.stdout


def test_cache_parity_resolver_rejects_missing_and_duplicate_cache(tmp_path):
    cache_root = tmp_path / "partner"
    flat_root = tmp_path / "flat"
    with pytest.raises(FileNotFoundError, match="exactly one cache"):
        resolve_unique_cache_path(cache_root, flat_root, 2, "P1", "patch_x1_y2")

    first = cache_root / "MPP2_UNI" / "P1" / "patch_x1_y2.pt"
    second = flat_root / "2" / "P1" / "patch_x1_y2.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    torch.save(torch.zeros(4), first)
    torch.save(torch.zeros(4), second)
    with pytest.raises(FileNotFoundError, match="exactly one cache"):
        resolve_unique_cache_path(cache_root, flat_root, 2, "P1", "patch_x1_y2")


def test_cache_parity_extracts_cls_from_online_token_sequence():
    class TokenBackbone:
        def forward_features(self, images):
            batch = images.shape[0]
            return torch.arange(batch * 3 * 4, dtype=torch.float32).reshape(batch, 3, 4)

    images = torch.zeros(2, 3, 4, 4)
    cls = extract_online_cls_feature(TokenBackbone(), images, feature_dim=4)
    assert cls.shape == (2, 4)
    assert torch.equal(cls, torch.tensor([[0., 1., 2., 3.], [12., 13., 14., 15.]]))


def test_online_mpp_model_extracts_cls_from_token_backbone():
    model = OnlineMPPModel(
        backbone=TokenBackbone(), feature_dim=4, hidden_dim=8,
        output_dim=3, dropout=0.0,
    )
    output = model(torch.zeros(2, 3, 4, 4))
    assert output.shape == (2, 3)


def test_cache_parity_help_requires_resource_release_acknowledgement():
    project_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "scripts/check_mpp_online_cache_parity.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "--resource_release_ack" in completed.stdout
