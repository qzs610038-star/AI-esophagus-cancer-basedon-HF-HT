from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from adapters import (
    EncoderAdapter,
    TokenLayout,
    load_model_specs,
    pool_tokens,
    strict_load_state,
)
from errors import ConfigError, NonFiniteDataError, OfflineModelError


def _tokens(batch: int, token_count: int, dim: int) -> torch.Tensor:
    token_ids = torch.arange(token_count, dtype=torch.float32).view(1, token_count, 1)
    dims = torch.arange(dim, dtype=torch.float32).view(1, 1, dim)
    return (token_ids * 10_000 + dims).expand(batch, -1, -1).clone()


@pytest.mark.parametrize(
    ("layout", "patch_start"),
    [
        (TokenLayout(197, 1024, 0, 1, 196, 0), 1),
        (TokenLayout(261, 1280, 0, 5, 256, 4), 5),
        (TokenLayout(265, 1536, 0, 9, 256, 8), 9),
    ],
)
def test_explicit_token_layout_controls_cls_and_mean_patch(layout, patch_start):
    tokens = _tokens(2, layout.total_tokens, layout.feature_dim)
    cls, mean_patch = pool_tokens(tokens, layout)
    torch.testing.assert_close(cls, tokens[:, 0, :])
    torch.testing.assert_close(mean_patch, tokens[:, patch_start:, :].mean(dim=1))


def test_pooling_rejects_ambiguous_or_nonfinite_outputs():
    layout = TokenLayout(197, 4, 0, 1, 196, 0)
    with pytest.raises(ConfigError, match="三维"):
        pool_tokens(torch.ones(2, 4), layout)
    bad = torch.ones(1, 197, 4)
    bad[0, 3, 2] = float("nan")
    with pytest.raises(NonFiniteDataError):
        pool_tokens(bad, layout)
    with pytest.raises(ConfigError, match="float32"):
        pool_tokens(torch.ones(1, 197, 4, dtype=torch.float16), layout)


class _FakeEncoder(nn.Module):
    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self._tokens = tokens

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._tokens[: images.shape[0]] + self.bias


class _HalfEncoder(nn.Module):
    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 197, 8, dtype=torch.float16, device=images.device)


def test_adapter_freezes_encoder_and_returns_two_separate_vectors():
    layout = TokenLayout(197, 8, 0, 1, 196, 0)
    model = _FakeEncoder(_tokens(2, 197, 8))
    adapter = EncoderAdapter("uni", model, layout, torch.device("cpu"))
    cls, mean_patch = adapter.encode(torch.zeros(2, 3, 224, 224))
    assert cls.shape == mean_patch.shape == (2, 8)
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_adapter_does_not_silently_cast_non_fp32_encoder_outputs():
    adapter = EncoderAdapter(
        "uni", _HalfEncoder(), TokenLayout(197, 8, 0, 1, 196, 0), torch.device("cpu")
    )
    with pytest.raises(ConfigError, match="float32"):
        adapter.encode(torch.zeros(1, 3, 224, 224))


def test_strict_loader_rejects_missing_or_unexpected_keys():
    model = nn.Linear(2, 2)
    with pytest.raises(OfflineModelError, match="严格"):
        strict_load_state(model, {"weight": torch.ones(2, 2)})


def test_model_manifest_is_pinned_and_never_marks_uni2h_for_extraction(tmp_path: Path):
    manifest = {
        "schema_version": "1.0",
        "models": {
            "uni2h": {
                "role": "historical_reference_only",
                "repo_id": "MahmoodLab/UNI2-h",
                "revision": None,
                "architecture": "vit_giant_patch14_224",
                "cls_dim": 1536,
                "mean_patch_dim": 1536,
                "total_tokens": 265,
                "cls_index": 0,
                "patch_start": 9,
                "patch_count": 256,
                "register_count": 8,
                "extract_in_this_package": False,
            },
            "uni": {
                "role": "candidate",
                "repo_id": "MahmoodLab/UNI",
                "revision": "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
                "architecture": "vit_large_patch16_224",
                "checkpoint_filename": "pytorch_model.bin",
                "config_filename": "config.json",
                "snapshot_path": str(tmp_path / "uni"),
                "cls_dim": 1024,
                "mean_patch_dim": 1024,
                "total_tokens": 197,
                "cls_index": 0,
                "patch_start": 1,
                "patch_count": 196,
                "register_count": 0,
                "extract_in_this_package": True,
            },
            "virchow2": {
                "role": "candidate",
                "repo_id": "paige-ai/Virchow2",
                "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
                "architecture": "vit_huge_patch14_224",
                "checkpoint_filename": "model.safetensors",
                "config_filename": "config.json",
                "snapshot_path": str(tmp_path / "v"),
                "cls_dim": 1280,
                "mean_patch_dim": 1280,
                "total_tokens": 261,
                "cls_index": 0,
                "patch_start": 5,
                "patch_count": 256,
                "register_count": 4,
                "extract_in_this_package": True,
            },
        },
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    specs = load_model_specs(path, require_local_files=False)
    assert specs["uni"].revision == "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0"
    assert specs["virchow2"].layout.patch_start == 5
    assert specs["uni2h"].extract_in_this_package is False


def test_unpinned_candidate_revision_fails_closed(tmp_path: Path):
    shipped = Path(__file__).resolve().parents[1] / "inputs" / "model_manifest.json"
    payload = json.loads(shipped.read_text(encoding="utf-8"))
    payload["models"]["uni"]["revision"] = None
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfflineModelError, match="revision"):
        load_model_specs(path, require_local_files=False)


def test_different_but_well_formed_candidate_revision_is_protocol_drift(tmp_path: Path):
    shipped = Path(__file__).resolve().parents[1] / "inputs" / "model_manifest.json"
    payload = json.loads(shipped.read_text(encoding="utf-8"))
    payload["models"]["virchow2"]["revision"] = "f" * 40
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfflineModelError, match="固定合同"):
        load_model_specs(path, require_local_files=False)
