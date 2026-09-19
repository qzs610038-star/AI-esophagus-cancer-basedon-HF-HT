from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from adapters import ModelSpec, TokenLayout
from config import load_config
from data import IdentityRecord
from feature_cache import load_feature_cache
from feature_extract import extract_one_model, prepare_feature_caches
from errors import IdentityMismatchError


class _FakeAdapter:
    def encode(self, images: torch.Tensor):
        base = images.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
        cls = base.repeat(1, 4).float()
        mean = (base + 1).repeat(1, 4).float()
        return cls, mean


def test_extraction_saves_cls_and_auxiliary_mean_without_labels(tmp_path: Path):
    rows = []
    for index, value in enumerate((0, 127, 255)):
        path = tmp_path / f"patch_x{index}_y{index}.png"
        Image.new("RGB", (224, 224), color=(value, value, value)).save(path)
        rows.append(
            IdentityRecord(
                index,
                "A",
                "MPP2_A_source01",
                path.stem,
                "train" if index < 2 else "internal_val",
                index,
                index,
                str(path),
                224,
                224,
            )
        )
    spec = ModelSpec(
        "uni",
        "candidate",
        "repo",
        "a" * 40,
        "fake",
        "weights.bin",
        "config.json",
        str(tmp_path / "snapshot"),
        TokenLayout(197, 4, 0, 1, 196, 0),
        True,
    )
    output = tmp_path / "cache"
    result = extract_one_model(
        spec,
        rows,
        output,
        device="cpu",
        batch_sizes=[2],
        adapter_factory=lambda _spec, _device: _FakeAdapter(),
        metadata_context={
            "experiment_id": "synthetic_experiment",
            "protocol_version": "synthetic_protocol",
            "common_identity_manifest": "synthetic/common_identity_manifest.csv",
        },
    )
    assert result["status"] == "complete"
    assert result["actual_batch_size"] == 2
    identities = [row.identity_key for row in rows]
    cache = load_feature_cache(output, expected_identities=identities, expected_dim=4)
    assert cache["cls"].shape == (3, 4)
    auxiliary = np.load(output / "mean_patch.npy", allow_pickle=False)
    assert np.allclose(auxiliary, np.asarray(cache["cls"]) + 1)
    assert cache["metadata"]["experiment_id"] == "synthetic_experiment"
    assert cache["metadata"]["protocol_version"] == "synthetic_protocol"


def test_official_feature_preparation_cannot_mark_a_partial_model_set_complete(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")
    with np.testing.assert_raises_regex(ValueError, "必须同时按固定顺序"):
        prepare_feature_caches(
            config,
            device="cpu",
            inspection_report=tmp_path / "not_read.json",
            models=("uni",),
        )


def test_extraction_rejects_an_image_changed_after_the_common_manifest_was_built(tmp_path: Path):
    path = tmp_path / "patch_x0_y0.png"
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(path)
    row = IdentityRecord(0, "A", "MPP2_A_source01", path.stem, "train", 0, 0, str(path), 12, 10)
    spec = ModelSpec(
        "uni",
        "candidate",
        "repo",
        "a" * 40,
        "fake",
        "weights.bin",
        "config.json",
        str(tmp_path / "snapshot"),
        TokenLayout(197, 4, 0, 1, 196, 0),
        True,
    )
    with pytest.raises(IdentityMismatchError, match="像素尺寸"):
        extract_one_model(
            spec,
            [row],
            tmp_path / "cache",
            device="cpu",
            batch_sizes=[1],
            adapter_factory=lambda _spec, _device: _FakeAdapter(),
        )
