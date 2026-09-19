from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from adapters import ModelSpec, TokenLayout, pool_tokens
from errors import ConfigError, IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache
from feature_extract import cache_contract, extract_one_model
from input_data import IdentityRecord, select_dataset_rows
from transforms import FULL_FOV_PROTOCOL, LEGACY_CROP_PROTOCOL, build_transform


def test_full_fov_retains_edge_marker_while_legacy_crop_excludes_it():
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[:, :] = (10, 20, 30)
    pixels[0:8, :] = (255, 0, 0)
    image = Image.fromarray(pixels, "RGB")
    full = np.asarray(build_transform(FULL_FOV_PROTOCOL).prepare_image(image))
    legacy = np.asarray(build_transform(LEGACY_CROP_PROTOCOL).prepare_image(image))
    assert tuple(full[0, 100]) == (255, 0, 0)
    assert tuple(legacy[0, 100]) != (255, 0, 0)


def test_full_fov_rejects_non_square_input():
    with pytest.raises(ConfigError, match="方形"):
        build_transform(FULL_FOV_PROTOCOL)(Image.new("RGB", (224, 223)))


def test_cache_requires_complete_identity_metadata_and_optional_mean(tmp_path: Path):
    output = tmp_path / "encoder" / FULL_FOV_PROTOCOL / "all"
    writer = FeatureCacheWriter(output, n_rows=2, feature_dim=3, identities=["p|s|a", "p|s|b"], include_mean_patch=True)
    writer.write_rows(0, np.ones((2, 3), dtype=np.float32), np.full((2, 3), 2, dtype=np.float32))
    writer.finalize({"experiment_id": "synthetic", "model": "uni2h", "revision": "a" * 40, "preprocessing": {"protocol": FULL_FOV_PROTOCOL}, "dataset": "all"})
    loaded = load_feature_cache(output, expected_identities=["p|s|a", "p|s|b"], expected_dim=3, expected_metadata={"model": "uni2h", "dataset": "all"}, require_mean_patch=True)
    assert loaded["cls"].shape == (2, 3)
    assert np.asarray(loaded["mean_patch"])[0, 0] == 2
    with pytest.raises(IdentityMismatchError, match="元数据"):
        load_feature_cache(output, expected_identities=["p|s|a", "p|s|b"], expected_dim=3, expected_metadata={"dataset": "train"})


def test_pool_tokens_uses_cls_and_skips_register_tokens():
    layout = TokenLayout(2, 5, 0, 2, 3, 1)
    tokens = torch.tensor([[[1.0, 2.0], [99, 99], [3, 4], [5, 6], [7, 8]]], dtype=torch.float32)
    cls, mean = pool_tokens(tokens, layout, include_mean_patch=True)
    torch.testing.assert_close(cls, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(mean, torch.tensor([[5.0, 6.0]]))


class _FakeAdapter:
    def encode(self, images: torch.Tensor, *, include_mean_patch: bool = False):
        cls = images.mean(dim=(1, 2, 3)).reshape(-1, 1).repeat(1, 4).float()
        return cls, (cls + 1 if include_mean_patch else None)


def test_extraction_writes_protocol_scoped_cls_cache(tmp_path: Path):
    image_path = tmp_path / "patch_x0_y0.png"
    Image.new("RGB", (224, 224), (30, 40, 50)).save(image_path)
    row = IdentityRecord(0, "P", "MPP2_P_source01", image_path.stem, "train", 0, 0, str(image_path), 224, 224)
    spec = ModelSpec("uni2h", "candidate", "repo", "a" * 40, "fake", "weights", None, str(tmp_path / "snapshots" / ("a" * 40)), TokenLayout(4, 5, 0, 2, 3, 1))
    result = extract_one_model(spec, [row], tmp_path / "uni2h" / FULL_FOV_PROTOCOL / "train", protocol=FULL_FOV_PROTOCOL, device="cpu", batch_sizes=[1], include_mean_patch=True, adapter_factory=lambda *_: _FakeAdapter(), metadata_context={"experiment_id": "synthetic", "model": "uni2h", "revision": "a" * 40, "preprocessing": {"protocol": FULL_FOV_PROTOCOL}, "dataset": "train"})
    assert result["status"] == "complete"
    cache = load_feature_cache(result["cache_dir"], expected_identities=[row.identity_key], expected_dim=4, require_mean_patch=True)
    assert cache["metadata"]["preprocessing"]["protocol"] == FULL_FOV_PROTOCOL


def test_dataset_selection_rejects_non_square_full_fov():
    row = IdentityRecord(0, "P", "MPP2_P_source01", "patch_x0_y0", "train", 0, 0, "unused.png", 224, 220)
    with pytest.raises(IdentityMismatchError, match="非方图"):
        select_dataset_rows([row], "train", protocol=FULL_FOV_PROTOCOL)


def test_cache_contract_binds_snapshot_and_checkpoint_path(tmp_path: Path):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    spec = ModelSpec("uni2h", "candidate", "repo", "a" * 40, "fake", "weights.bin", None, str(snapshot), TokenLayout(4, 5, 0, 2, 3, 1))
    contract = cache_contract({"experiment_id": "synthetic", "protocol_version": "v"}, spec, protocol=FULL_FOV_PROTOCOL, dataset="all", manifest_path=tmp_path / "identities.csv")
    assert contract["snapshot_path"] == str(snapshot.resolve())
    assert contract["checkpoint_path"] == str((snapshot / "weights.bin").resolve())


def test_primary_cls_cache_works_without_optional_mean_patch(tmp_path: Path):
    output = tmp_path / "cache"
    writer = FeatureCacheWriter(output, n_rows=1, feature_dim=2,
                                identities=["p|s|a"], include_mean_patch=False)
    writer.write_rows(0, np.asarray([[1., 2.]], dtype=np.float32))
    writer.finalize({"model": "uni2h", "dataset": "all"})
    cache = load_feature_cache(output, expected_identities=["p|s|a"], expected_dim=2,
                               expected_metadata={"model": "uni2h", "dataset": "all"},
                               require_mean_patch=False)
    assert "mean_patch" not in cache
    np.testing.assert_array_equal(cache["cls"], [[1., 2.]])
