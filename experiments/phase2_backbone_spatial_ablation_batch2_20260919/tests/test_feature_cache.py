from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from errors import IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache


def test_cache_requires_complete_identity_and_contract(tmp_path: Path):
    output = tmp_path / "hoptimus0" / "full_fov_224_bicubic_v1__hoptimus_rgb_v1" / "all"
    writer = FeatureCacheWriter(output, n_rows=2, feature_dim=3, identities=["p|s|a", "p|s|b"])
    writer.write_rows(0, np.ones((2, 3), dtype=np.float32))
    writer.finalize({
        "model": "hoptimus0",
        "revision": "a" * 40,
        "output_mode": "timm_final_embedding",
        "feature_dim": 3,
        "geometry_protocol": "full_fov_224_bicubic_v1",
        "normalization_profile": "hoptimus_rgb_v1",
        "dataset": "all",
    })
    loaded = load_feature_cache(
        output,
        expected_identities=["p|s|a", "p|s|b"],
        expected_dim=3,
        expected_metadata={"model": "hoptimus0", "normalization_profile": "hoptimus_rgb_v1", "revision": "a" * 40},
    )
    assert loaded["features"].shape == (2, 3)
    assert (output / "COMPLETE").is_file()
    assert (output / "metadata.json").is_file()
    with pytest.raises(IdentityMismatchError, match="元数据"):
        load_feature_cache(output, expected_identities=["p|s|a", "p|s|b"], expected_dim=3, expected_metadata={"revision": "b" * 40})
    with pytest.raises(IdentityMismatchError, match="身份"):
        load_feature_cache(output, expected_identities=["p|s|a", "p|s|c"], expected_dim=3)
    incomplete = tmp_path / "broken"
    incomplete.mkdir()
    with pytest.raises(IdentityMismatchError, match="COMPLETE"):
        load_feature_cache(incomplete, expected_identities=["p|s|a"], expected_dim=3)


def test_writer_rejects_mean_patch(tmp_path: Path):
    writer = FeatureCacheWriter(tmp_path / "cache", n_rows=1, feature_dim=2, identities=["p|s|a"])
    with pytest.raises(IdentityMismatchError, match="mean_patch"):
        writer.write_rows(0, np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    writer.abort()
