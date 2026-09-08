from __future__ import annotations

from pathlib import Path

import numpy as np

from data import (
    build_unlabeled_point_table_from_feature_directory,
    inverse_transform,
    load_normalization,
    load_pathway_names,
    load_split_point_table,
    make_point_table,
    require_verified_slides,
)
from errors import SlideMappingMissingError
from helpers import PACKAGE, package_config


def test_split_counts_and_no_slide_id():
    config = package_config()
    table = load_split_point_table(config)
    assert len(table) == 9472 + 1078
    assert int((table.split == "train").sum()) == 9472
    assert int((table.split == "internal_val").sum()) == 1078
    assert table.slide_status == "missing"
    assert all(not ident.has_verified_slide() for ident in table.identities)
    assert table.identities[0].key() <= table.identities[-1].key()


def test_slide_mapping_gap_is_diagnosable():
    config = package_config()
    table = load_split_point_table(config)
    try:
        require_verified_slides(table, context="test")
        raise AssertionError("missing mapping must fail")
    except SlideMappingMissingError as exc:
        assert exc.error_code == "SLIDE_MAPPING_UNVERIFIED"
        assert "patient_id" in str(exc)
        assert "point/relation" in str(exc)


def test_identity_join_does_not_use_file_order():
    labels = np.arange(6, dtype=np.float32).reshape(2, 3)
    table = make_point_table(
        patient_ids=["B", "A"],
        spot_ids=["z", "a"],
        splits=["train", "train"],
        x=[1, 0],
        y=[1, 0],
        labels_z=labels,
        sort_identities=True,
    )
    assert table.identities[0].patient_id == "A"
    assert table.identities[0].spot_id == "a"
    np.testing.assert_allclose(table.labels_z[0], labels[1])


def test_unlabeled_feature_directory_does_not_use_labels(tmp_path: Path):
    import torch

    folder = tmp_path / "feats"
    folder.mkdir()
    torch.save(torch.ones(2, 1536), folder / "patch_x10_y20.pt")
    torch.save(torch.arange(1536, dtype=torch.float32), folder / "patch_x30_y40.pt")
    table = build_unlabeled_point_table_from_feature_directory(
        folder,
        patient_id="XZY",
        split="external",
        load_vectors=True,
    )
    assert table.labels_z is None
    assert table.slide_status == "missing"
    assert len(table) == 2
    assert table.features.shape == (2, 1536)
    np.testing.assert_allclose(table.features[0], 1.0)


def test_inverse_transform_once_no_clip():
    names = load_pathway_names(PACKAGE / "inputs" / "mpp2" / "zscore_manifest.json")
    norm = load_normalization(PACKAGE / "inputs" / "mpp2" / "zscore_params_from_train.json", names)
    assert len(norm.pathway_names) == 30
    z = np.zeros((2, 30), dtype=np.float64)
    z[1] = 1.0
    raw = inverse_transform(z, norm)
    np.testing.assert_allclose(raw[0], norm.mean)
    np.testing.assert_allclose(raw[1], norm.mean + norm.std)
    z_big = np.full((1, 30), 1000.0)
    raw_big = inverse_transform(z_big, norm)
    clipped = np.clip(raw_big, norm.clip_range[0], norm.clip_range[1])
    assert float(np.max(np.abs(raw_big))) > 100
    assert not np.allclose(raw_big, clipped)
    twice = inverse_transform(raw, norm)
    assert not np.allclose(twice, raw)
