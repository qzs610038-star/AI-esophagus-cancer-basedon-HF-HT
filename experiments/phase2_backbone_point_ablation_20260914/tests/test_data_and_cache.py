from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data import (
    build_common_identity_manifest,
    load_common_manifest,
    load_labels_for_rows,
)
from errors import IdentityMismatchError
from feature_cache import FeatureCacheWriter, load_feature_cache


def _write_image(path: Path, size: tuple[int, int] = (12, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 30, 40)).save(path)


def _tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    rows = [
        {"mpp_id": 2, "patient": "A", "patch_stem": "patch_x1_y2", "x": 1, "y": 2, "split": "train", "block_id": "a"},
        {"mpp_id": 2, "patient": "A", "patch_stem": "patch_x3_y4", "x": 3, "y": 4, "split": "internal_val", "block_id": "b"},
    ]
    split_path = tmp_path / "split.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        _write_image(image_root / row["patient"] / "patch_images" / f"{row['patch_stem']}.png")
    _write_image(image_root / "XZY" / "patch_images" / "patch_x9_y8.png", (14, 11))
    return split_path, image_root


def test_common_manifest_joins_by_identity_and_records_real_pixel_size(tmp_path: Path):
    split_path, image_root = _tiny_inputs(tmp_path)
    output = tmp_path / "common.csv"
    rows = build_common_identity_manifest(
        split_path,
        image_root,
        output,
        mpp_id=2,
        external_patient="XZY",
        expected_counts={"train": 1, "internal_val": 1, "external_test": 1},
    )
    assert output.is_file()
    assert [row.split for row in rows].count("external_test") == 1
    external = next(row for row in rows if row.patient_id == "XZY")
    assert (external.width_px, external.height_px) == (14, 11)
    assert external.source_group == "MPP2_XZY_source01"
    loaded = load_common_manifest(
        output, expected_counts={"train": 1, "internal_val": 1, "external_test": 1}
    )
    assert [row.identity_key for row in loaded] == [row.identity_key for row in rows]


def test_common_manifest_fails_on_missing_or_duplicate_identity(tmp_path: Path):
    split_path, image_root = _tiny_inputs(tmp_path)
    # Remove an image: the manifest must not silently drop the row.
    (image_root / "A" / "patch_images" / "patch_x1_y2.png").unlink()
    with pytest.raises(IdentityMismatchError, match="图像不存在"):
        build_common_identity_manifest(
            split_path,
            image_root,
            tmp_path / "common.csv",
            mpp_id=2,
            external_patient="XZY",
            expected_counts={"train": 1, "internal_val": 1, "external_test": 1},
        )


def test_external_rows_cannot_be_passed_to_the_training_label_loader(tmp_path: Path):
    split_path, image_root = _tiny_inputs(tmp_path)
    rows = build_common_identity_manifest(
        split_path,
        image_root,
        tmp_path / "common.csv",
        mpp_id=2,
        external_patient="XZY",
        expected_counts={"train": 1, "internal_val": 1, "external_test": 1},
    )
    external = [row for row in rows if row.split == "external_test"]
    with pytest.raises(IdentityMismatchError, match="外部"):
        load_labels_for_rows(external, tmp_path / "labels", ["p"])


def test_feature_cache_commits_both_vectors_atomically_and_loads_cls_only(tmp_path: Path):
    _, image_root = _tiny_inputs(tmp_path)
    identities = ["A|MPP2_A_source01|patch_x1_y2", "XZY|MPP2_XZY_source01|patch_x9_y8"]
    final = tmp_path / "feature_caches" / "uni" / "v1"
    writer = FeatureCacheWriter(final, n_rows=2, feature_dim=4, identities=identities)
    writer.write_rows(
        0,
        np.arange(8, dtype=np.float32).reshape(2, 4),
        np.arange(8, 16, dtype=np.float32).reshape(2, 4),
    )
    writer.finalize({"model": "uni", "image_root": str(image_root), "status": "complete"})

    assert (final / "COMPLETE").is_file()
    cache = load_feature_cache(final, expected_identities=identities, expected_dim=4)
    assert set(cache) == {"cls", "identities", "metadata", "cache_dir"}
    assert cache["cls"].shape == (2, 4)
    assert (final / "mean_patch.npy").is_file()


def test_incomplete_feature_cache_is_never_accepted(tmp_path: Path):
    final = tmp_path / "cache"
    writer = FeatureCacheWriter(final, n_rows=2, feature_dim=2, identities=["a", "b"])
    writer.write_rows(0, np.ones((1, 2), np.float32), np.ones((1, 2), np.float32))
    with pytest.raises(IdentityMismatchError, match="尚未写满"):
        writer.finalize({"model": "uni"})
    assert not final.exists()


def test_feature_cache_rejects_identity_reordering(tmp_path: Path):
    final = tmp_path / "cache"
    writer = FeatureCacheWriter(final, n_rows=2, feature_dim=2, identities=["a", "b"])
    writer.write_rows(0, np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))
    writer.finalize({"model": "uni"})
    with pytest.raises(IdentityMismatchError, match="身份"):
        load_feature_cache(final, expected_identities=["b", "a"], expected_dim=2)
