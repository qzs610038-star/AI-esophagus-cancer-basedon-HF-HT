from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import load_bundle, preflight

from helpers import make_data_package, replace_npz


def test_load_bundle_preserves_verified_identity_order_coordinates_and_targets(tmp_path: Path) -> None:
    config, prediction_arrays, graph_arrays = make_data_package(tmp_path)

    bundle = load_bundle(config, tmp_path / "package")

    train = bundle.partitions["train"]
    assert train.patient_ids.tolist() == ["P2", "P1"]
    assert train.spot_ids.tolist() == ["patch_x20_y20", "patch_x10_y10"]
    np.testing.assert_array_equal(train.coordinates, [[20.0, 20.0], [10.0, 10.0]])
    np.testing.assert_array_equal(train.native_steps, [20.0, 10.0])
    np.testing.assert_array_equal(train.features, prediction_arrays["cls_train"])
    np.testing.assert_array_equal(train.graph_features, graph_arrays["cls_train"])
    np.testing.assert_array_equal(train.targets, prediction_arrays["target_train"])
    assert bundle.pathway_names == ("pathway_0", "pathway_1")


def test_load_bundle_rejects_prediction_and_graph_identity_order_mismatch(tmp_path: Path) -> None:
    config, prediction_arrays, _ = make_data_package(tmp_path)
    prediction_arrays["patient_train"] = np.asarray(["P1", "P2"])
    replace_npz(config["inputs"]["feature_cache"], prediction_arrays)

    with pytest.raises(ValueError, match="patient order differs"):
        load_bundle(config, tmp_path / "package")


def test_load_bundle_rejects_same_patient_row_reordering_via_frozen_cls(tmp_path: Path) -> None:
    config, _, graph_arrays = make_data_package(tmp_path)
    graph_arrays["cls_train"] = graph_arrays["cls_train"][::-1].copy()
    replace_npz(config["inputs"]["graph_feature_cache"], graph_arrays)

    with pytest.raises(ValueError, match="CLS order/value mismatch"):
        load_bundle(config, tmp_path / "package")


@pytest.mark.parametrize("invalid_part", ["feature_dimension", "target_dimension", "missing_target"])
def test_load_bundle_rejects_invalid_feature_or_target_contract(
    tmp_path: Path, invalid_part: str,
) -> None:
    config, prediction_arrays, _ = make_data_package(tmp_path)
    if invalid_part == "feature_dimension":
        prediction_arrays["cls_train"] = np.zeros((2, 4), dtype=np.float32)
        expected = "expected 3-dimensional frozen features"
    elif invalid_part == "target_dimension":
        prediction_arrays["target_train"] = np.zeros((2, 3), dtype=np.float32)
        expected = "labels must have 2 pathways"
    else:
        prediction_arrays.pop("target_train")
        expected = "missing required labels target_train"
    replace_npz(config["inputs"]["feature_cache"], prediction_arrays)

    with pytest.raises((ValueError, KeyError), match=expected):
        load_bundle(config, tmp_path / "package")


@pytest.mark.parametrize("missing_source", ["feature_cache", "graph_feature_cache"])
def test_load_bundle_reports_the_exact_missing_cache(tmp_path: Path, missing_source: str) -> None:
    config, _, _ = make_data_package(tmp_path)
    missing = tmp_path / f"missing_{missing_source}.npz"
    config["inputs"][missing_source] = str(missing)

    with pytest.raises(FileNotFoundError, match=str(missing).replace("\\", r"\\")):
        load_bundle(config, tmp_path / "package")


def test_preflight_does_not_require_or_read_external_inputs(tmp_path: Path) -> None:
    config, _, _ = make_data_package(tmp_path)
    external_path = Path(config["inputs"]["xzy_source"])
    assert not external_path.exists()

    report = preflight(config, tmp_path / "package")

    assert report["status"] == "ok"
    assert report["external_labels_read"] is False
    assert report["partitions"] == {"train": 2, "internal_val": 1}
    assert "external_test" not in report["partitions"]
    assert not external_path.exists()


def test_preflight_uses_declared_bf16_cache_tolerance(tmp_path: Path) -> None:
    config, prediction_arrays, _ = make_data_package(tmp_path)
    config["inputs"]["source_prediction_precision"] = "bf16"
    config["parameters"]["source_cache_prediction_max_abs_tolerance"] = 0.02
    prediction_arrays["pred_train"].fill(0.01377105712890625)
    prediction_arrays["pred_val"].fill(0.01377105712890625)
    replace_npz(config["inputs"]["feature_cache"], prediction_arrays)

    report = preflight(config, tmp_path / "package")
    assert report["source_prediction_consistency"]["tolerance"] == 0.02
    assert report["source_prediction_consistency"]["source_precision"] == "bf16"

    prediction_arrays["pred_train"].fill(0.021)
    replace_npz(config["inputs"]["feature_cache"], prediction_arrays)
    with pytest.raises(AssertionError, match="source H/C prediction mismatch"):
        preflight(config, tmp_path / "package")
