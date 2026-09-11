from __future__ import annotations

from pathlib import Path

import numpy as np

from src.main import _inference_input, _prepare_xzy_input


def test_prepare_xzy_input_maps_existing_fixed_e5_cache(tmp_path: Path) -> None:
    source = tmp_path / "frozen_regression_e5.npz"
    output = tmp_path / "xzy_standardized_input.npz"
    cls = np.arange(12, dtype=np.float32).reshape(3, 4)
    prediction_z = np.arange(6, dtype=np.float32).reshape(3, 2)
    target_z = prediction_z + 0.5
    np.savez_compressed(
        source,
        cls=cls,
        prediction_z=prediction_z,
        target_z=target_z,
        patient=np.asarray(["XZY"] * 3),
        patch_stem=np.asarray(["a", "b", "c"]),
        x=np.asarray([0, 224, 448]),
        y=np.asarray([0, 0, 0]),
    )

    summary = _prepare_xzy_input(source, output, native_step=224)
    converted = _inference_input(output, require_targets=True)

    assert summary["point_count"] == 3
    assert summary["external_used_for_selection"] is False
    np.testing.assert_array_equal(converted["features"], cls)
    np.testing.assert_array_equal(converted["graph_features"], prediction_z)
    np.testing.assert_array_equal(converted["target"], target_z)
    np.testing.assert_array_equal(converted["native_step"], [224, 224, 224])
    np.testing.assert_array_equal(converted["partition"], ["external_test"] * 3)
