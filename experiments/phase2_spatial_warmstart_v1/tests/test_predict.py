from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import SpatialWarmstartModel
from src.predict import (
    evaluate_external,
    fixed_beta_one_smoothing,
    load_model_bundle,
    predict_features,
    save_model_bundle,
)
from src.spatial import SpatialPoint, build_spatial_graph


def _metadata() -> dict:
    return {
        "graph_parameters": {
            "radius_in_native_steps": 1.5,
            "max_neighbors": 8,
            "distance_sigma": 1.0,
            "image_temperature": 0.2,
            "self_raw_weight": 1.0,
            "scope": "within_patient_and_partition",
            "use_labels": False,
        },
        "pathway_names": ["p0", "p1"],
        "normalization": {
            "scale": "training_zscore",
            "fit_split": "train",
            "parameters_embedded": False,
        },
        "feature_contract": {
            "backbone": "MahmoodLab/UNI2-h",
            "token_policy": "forward_features_first_CLS",
            "dimension": 3,
            "dtype": "float32",
        },
        "native_step_contract": {
            "source": "packaged_split_info",
            "unit": "source_coordinate_unit",
            "positive": True,
        },
        "coordinate_contract": {
            "fields": ["x", "y"],
            "identity_fields": ["patient_id", "spot_id"],
            "graph_scope": "within_patient_and_partition",
        },
    }


def _point_model() -> SpatialWarmstartModel:
    model = SpatialWarmstartModel(input_dim=3, hidden_dim=2, output_dim=2, dropout=0.0)
    with torch.no_grad():
        model.shared.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        model.shared.linear.bias.zero_()
        model.regression.linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        model.regression.linear.bias.copy_(torch.tensor([0.25, -0.5]))
    model.eval()
    return model


def test_point_bundle_roundtrip_is_strict_and_prediction_interface_has_no_labels(
    tmp_path: Path,
) -> None:
    forbidden = {"label", "labels", "target", "targets", "labels_z", "pathway"}
    assert not (set(inspect.signature(predict_features).parameters) & forbidden)

    model = _point_model()
    features = np.asarray([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]], dtype=np.float32)
    expected = predict_features(model, features, spatial_enabled=False, device="cpu")
    path = tmp_path / "model_bundle.pt"
    save_model_bundle(
        path,
        model,
        arm="point_continue",
        spatial_enabled=False,
        **_metadata(),
    )

    loaded, bundle = load_model_bundle(path, device="cpu")
    actual = predict_features(loaded, features, spatial_enabled=False, device="cpu")

    np.testing.assert_array_equal(actual, expected)
    assert bundle["arm"] == "point_continue"
    assert bundle["spatial_enabled"] is False
    assert bundle["pathway_names"] == ["p0", "p1"]
    assert torch.count_nonzero(loaded.B).item() == 0
    assert set(bundle["model_state_dict"]) == {
        "shared.linear.weight",
        "shared.linear.bias",
        "regression.linear.weight",
        "regression.linear.bias",
        "B",
    }

    damaged = dict(bundle)
    damaged["model_state_dict"] = dict(bundle["model_state_dict"])
    damaged["model_state_dict"].pop("regression.linear.bias")
    damaged_path = tmp_path / "damaged.pt"
    torch.save(damaged, damaged_path)
    with pytest.raises((KeyError, RuntimeError, ValueError), match="regression|state|missing"):
        load_model_bundle(damaged_path, device="cpu")


def test_fixed_beta_one_smoothing_includes_self_weight_and_preserves_isolated_point() -> None:
    points = [
        SpatialPoint("a", "P", "external_test", 0.0, 0.0),
        SpatialPoint("b", "P", "external_test", 1.0, 0.0),
        SpatialPoint("c", "P", "external_test", 10.0, 0.0),
    ]
    graph = build_spatial_graph(
        points,
        np.ones((3, 2), dtype=np.float32),
        native_steps={"P": 1.0},
        radius_in_native_steps=1.5,
        max_neighbors=8,
        distance_sigma=1.0,
        image_temperature=0.2,
        self_raw_weight=1.0,
    )
    prediction = np.asarray([[1.0], [3.0], [9.0]], dtype=np.float64)

    smoothed = fixed_beta_one_smoothing(prediction, graph)

    raw_neighbor = np.exp(-0.5)
    self_weight = 1.0 / (1.0 + raw_neighbor)
    neighbor_weight = raw_neighbor / (1.0 + raw_neighbor)
    expected = np.asarray(
        [
            [self_weight * 1.0 + neighbor_weight * 3.0],
            [self_weight * 3.0 + neighbor_weight * 1.0],
            [9.0],
        ]
    )
    np.testing.assert_allclose(smoothed, expected, rtol=0, atol=1e-7)


def test_spatial_bundle_roundtrip_preserves_nonzero_b_and_inference_contract(
    tmp_path: Path,
) -> None:
    model = _point_model()
    with torch.no_grad():
        model.B.copy_(torch.tensor([[0.5, -0.25], [-0.75, 0.125]]))
    points = [
        SpatialPoint("a", "P", "internal_val", 0.0, 0.0),
        SpatialPoint("b", "P", "internal_val", 1.0, 0.0),
    ]
    graph = build_spatial_graph(
        points,
        np.ones((2, 3), dtype=np.float32),
        native_steps={"P": 1.0},
    )
    features = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    expected = predict_features(
        model, features, spatial_enabled=True, graph=graph, device="cpu"
    )
    path = tmp_path / "spatial_bundle.pt"
    metadata = _metadata()
    save_model_bundle(
        path,
        model,
        arm="spatial_joint",
        spatial_enabled=True,
        **metadata,
    )

    loaded, bundle = load_model_bundle(path, device="cpu")
    actual = predict_features(
        loaded, features, spatial_enabled=True, graph=graph, device="cpu"
    )

    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(loaded.B, model.B)
    for field in (
        "graph_parameters",
        "pathway_names",
        "normalization",
        "feature_contract",
        "native_step_contract",
        "coordinate_contract",
    ):
        assert bundle[field] == metadata[field]


def test_external_evaluation_is_separate_and_reports_both_pcc_definitions() -> None:
    model = _point_model()
    features = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float64
    )

    result = evaluate_external(
        model,
        features,
        targets=targets,
        patient_ids=["A", "A", "B", "B"],
        spatial_enabled=False,
        device="cpu",
    )

    assert result["split"] == "external_test"
    assert result["selection_used"] is False
    assert result["training_performed"] is False
    assert result["prediction"].shape == targets.shape
    assert result["target"].shape == targets.shape
    assert "patient_macro_pathway_pcc" in result["metrics"]
    assert "flattened_pooled_pcc" in result["metrics"]
    assert result["metrics"]["pooled_pcc"] == result["metrics"]["flattened_pooled_pcc"]
    assert result["metrics"]["patient_macro_pathway_pcc"] == pytest.approx(1.0)
    assert result["metrics"]["valid_patient_pathway_count"] == 2
    assert result["metrics"]["invalid_patient_pathway_count"] == 2
    assert result["metrics"]["flattened_pooled_pcc"] == pytest.approx(
        np.corrcoef(result["prediction"].ravel(), targets.ravel())[0, 1]
    )
