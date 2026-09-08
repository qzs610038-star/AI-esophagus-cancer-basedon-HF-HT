from __future__ import annotations

import numpy as np

from analyze_local import compute_metric_bundle, fixed_graph_smoothing


def test_metric_bundle_preserves_defined_and_not_applicable_metrics():
    target = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    pred = target.copy()
    result = compute_metric_bundle(
        pred,
        target,
        mean=np.array([10.0, 20.0]),
        std=np.array([2.0, 4.0]),
        patient_ids=["a", "a", "b"],
        pathway_names=["p0", "p1"],
    )
    assert result["pooled_pathway_pcc"]["mean"] == 1.0
    assert result["flattened_pcc"] == 1.0
    assert result["z_rmse"] == 0.0
    assert result["raw_mae"] == 0.0
    assert result["ccc"] == 1.0
    assert result["masked_ssim"]["status"] == "not_applicable"
    assert result["top_k_overlap"]["status"] == "not_applicable"


def test_beta_one_smoothing_and_isolated_identity():
    pred = np.array([[1.0], [3.0], [9.0]])
    identities = [("p", "s", "a"), ("p", "s", "b"), ("p", "s", "c")]
    edges = [
        {"patient_id": "p", "slide_id": "s", "spot_id": "a", "neighbor_patient_id": "p", "neighbor_slide_id": "s", "neighbor_spot_id": "b", "a_self": "0.5", "a": "0.5", "degree": "1"},
        {"patient_id": "p", "slide_id": "s", "spot_id": "b", "neighbor_patient_id": "p", "neighbor_slide_id": "s", "neighbor_spot_id": "a", "a_self": "0.5", "a": "0.5", "degree": "1"},
        {"patient_id": "p", "slide_id": "s", "spot_id": "c", "neighbor_patient_id": "", "neighbor_slide_id": "", "neighbor_spot_id": "", "a_self": "1", "a": "", "degree": "0"},
    ]
    smooth, degree = fixed_graph_smoothing(pred, identities, edges, beta=1.0)
    np.testing.assert_allclose(smooth[:, 0], [2.0, 2.0, 9.0])
    np.testing.assert_array_equal(degree, [1, 1, 0])
