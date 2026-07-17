import numpy as np

from scripts.mpp2_pathway_ridge_calibration import (
    choose_lambda_one_se,
    fit_positive_affine,
    patient_balanced_mse,
)


def test_positive_affine_recovers_known_patient_balanced_relation():
    patients = np.array(["A"] * 2 + ["B"] * 8)
    prediction = np.linspace(-2, 2, 10)
    truth = 1.5 * prediction - 0.25
    result = fit_positive_affine(truth, prediction, patients, 0.1)
    assert result.fallback_reason is None
    assert result.slope > 0
    fitted_mse = patient_balanced_mse(truth[:, None], (prediction * result.slope + result.intercept)[:, None], patients)
    identity_mse = patient_balanced_mse(truth[:, None], prediction[:, None], patients)
    # lambda=0.1 intentionally shrinks the exact relation toward identity.
    assert fitted_mse < 0.002
    assert fitted_mse < identity_mse


def test_degenerate_prediction_uses_identity_fallback():
    result = fit_positive_affine(np.array([1.0, 2.0, 3.0]), np.ones(3), ["A", "A", "B"], 1.0)
    assert (result.slope, result.intercept, result.fallback_reason) == (1.0, 0.0, "pred_z_variance_below_1e-6")


def test_one_se_rule_prefers_stronger_regularization_when_eligible():
    patients = np.repeat(np.array(["A", "B", "C"]), 4)
    prediction = np.arange(12, dtype=float)[:, None]
    truth = prediction.copy()
    selected, evidence = choose_lambda_one_se(truth, prediction, patients)
    assert selected == max(evidence["one_se_eligible"])
    assert selected in {0.1, 1.0, 10.0}
