from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.explorations.mpp2_r2_root_cause_audit import (
    AuditError,
    bootstrap_mean_ci,
    metrics,
    oracle_intercept,
    oracle_location_scale,
    oracle_positive_affine,
    require_sha256,
    r2_score,
    sha256_file,
    validate_prediction_frame,
)


class RootCauseMathTests(unittest.TestCase):
    def test_pure_intercept_shift_is_removed(self) -> None:
        truth = np.arange(20, dtype=float)
        prediction = truth + 7.0
        corrected = oracle_intercept(truth, prediction)
        self.assertLess(r2_score(truth, prediction), 1.0)
        self.assertAlmostEqual(r2_score(truth, corrected), 1.0)

    def test_pure_scale_shift_is_removed(self) -> None:
        truth = np.linspace(-2.0, 2.0, 21)
        prediction = 0.25 * truth + 3.0
        corrected = oracle_location_scale(truth, prediction)
        self.assertTrue(np.allclose(corrected, truth))

    def test_positive_affine_ceiling_with_noise(self) -> None:
        rng = np.random.default_rng(42)
        prediction = np.linspace(-2.0, 2.0, 200)
        truth = 2.0 * prediction + 1.0 + rng.normal(0, 0.4, len(prediction))
        corrected, slope, _ = oracle_positive_affine(truth, prediction)
        self.assertGreater(slope, 0)
        self.assertGreater(r2_score(truth, corrected), r2_score(truth, prediction))
        self.assertLess(r2_score(truth, corrected), 1.0)

    def test_zero_variance_truth_returns_nan_r2(self) -> None:
        truth = np.ones(5)
        prediction = np.arange(5, dtype=float)
        self.assertTrue(np.isnan(r2_score(truth, prediction)))

    def test_mse_bias_variance_identity(self) -> None:
        truth = np.array([1.0, 2.0, 4.0, 8.0])
        prediction = np.array([2.0, 2.0, 5.0, 9.0])
        result = metrics(truth, prediction)
        self.assertAlmostEqual(
            result.mse, result.error_variance + result.bias**2, places=14
        )

    def test_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci([1.0, 2.0, 3.0], replicates=1000)
        second = bootstrap_mean_ci([1.0, 2.0, 3.0], replicates=1000)
        self.assertEqual(first, second)

    def test_hash_changes_when_content_changes(self) -> None:
        path = Path(__file__)
        expected = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(path), expected)

    def test_audit_error_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(AuditError, RuntimeError))

    def test_duplicate_key_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["A", "A"],
                "patch_id": ["p1", "p1"],
                "x": [1, 1],
                "y": [2, 2],
                "true_pw": [0.0, 1.0],
                "pred_pw": [0.0, 1.0],
            }
        )
        with self.assertRaisesRegex(AuditError, "键重复"):
            validate_prediction_frame(frame, ["pw"], ["true_", "pred_"], "test")

    def test_missing_column_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {"patient_id": ["A"], "patch_id": ["p1"], "x": [1], "y": [2]}
        )
        with self.assertRaisesRegex(AuditError, "缺列"):
            validate_prediction_frame(frame, ["pw"], ["true_", "pred_"], "test")

    def test_nan_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["A"],
                "patch_id": ["p1"],
                "x": [1],
                "y": [2],
                "true_pw": [np.nan],
                "pred_pw": [0.0],
            }
        )
        with self.assertRaisesRegex(AuditError, "NaN/Inf"):
            validate_prediction_frame(frame, ["pw"], ["true_", "pred_"], "test")

    def test_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditError, "SHA-256"):
            require_sha256(Path(__file__), "0" * 64, "test asset")


if __name__ == "__main__":
    unittest.main()
