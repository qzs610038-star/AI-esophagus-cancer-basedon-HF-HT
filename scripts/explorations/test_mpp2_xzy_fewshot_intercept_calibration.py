from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.explorations.mpp2_xzy_fewshot_intercept_calibration import (
    CalibrationError,
    apply_intercepts,
    choose_anchor_indices,
    evaluate_split,
    fit_intercepts,
    summarize_pathways,
)


class FewShotInterceptCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "patient_id": ["XZY"] * 20,
                "patch_id": [f"p{i:02d}" for i in range(20)],
                "x": np.tile(np.arange(5), 4),
                "y": np.repeat(np.arange(4), 5),
            }
        )

    def test_random_anchor_split_is_exact_disjoint_and_deterministic(self) -> None:
        first = choose_anchor_indices(self.frame, 0.2, 42, "random")
        second = choose_anchor_indices(self.frame, 0.2, 42, "random")
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 4)
        self.assertEqual(len(np.unique(first)), 4)
        self.assertTrue(np.all((first >= 0) & (first < len(self.frame))))

    def test_spatial_stratified_anchor_split_is_deterministic(self) -> None:
        first = choose_anchor_indices(
            self.frame, 0.25, 20260725, "spatial_stratified"
        )
        second = choose_anchor_indices(
            self.frame, 0.25, 20260725, "spatial_stratified"
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 5)

    def test_intercepts_are_fit_from_anchor_rows_only(self) -> None:
        prediction = np.zeros((6, 2), dtype=float)
        truth = np.array(
            [
                [2.0, -3.0],
                [2.0, -3.0],
                [2.0, -3.0],
                [999.0, 999.0],
                [999.0, 999.0],
                [999.0, 999.0],
            ]
        )
        intercepts = fit_intercepts(truth, prediction, np.array([0, 1, 2]))
        self.assertTrue(np.allclose(intercepts, [2.0, -3.0]))

    def test_pure_shift_improves_heldout_r2(self) -> None:
        prediction = np.column_stack(
            [np.linspace(-2.0, 2.0, 20), np.linspace(1.0, 5.0, 20)]
        )
        truth = prediction + np.array([3.0, -2.0])
        anchors = np.array([0, 5, 10, 15])
        intercepts = fit_intercepts(truth, prediction, anchors)
        calibrated = apply_intercepts(prediction, intercepts)
        result = evaluate_split(
            truth,
            prediction,
            calibrated,
            anchors,
            raw_stds=np.ones(2),
        )
        self.assertAlmostEqual(result["calibrated_mean_r2"], 1.0)
        self.assertGreater(result["delta_mean_r2"], 0.0)
        self.assertEqual(result["n_calibration"], 4)
        self.assertEqual(result["n_test"], 16)

    def test_invalid_fraction_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            choose_anchor_indices(self.frame, 0.0, 42, "random")
        with self.assertRaises(CalibrationError):
            choose_anchor_indices(self.frame, 1.0, 42, "random")

    def test_unknown_sampling_method_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            choose_anchor_indices(self.frame, 0.2, 42, "unknown")

    def test_direct_script_cli_is_runnable(self) -> None:
        script = (
            Path(__file__).parent / "mpp2_xzy_fewshot_intercept_calibration.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--result-dir", completed.stdout)

    def test_pathway_summary_reports_repeat_distribution(self) -> None:
        rows = pd.DataFrame(
            {
                "sampling_method": ["spatial_stratified"] * 3,
                "calibration_fraction": [0.1] * 3,
                "pathway": ["pw"] * 3,
                "intercept_z": [1.0, 2.0, 3.0],
                "intercept_raw": [10.0, 20.0, 30.0],
                "baseline_raw_r2": [-1.0, -1.0, -1.0],
                "calibrated_raw_r2": [0.0, 0.2, 0.4],
                "delta_raw_r2": [1.0, 1.2, 1.4],
                "baseline_raw_mae": [5.0, 5.0, 5.0],
                "calibrated_raw_mae": [3.0, 2.0, 1.0],
            }
        )
        summary = summarize_pathways(rows, "spatial_stratified", 0.1)
        self.assertEqual(summary.loc[0, "pathway"], "pw")
        self.assertAlmostEqual(summary.loc[0, "delta_raw_r2_median"], 1.2)
        self.assertAlmostEqual(summary.loc[0, "calibrated_raw_r2_median"], 0.2)


if __name__ == "__main__":
    unittest.main()
