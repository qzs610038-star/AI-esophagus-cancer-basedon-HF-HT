import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_xzy_prediction import (
    align_prediction_rows,
    classify_pathway_pattern,
    compute_coordinate_cv_r2,
    compute_rank_fidelity,
    resolve_prediction_output_root,
)


class TestXzyPredictionAnalysis(unittest.TestCase):
    def test_01_aligns_returned_rows_to_audited_coordinates(self):
        original = pd.DataFrame({
            "true_A": [0.1, 0.2],
            "pred_A": [0.15, 0.25],
        })
        bridge = pd.DataFrame({
            "patient_id": ["XZY", "XZY"],
            "patch_id": ["patch_x0_y0", "patch_x1_y0"],
            "x": [0, 1], "y": [0, 0],
            "true_A": [0.10000001, 0.19999999],
            "pred_base_A": [0.15000001, 0.24999999],
        })
        aligned, qc = align_prediction_rows(original, bridge, tolerance=1e-5)
        self.assertEqual(aligned["barcode"].tolist(), ["patch_x0_y0", "patch_x1_y0"])
        self.assertEqual(qc["pathway_count"], 1)
        self.assertLess(qc["max_prediction_abs_difference"], 1e-5)

    def test_02_rejects_unproven_row_order(self):
        original = pd.DataFrame({"true_A": [0.1, 9.0], "pred_A": [0.2, 0.3]})
        bridge = pd.DataFrame({
            "patient_id": ["XZY", "XZY"],
            "patch_id": ["patch_x0_y0", "patch_x1_y0"], "x": [0, 1], "y": [0, 0],
            "true_A": [0.1, 0.2], "pred_base_A": [0.2, 0.3],
        })
        with self.assertRaises(ValueError):
            align_prediction_rows(original, bridge, tolerance=1e-5)

    def test_03_rank_fidelity_distinguishes_identity_and_reversal(self):
        truth = np.arange(10, dtype=float)
        same = compute_rank_fidelity(truth, truth)
        reverse = compute_rank_fidelity(truth, truth[::-1])
        self.assertAlmostEqual(same["spearman_rho"], 1.0)
        self.assertAlmostEqual(same["top10_jaccard"], 1.0)
        self.assertAlmostEqual(reverse["spearman_rho"], -1.0)
        self.assertAlmostEqual(reverse["top10_jaccard"], 0.0)

    def test_04_coordinate_cv_detects_a_smooth_coordinate_signal(self):
        coords = np.array([[x, y] for x in range(10) for y in range(10)], dtype=float)
        values = 2.0 * coords[:, 0] - 0.5 * coords[:, 1]
        result = compute_coordinate_cv_r2(values, coords, block_pixels=2.0, n_splits=5)
        self.assertGreater(result["cv_r2"], 0.95)
        self.assertGreaterEqual(result["n_blocks"], 20)

    def test_05_pattern_classification_keeps_direction_and_error_structure_separate(self):
        self.assertEqual(classify_pathway_pattern(0.4, 0.2, 0.3, 0.01),
                         "prediction_less_smooth_structured_error")
        self.assertEqual(classify_pathway_pattern(0.2, 0.4, 0.3, 0.01),
                         "prediction_more_smooth_structured_error")
        self.assertEqual(classify_pathway_pattern(0.2, 0.4, 0.0, 0.5),
                         "no_significant_structured_error")

    def test_06_prediction_outputs_are_isolated_below_v001(self):
        config = {"output_root": "analysis/mpp2_raw_spatial_rank_v001"}
        self.assertEqual(
            resolve_prediction_output_root(config),
            Path("analysis/mpp2_raw_spatial_rank_v001/xzy_baseline_prediction"),
        )


if __name__ == "__main__":
    unittest.main()
