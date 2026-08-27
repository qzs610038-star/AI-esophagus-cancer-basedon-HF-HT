from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explorations.phase2_phase3_reanalysis_v002 import run_reanalysis as reanalysis


run_reanalysis = reanalysis.run_reanalysis


class Phase2Phase3ReanalysisV002Tests(unittest.TestCase):
    def test_patient_balanced_formula_ignores_unequal_spot_counts(self) -> None:
        patient_a_truth = np.array([0.0, 1.0, 2.0])
        patient_b_truth = np.arange(9.0)
        table = pd.DataFrame(
            {
                "patient": ["A"] * 3 + ["B"] * 9,
                "truth_z__P1": np.concatenate([patient_a_truth, patient_b_truth]),
                "prediction_z__P1": np.concatenate(
                    [patient_a_truth, patient_b_truth[::-1]]
                ),
                "truth_raw__P1": np.concatenate([patient_a_truth, patient_b_truth]),
                "prediction_raw__P1": np.concatenate(
                    [patient_a_truth, patient_b_truth[::-1]]
                ),
            }
        )

        balanced, patient_level = reanalysis.compute_patient_balanced_pathway_metrics(
            table, arm="SYNTHETIC", split="synthetic"
        )

        self.assertEqual(len(balanced), 1)
        self.assertEqual(len(patient_level), 2)
        self.assertEqual(balanced.loc[0, "aggregation"], "patient_balanced_mean")
        self.assertEqual(balanced.loc[0, "n_patients"], 2)
        self.assertEqual(balanced.loc[0, "n_spots"], 12)
        self.assertAlmostEqual(balanced.loc[0, "pcc"], 0.0, places=12)
        self.assertAlmostEqual(
            balanced.loc[0, "pcc"], patient_level["pcc"].mean(), places=12
        )
        pooled_pcc = np.corrcoef(table["truth_z__P1"], table["prediction_z__P1"])[0, 1]
        self.assertNotAlmostEqual(balanced.loc[0, "pcc"], pooled_pcc, places=6)

    def test_run_reanalysis_writes_w007_pathway_metric_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            summary = run_reanalysis(PROJECT_ROOT, output_dir)

            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            metrics = pd.read_csv(output_dir / "w007_pathway_metrics.csv")
            patient_metrics = pd.read_csv(
                output_dir / "w007_patient_pathway_metrics.csv"
            )

            self.assertEqual(summary["analysis_id"], "phase2_phase3_reanalysis_v002")
            self.assertEqual(metadata["analysis_class"], "exploratory_reanalysis")
            self.assertEqual(metadata["evidence_status"], "pending_user_review")
            self.assertFalse(metadata["registry_mutated"])
            self.assertEqual(
                metadata["w007_aggregation"],
                "patient_level_then_equal_patient_arithmetic_mean",
            )
            self.assertEqual(metadata["w007_internal_val_patient_count"], 6)
            self.assertEqual(metadata["w007_xzy_patient_count"], 1)
            self.assertEqual(len(metrics), 4 * 2 * 30)
            self.assertEqual(set(metrics["arm"]), {"FBR", "RCC", "HCR", "CPGCR"})
            self.assertEqual(set(metrics["split"]), {"internal_val", "XZY"})
            self.assertEqual(set(metrics["aggregation"]), {"patient_balanced_mean"})
            self.assertEqual(
                set(metrics.loc[metrics["split"].eq("internal_val"), "n_patients"]),
                {6},
            )
            self.assertEqual(
                set(metrics.loc[metrics["split"].eq("XZY"), "n_patients"]), {1}
            )
            self.assertEqual(len(patient_metrics), 4 * (6 + 1) * 30)
            self.assertEqual(set(patient_metrics["aggregation"]), {"patient_level"})
            self.assertEqual(
                set(metrics["analysis_class"]), {"exploratory_reanalysis"}
            )
            self.assertEqual(set(metrics["evidence_status"]), {"pending_user_review"})

            required = {
                "pcc",
                "ccc",
                "z_rmse",
                "raw_r2",
                "spearman",
                "pcc_minus_ccc",
                "pcc_squared_minus_raw_r2",
                "z_mse",
                "bias_squared",
                "scale_mismatch_squared",
                "decorrelation_component",
                "decomposition_sum",
                "decomposition_residual",
            }
            self.assertTrue(required.issubset(metrics.columns))
            self.assertTrue(np.isfinite(metrics[list(required)]).all().all())
            self.assertTrue(
                np.isfinite(patient_metrics[list(required)]).all().all()
            )
            expected_balanced = (
                patient_metrics.groupby(["arm", "split", "pathway"], as_index=False)[
                    list(required)
                ]
                .mean()
                .sort_values(["arm", "split", "pathway"])
                .reset_index(drop=True)
            )
            observed_balanced = (
                metrics.sort_values(["arm", "split", "pathway"])
                .reset_index(drop=True)
            )
            self.assertTrue(
                np.allclose(
                    observed_balanced[list(required)],
                    expected_balanced[list(required)],
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    metrics["z_mse"],
                    metrics["decomposition_sum"],
                    rtol=0.0,
                    atol=1e-10,
                )
            )
            self.assertTrue(
                np.allclose(metrics["decomposition_residual"], 0.0, atol=1e-10)
            )

    def test_run_reanalysis_writes_w004_and_spatial_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_reanalysis(PROJECT_ROOT, output_dir)

            per_fit = pd.read_csv(output_dir / "w004_per_fit_metrics.csv")
            pooled = pd.read_csv(output_dir / "w004_pooled_metrics.csv")
            auc_differences = pd.read_csv(
                output_dir / "w004_fixed_auc_differences.csv"
            )
            case_posthoc = pd.read_csv(output_dir / "w004_case_posthoc.csv")
            w007_spatial = pd.read_csv(output_dir / "w007_spatial_metrics.csv")
            w007_hotspots = pd.read_csv(output_dir / "w007_hotspot_summary.csv")
            w004_spatial = pd.read_csv(output_dir / "w004_spatial_status.csv")
            review_report = (output_dir / "review_report.md").read_text(
                encoding="utf-8"
            )

            self.assertEqual(len(per_fit), 4 * 2 * 4 * 5 * 2)
            self.assertEqual(len(pooled), 4 * 2 * 2)
            self.assertEqual(set(per_fit["arm"]), {"ABS", "ORD", "SPATIAL_IDENTITY", "SPATIAL_SHUFFLE"})
            self.assertEqual(set(per_fit["task"]), {"pCR", "MPR"})
            self.assertEqual(set(per_fit["split"]), {"validation", "test"})
            for table in (per_fit, pooled):
                self.assertTrue(
                    {"auc", "auprc", "brier", "calibration_intercept", "calibration_slope", "calibration_status"}.issubset(
                        table.columns
                    )
                )
                self.assertTrue(np.isfinite(table[["auc", "auprc", "brier"]]).all().all())
                self.assertTrue(table["auc"].between(0.0, 1.0).all())
                self.assertTrue(table["auprc"].between(0.0, 1.0).all())
                self.assertTrue(table["brier"].between(0.0, 1.0).all())
                computed = table["calibration_status"].eq("computed")
                self.assertTrue(
                    np.isfinite(
                        table.loc[computed, ["calibration_intercept", "calibration_slope"]]
                    ).all().all()
                )

            self.assertEqual(len(auc_differences), 4)
            self.assertEqual(
                set(auc_differences["contrast"]),
                {"ORD_MINUS_ABS", "SPATIAL_IDENTITY_MINUS_SPATIAL_SHUFFLE"},
            )
            self.assertEqual(set(auc_differences["task"]), {"pCR", "MPR"})
            self.assertEqual(set(auc_differences["n_paired_fits"]), {20})
            self.assertGreater(len(case_posthoc), 0)
            self.assertEqual(set(case_posthoc["interpretation"]), {"post_hoc_descriptive"})

            self.assertEqual(set(w007_spatial["computation_status"]), {"computed"})
            self.assertTrue(np.isfinite(w007_spatial["morans_i"]).all())
            self.assertEqual(set(w007_hotspots["computation_status"]), {"computed_descriptive"})
            self.assertEqual(set(w004_spatial["computation_status"]), {"not_computable"})
            self.assertEqual(
                set(w004_spatial["reason"]),
                {"prediction_csv_has_no_coordinate_columns"},
            )
            self.assertIn("exploratory_reanalysis", review_report)
            self.assertIn("pending_user_review", review_report)
            self.assertIn("prediction_csv_has_no_coordinate_columns", review_report)
            self.assertIn("Registry", review_report)

            for table in (
                per_fit,
                pooled,
                auc_differences,
                case_posthoc,
                w007_spatial,
                w007_hotspots,
                w004_spatial,
            ):
                self.assertEqual(set(table["analysis_class"]), {"exploratory_reanalysis"})
                self.assertEqual(set(table["evidence_status"]), {"pending_user_review"})


if __name__ == "__main__":
    unittest.main()
