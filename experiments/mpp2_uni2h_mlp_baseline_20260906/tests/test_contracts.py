"""Local contract tests. No repository imports and no server paths."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
sys.path.insert(0, str(SRC))

from export import write_prediction_table
from run_presets import resolve_run_kind
from sampling.registry import load_sampler
from targets.gene_expression import MISSING
from targets.protocol import MissingTargetContract, TargetSpec
from targets.registry import load_target


class ContractTest(unittest.TestCase):
    def test_run_mode_presets(self):
        self.assertEqual(resolve_run_kind("smoke")["num_epochs"], 2)
        self.assertEqual(resolve_run_kind("full")["num_epochs"], 50)
        self.assertEqual(resolve_run_kind("predict")["mode"], "predict")
        with self.assertRaises(ValueError):
            resolve_run_kind("debug")

    def test_gene_target_fails_with_missing_list(self):
        with self.assertRaises(MissingTargetContract) as ctx:
            load_target("gene_expression", PACKAGE / "assets" / "group_2" / "zscore_manifest.json").spec()
        text = str(ctx.exception)
        for item in MISSING:
            self.assertIn(item, text)

    def test_pathway_names_come_from_spec(self):
        target = load_target("pathway_ssgsea", PACKAGE / "assets" / "group_2" / "zscore_manifest.json")
        spec = target.spec()
        self.assertEqual(spec.n_outputs, 30)
        self.assertEqual(spec.names[0], "tls")
        self.assertNotIn("30", str(spec.n_outputs) if False else spec.target_id)

    def test_spatial_stride_only_filters_train(self):
        rows = []
        for x in (0, 1, 2, 3):
            for y in (0, 1, 2, 3):
                rows.append({"patient": "P1", "patch_stem": f"patch_x{x}_y{y}", "x": x, "y": y, "split": "train"})
        rows.append({"patient": "P1", "patch_stem": "patch_x9_y9", "x": 9, "y": 9, "split": "internal_val"})
        rows.append({"patient": "XZY", "patch_stem": "patch_x1_y1", "x": 1, "y": 1, "split": "external"})
        frame = pd.DataFrame(rows)
        filtered = load_sampler("spatial_stride", {"stride_x": 2, "stride_y": 2}).filter_manifest(frame, "train")
        self.assertEqual(int((filtered["split"] == "internal_val").sum()), 1)
        self.assertEqual(int((filtered["split"] == "external").sum()), 1)
        self.assertLess(int((filtered["split"] == "train").sum()), int((frame["split"] == "train").sum()))
        dense = load_sampler("dense", {}).filter_manifest(frame, "train")
        self.assertEqual(len(dense), len(frame))

    def test_export_columns_follow_target_spec(self):
        spec = TargetSpec("demo", ["a", "b"], 2, "demo", "test")
        records = [{"patient_id": "P", "patch_id": "patch_x0_y0", "x": 0, "y": 0, "split": "train"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.csv"
            write_prediction_table(
                path, records,
                np.array([[1.0, 2.0]]),
                np.array([[1.1, 2.1]]),
                spec, "uni2h_mlp", "dense", "v001", "ckpt",
            )
            columns = pd.read_csv(path).columns.tolist()
        for name in ("patient_id", "patch_id", "x", "y", "split", "true_a", "pred_a", "true_b", "pred_b", "target_id"):
            self.assertIn(name, columns)
        self.assertNotIn("true_tls", columns)


if __name__ == "__main__":
    unittest.main()
