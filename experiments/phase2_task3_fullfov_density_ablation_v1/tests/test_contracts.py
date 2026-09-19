from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
import copy

import numpy as np
import pandas as pd
from PIL import Image
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

from config import load_config
from data import Normalization, make_point_table
from data_runtime import RuntimeData
from graph import build_split_graph, gather_neighbors
from label_audit import _load_true_only, run_label_audit
from minimal_checks import check_result_coverage
from model import assert_spatial_zero_initialized, build_model
from sampling import arm_indices, assert_random_equal_contract, counts_by_patient
from train_density import _load_model_checkpoint, _selection_eval, check_frozen_reference_compatibility, train_arm


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        split = pd.read_csv(cls.config["inputs"]["split_manifest"])
        mapping = pd.read_csv(cls.config["inputs"]["slide_mapping"]).set_index("patient_id")["slide_id"].to_dict()
        cls.local_table = make_point_table(
            patient_ids=split["patient"].astype(str).tolist(),
            spot_ids=split["patch_stem"].astype(str).tolist(),
            splits=split["split"].astype(str).tolist(),
            x=split["x"].tolist(), y=split["y"].tolist(),
            slide_ids=[mapping[str(patient)] for patient in split["patient"]],
            slide_status="verified", sort_identities=True,
        )

    def test_fullfov_and_model_contract(self):
        cfg = self.config
        self.assertEqual(cfg["data"]["full_fov_protocol"], "full_fov_224_bicubic_v1")
        self.assertEqual((cfg["data"]["resize"], cfg["data"]["crop"], cfg["data"]["augmentation"]), (224, None, False))
        self.assertEqual((cfg["data"]["input_dim"], cfg["model"]["hidden_dim"], cfg["data"]["output_dim"]), (1536, 256, 30))
        self.assertEqual(cfg["model"]["activation"], "gelu_exact")
        self.assertEqual(cfg["model"]["dropout"], .3)
        torch.manual_seed(45); model = build_model(cfg, "spatial"); assert_spatial_zero_initialized(model)

    def test_frozen_reference_checkpoint_has_no_spatial_bias(self):
        cfg = copy.deepcopy(self.config)
        self.assertFalse(cfg["model"]["spatial_bias"])
        accepted_model = build_model(cfg, "spatial")
        self.assertIsNone(accepted_model.spatial_head.bias)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "accepted.pt"
            torch.save({"model_state_dict": accepted_model.state_dict()}, checkpoint)
            incompatible = copy.deepcopy(cfg); incompatible["model"]["spatial_bias"] = True
            with self.assertRaisesRegex(RuntimeError, "spatial_head.bias"):
                _load_model_checkpoint(checkpoint, incompatible, torch.device("cpu"))
            loaded, _ = _load_model_checkpoint(checkpoint, cfg, torch.device("cpu"))
            self.assertIsNone(loaded.spatial_head.bias)
            cfg["paths"]["frozen_seed45_checkpoint"] = str(checkpoint)
            compatibility = check_frozen_reference_compatibility(cfg)
            self.assertEqual(compatibility["status"], "PASS")
            self.assertFalse(compatibility["spatial_bias"])

    def test_registry_updates_only_task3_target(self):
        registry = json.loads((PACKAGE_ROOT.parents[1] / "experiments" / "experiment_registry.json").read_text(encoding="utf-8"))
        task2 = next(item for item in registry["experiments"] if item["id"] == "phase2_gene_reconstruction_comparison_v001_20260904")
        task3 = next(item for item in registry["experiments"] if item["id"] == "phase2_spatial_density_comparison_v001_20260904")
        self.assertEqual((task2["status"], task2["baseline_experiment_id"]), ("planned", "phase2_softlink_local_v2"))
        self.assertNotIn("experiment_package", task2)
        self.assertEqual((task3["status"], task3["evidence_status"]), ("planned", "planned"))
        self.assertEqual(task3["baseline_experiment_id"], "phase2_fullfov_hpo_v1")
        self.assertEqual(task3["experiment_package"], "experiments/phase2_task3_fullfov_density_ablation_v1")

    def test_graph_optimizer_and_budget_contract(self):
        cfg = self.config
        self.assertEqual((cfg["graph"]["radius_in_native_steps"], cfg["graph"]["max_neighbors"], cfg["graph"]["distance_sigma"], cfg["graph"]["image_temperature"], cfg["graph"]["self_raw_weight"]), (2.0, 12, .5, 0.7716396069760084, .5))
        self.assertEqual((cfg["optimizer"]["name"], cfg["optimizer"]["learning_rate"], cfg["optimizer"]["weight_decay"], cfg["optimizer"]["batch_size"], cfg["optimizer"]["schedule"]), ("Adam", 1e-4, 0.0, 32, "constant"))
        self.assertEqual(cfg["training"]["max_updates"], 14800)
        self.assertFalse(cfg["training"]["early_stopping"])
        self.assertEqual(cfg["selection"]["primary"]["every_updates"] * cfg["selection"]["primary"]["opportunities"], 14800)
        self.assertEqual(cfg["selection"]["sensitivity"]["epochs"], 50)
        self.assertEqual(cfg["selection"]["sensitivity"]["opportunities"], 50)

    def test_three_arm_sampling_counts_and_seed_stream(self):
        cfg, table = self.config, self.local_table
        dense = arm_indices(table, "dense", 45, cfg)
        stride = arm_indices(table, "stride_2x2", 45, cfg)
        self.assertEqual((len(dense), len(stride)), (9472, 2383))
        expected_counts = {"HYZ15040": 327, "JFX": 460, "LMZ12939": 760, "TGC": 251, "XSL": 343, "ZHZ": 242}
        self.assertEqual(counts_by_patient(table, stride), expected_counts)
        random_masks = [arm_indices(table, "random_equal", seed, cfg) for seed in (45, 46, 47)]
        for mask in random_masks:
            self.assertEqual(counts_by_patient(table, mask), expected_counts)
        self.assertEqual(len({tuple(mask.tolist()) for mask in random_masks}), 3)
        assert_random_equal_contract(table, cfg, (45, 46, 47))

    def test_same_seed_initialization_is_exactly_paired(self):
        states = []
        for _ in range(3):
            torch.manual_seed(46)
            states.append({name: value.detach().clone() for name, value in build_model(self.config, "spatial").state_dict().items()})
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]))
            self.assertTrue(torch.equal(states[0][name], states[2][name]))

    def test_spatial_graph_and_single_optimizer_step(self):
        rng = np.random.default_rng(7)
        table = make_point_table(
            patient_ids=["P"] * 4,
            spot_ids=["x0_y0", "x1_y0", "x0_y1", "x20_y20"],
            splits=["train"] * 4,
            x=[0, 1, 0, 20], y=[0, 0, 1, 20],
            slide_ids=["S"] * 4,
            features=rng.normal(size=(4, 1536)).astype(np.float32),
            labels_z=rng.normal(size=(4, 30)).astype(np.float32),
            slide_status="verified",
            pathway_names=[f"p{i}" for i in range(30)],
        )
        geometry = pd.DataFrame([{"patient_id": "P", "slide_id": "S", "s": 1.0, "coordinate_unit": "native", "patch_coverage_size": 1.0, "source": "synthetic", "status": "verified"}])
        graph = build_split_graph(table, "train", self.config, geometry_table=geometry)
        for identity in graph.identities:
            node = graph.node(identity)
            self.assertAlmostEqual(node.a_self + sum(edge.a for edge in node.edges), 1.0, places=12)
            self.assertTrue(all(edge.distance <= 2.0 for edge in node.edges))
        isolated = graph.node(next(identity for identity in graph.identities if identity.spot_id == "x20_y20"))
        self.assertEqual(isolated.degree, 0)
        torch.manual_seed(45); model = build_model(self.config, "spatial"); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        neighbors = gather_neighbors(graph, table.identities)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        output = model(torch.as_tensor(table.features), torch.as_tensor(neighbors.neighbor_features(table)), torch.as_tensor(neighbors.neighbor_weights, dtype=torch.float32), torch.as_tensor(neighbors.neighbor_mask))["y_hat"]
        loss = torch.mean((output - torch.as_tensor(table.labels_z)) ** 2)
        self.assertTrue(torch.isfinite(loss)); loss.backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_planned_result_coverage_and_partial_detection(self):
        expected = [(arm, seed, split) for arm in self.config["training"]["arms"] for seed in self.config["training"]["seeds"] for split in ("internal_val", "external_test")]
        complete = [{"model": arm, "seed": seed, "split": split, "status": "succeeded"} for arm, seed, split in expected]
        self.assertTrue(check_result_coverage(expected, complete, batch_status="succeeded")["complete"])
        self.assertFalse(check_result_coverage(expected, complete[:-1], batch_status="partial")["complete"])

    def test_real_training_entrypoint_smoke_on_synthetic_data(self):
        cfg = copy.deepcopy(self.config)
        cfg["sampling"]["expected_dense"] = 24
        cfg["sampling"]["expected_sparse"] = 6
        rng = np.random.default_rng(19); pathways = tuple(f"p{i}" for i in range(30))
        patients = []; spots = []; splits = []; xs = []; ys = []; slides = []
        for p in range(6):
            for j, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                patients.append(f"P{p}"); spots.append(f"train_{p}_{j}"); splits.append("train"); xs.append(x); ys.append(y); slides.append(f"S{p}")
            for j, (x, y) in enumerate(((0, 0), (1, 0))):
                patients.append(f"P{p}"); spots.append(f"val_{p}_{j}"); splits.append("internal_val"); xs.append(x); ys.append(y); slides.append(f"S{p}")
        features = rng.normal(size=(len(patients), 1536)).astype(np.float32)
        labels = rng.normal(size=(len(patients), 30)).astype(np.float32)
        table = make_point_table(patient_ids=patients, spot_ids=spots, splits=splits, x=xs, y=ys, slide_ids=slides, features=features, labels_z=labels, slide_status="verified", pathway_names=pathways)
        norm = Normalization(pathways, np.zeros(30), np.ones(30), "train", 1, True, (-100., 100.), 24, "synthetic")
        runtime = RuntimeData(table, np.asarray(table.labels_z, dtype=np.float64), norm, "synthetic", "synthetic")
        with tempfile.TemporaryDirectory() as temporary:
            geometry_path = Path(temporary) / "geometry.csv"
            pd.DataFrame([{"patient_id": f"P{p}", "slide_id": f"S{p}", "s": 1.0, "coordinate_unit": "native", "patch_coverage_size": 1.0, "source": "synthetic", "status": "verified"} for p in range(6)]).to_csv(geometry_path, index=False)
            cfg["inputs"]["slide_geometry"] = str(geometry_path)
            summary = train_arm(cfg, runtime, "dense", 45, Path(temporary) / "run", Path(temporary) / "weights", mode="smoke", requested_device="cpu")
            self.assertEqual(summary["status"], "smoke_complete_non_evidence")
            self.assertEqual(summary["fixed_updates"], 2)
            self.assertEqual(summary["selection_opportunities"], {"equal_updates": 2, "equal_50_epochs": 1})
            self.assertTrue(all(Path(path).is_file() for path in summary["checkpoints"].values()))

    def test_external_is_not_a_selection_input(self):
        self.assertFalse(self.config["selection"]["external_used_for_selection"])
        selection_source = inspect.getsource(_selection_eval)
        self.assertNotIn("external", selection_source)
        train_source = inspect.getsource(train_arm)
        self.assertLess(train_source.index("checkpoints ="), train_source.index("graph_external ="))


class LabelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_reader_selects_true_columns_only(self):
        audit = self.config["label_audit"]
        root = Path(audit["source_root"])
        pathways = json.loads(Path(self.config["inputs"]["zscore_manifest"]).read_text(encoding="utf-8-sig"))["pathway_names"]
        frame = _load_true_only(root / audit["original_internal"], audit["join_keys"], pathways)
        self.assertFalse(any(column.startswith("pred_") or column.startswith("pred_raw_") for column in frame.columns))
        self.assertEqual(len(frame), 1078)
        self.assertEqual(sum(column.startswith("true_") and not column.startswith("true_raw_") for column in frame.columns), 30)
        self.assertEqual(sum(column.startswith("true_raw_") for column in frame.columns), 30)

    def test_rebuild_is_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_label_audit(self.config, temporary)
            summary = result["summary"]
            self.assertEqual((summary["n_internal"], summary["n_external"], summary["n_total_aligned"], summary["n_pathways"]), (1078, 1039, 2117, 30))
            self.assertAlmostEqual(summary["overall_internal_patient_pathway_mean_pcc"], 0.7437947834175882, places=12)
            self.assertAlmostEqual(summary["overall_external_pathway_mean_pcc"], 0.7517970361669122, places=12)
            frame = pd.read_csv(result["csv"])
            self.assertEqual(len(frame), 30)
            self.assertEqual(frame.iloc[:3]["pathway"].tolist(), ["Apoptosis", "mTOR_Signaling", "Unfolded_Protein_Response"])
            manifest = json.loads(Path(result["source_manifest"]).read_text(encoding="utf-8"))
            self.assertIn("禁止pred_*", manifest["loaded_columns_policy"])
            self.assertFalse(manifest["hashes_computed"])
            self.assertFalse(manifest["original_files_modified"])
            with Image.open(result["png"]) as image:
                self.assertGreaterEqual(image.width, 4000)
                self.assertGreaterEqual(image.height, 3500)

    def test_old_two_branch_truth_drift_is_recorded(self):
        summary = json.loads((PACKAGE_ROOT / "analysis" / "task2_true_label_comparison_summary.json").read_text(encoding="utf-8"))
        drift = summary["confirmed_legacy_common_truth_drift"]
        self.assertGreater(drift["internal"]["n_nonzero_z_cells"], 0)
        self.assertGreater(drift["external"]["n_nonzero_z_cells"], 0)


if __name__ == "__main__":
    unittest.main()
