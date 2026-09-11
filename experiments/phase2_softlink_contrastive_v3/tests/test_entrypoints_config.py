"""Static package-contract tests; never train, download or inspect large inputs."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
import unittest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_DIR / "config.json"
PACKAGE_JSON_PATH = PACKAGE_DIR / "package.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_runner():
    spec = importlib.util.spec_from_file_location("phase2_runner_for_contract_test", PACKAGE_DIR / "runner.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntryPointAndConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_json(CONFIG_PATH)
        cls.package = load_json(PACKAGE_JSON_PATH)
        cls.original_text = (PACKAGE_DIR / "run_original_split.ps1").read_text(encoding="utf-8-sig")
        cls.lopo_text = (PACKAGE_DIR / "run_lopo6.ps1").read_text(encoding="utf-8-sig")
        cls.shared_text = (PACKAGE_DIR / "run.ps1").read_text(encoding="utf-8-sig")

    def test_package_and_design_version(self):
        self.assertEqual(self.package["experiment_id"], "phase2_softlink_contrastive_v3")
        self.assertEqual(self.package["plan_version"], "v4")
        self.assertEqual(self.config["experiment_id"], self.package["experiment_id"])
        self.assertEqual(self.config["plan_version"], "v4")
        self.assertEqual(self.package["entrypoint"], "runner.py")
        self.assertFalse(self.package["demo_only"])

    def test_config_covers_v4_design(self):
        self.assertEqual(
            self.config["arms"]["primary"],
            [
                "frozen_regression",
                "frozen_centered_contrastive",
                "r8_regression",
                "r8_global_contrastive",
                "r8_centered_contrastive",
            ],
        )
        self.assertEqual(self.config["arms"]["capacity_diagnostic"], ["r2_regression"])
        adaptation = self.config["adaptation"]
        self.assertEqual(adaptation["adapted_blocks"], [20, 21, 22, 23])
        self.assertTrue(adaptation["same_scope_for_r2"])
        self.assertEqual(self.config["training"]["fixed_epochs"], 5)
        self.assertEqual(self.config["training"]["common_warmup"]["epochs"], 60)
        self.assertEqual(self.config["training"]["common_warmup"]["max_epochs"], 60)
        self.assertEqual(self.config["training"]["batch_size_candidates"], [64, 128])
        self.assertEqual(self.config["training"]["learning_rates"]["lora_candidates"], [1e-5, 3e-5, 1e-4])
        self.assertEqual(self.config["training"]["learning_rates"]["h_lr"], 3e-5)
        self.assertEqual(self.config["training"]["learning_rates"]["c_lr"], 3e-5)
        self.assertEqual(self.config["training"]["learning_rates"]["t_lr"], 3e-4)
        self.assertEqual(self.config["teacher"]["rho"], 0.25)
        self.assertEqual(self.config["teacher"]["tau_z"], 0.1)
        self.assertEqual(self.config["graph"]["radius_in_native_steps"], 1.5)
        self.assertEqual(self.config["graph"]["max_neighbors"], 8)
        self.assertTrue(self.config["selection"]["best_epoch_sensitivity"]["enabled"])
        self.assertEqual(self.config["selection"]["formal_start_epoch"], 6)
        self.assertEqual(self.config["selection"]["early_stop_count_start_epoch"], 16)
        self.assertEqual(self.config["selection"]["patience"], 10)
        self.assertEqual(self.config["selection"]["early_stop_patience"], 10)
        self.assertFalse(self.config["disabled_features"]["EVA"]["enabled"])
        self.assertFalse(self.config["disabled_features"]["label_queue"]["enabled"])
        self.assertFalse(self.config["disabled_features"]["activation_centering"]["enabled"])
        short_test = self.config["resource_short_test"]
        self.assertEqual(short_test["warmup_updates_max"], 10)
        self.assertEqual(short_test["timed_updates"], 30)
        self.assertEqual(short_test["minimum_memory_margin_bytes"], 2 * 1024**3)
        self.assertEqual(short_test["insufficient_memory_margin_policy"], "warn_and_continue_with_batch64")
        self.assertTrue(short_test["shared_memory_slowdown_is_non_blocking"])
        self.assertTrue(self.config["training"]["gradient_checkpointing"]["enabled"])
        self.assertEqual(self.config["training"]["precision"]["stage1"], "bf16")
        self.assertEqual(self.config["training"]["precision"]["softmax_and_normalization"], "float32")

    def test_path_contract_separates_code_runs_and_weights(self):
        runtime = self.config["runtime"]
        self.assertEqual(runtime["server_code_directory"], r"D:\AIPatho\qzs\code\phase2_softlink_contrastive_v3")
        self.assertEqual(runtime["run_layout"], r"D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\<batch>\<run>")
        self.assertEqual(runtime["weights_layout"], r"D:\AIPatho\qzs\weights\phase2_softlink_contrastive_v3\<batch>\<run>")
        self.assertNotEqual(runtime["runs_root"], runtime["weights_root"])
        self.assertTrue(self.config["data"]["large_inputs_are_absolute"])
        for key in ("patch_images_root", "labels_root", "feature_cache_root", "uni2_h_weights"):
            self.assertRegex(self.config["data"][key], r"^[A-Za-z]:\\")
        self.assertEqual(self.config["data"]["uni2_h_checkpoint"], r"D:\AIPatho\shared\.cache\huggingface\hub\models--MahmoodLab--UNI2-h\snapshots\d517a8dd47902dd7c308b3c36f63bce47e7b9a43")
        self.assertNotIn("<", self.config["data"]["uni2_h_checkpoint"])
        self.assertIn("{patient}_ssGSEA.csv", self.config["data"]["raw_label_template"])
        self.assertIn("refit_zscore_on_training_blocks", self.config["data"]["label_fit_policy"])

    def test_protocol_defaults_and_wrapper_isolation(self):
        original = self.config["protocol_defaults"]["original"]
        lopo6 = self.config["protocol_defaults"]["lopo6"]
        self.assertEqual(original["protocol"], "original")
        self.assertEqual(original["seeds"], [42])
        self.assertFalse(original["lopo"])
        self.assertEqual(lopo6["protocol"], "lopo6")
        self.assertEqual(len(lopo6["folds"]), 6)
        self.assertTrue(lopo6["lopo"])

        self.assertIn("Protocol = 'original'", self.original_text)
        self.assertIn("[int[]]$Seeds = @(42)", self.original_text)
        self.assertIn("[string]$ResumeFrom", self.original_text)
        self.assertIn("ResumeFrom = $ResumeFrom", self.original_text)
        self.assertNotIn("run_lopo6.ps1", self.original_text)
        self.assertIn("Protocol = 'lopo6'", self.lopo_text)
        self.assertIn("resume_history.jsonl", self.shared_text)
        self.assertIn("code_version = $package.code_version", self.shared_text)
        self.assertIn("event = 'completed'", self.shared_text)
        self.assertIn("[string[]]$Folds = @('HYZ15040', 'JFX', 'LMZ12939', 'TGC', 'XSL', 'ZHZ')", self.lopo_text)
        self.assertIn("[int[]]$Seeds = @(42, 43, 44)", self.lopo_text)
        self.assertIn("[string]$ResumeFrom", self.lopo_text)
        self.assertIn("ResumeFrom = $ResumeFrom", self.lopo_text)
        self.assertIn("if ($Protocol -eq 'lopo6')", self.shared_text)
        self.assertIn("$Folds = @()", self.shared_text)
        self.assertIn("$batchRoot", self.shared_text)
        self.assertIn("$weightBatchRoot", self.shared_text)
        self.assertIn("$Resume -and -not $hasResumeFrom", self.shared_text)
        self.assertIn("-not $Resume -and $hasResumeFrom", self.shared_text)
        self.assertIn("existingRecord.experiment_id", self.shared_text)
        self.assertIn("existingRecord.protocol", self.shared_text)
        self.assertIn("existingRecord.batch_id", self.shared_text)
        self.assertIn("Do not rewrite run.json/config/package on resume", self.shared_text)
        self.assertIn("Ensure-ErrorLog", self.shared_text)

    def test_runner_is_thin_and_calls_shared_orchestrator(self):
        runner = load_runner()
        calls = []

        fake_src = types.ModuleType("src")
        fake_src.__path__ = []
        fake_orchestrator = types.ModuleType("src.orchestrator")

        def execute_plan(**kwargs):
            calls.append(kwargs)
            return {"exit_code": 0}

        fake_orchestrator.execute_plan = execute_plan
        old_src = sys.modules.get("src")
        old_orchestrator = sys.modules.get("src.orchestrator")
        sys.modules["src"] = fake_src
        sys.modules["src.orchestrator"] = fake_orchestrator
        try:
            result = runner.main(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--run-dir",
                    str(PACKAGE_DIR / "_test_run"),
                    "--weights-dir",
                    str(PACKAGE_DIR / "_test_weights"),
                    "--protocol",
                    "original",
                    "--seeds",
                    "42",
                    "--plan-only",
                ]
            )
        finally:
            if old_src is None:
                sys.modules.pop("src", None)
            else:
                sys.modules["src"] = old_src
            if old_orchestrator is None:
                sys.modules.pop("src.orchestrator", None)
            else:
                sys.modules["src.orchestrator"] = old_orchestrator

        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["protocol"], "original")
        self.assertEqual(calls[0]["seeds"], (42,))
        self.assertTrue(calls[0]["plan_only"])
        self.assertFalse(calls[0]["resume"])


if __name__ == "__main__":
    unittest.main()
