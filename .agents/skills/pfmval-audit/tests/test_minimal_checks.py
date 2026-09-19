"""Behavior checks for the portable helpers; no training or project assets."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "assets" / "minimal_checks.py"
spec = importlib.util.spec_from_file_location("minimal_checks", HELPER)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class IdentityTests(unittest.TestCase):
    def test_valid_composite_keys_and_pathway_order(self):
        keys = [("patient_a", "spot1"), ("patient_b", "spot1")]
        self.assertEqual(checks.check_identity_order(keys, keys)["status"], "PASS")
        result = checks.check_identity_order(["p1", "p2"], ["p2", "p1"], "pathway")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["order_mismatch_count"], 2)
        self.assertEqual(result["order_mismatch_examples"][0]["index"], 0)

    def test_missing_extra_and_duplicate_are_distinct(self):
        result = checks.check_identity_order(["a", "b"], ["a", "a", "c"])
        self.assertEqual(result["missing"], ["b"])
        self.assertEqual(result["extra"], ["c"])
        self.assertEqual(result["duplicates_actual"], {"a": 2})
        self.assertEqual(result["status"], "FAIL")

    def test_repeated_expectations_do_not_pass(self):
        result = checks.check_identity_order(["a", "a"], ["a", "a"])
        self.assertEqual(result["duplicates_expected"], {"a": 2})
        self.assertEqual(result["status"], "FAIL")

    def test_empty_inputs_are_not_evidence(self):
        self.assertEqual(checks.check_identity_order([], [])["status"], "UNVERIFIED")


class ContractTests(unittest.TestCase):
    def test_declared_runtime_values_and_explicit_none(self):
        result = checks.compare_contract({"lr": 0.0003, "resume": None},
                                         {"lr": 0.0003, "resume": None, "other": 9})
        self.assertEqual(result["status"], "PASS")

    def test_unavailable_server_evidence_is_unverified(self):
        result = checks.compare_contract({"lr": 0.0003, "dtype": "float32"}, {"lr": 0.0003})
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertEqual(result["unverified"], ["dtype"])
        self.assertEqual(result["mismatches"], [])

    def test_runtime_override_and_unknown_are_both_retained(self):
        result = checks.compare_contract({"lr": 0.0003, "dtype": "float32"}, {"lr": 0.001})
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["mismatches"][0]["field"], "lr")
        self.assertEqual(result["unverified"], ["dtype"])

    def test_no_builtin_optimal_values(self):
        values = {"lr": 0.007, "crop": "center_crop", "aux_required": False}
        self.assertEqual(checks.compare_contract(values, values.copy())["status"], "PASS")

    def test_tolerance_is_explicit(self):
        self.assertEqual(checks.compare_contract({"x": 1.0}, {"x": 1.00001})["status"], "FAIL")
        self.assertEqual(checks.compare_contract({"x": 1.0}, {"x": 1.00001},
                                                {"x": (0, 0.0001)})["status"], "PASS")
        for tolerances in ({"y": (0, 1)}, {"x": (-1, 0)}, {"x": (float("inf"), 0)}):
            with self.assertRaises(ValueError):
                checks.compare_contract({"x": 1}, {"x": 1}, tolerances)

    def test_nonfinite_and_boolean_mismatch(self):
        for wanted, got in ((1, True), (False, 0), (1, float("nan")),
                            (float("inf"), float("inf"))):
            self.assertEqual(checks.compare_contract({"x": wanted}, {"x": got})["status"], "FAIL")

    def test_no_expectations_is_unverified(self):
        self.assertEqual(checks.compare_contract({}, {})["status"], "UNVERIFIED")

    def test_exact_large_integer_values_do_not_round_to_same_float(self):
        self.assertEqual(checks.compare_contract({"seed": 2**60},
                                                {"seed": 2**60 + 1})["status"], "FAIL")


def records(keys, status="succeeded"):
    return [dict(model=model, seed=seed, split=split, status=status) for model, seed, split in keys]


class CoverageTests(unittest.TestCase):
    keys = [("uni2h", 42, "internal_val"), ("uni", 42, "internal_val"),
            ("virchow2", 42, "internal_val")]

    def test_complete_single_seed_is_enough(self):
        result = checks.check_result_coverage(self.keys, records(self.keys), "succeeded")
        self.assertTrue(result["complete"])
        self.assertEqual(result["status"], "PASS")

    def test_missing_model_does_not_become_complete_single_seed(self):
        result = checks.check_result_coverage(self.keys, records(self.keys[:2]), "failed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], [self.keys[2]])
        self.assertEqual(result["status"], "WARN")

    def test_failed_not_run_and_unknown_are_visible(self):
        for status in ("failed", "not_run", "running", "completed", None):
            rows = records(self.keys)
            rows[1]["status"] = status
            result = checks.check_result_coverage(self.keys, rows)
            self.assertFalse(result["complete"])
            self.assertEqual(result["unsuccessful"], [{"key": self.keys[1], "status": status}])
        rows = records(self.keys)
        del rows[0]["status"]
        self.assertEqual(checks.check_result_coverage(self.keys, rows)["unsuccessful"][0]["status"], "unknown")

    def test_duplicate_and_unexpected_results(self):
        rows = records(self.keys + [self.keys[0], ("uni", 43, "external_test")])
        result = checks.check_result_coverage(self.keys, rows)
        self.assertFalse(result["complete"])
        self.assertEqual(result["duplicates_actual"], {self.keys[0]: 2})
        self.assertEqual(result["extra"], [("uni", 43, "external_test")])

    def test_missing_external_or_seed_and_duplicate_expectation(self):
        wanted = self.keys + [("uni", 43, "internal_val"), ("uni", 42, "external_test")]
        result = checks.check_result_coverage(wanted, records(self.keys))
        self.assertEqual(result["missing"], wanted[-2:])
        result = checks.check_result_coverage(self.keys + [self.keys[0]], records(self.keys))
        self.assertFalse(result["complete"])
        self.assertEqual(result["duplicates_expected"], {self.keys[0]: 2})

    def test_batch_failure_even_when_requested_rows_succeeded(self):
        result = checks.check_result_coverage(self.keys, records(self.keys), "failed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["batch_status"], "failed")

    def test_no_batch_status_does_not_claim_batch_success(self):
        result = checks.check_result_coverage(self.keys, records(self.keys))
        self.assertTrue(result["complete"])
        self.assertFalse(result["batch_status_checked"])

    def test_no_declared_scope_is_unverified(self):
        result = checks.check_result_coverage([], records(self.keys))
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertFalse(result["complete"])


class PortabilityTests(unittest.TestCase):
    def test_copy_runs_without_repository_or_site_packages(self):
        with tempfile.TemporaryDirectory(prefix="pfmval_minimal_audit_") as folder:
            destination = Path(folder) / "minimal_checks.py"
            shutil.copyfile(HELPER, destination)
            program = (
                "import runpy; c=runpy.run_path('minimal_checks.py'); "
                "assert c['check_identity_order'](['a'],['a'])['status']=='PASS'; "
                "assert c['compare_contract']({'lr':1},{})['status']=='UNVERIFIED'; "
                "assert not c['check_result_coverage']([('m',42,'val')],[])['complete']"
            )
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            run = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program],
                                 cwd=folder, env=environment, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
