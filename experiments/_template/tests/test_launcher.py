"""Exercise the public PowerShell launcher, with no repository imports or training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class LauncherTest(unittest.TestCase):
    def test_isolated_run_failure_and_code_replacement(self):
        template = Path(__file__).resolve().parents[1]
        scratch = Path(os.environ.get("PFMVAL_TEST_ROOT", str(template / "tests" / "work")))
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="独立包 验收 ", dir=scratch))
        package = root / "code" / "中文 实验包"
        shutil.copytree(template, package, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        runs = root / "runs"
        weights = root / "weights"
        # A config filename with spaces exercises PowerShell argument handling too.
        config_file = package / "实际 配置.json"
        config = json.loads((package / "config.json").read_text(encoding="utf-8"))
        config.update(python_interpreter=sys.executable, runs_root=str(runs), weights_root=str(weights))
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

        def launch(python_override=None):
            command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(package / "run.ps1"), "-Config", str(config_file)]
            if python_override:
                command += ["-PythonInterpreter", python_override]
            # No inherited PYTHONPATH: a repository reference must not make this pass.
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            return subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=30)

        success = launch()
        self.assertEqual(success.returncode, 0, repr(success.stderr))
        first = next((runs / "template_demo").iterdir())
        first_record = json.loads((first / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(first_record["status"], "succeeded")
        self.assertEqual(first_record["exit_code"], 0)
        self.assertTrue(first_record["demo_only"])
        weight_dir = weights / "template_demo" / first.name
        self.assertEqual(Path(first_record["weight_directory"]), weight_dir.resolve())
        self.assertTrue(weight_dir.is_dir())
        self.assertFalse((first / "checkpoints").exists())
        registry = json.loads((first / "model_weights.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(registry["weight_directory"]), weight_dir.resolve())
        self.assertEqual(registry["files"], [])
        self.assertIn("零训练演示", (first / "logs/stdout.log").read_text(encoding="utf-8"))
        self.assertTrue((first / "raw/demo.json").is_file())
        self.assertEqual(json.loads((first / "metrics.json").read_text())["metrics"], {})
        original = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}

        # Replace only the test-owned code directory by moving it aside; keep it for review.
        self.assertTrue(package.resolve().is_relative_to(root.resolve()))
        self.assertTrue((root / "previous_code").resolve().is_relative_to(root.resolve()))
        package.rename(root / "previous_code")
        shutil.copytree(template, package, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
        metadata.update(code_version="v002", args=["--fail"])
        (package / "package.json").write_text(json.dumps(metadata), encoding="utf-8")
        failure = launch()
        self.assertNotEqual(failure.returncode, 0)
        directories = list((runs / "template_demo").iterdir())
        self.assertEqual(len(directories), 2)
        second = next(p for p in directories if p != first)
        record = json.loads((second / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["exit_code"], failure.returncode)
        self.assertEqual(record["code_version"], "v002")
        self.assertIn("RuntimeError", (second / "logs/errors.log").read_text(encoding="utf-8"))
        self.assertIn("主动报错", (second / "logs/console.log").read_text(encoding="utf-8"))

        missing = launch(str(root / "missing_python.exe"))
        self.assertNotEqual(missing.returncode, 0)
        third = next(p for p in (runs / "template_demo").iterdir() if p not in directories)
        self.assertEqual(json.loads((third / "run.json").read_text(encoding="utf-8"))["status"], "failed")
        self.assertIn("missing_python", (third / "logs/errors.log").read_text(encoding="utf-8"))
        self.assertEqual(original, {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()})
        print(f"Verification artifacts: {root}")


if __name__ == "__main__":
    unittest.main()
