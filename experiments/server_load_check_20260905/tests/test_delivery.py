"""One tiny end-to-end folder delivery check, using synthetic local inputs."""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DeliveryTest(unittest.TestCase):
    def test_folder_loads_samples_and_returns_failures_without_archives(self):
        import torch
        from PIL import Image
        package = Path(__file__).resolve().parents[1]
        work = package / "tests/work"
        work.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="本地 文件夹验收 ", dir=work))
        uploaded = root / package.name
        shutil.copytree(package / "deliverables" / package.name, uploaded)
        self.assertEqual(sum(p.is_file() for p in uploaded.rglob("*")), 8)
        inputs = root / "模拟 只读输入"
        inputs.mkdir()
        torch.save(torch.ones(1536), inputs / "feature.pt")
        torch.save({"layer.weight": torch.ones(2, 3)}, inputs / "weights.pth")
        Image.new("RGB", (32, 32)).save(inputs / "image.png")
        (inputs / "labels.csv").write_text("id,pathway\nsample1,0.5\n", encoding="utf-8")
        before = {p.name: p.read_bytes() for p in inputs.iterdir()}
        checks = []
        for name, kind in [("image.png", "image"), ("labels.csv", "csv"), ("feature.pt", "tensor"), ("weights.pth", "weights"),
                           ("missing.csv", "csv"), ("optional.csv", "csv"), ("labels.csv", "csv")]:
            checks.append(dict(id=str(len(checks)), label=name, kind=kind, path=str(inputs / name), source="synthetic_local_test",
                               required=name != "optional.csv"))
        checks[2]["expected_dim"] = 1536
        config = dict(python_interpreter=sys.executable, runs_root=str(root / "runs"), checks=checks)
        config_path = uploaded / "本地 配置.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(uploaded / "run.ps1"), "-Config", str(config_path)], cwd=root,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 1, repr(result.stderr))
        runs = root / "runs" / package.name
        run = next(p for p in runs.iterdir() if p.is_dir())
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"], {"PASS": 5, "WARN": 1, "FAIL": 1})
        self.assertEqual(report["checks"][-1]["status"], "PASS")
        self.assertIn("FileNotFoundError", (run / "logs/errors.log").read_text(encoding="utf-8"))
        self.assertEqual(before, {p.name: p.read_bytes() for p in inputs.iterdir()})
        self.assertEqual(json.loads((run / "run.json").read_text(encoding="utf-8"))["status"], "failed")
        self.assertTrue((run / "REPORT.md").is_file())
        self.assertEqual(list(root.rglob("*.zip")), [])
        self.assertIn(b"Copy this return folder:", result.stdout)
        validation = dict(local_only=True, server_tested=False, counts=report["counts"],
                          checks="folder copy, image/CSV/tensor/weight loading, continue after failure, folder return without ZIP, unchanged inputs",
                          artifacts=str(root))
        (work / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
