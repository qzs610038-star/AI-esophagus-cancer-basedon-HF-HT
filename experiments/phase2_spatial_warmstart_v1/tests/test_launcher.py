from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from conftest import PACKAGE_DIR


def _make_launcher_package(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "launcher_package"
    (package / "src").mkdir(parents=True)
    shutil.copy2(PACKAGE_DIR / "run.ps1", package / "run.ps1")
    shutil.copy2(PACKAGE_DIR / "runner.py", package / "runner.py")
    (package / "package.json").write_text(
        json.dumps(
            {
                "experiment_id": "synthetic_launcher",
                "code_version": "test",
                "entrypoint": "src/main.py",
                "args": [],
                "demo_only": True,
                "implementation_status": "implemented",
            }
        ),
        encoding="utf-8",
    )
    (package / "config.json").write_text(
        json.dumps(
            {
                "batch_id": "batch_contract",
                "python_interpreter": sys.executable,
                "runs_root": str(tmp_path / "runs"),
                "weights_root": str(tmp_path / "weights"),
            }
        ),
        encoding="utf-8",
    )
    (package / "src" / "main.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--config', required=True)\n"
        "p.add_argument('--run-dir', required=True, type=Path)\n"
        "p.add_argument('--weights-dir', required=True, type=Path)\n"
        "a=p.parse_args()\n"
        "(a.run_dir/'raw'/'entrypoint.json').write_text(json.dumps({\n"
        "    'run_dir': str(a.run_dir), 'weights_dir': str(a.weights_dir)\n"
        "}), encoding='utf-8')\n"
        ,
        encoding="utf-8",
    )
    return package, tmp_path / "runs", tmp_path / "weights"


def _launch(package: Path, runs_root: Path, weights_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("powershell")
    if powershell is None:
        raise RuntimeError("Windows PowerShell is required for the launcher contract test")
    return subprocess.run(
        [
            powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(package / "run.ps1"),
            "-PythonInterpreter", sys.executable,
            "-RunsRoot", str(runs_root),
            "-WeightsRoot", str(weights_root),
            *extra,
        ],
        cwd=package,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_launcher_uses_batch_layer_and_creates_new_run_instead_of_overwriting(tmp_path: Path) -> None:
    package, runs_root, weights_root = _make_launcher_package(tmp_path)

    first = _launch(package, runs_root, weights_root)
    second = _launch(package, runs_root, weights_root)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    run_batch = runs_root / "synthetic_launcher" / "batch_contract"
    weight_batch = weights_root / "synthetic_launcher" / "batch_contract"
    run_dirs = sorted(path for path in run_batch.iterdir() if path.is_dir())
    weight_dirs = sorted(path for path in weight_batch.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert len(weight_dirs) == 2
    assert {path.name for path in run_dirs} == {path.name for path in weight_dirs}

    for run_dir in run_dirs:
        run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
        weights_record = json.loads(
            (run_dir / "model_weights.json").read_text(encoding="utf-8-sig")
        )
        entrypoint_record = json.loads(
            (run_dir / "raw" / "entrypoint.json").read_text(encoding="utf-8")
        )
        assert run_record["batch_id"] == "batch_contract"
        assert weights_record["batch_id"] == "batch_contract"
        assert run_record["status"] == "succeeded"
        assert run_record["run_id"] == run_dir.name
        assert Path(run_record["weight_directory"]).parent == weight_batch
        assert Path(entrypoint_record["run_dir"]) == run_dir
        assert Path(entrypoint_record["weights_dir"]).name == run_dir.name


def test_launcher_rejects_resume_checkpoint_in_train_mode(tmp_path: Path) -> None:
    package, runs_root, weights_root = _make_launcher_package(tmp_path)
    result = _launch(
        package, runs_root, weights_root,
        "-Mode", "train", "-ResumeCheckpoint", str(tmp_path / "not_used.pt"),
    )
    assert result.returncode != 0
    assert "only valid when Mode=resume" in (result.stdout + result.stderr)
