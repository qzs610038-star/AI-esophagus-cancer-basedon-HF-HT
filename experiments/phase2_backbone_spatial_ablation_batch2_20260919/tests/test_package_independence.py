from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_source_has_no_parent_repository_import():
    package = Path(__file__).resolve().parents[1]
    forbidden = ("sys.path.append", "sys.path.insert", "phase2_fullfov_hpo_v1.src", "parents[2]", "from search import")
    for path in (package / "src").glob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert all(value not in content for value in forbidden), path.name
        assert "AutoImageProcessor" not in content
    launcher = (package / "run.ps1").read_text(encoding="utf-8")
    assert "[string]$Action = 'check-inputs'" in launcher
    assert "search" not in launcher.lower() or "ValidateSet('check-environment','check-inputs','prepare-features','train-spatial','external-eval','analyze-local')" in launcher.replace(" ", "")


def test_runner_help_works_after_copy_without_parent_pythonpath(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    copied = tmp_path / "standalone_package"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc", "tests"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(copied / "runner.py"), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--action" in result.stdout or "usage" in result.stdout.lower()
