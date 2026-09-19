from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_runner_help_works_after_package_is_copied_without_parent_pythonpath(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    copied = tmp_path / "standalone_package"
    shutil.copytree(
        package,
        copied,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc"),
    )
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
    assert "seed42" in result.stdout


def test_source_has_no_parent_repository_import_or_path_injection():
    package = Path(__file__).resolve().parents[1]
    forbidden = ("sys.path.append", "sys.path.insert", "phase2_softlink_local_v2.src", "parents[2]")
    for path in (package / "src").glob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert all(value not in content for value in forbidden), path.name
