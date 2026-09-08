from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
RESOLVER = PACKAGE / "resolve_local_python.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def _run_resolver(
    tmp_path: Path,
    *arguments: str,
    conda_envs: list[Path],
    use_conda_cli: bool = True,
) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL, "本机需要 Windows PowerShell 来验证本地启动脚本"
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    if use_conda_cli:
        payload = json.dumps({"envs": [str(path) for path in conda_envs]})
        (fake_bin / "conda.bat").write_text(f"@echo off\r\necho {payload}\r\n", encoding="utf-8")

    fake_profile = tmp_path / "profile"
    registry = fake_profile / ".conda" / "environments.txt"
    registry.parent.mkdir(parents=True)
    registry.write_text("\n".join(str(path) for path in conda_envs), encoding="utf-8")
    environment = os.environ | {
        "PATH": f"{fake_bin};{os.environ['PATH']}" if use_conda_cli else str(fake_bin),
        "USERPROFILE": str(fake_profile),
    }
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RESOLVER), *arguments],
        cwd=PACKAGE,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_explicit_python_interpreter_takes_priority_over_conda(tmp_path: Path):
    explicit = tmp_path / "custom_python.exe"
    explicit.touch()

    result = _run_resolver(tmp_path, "-PythonInterpreter", str(explicit), conda_envs=[])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(explicit.resolve())


def test_registered_conda_environment_name_resolves_python(tmp_path: Path):
    environment = tmp_path / "relocated_envs" / "pfmval_py310"
    environment.mkdir(parents=True)
    interpreter = environment / "python.exe"
    interpreter.touch()

    result = _run_resolver(tmp_path, conda_envs=[environment], use_conda_cli=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(interpreter.resolve())


def test_missing_registered_conda_environment_fails_with_override_hint(tmp_path: Path):
    result = _run_resolver(tmp_path, conda_envs=[], use_conda_cli=False)

    assert result.returncode != 0
    details = f"{result.stdout}\n{result.stderr}"
    assert "pfmval_py310" in details
    assert "-PythonInterpreter" in details
