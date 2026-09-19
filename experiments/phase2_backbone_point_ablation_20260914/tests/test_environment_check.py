from __future__ import annotations

import sys
from pathlib import Path

from config import load_config
from environment_check import build_environment_report


def test_environment_report_includes_verified_models_and_complete_feature_registry_state(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")
    config["python_interpreter"] = sys.executable
    config["paths"]["code_root"] = str(package.parent)
    config["paths"]["feature_caches_root"] = str(tmp_path / "feature_caches")
    report = build_environment_report(config)
    checks = {item["name"]: item for item in report["checks"]}
    assert "feature_registry" in checks
    assert checks["feature_registry"]["status"] == "error"
    assert checks["model_snapshot:uni"]["strict_load_status"] == "not_run"
    assert checks["model_snapshot:virchow2"]["download_status"] == "not_downloaded"
