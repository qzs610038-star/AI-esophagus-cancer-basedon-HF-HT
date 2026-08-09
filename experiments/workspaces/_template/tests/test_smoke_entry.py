from pathlib import Path

from experiments.workspaces._template.code.smoke_entry import load_config


def test_template_loads_no_training_config():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "smoke.json")
    assert config == {"mode": "no_training", "workspace_id": "W###"}
