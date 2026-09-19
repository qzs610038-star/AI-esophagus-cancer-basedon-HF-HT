from __future__ import annotations

import copy
from pathlib import Path

import pytest

from config import load_config, validate_config
from errors import ConfigError


def test_shipped_config_freezes_the_v11_scientific_contract():
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")

    assert config["experiment_id"] == "phase2_backbone_point_ablation_20260914"
    assert config["protocol_version"] == "v1.1-20260915"
    assert config["data"]["split_id"] == "MPP2/group_2"
    assert config["data"]["expected_counts"] == {
        "train": 9472,
        "internal_val": 1078,
        "external_test": 1039,
    }
    assert config["execution"]["default_scope"] == "seed42"
    assert config["execution"]["train_models"] == ["uni", "virchow2"]
    assert config["execution"]["scopes"] == {"seed42": [42], "remaining": [43, 44]}
    assert config["training"]["batch_size"] == 256
    assert config["training"]["max_epochs"] == 60
    assert config["training"]["precision"] == "float32"
    assert config["training"]["amp"] is False
    assert config["training"]["tf32"] is False
    assert config["selection"]["formal_start_epoch"] == 6
    assert config["selection"]["constant_prediction_selection_penalty"] == -1.0
    assert config["preprocessing"]["steps"] == [
        "rgb",
        "resize_shorter_side_256_bicubic_antialias",
        "center_crop_224",
        "to_float_tensor_0_1",
        "imagenet_normalize",
    ]
    assert config["paths"]["code_root"] == r"D:\AIPatho\qzs\code"
    assert config["paths"]["feature_caches_root"] == r"D:\AIPatho\qzs\feature_caches"
    assert config["paths"]["weights_root"] == r"D:\AIPatho\qzs\weights"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["training"].__setitem__("amp", True), "AMP"),
        (lambda c: c["training"].__setitem__("learning_rate", 1e-4), "learning_rate"),
        (lambda c: c["execution"].__setitem__("default_scope", "remaining"), "default_scope"),
        (lambda c: c["execution"]["train_models"].append("uni2h"), "UNI2-h"),
        (lambda c: c["data"]["expected_counts"].__setitem__("external_test", 1000), "external_test"),
    ],
)
def test_scientific_contract_rejects_silent_protocol_drift(mutation, message):
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(ConfigError, match=message):
        validate_config(changed)


def test_package_inputs_resolve_from_the_package_not_the_callers_cwd():
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")
    assert Path(config["inputs"]["split_manifest"]).is_absolute()
    assert Path(config["inputs"]["split_manifest"]).parent.name == "mpp2"
    assert Path(config["inputs"]["model_manifest"]).parent.name == "inputs"


def test_small_scientific_inputs_cannot_be_redirected_outside_the_package(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    config = load_config(package / "config.json")
    changed = copy.deepcopy(config)
    changed["inputs"]["split_manifest"] = str(tmp_path / "replacement.csv")
    with pytest.raises(ConfigError, match="inputs.split_manifest"):
        validate_config(changed)
