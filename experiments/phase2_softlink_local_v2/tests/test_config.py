from __future__ import annotations

from copy import deepcopy

import pytest

from config import load_config, validate_config
from errors import UnsupportedAlgorithmError
from helpers import PACKAGE, package_config


def test_real_config_loads_and_keeps_dims():
    config = package_config()
    assert config["model"]["hidden_dim"] == 256
    assert config["model"]["relation_dim"] == 256
    assert config["data"]["expected_train_points"] == 9472
    assert config["relation"]["far_patient_scope"] == "all"
    assert config["graph"]["use_image_similarity"] is True


def test_rejects_dense_q_and_multihop():
    raw = load_config(PACKAGE / "config.json", package_dir=PACKAGE)
    dense = deepcopy(raw)
    dense["relation"]["support"] = "dense"
    dense.pop("_package_root", None)
    dense.pop("_config_path", None)
    with pytest.raises(UnsupportedAlgorithmError, match="truncated"):
        validate_config(dense)

    hops = deepcopy(raw)
    hops["graph"]["hops"] = 2
    hops.pop("_package_root", None)
    hops.pop("_config_path", None)
    with pytest.raises(UnsupportedAlgorithmError, match="hops"):
        validate_config(hops)


def test_rejects_rkd_distance_and_amp():
    raw = load_config(PACKAGE / "config.json", package_dir=PACKAGE)
    rkd = deepcopy(raw)
    rkd["relation"]["distance"] = "rkd"
    rkd.pop("_package_root", None)
    rkd.pop("_config_path", None)
    with pytest.raises(UnsupportedAlgorithmError):
        validate_config(rkd)

    amp = deepcopy(raw)
    amp["training"]["amp"] = True
    amp.pop("_package_root", None)
    amp.pop("_config_path", None)
    with pytest.raises(UnsupportedAlgorithmError, match="AMP"):
        validate_config(amp)


def test_same_patient_scope_is_allowed():
    raw = load_config(PACKAGE / "config.json", package_dir=PACKAGE)
    raw["relation"]["far_patient_scope"] = "same_patient"
    raw.pop("_package_root", None)
    raw.pop("_config_path", None)
    validate_config(raw)
