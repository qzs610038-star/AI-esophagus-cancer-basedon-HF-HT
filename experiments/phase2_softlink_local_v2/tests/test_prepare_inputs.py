from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from prepare_inputs import prepare_inputs


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_package(tmp_path: Path, *, external_points=4):
    package = tmp_path / "package"
    inputs = package / "inputs"
    inputs.mkdir(parents=True)
    partner = tmp_path / "partner"
    flat = tmp_path / "flat"
    internal = partner / "MPP2_UNI" / "P1"
    external = flat / "2" / "XZY"
    internal.mkdir(parents=True)
    external.mkdir(parents=True)
    rows = []
    for i, (x, y, split) in enumerate([(0, 0, "train"), (2, 0, "train"), (0, 2, "internal_val"), (2, 2, "internal_val")]):
        stem = f"patch_x{x}_y{y}"
        torch.save(torch.arange(4, dtype=torch.float32), internal / f"{stem}.pt")
        rows.append({"mpp_id": 2, "patient": "P1", "patch_stem": stem, "x": x, "y": y, "split": split, "block_id": i})
    with (inputs / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    external_coords = [(0, 0), (5, 0), (0, 5), (5, 5)][:external_points]
    for x, y in external_coords:
        torch.save(torch.arange(4, dtype=torch.float32), external / f"patch_x{x}_y{y}.pt")
    _write_json(inputs / "feature_source_manifest.json", {
        "mpp_id": 2,
        "expected_dim": 4,
        "partner_cache_root": str(partner),
        "flat_cache_root": str(flat),
    })
    config = {
        "data": {
            "mpp_id": 2,
            "input_dim": 4,
            "split_manifest_file": "inputs/split_manifest.csv",
            "feature_source_manifest": "inputs/feature_source_manifest.json",
            "slide_mapping_file": "inputs/slide_mapping.csv",
            "slide_mapping_status_file": "inputs/slide_mapping_status.json",
            "slide_geometry_file": "inputs/slide_geometry.csv",
            "external_point_table": None,
            "input_status_file": "inputs/input_status.json",
            "expected_train_points": 2,
            "expected_internal_val_points": 2,
            "expected_external_points": 4,
            "external_patient": "XZY",
        }
    }
    _write_json(package / "config.json", config)
    return package, config


def test_prepare_inputs_generates_auditable_groups_geometry_and_xzy_table(tmp_path):
    package, config = _make_package(tmp_path)
    report = prepare_inputs(config, package_dir=package)
    assert report["status"] == "prepared"
    assert report["internal_points"] == 4
    assert report["external_points"] == 4
    mapping = list(csv.DictReader((package / "inputs" / "slide_mapping.csv").open(encoding="utf-8-sig")))
    assert [(row["patient_id"], row["slide_id"], row["status"]) for row in mapping] == [
        ("P1", "MPP2_P1_source01", "verified"),
        ("XZY", "MPP2_XZY_source01", "verified"),
    ]
    geometry = {row["slide_id"]: float(row["s"]) for row in csv.DictReader((package / "inputs" / "slide_geometry.csv").open(encoding="utf-8-sig"))}
    assert geometry == {"MPP2_P1_source01": 2.0, "MPP2_XZY_source01": 5.0}
    external = list(csv.DictReader((package / "inputs" / "external_point_table.csv").open(encoding="utf-8-sig")))
    assert len(external) == 4
    assert set(external[0]) == {"patient_id", "slide_id", "spot_id", "x", "y", "feature_path"}
    saved_config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    assert saved_config["data"]["external_point_table"] == "inputs/external_point_table.csv"


def test_prepare_inputs_rejects_incomplete_external_cache(tmp_path):
    package, config = _make_package(tmp_path, external_points=3)
    with pytest.raises(ValueError, match="外部点数"):
        prepare_inputs(config, package_dir=package)
