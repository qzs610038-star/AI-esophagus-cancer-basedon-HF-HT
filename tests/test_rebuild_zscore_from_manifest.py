import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from dataset_mpp import MPPFeatureDataset
from scripts.rebuild_zscore_from_manifest import TRAIN_PATIENTS


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_zscore_from_manifest.py"


def _write_rebuild_fixture(root: Path, *, duplicate_raw: bool = False) -> tuple[Path, Path]:
    splits_root = root / "splits"
    mpp_root = root / "raw"
    group = splits_root / "group_2"
    group.mkdir(parents=True)

    manifest_rows = []
    for index, patient in enumerate(TRAIN_PATIENTS):
        train_barcode = "patch_x1_y1"
        val_barcode = "patch_x2_y2"
        manifest_rows.extend(
            [
                {"patient": patient, "patch_stem": train_barcode, "split": "train"},
                {"patient": patient, "patch_stem": val_barcode, "split": "internal_val"},
            ]
        )
        raw_rows = [
            {"barcode": train_barcode, "pathway_a": float(index + 1)},
            {"barcode": val_barcode, "pathway_a": float(index + 11)},
        ]
        if duplicate_raw and index == 0:
            raw_rows.append({"barcode": train_barcode, "pathway_a": 999.0})
        label = mpp_root / "2" / patient / f"{patient}_ssGSEA.csv"
        label.parent.mkdir(parents=True)
        pd.DataFrame(raw_rows).to_csv(label, index=False)

    pd.DataFrame(manifest_rows).to_csv(group / "split_manifest.csv", index=False)
    external = mpp_root / "2" / "XZY" / "XZY_ssGSEA.csv"
    external.parent.mkdir(parents=True)
    pd.DataFrame([{"barcode": "patch_x9_y9", "pathway_a": 20.0}]).to_csv(external, index=False)
    return splits_root, mpp_root


def _run_rebuild(tmp_path: Path, *, version: str, duplicate_raw: bool = False):
    splits_root, mpp_root = _write_rebuild_fixture(tmp_path, duplicate_raw=duplicate_raw)
    staging_root = tmp_path / "staging"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--splits-root",
            str(splits_root),
            "--mpp-root",
            str(mpp_root),
            "--mpp-ids",
            "2",
            "--staging-root",
            str(staging_root),
            "--staging-version",
            version,
            "--source-commit",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, staging_root / version


def test_cli_keeps_same_barcode_isolated_by_patient_and_emits_asset_audit(tmp_path):
    result, version_dir = _run_rebuild(tmp_path, version="repair-v001")

    assert result.returncode == 0, result.stdout + result.stderr
    for patient in TRAIN_PATIENTS:
        label = version_dir / "group_2" / "labels" / "train" / patient / f"{patient}_ssGSEA_zscore.csv"
        frame = pd.read_csv(label)
        assert frame["barcode"].tolist() == ["patch_x1_y1"]

    audit_path = version_dir / "server_asset_audit_manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["staging_version"] == "repair-v001"
    assert audit["source_commit"] == "a" * 40
    assert audit["server_transport"] == "gitee_only"
    assert audit["validation"]["patient_barcode_one_to_one"] is True
    assert audit["validation"]["dataset_labels_unique"] is True
    assert audit["generated_assets"]
    assert all(len(item["sha256"]) == 64 for item in audit["generated_assets"])
    label_assets = [item for item in audit["generated_assets"] if item["role"] == "standardized_label"]
    assert label_assets
    assert all(item["duplicate_barcode_count"] == 0 for item in label_assets)


def test_cli_rejects_duplicate_patient_barcode_without_publishing_version(tmp_path):
    result, version_dir = _run_rebuild(tmp_path, version="repair-v002", duplicate_raw=True)

    assert result.returncode != 0
    assert "patient+barcode" in result.stdout + result.stderr
    assert not version_dir.exists()


def test_dataset_hard_fails_on_duplicate_label_key_even_in_allow_missing_mode(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save(torch.zeros(4), cache / "patch_x1_y1.pt")
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"barcode": "patch_x1_y1", "pathway_a": 1.0},
            {"barcode": "patch_x1_y1", "pathway_a": 2.0},
        ]
    ).to_csv(labels, index=False)

    with pytest.raises(ValueError, match="duplicate label key"):
        MPPFeatureDataset(str(cache), str(labels), allow_missing=True)
