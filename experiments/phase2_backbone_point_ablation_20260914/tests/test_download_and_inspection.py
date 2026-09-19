from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from download_models import record_snapshot
from errors import IdentityMismatchError, OfflineModelError
from inspect_existing_uni2h import (
    describe_registered_historical_feature,
    extract_registered_historical_cls,
)


def test_download_record_accepts_only_the_pinned_snapshot_directory(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    payload = json.loads((package / "inputs" / "model_manifest.json").read_text(encoding="utf-8"))
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    revision = payload["models"]["uni"]["revision"]
    snapshot = tmp_path / "hub" / "models--MahmoodLab--UNI" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"0" * 2048)
    record_snapshot(manifest, "uni", snapshot)
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["models"]["uni"]["snapshot_path"] == str(snapshot.resolve())
    assert updated["models"]["uni"]["revision"] == revision
    assert updated["models"]["uni"]["download_status"] == "files_present_unverified"
    assert updated["models"]["uni"]["strict_load_status"] == "pending"


def test_download_record_rejects_a_different_revision(tmp_path: Path):
    package = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "models.json"
    manifest.write_text((package / "inputs" / "model_manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    wrong = tmp_path / "snapshots" / ("f" * 40)
    wrong.mkdir(parents=True)
    with pytest.raises(OfflineModelError, match="revision"):
        record_snapshot(manifest, "uni", wrong)


def test_only_explicit_historical_loader_can_take_first_row_of_2d_tensor():
    vector = torch.arange(1536, dtype=torch.float32)
    torch.testing.assert_close(extract_registered_historical_cls(vector), vector)
    matrix = torch.stack([vector, vector + 1])
    torch.testing.assert_close(extract_registered_historical_cls(matrix), vector)
    with pytest.raises(IdentityMismatchError):
        extract_registered_historical_cls(torch.ones(2, 2, 1536))
    with pytest.raises(IdentityMismatchError):
        extract_registered_historical_cls(torch.ones(2, 1024))


def test_historical_inspection_records_source_dtype_without_claiming_full_precision():
    description = describe_registered_historical_feature(
        torch.zeros(265, 1536, dtype=torch.float16)
    )
    assert description["source_dtype"] == "float16"
    assert description["shape"] == [265, 1536]
    assert description["mean_patch_available"] is True
    assert description["historical_compute_precision_fully_recorded"] is False
