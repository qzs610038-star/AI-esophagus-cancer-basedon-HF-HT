from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from src.backbone import forward_cls, resolve_uni2_checkpoint
from src.data import OnlineRecord, fit_zscore, read_split_manifest, view_seed
from src.data import load_patient_raw_labels


def _record(patient: str, stem: str, value: float) -> OnlineRecord:
    return OnlineRecord(
        2, patient, stem, 0, 0, "train", "b", Path("unused.png"),
        np.full(30, value, dtype=np.float32),
    )


def test_fit_zscore_records_exact_training_identity() -> None:
    records = [_record("A", "a", 1.0), _record("B", "b", 3.0)]
    artifact = fit_zscore(records)
    np.testing.assert_allclose(artifact["mean"], 2.0)
    np.testing.assert_allclose(artifact["std"], 1.0)
    assert artifact["fit_patients"] == ["A", "B"]
    assert artifact["fit_identities"] == [record.identity for record in records]


def test_manifest_rejects_duplicate_and_xzy(tmp_path: Path) -> None:
    row = {"mpp_id": 2, "patient": "A", "patch_stem": "p", "x": 0,
           "y": 0, "split": "train", "block_id": "b"}
    path = tmp_path / "manifest.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_split_manifest(path)
    row["patient"] = "XZY"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="XZY"):
        read_split_manifest(path)


def test_view_seed_is_cell_shared_and_slot_specific() -> None:
    left = view_seed("original", "all", 42, 1, 2, 3)
    assert left == view_seed("original", "all", 42, 1, 2, 3)
    assert left != view_seed("original", "all", 42, 1, 2, 4)


def test_checkpoint_resolution_requires_one_complete_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"x")
    assert resolve_uni2_checkpoint(tmp_path) == checkpoint.resolve()
    nested = tmp_path / "other"
    nested.mkdir()
    (nested / "pytorch_model.bin").write_bytes(b"y")
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_uni2_checkpoint(tmp_path)


def test_forward_cls_uses_first_token_and_checks_dimension() -> None:
    class Tiny(nn.Module):
        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            return torch.ones(images.shape[0], 4, 1536)

    result = forward_cls(Tiny(), torch.zeros(2, 3, 8, 8))
    assert result.shape == (2, 1536)


def test_raw_label_loader_excludes_numeric_coordinate_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "barcode": ["patch_x0_y0", "patch_x1_y1"],
        "x": [0, 1], "y": [0, 1],
        **{f"pathway_{index}": [float(index), float(index + 1)] for index in range(30)},
    })
    path = tmp_path / "labels.csv"
    frame.to_csv(path, index=False)
    labels, names = load_patient_raw_labels(path)
    assert len(names) == 30 and "x" not in names and "y" not in names
    assert labels["patch_x0_y0"].shape == (30,)
