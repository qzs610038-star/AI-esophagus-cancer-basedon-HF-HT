from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_cache import load_feature_cache
from feature_extract import cache_contract, cache_path, extract_one_model, prepare_feature_caches
from input_data import IdentityRecord, select_dataset_rows
from model_adapters import fake_spec
from transforms import GEOMETRY_PROTOCOL, HOPTIMUS_RGB_V1, IMAGENET_RGB_V1
from errors import IdentityMismatchError
from data import load_normalization, load_pathway_names


class _FakeAdapter:
    def encode(self, images: torch.Tensor):
        return images.mean(dim=(2, 3)).float()


def test_extraction_writes_profile_scoped_cache(tmp_path: Path):
    image_path = tmp_path / "patch_x0_y0.png"
    Image.new("RGB", (224, 224), (30, 40, 50)).save(image_path)
    row = IdentityRecord(0, "P", "MPP2_P_source01", image_path.stem, "train", 0, 0, str(image_path), 224, 224)
    spec = fake_spec(output_mode="timm_final_embedding", feature_dim=3, snapshot_path=str(tmp_path / "snapshots" / ("b" * 40)))
    result = extract_one_model(
        spec, [row], tmp_path / "hoptimus0" / "full_fov_224_bicubic_v1__hoptimus_rgb_v1" / "all",
        device="cpu", batch_sizes=[1], adapter_factory=lambda *_: _FakeAdapter(),
        metadata_context={"model": "hoptimus0", "revision": spec.revision, "output_mode": spec.output_mode,
                          "geometry_protocol": GEOMETRY_PROTOCOL, "normalization_profile": HOPTIMUS_RGB_V1, "dataset": "all"},
    )
    cache = load_feature_cache(result["cache_dir"], expected_identities=[row.identity_key], expected_dim=3)
    assert cache["metadata"]["normalization_profile"] == HOPTIMUS_RGB_V1
    assert cache["features"].shape == (1, 3)


def test_non_square_dataset_rejected_and_label_zscore_is_train_only():
    row = IdentityRecord(0, "P", "MPP2_P_source01", "patch_x0_y0", "train", 0, 0, "unused.png", 224, 220)
    try:
        select_dataset_rows([row], "train", protocol=GEOMETRY_PROTOCOL)
        raise AssertionError("non-square should fail")
    except IdentityMismatchError:
        pass
    package = Path(__file__).resolve().parents[1]
    names = load_pathway_names(package / "inputs" / "mpp2" / "zscore_manifest.json")
    norm = load_normalization(package / "inputs" / "mpp2" / "zscore_params_from_train.json", names)
    assert len(names) == 30
    assert list(norm.pathway_names) == names
    payload = json.loads((package / "inputs" / "mpp2" / "zscore_params_from_train.json").read_text(encoding="utf-8-sig"))
    assert payload.get("fit_split") == "train"
    assert "XZY" not in payload["fit_patients"]


def test_cache_contract_includes_loader_and_mpp_warning(tmp_path: Path):
    spec = fake_spec(snapshot_path=str(tmp_path / "snapshots" / ("b" * 40)))
    contract = cache_contract(
        {"experiment_id": "phase2_backbone_spatial_ablation_batch2_20260919", "protocol_version": "fullfov-spatial-backbone-batch2-v1",
         "data": {"split_id": "MPP2/group_2", "label_version": "barcode-repair-v003"}},
        spec, dataset="all", manifest_path=tmp_path / "identities.csv",
    )
    assert contract["loader_type"] == "timm_local_snapshot"
    assert contract["output_mode"] == "timm_final_embedding"
    assert contract["official_expected_mpp"] == 0.5
    assert contract["project_mpp_status"] == "unverified"
    assert contract["uses_auto_image_processor"] is False
    assert contract["image_normalization_object"] == "image_input_before_encoder"
    path = cache_path({"paths": {"feature_caches_root": str(tmp_path)}, "experiment_id": "exp"}, model="hoptimus0")
    assert "hoptimus_rgb_v1" in str(path)
    assert "imagenet_rgb_v1" not in str(path)
    cls_spec = fake_spec(output_mode="timm_tokens_cls", snapshot_path=str(tmp_path / "snapshots" / ("b" * 40)))
    try:
        extract_one_model(cls_spec, [], tmp_path / "bad", device="cpu", batch_sizes=[1], adapter_factory=lambda *_: _FakeAdapter())
        raise AssertionError("timm_tokens_cls should be rejected for this package")
    except ValueError:
        pass


def test_prepare_features_continues_after_missing_hoptimus1(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "identities.csv"
    manifest.write_text("unused\n", encoding="utf-8")
    row = IdentityRecord(0, "P", "MPP2_P_source01", "patch_x0_y0", "train", 0, 0, str(tmp_path / "unused.png"), 224, 224)
    revision = "b" * 40
    specs = {
        "hoptimus0": fake_spec(name="hoptimus0", feature_dim=3, snapshot_path=str(tmp_path / "snapshots" / revision)),
        "hoptimus1": fake_spec(name="hoptimus1", feature_dim=3, snapshot_path=str(tmp_path / "snapshots" / revision)),
        "phikonv2": fake_spec(
            name="phikonv2", output_mode="transformers_last_hidden_cls", feature_dim=3,
            normalization_profile=IMAGENET_RGB_V1, snapshot_path=str(tmp_path / "snapshots" / revision),
        ),
    }
    calls = []
    access = {"hoptimus1": False}

    monkeypatch.setattr("feature_extract.ensure_common_identity_manifest", lambda *_args, **_kwargs: (manifest, [row]))
    monkeypatch.setattr("feature_extract.load_model_specs", lambda *_args, **_kwargs: specs)

    def fake_validate(spec):
        if spec.name == "hoptimus1" and not access["hoptimus1"]:
            raise RuntimeError("gated")
        return tmp_path / "config.json", tmp_path / spec.checkpoint_filename

    def fake_extract(spec, rows, output_dir, **kwargs):
        calls.append(spec.name)
        return {"status": "complete", "model": spec.name, "cache_dir": str(output_dir), "n_rows": len(rows), "feature_dim": spec.feature_dim}

    monkeypatch.setattr("feature_extract.validate_local_snapshot", fake_validate)
    monkeypatch.setattr("feature_extract.extract_one_model", fake_extract)
    config = {
        "experiment_id": "phase2_backbone_spatial_ablation_batch2_20260919",
        "protocol_version": "fullfov-spatial-backbone-batch2-v1",
        "inputs": {"model_manifest": str(tmp_path / "model_manifest.json")},
        "paths": {"feature_caches_root": str(tmp_path / "caches")},
        "feature_extraction": {"initial_batch_size": 2, "fallback_batch_sizes": [1]},
        "data": {"split_id": "MPP2/group_2", "label_version": "barcode-repair-v003"},
    }
    result = prepare_feature_caches(config, device="cpu")
    assert calls == ["hoptimus0", "phikonv2"]
    assert result["status"] == "partial"
    assert result["overall_status"] == "partial"
    assert [item["model"] for item in result["selected_entries"]] == ["hoptimus0", "hoptimus1", "phikonv2"]
    assert result["selected_entries"][1]["status"] == "failed"
    access["hoptimus1"] = True
    resumed = prepare_feature_caches(config, device="cpu", models=["hoptimus1"])
    assert resumed["status"] == "completed"
    assert resumed["overall_status"] == "complete"
    assert [item["model"] for item in resumed["entries"]] == ["hoptimus0", "hoptimus1", "phikonv2"]
