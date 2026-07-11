import json
from pathlib import Path

import pytest

from deploy.pfmval_ops import bound_job_parameter_argv
from train_mpp_uni2h_mlp import resolve_manifest_data_roots


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_manifest_routing_keeps_pinned_splits_separate_from_repaired_labels(tmp_path):
    split_root = tmp_path / "pinned-splits"
    label_root = tmp_path / "repaired-labels"
    split_group = split_root / "group_2"
    label_group = label_root / "group_2"
    split_group.mkdir(parents=True)
    (split_group / "split_manifest.csv").write_text("patient,patch_stem,split\n", encoding="utf-8")
    _write_json(split_group / "split_info.json", {"leakage_pairs": 0})
    _write_json(label_group / "zscore_manifest.json", {"fit_patients": ["P1"]})
    _write_json(label_group / "zscore_params_from_train.json", {"pathways": {}})
    (label_group / "labels").mkdir()

    roots = resolve_manifest_data_roots(split_root, label_root, 2)

    assert roots["split_manifest"] == split_group / "split_manifest.csv"
    assert roots["split_info"] == split_group / "split_info.json"
    assert roots["zscore_manifest"] == label_group / "zscore_manifest.json"
    assert roots["zscore_params"] == label_group / "zscore_params_from_train.json"
    assert roots["labels"] == label_group / "labels"


def test_server_job_injects_only_the_activated_repaired_label_root(tmp_path):
    stage = r"D:\staging\barcode-repair-v003"
    canonical_audit = (
        tmp_path / "project_state" / "evidence" / "mpp" / "repair-v003"
        / "server_asset_audit_manifest.json"
    )
    _write_json(canonical_audit, {"generated_assets": []})
    import hashlib
    audit_sha = hashlib.sha256(canonical_audit.read_bytes()).hexdigest()
    _write_json(tmp_path / "project_state" / "current_state.json", {
        "mpp_repair": {"active_data_manifest_id": "repair-v003:abc"},
    })
    _write_json(tmp_path / "project_state" / "mpp_repair_registry.json", {
        "active_data_manifest_id": "repair-v003:abc",
        "repairs": [{
            "evidence_id": "repair-v003",
            "data_manifest_id": "repair-v003:abc",
            "status": "active",
            "server_stage_path": stage,
            "audit_sha256": audit_sha,
        }],
    })
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "server_paths.yaml").write_text(
        "schema_version: '1.0'\npaths:\n"
        "  mpp_data_root: {path: 'D:\\raw', required_on: server}\n"
        "  server_mpp_flat_cache: {path: 'D:\\cache', required_on: server}\n"
        "  server_mpp_results: {path: 'D:\\results', required_on: server}\n",
        encoding="utf-8",
    )
    manifest = {"data_manifest_id": "repair-v003:abc", "parameters": {"num_epochs": 2}}
    experiment = {"script": "train_mpp_uni2h_mlp.py"}

    argv = bound_job_parameter_argv(tmp_path, manifest, experiment)

    manifest_index = argv.index("--manifest_labels_root")
    splits_index = argv.index("--splits_root")
    assert argv[manifest_index + 1] == stage
    assert Path(argv[splits_index + 1]) == (tmp_path / "mpp_standard_splits").resolve()

    with pytest.raises(ValueError, match="does not match active repaired labels"):
        bound_job_parameter_argv(
            tmp_path,
            {"data_manifest_id": "wrong", "parameters": {"num_epochs": 2}},
            experiment,
        )
