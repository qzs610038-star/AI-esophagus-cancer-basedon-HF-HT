import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import point_table_from_cache_arrays
from export_phase3 import build_phase3_contract, export_phase3_bundle, predict_from_phase3_contract, write_phase3_contract
from local_report import discover_prediction_files, main as local_report_main, evaluate_prediction_file, summarize_prediction_files
from model import build_model
from predict import load_formal_model, predict_table, save_predictions


def _config(tmp_path):
    names = [f"pathway_{i}" for i in range(30)]
    zscore = tmp_path / "zscore.json"
    zscore.write_text(json.dumps({"pathway_names": names}), encoding="utf-8")
    norm = tmp_path / "normalization.json"
    norm.write_text(json.dumps({"fit_split": "train", "ddof": 1, "n_train_samples": 8, "pathways": {name: {"mean": 0.0, "std": 1.0} for name in names}}), encoding="utf-8")
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"models": {"uni2h": {"cls_dim": 4, "architecture": "test", "repo_id": "test/uni2h", "cls_index": 0}}}), encoding="utf-8")
    geometry = tmp_path / "geometry.csv"
    pd.DataFrame({"patient_id": ["XZY"], "slide_id": ["slide-x"], "s": [1.0], "coordinate_unit": ["grid"], "patch_coverage_size": [np.nan], "source": ["test"], "status": ["verified"]}).to_csv(geometry, index=False)
    return {
        "data": {"input_dim": 4, "hidden_dim": 3, "output_dim": 30, "slide_geometry_file": str(geometry), "expected_counts": {"train": 8}},
        "model": {"hidden_dim": 3, "dropout": 0.0, "shared_bias": True, "readout_bias": True, "spatial_bias": False},
        "graph": {"hops": 1, "directed": True, "radius_in_native_steps": 1.5, "max_neighbors": 4, "distance_sigma": 1.0, "use_image_similarity": True, "image_temperature": 0.2, "self_raw_weight": 1.0},
        "training": {"batch_size": 2, "keep_last_batch": True},
        "preprocessing": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6], "protocols": {"full_fov_224_bicubic_v1": ["rgb", "require_square"]}},
        "inputs": {"model_manifest": str(models), "zscore_manifest": str(zscore), "normalization": str(norm)},
        "model_name": "uni2h", "selection_mode": "search",
    }


def _external_table():
    rng = np.random.default_rng(8)
    return point_table_from_cache_arrays(
        features=rng.normal(size=(4, 4)).astype("float32"), patient_ids=["XZY"] * 4,
        slide_ids=["slide-x"] * 4, spot_ids=[f"x{i}" for i in range(4)],
        splits=["external_test"] * 4, x=[0, 1, 0, 1], y=[0, 0, 1, 1], pathway_names=[f"pathway_{i}" for i in range(30)],
    )


def _formal_checkpoint(config, path):
    model = build_model(config, "spatial")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "epoch": 1,
                "selection": {"kind": "formal", "epoch": 1}, "kind": "formal", "config": config,
                "arm": "spatial", "seed": 45, "input_dim": 4, "hidden_dim": 3, "output_dim": 30}, path)


def test_export_contract_preserves_dynamic_model_and_prediction_equivalence(tmp_path):
    config = _config(tmp_path)
    checkpoint = tmp_path / "formal.pt"
    _formal_checkpoint(config, checkpoint)
    contract = build_phase3_contract(config, checkpoint)
    contract_path = write_phase3_contract(contract, tmp_path / "export")
    table = _external_table()
    model, _ = load_formal_model(config, checkpoint, arm="spatial", seed=45, device="cpu")
    phase2 = predict_table(config, arm="spatial", model=model, table=table, device="cpu")
    phase3 = predict_from_phase3_contract(contract_path, table, device="cpu")
    np.testing.assert_allclose(phase2, phase3, rtol=0, atol=0)
    loaded = json.loads(contract_path.read_text(encoding="utf-8"))
    assert loaded["dynamic_dimensions"] == {"input_dim": 4, "hidden_dim": 3, "output_dim": 30}
    assert loaded["checkpoint"]["weights_copied"] is False
    assert loaded["geometry"]["groups"][0]["coordinate_unit"] == "grid"
    assert loaded["geometry"]["physical_patch_coverage_verified"] is False
    traced = export_phase3_bundle(config, checkpoint_path=checkpoint, output_dir=tmp_path / "traced",
                                  seed_sources=[{"seed": seed, "selected_for_phase3": seed == 45} for seed in (45, 46, 47)])
    assert len(json.loads(traced.read_text(encoding="utf-8"))["seed_sources"]) == 3


def test_local_report_separates_two_pcc_definitions_and_seed_sd(tmp_path):
    table = _external_table()
    names = list(table.pathway_names)
    mean, std = np.zeros(30), np.ones(30)
    rng = np.random.default_rng(6)
    target = rng.normal(size=(4, 30))
    paths = []
    for seed, noise in ((45, 0.05), (46, 0.10)):
        path = tmp_path / f"seed{seed}.npz"
        task_id = f"final-ablation__uni2h__full_fov_224_bicubic_v1__spatial__{seed}__frozen_spatial"
        save_predictions(path, pred_z=target + rng.normal(scale=noise, size=target.shape), target_z=target,
                         mean=mean, std=std, pathway_names=names, table=table,
                             checkpoint_metadata={"seed": seed, "task_id": task_id, "split": "external_test"})
        paths.append(path)
    one = evaluate_prediction_file(paths[0])
    assert one["status"] == "scored"
    assert one["patient_macro_pathway_pcc"] is not None
    assert one["pooled_pcc"] is not None
    summary = summarize_prediction_files(paths)
    assert summary["computed_locally"] is True
    assert len(summary["recipe_groups"]) == 1
    group = summary["recipe_groups"][0]
    assert group["model"] == "uni2h" and group["arm"] == "spatial"
    assert group["seed_summary"]["patient_macro_pathway_pcc"]["n"] == 2
    assert group["seed_summary"]["patient_macro_pathway_pcc"]["sd"] is not None


def test_local_report_cli_recurses_and_skips_non_prediction_npz(tmp_path):
    table = _external_table()
    target = np.tile(np.arange(30, dtype=float), (4, 1)) + np.arange(4)[:, None]
    raw = tmp_path / "returned" / "raw"
    raw.mkdir(parents=True)
    save_predictions(raw / "external.npz", pred_z=target + 0.1, target_z=target,
                     mean=np.zeros(30), std=np.ones(30), pathway_names=list(table.pathway_names), table=table,
                     checkpoint_metadata={"task_id": "final-ablation__uni__full_fov_224_bicubic_v1__point__45__frozen_point"})
    np.savez(raw / "internal_best.npz", pred_z=target)  # Expected raw non-export artifact.
    files, skipped = discover_prediction_files(tmp_path / "returned")
    assert [path.name for path in files] == ["external.npz"] and len(skipped) == 1
    output = tmp_path / "analysis" / "report.json"
    assert local_report_main(["--prediction-dir", str(tmp_path / "returned"), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_files"] == 1 and len(payload["skipped_npz_files"]) == 1
