from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from src.xzy_external import (
    build_external_plan,
    build_xzy_records,
    evaluate_stage2_checkpoint,
    external_output_directory,
    load_external_feature_cache,
)
from src.models import SharedProjection, Stage2Head
from xzy_runner import main as xzy_main


PATHWAYS = tuple(f"pathway_{index}" for index in range(30))


def _write_xzy_inputs(root: Path, stems: tuple[str, ...]) -> dict:
    label_path = root / "XZY_ssGSEA.csv"
    frame = pd.DataFrame({"spot_id": stems})
    for index, pathway in enumerate(PATHWAYS):
        frame[pathway] = [float(index + row) for row in range(len(stems))]
    frame.to_csv(label_path, index=False)
    image_directory = root / "patch_images"
    image_directory.mkdir()
    for stem in stems:
        (image_directory / f"{stem}.png").write_bytes(b"read-only fixture")
    return {
        "external_xzy_labels": str(label_path),
        "external_patient": "XZY",
        "external_mpp_id": 2,
        "image_template": str(image_directory / "{patch_stem}.png"),
        "expected_external_points": len(stems),
    }


def test_build_xzy_records_is_read_only_and_auditable(tmp_path: Path) -> None:
    data_config = _write_xzy_inputs(
        tmp_path, ("patch_x224_y448", "patch_x448_y448")
    )
    label_path = Path(data_config["external_xzy_labels"])
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [label_path, *sorted((tmp_path / "patch_images").iterdir())]
    }

    records, audit = build_xzy_records(
        data_config, pathway_names=PATHWAYS, native_step=224
    )

    assert [(record.patient, record.x, record.y) for record in records] == [
        ("XZY", 224, 448),
        ("XZY", 448, 448),
    ]
    assert audit["point_count"] == 2
    assert audit["native_step"] == 224
    assert audit["identity_source"] == "raw_label_first_column"
    assert audit["input_mode"] == "read_only"
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in before
    }
    assert after == before


def test_build_xzy_records_rejects_missing_images_and_off_grid_coordinates(
    tmp_path: Path,
) -> None:
    data_config = _write_xzy_inputs(
        tmp_path, ("patch_x224_y448", "patch_x500_y448")
    )
    (tmp_path / "patch_images" / "patch_x500_y448.png").unlink()
    with pytest.raises(ValueError, match="image identities do not match"):
        build_xzy_records(data_config, pathway_names=PATHWAYS, native_step=224)

    (tmp_path / "patch_images" / "patch_x500_y448.png").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="native spatial grid"):
        build_xzy_records(data_config, pathway_names=PATHWAYS, native_step=224)


def test_external_output_directory_is_fixed_under_the_existing_run(
    tmp_path: Path,
) -> None:
    assert external_output_directory(tmp_path) == tmp_path.resolve() / "external_xzy"


def test_external_feature_cache_rejects_identity_mismatch(tmp_path: Path) -> None:
    raw_target = torch.zeros(30).numpy()
    records = [
        __import__("src.data", fromlist=["OnlineRecord"]).OnlineRecord(
            2, "XZY", "patch_x224_y448", 224, 448, "external_test",
            "external", tmp_path / "patch_x224_y448.png", raw_target,
        )
    ]
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        cls=np.zeros((1, 1536), dtype="float32"),
        patient=np.asarray(["XZY"]),
        patch_stem=np.asarray(["patch_x448_y448"]),
        x=np.asarray([224]),
        y=np.asarray([448]),
    )

    with pytest.raises(AssertionError, match="identity mismatch"):
        load_external_feature_cache(cache, records)


def test_build_external_plan_uses_existing_fixed_and_sensitivity_tasks(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    seed = run / "raw" / "original" / "original" / "seed_42"
    (seed / "stage1").mkdir(parents=True)
    (seed / "fixed_e5").mkdir()
    (seed / "best_epoch_sensitivity").mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    stage1_directory = weights / "r8_regression"
    stage1_directory.mkdir()
    for epoch in (3, 5):
        (stage1_directory / f"epoch_{epoch}.pt").write_bytes(b"checkpoint")
    stage1 = {
        "checkpoint_directory": str(stage1_directory),
        "warmup_checkpoint": str(weights / "warmup.pt"),
        "history": [
            {"epoch": 1, "patient_macro_pathway_pcc": 0.1, "zMSE": 1.0},
            {"epoch": 2, "patient_macro_pathway_pcc": 0.2, "zMSE": 0.9},
            {"epoch": 3, "patient_macro_pathway_pcc": 0.3, "zMSE": 0.8},
            {"epoch": 4, "patient_macro_pathway_pcc": 0.25, "zMSE": 0.85},
            {"epoch": 5, "patient_macro_pathway_pcc": 0.2, "zMSE": 0.9},
        ],
    }
    (seed / "stage1" / "r8_regression.json").write_text(
        json.dumps(stage1), encoding="utf-8"
    )
    fixed_checkpoint = weights / "fixed_formal.pt"
    best_checkpoint = weights / "best_formal.pt"
    fixed_checkpoint.write_bytes(b"checkpoint")
    best_checkpoint.write_bytes(b"checkpoint")
    fixed = [{
        "task": {"cell": "r8_regression", "head": "point", "h_mode": "inherit", "endpoint": 5},
        "result": {"checkpoint": str(fixed_checkpoint)},
    }]
    best = [{
        "task": {"cell": "r8_regression", "head": "spatial", "h_mode": "inherit", "endpoint": 3},
        "result": {"checkpoint": str(best_checkpoint)},
    }]
    (seed / "fixed_e5" / "stage2_results.json").write_text(
        json.dumps(fixed), encoding="utf-8"
    )
    (seed / "best_epoch_sensitivity" / "stage2_results.json").write_text(
        json.dumps(best), encoding="utf-8"
    )

    plan = build_external_plan(run, weights, seed=42)

    assert [item.kind for item in plan] == [
        "stage1_fixed_e5",
        "stage1_best_epoch_sensitivity",
        "stage2_fixed_e5",
        "stage2_best_epoch_sensitivity",
    ]
    assert [item.endpoint for item in plan] == [5, 3, 5, 3]
    assert all(Path(item.checkpoint).is_file() for item in plan)


def test_evaluate_stage2_checkpoint_is_external_only_and_reports_both_pccs(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    head = Stage2Head(SharedProjection(), mode="point")
    checkpoint = tmp_path / "stage2_point_formal.pt"
    torch.save(head.state_dict(), checkpoint)
    features = torch.randn(40, 1536).numpy()
    raw_target = torch.randn(40, 30).numpy()
    records = [
        __import__("src.data", fromlist=["OnlineRecord"]).OnlineRecord(
            2,
            "XZY",
            f"patch_x{224 * index}_y0",
            224 * index,
            0,
            "external_test",
            "external",
            tmp_path / f"{index}.png",
            raw_target[index],
        )
        for index in range(40)
    ]
    task = __import__("src.xzy_external", fromlist=["ExternalTask"]).ExternalTask(
        "stage2_fixed_e5",
        "r8_regression",
        5,
        str(checkpoint),
        head="point",
        h_mode="inherit",
    )

    result = evaluate_stage2_checkpoint(
        task,
        features=features,
        frozen_features=features,
        records=records,
        normalization={"mean": [0.0] * 30, "std": [1.0] * 30},
        native_step=224,
        device="cpu",
    )

    assert result["selection_used"] is False
    assert result["split"] == "external_xzy"
    assert result["prediction_z"].shape == (40, 30)
    assert "patient_macro_pathway_pcc" in result["metrics"]
    assert "pooled_pcc" in result["metrics"]


def test_cli_plan_only_validates_inputs_without_creating_external_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    seed = run / "raw" / "original" / "original" / "seed_42"
    (seed / "stage1").mkdir(parents=True)
    (seed / "fixed_e5").mkdir()
    (seed / "best_epoch_sensitivity").mkdir()
    (run / "logs").mkdir()
    weights = tmp_path / "weights"
    stage1_directory = weights / "r8_regression"
    stage1_directory.mkdir(parents=True)
    for epoch in (3, 5):
        (stage1_directory / f"epoch_{epoch}.pt").write_bytes(b"checkpoint")
    (weights / "warmup.pt").write_bytes(b"checkpoint")
    fixed_checkpoint = weights / "fixed_formal.pt"
    best_checkpoint = weights / "best_formal.pt"
    fixed_checkpoint.write_bytes(b"checkpoint")
    best_checkpoint.write_bytes(b"checkpoint")

    xzy_root = tmp_path / "xzy"
    xzy_root.mkdir()
    data_config = _write_xzy_inputs(
        xzy_root, ("patch_x224_y448", "patch_x448_y448")
    )
    zscore_manifest = tmp_path / "zscore_manifest.json"
    zscore_manifest.write_text(
        json.dumps({"pathway_names": list(PATHWAYS)}), encoding="utf-8"
    )
    (run / "run.json").write_text(
        json.dumps({
            "experiment_id": "phase2_softlink_contrastive_v3",
            "run_id": "fixture",
            "protocol": "original",
            "seeds": [42],
            "weight_directory": str(weights),
        }),
        encoding="utf-8",
    )
    (run / "config.json").write_text(
        json.dumps({
            "data": data_config | {"zscore_manifest_file": str(zscore_manifest)}
        }),
        encoding="utf-8",
    )
    (run / "raw" / "engine_state.json").write_text(
        json.dumps({"tasks": {"formal": {"status": "completed"}}}),
        encoding="utf-8",
    )
    (run / "logs" / "resume_history.jsonl").write_text(
        json.dumps({"status": "succeeded", "exit_code": 0}) + "\n",
        encoding="utf-8",
    )
    stage1_result = {
        "checkpoint_directory": str(stage1_directory),
        "warmup_checkpoint": str(weights / "warmup.pt"),
        "history": [
            {"epoch": epoch, "patient_macro_pathway_pcc": 0.3 if epoch == 3 else 0.1,
             "zMSE": float(epoch)}
            for epoch in range(1, 6)
        ],
    }
    (seed / "stage1" / "r8_regression.json").write_text(
        json.dumps(stage1_result), encoding="utf-8"
    )
    fixed = [{
        "task": {"cell": "r8_regression", "head": "point", "h_mode": "inherit", "endpoint": 5},
        "result": {"checkpoint": str(fixed_checkpoint)},
    }]
    sensitivity = [{
        "task": {"cell": "r8_regression", "head": "spatial", "h_mode": "inherit", "endpoint": 3},
        "result": {"checkpoint": str(best_checkpoint)},
    }]
    (seed / "fixed_e5" / "stage2_results.json").write_text(
        json.dumps(fixed), encoding="utf-8"
    )
    (seed / "best_epoch_sensitivity" / "stage2_results.json").write_text(
        json.dumps(sensitivity), encoding="utf-8"
    )

    exit_code = xzy_main([
        "--run-dir", str(run), "--native-step", "224", "--plan-only"
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["task_count"] == 4
    assert payload["training_performed"] is False
    assert not (run / "external_xzy").exists()
