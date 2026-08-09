from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.workspaces.W004.code.phase3_pipeline.engine import (
    FoldTrainingResult,
)
from experiments.workspaces.W004.code.phase3_pipeline.run_stage2 import (
    REQUIRED_PREDICTION_COLUMNS,
    finalize_result_source_bundle,
    run_card,
    _decorate_predictions,
    _transform_pathway,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage2_transforms_are_explicit_and_deterministic():
    pathway = np.arange(18, dtype=np.float32).reshape(6, 3)
    coordinates = np.stack([np.arange(6), np.zeros(6)], axis=1).astype(np.float32)
    first = _transform_pathway(
        pathway,
        coordinates,
        card_id="S2-SPATIAL-SHUFFLE-001",
        slide_id="slide-1",
        seed=42,
        fold=0,
    )
    second = _transform_pathway(
        pathway,
        coordinates,
        card_id="S2-SPATIAL-SHUFFLE-001",
        slide_id="slide-1",
        seed=42,
        fold=0,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        _transform_pathway(
            pathway,
            coordinates,
            card_id="S2-ABS-001",
            slide_id="slide-1",
            seed=42,
            fold=0,
        ),
        pathway,
    )


def test_stage2_prediction_contract_has_explicit_identity_fields():
    base = pd.DataFrame(
        {
            "case_id": ["case-1"],
            "slide_id": ["slide-1"],
            "y_true": [1],
            "y_pred": [0.75],
        }
    )
    table = _decorate_predictions(
        base,
        task="pCR",
        seed=1,
        fold=0,
        card_id="S2-ABS-001",
        split="test",
        evidence_scope="slide_repeated_holdout_patient_nonindependent",
        warnings=["WARN_PATIENT_NONINDEPENDENCE"],
        prediction_kind="repeated_holdout_test",
    )
    assert set(REQUIRED_PREDICTION_COLUMNS).issubset(table.columns)
    assert table.loc[0, "sample_id"] == "slide-1"
    assert table.loc[0, "spatial_cluster_id"] == "NA"
    assert table.loc[0, "pathway_id"] == "all_30"
    assert table.loc[0, "prediction_kind"] == "repeated_holdout_test"


def test_stage2_card_runner_can_be_contract_tested_without_training(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    source_config = (
        source_root / "configs" / "phase3_stage2_training_config_v002_20260807.json"
    )
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["selected_effective_slide_count"] = 3
    config["expected_patient_count"] = 3
    config["pathology_dim"] = 2
    config["model"]["image_dim"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    records = []
    assignments = []
    for index, split in enumerate(("train", "val", "test")):
        slide_id = f"slide-{index}"
        case_id = f"case-{index}"
        pathology_path = asset_dir / f"{slide_id}_pathology.npy"
        pathway_path = asset_dir / f"{slide_id}_pathway.npy"
        coordinates_path = asset_dir / f"{slide_id}_coordinates.npy"
        np.save(pathology_path, np.ones((2, 2), dtype=np.float32) * index)
        np.save(pathway_path, np.ones((2, 30), dtype=np.float32) * (index + 1))
        np.save(coordinates_path, np.array([[0, 0], [1, 0]], dtype=np.float32))
        records.append(
            {
                "case_id": case_id,
                "slide_id": slide_id,
                "pathology_path": str(pathology_path),
                "pathology_sha256": _sha256(pathology_path),
                "pathway_path": str(pathway_path),
                "pathway_sha256": _sha256(pathway_path),
                "coordinates_path": str(coordinates_path),
                "coordinates_sha256": _sha256(coordinates_path),
            }
        )
        assignments.append(
            {"slide_id": slide_id, "case_id": case_id, "label": "pCR", "split": split}
        )
    asset_manifest = tmp_path / "asset_manifest.json"
    asset_manifest.write_text(
        json.dumps(
            {
                "pathway_input_mode": "legacy_zscore_compatibility",
                "compatibility_annotation": {
                    "source_kind": "legacy_zscore",
                    "raw_mpp2_claim": False,
                    "evidence_status": "pending_review",
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    split_root = tmp_path / "slide_seed_package_v001"
    assignment_path = split_root / "seed_1" / "pCR"
    assignment_path.mkdir(parents=True)
    pd.DataFrame(assignments).to_csv(
        assignment_path / "slide_assignments_0.csv", index=False
    )
    (split_root / "package_manifest.json").write_text("{}", encoding="utf-8")

    def no_training(
        model,
        train_examples,
        validation_examples,
        test_examples,
        *,
        seed,
        epochs,
        learning_rate,
        patience,
        patient_weighting,
        patient_overlap_policy,
        device,
    ):
        return FoldTrainingResult(
            model=model,
            history=[
                {"epoch": 0.0, "train_loss": 0.2, "validation_loss": 0.1}
            ],
            validation_predictions=pd.DataFrame(
                {"case_id": ["case-1"], "slide_id": ["slide-1"], "y_true": [1], "y_pred": [0.6]}
            ),
            test_predictions=pd.DataFrame(
                {"case_id": ["case-2"], "slide_id": ["slide-2"], "y_true": [0], "y_pred": [0.4]}
            ),
            evidence_scope="slide_repeated_holdout_patient_nonindependent",
            warnings=["WARN_PATIENT_NONINDEPENDENCE"],
        )

    output_root = tmp_path / "attempt"
    metadata = run_card(
        card_id="S2-ABS-001",
        config_path=config_path,
        asset_manifest_path=asset_manifest,
        split_package_root=split_root,
        output_root=output_root,
        device="cpu",
        trainer=no_training,
        fit_limit=1,
    )
    assert metadata["run_units"] == 1
    assert (output_root / "metrics.json").is_file()
    prediction = pd.read_csv(
        output_root / "predictions" / "pCR" / "seed_1" / "fold_0" / "test.csv"
    )
    assert set(REQUIRED_PREDICTION_COLUMNS).issubset(prediction.columns)
    assert not (output_root / "result_source_bundle" / "result.json").exists()


def test_result_finalizer_declares_prediction_return_for_successful_standard_job(tmp_path):
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    job = {
        "schema_version": "2.0",
        "job_id": "W004-A001",
        "experiment_id": "phase3_dual_baseline_spatial_pathway_transfer_v001_20260804",
        "workspace_id": "W004",
        "attempt_id": "A001",
        "protocol_revision": 2,
        "approval_id": "APR-W004-STAGE2-R002",
        "critical_contract_sha256": "b" * 64,
        "source_commit": "a" * 40,
        "phase": "formal",
        "run_units": 40,
        "command_id": "standard_training",
        "artifact_policy": {
            "critical": "sha256_required",
            "supporting": "size_inventory_default",
            "diagnostic": "non_evidence",
            "diagnostic_logs": "include_only_when_needed_for_agent_analysis",
            "large_artifacts": "registered_path_size_sha256_only",
            "raw_predictions": {
                "requirement": "required_inline_for_every_evaluated_split",
                "artifact_kind": "raw_prediction_table",
                "large_artifact_exemption": False,
                "required_columns": list(REQUIRED_PREDICTION_COLUMNS),
            },
        },
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    (output_root / "attempt_started.json").write_text(
        json.dumps({"event_type": "EXPERIMENT_STARTED", "job_id": "W004-A001", "attempt_id": "A001"}),
        encoding="utf-8",
    )
    (output_root / "attempt_terminal.json").write_text(
        json.dumps(
            {
                "event_type": "EXPERIMENT_TERMINAL",
                "job_id": "W004-A001",
                "attempt_id": "A001",
                "status": "completed",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    (output_root / "metrics.json").write_text("{}", encoding="utf-8")
    (output_root / "selection_proof.json").write_text("{}", encoding="utf-8")
    (output_root / "prediction_index.json").write_text("{}", encoding="utf-8")
    (output_root / "resolved_config.json").write_text("{}", encoding="utf-8")
    (output_root / "runner_metadata.json").write_text(
        json.dumps({"checkpoint_records": []}), encoding="utf-8"
    )
    (output_root / "runner.log").write_text("test\n", encoding="utf-8")
    prediction = output_root / "predictions" / "pCR" / "seed_1" / "fold_0" / "test.csv"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(
        "sample_id,spatial_cluster_id,pathway_id,y_true,y_pred\n"
        "slide-1,NA,all_30,0,0.4\n",
        encoding="utf-8",
    )
    history = output_root / "training_history" / "pcr_seed1_fold0.json"
    history.parent.mkdir(parents=True)
    history.write_text("{}", encoding="utf-8")

    report = finalize_result_source_bundle(
        output_root=output_root,
        job_manifest_path=job_path,
    )
    result = json.loads(
        (output_root / "result_source_bundle" / "result.json").read_text(encoding="utf-8")
    )
    assert report["prediction_artifact_count"] == 1
    assert result["prediction_return"]["applicability"] == "required"
    assert len(result["prediction_return"]["evaluated_splits"]) == 1


def test_result_finalizer_sanitizes_checkpoint_metadata_and_supports_repack_bundle(tmp_path):
    """A failed first packaging pass must not force loss of completed training output."""
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    job = {
        "schema_version": "2.0",
        "job_id": "W004-A002",
        "experiment_id": "phase3_dual_baseline_spatial_pathway_transfer_v001_20260804",
        "workspace_id": "W004",
        "attempt_id": "A002",
        "protocol_revision": 2,
        "approval_id": "APR-W004-STAGE2-R002",
        "critical_contract_sha256": "b" * 64,
        "source_commit": "a" * 40,
        "phase": "formal",
        "run_units": 40,
        "command_id": "standard_training",
        "artifact_policy": {
            "critical": "sha256_required",
            "supporting": "size_inventory_default",
            "diagnostic": "non_evidence",
            "diagnostic_logs": "include_only_when_needed_for_agent_analysis",
            "large_artifacts": "registered_path_size_sha256_only",
            "raw_predictions": {
                "requirement": "required_inline_for_every_evaluated_split",
                "artifact_kind": "raw_prediction_table",
                "large_artifact_exemption": False,
                "required_columns": list(REQUIRED_PREDICTION_COLUMNS),
            },
        },
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    (output_root / "attempt_started.json").write_text(
        json.dumps({"event_type": "EXPERIMENT_STARTED", "job_id": "W004-A002", "attempt_id": "A002"}),
        encoding="utf-8",
    )
    (output_root / "attempt_terminal.json").write_text(
        json.dumps({"event_type": "EXPERIMENT_TERMINAL", "job_id": "W004-A002", "attempt_id": "A002", "status": "completed", "returncode": 0}),
        encoding="utf-8",
    )
    for name in ("metrics.json", "selection_proof.json", "prediction_index.json", "resolved_config.json"):
        (output_root / name).write_text("{}", encoding="utf-8")
    (output_root / "runner_metadata.json").write_text(
        json.dumps({"checkpoint_records": [{
            "artifact_id": "checkpoint_pcr_seed1_fold0",
            "server_path": "D:\\runs\\checkpoint.pt",
            "size_bytes": 1024,
            "sha256": "c" * 64,
            "retention": "retain_on_server_until_explicit_user_disposition",
            "card_id": "S2-ORD-SLIDE-001",
            "task": "pCR",
            "seed": 1,
            "fold": 0,
        }]}),
        encoding="utf-8",
    )
    (output_root / "runner.log").write_text("original training log\n", encoding="utf-8")
    prediction = output_root / "predictions" / "pCR" / "seed_1" / "fold_0" / "test.csv"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(
        "sample_id,spatial_cluster_id,pathway_id,y_true,y_pred\nslide-1,NA,all_30,0,0.4\n",
        encoding="utf-8",
    )
    history = output_root / "training_history" / "pcr_seed1_fold0.json"
    history.parent.mkdir(parents=True)
    history.write_text("{}", encoding="utf-8")

    report = finalize_result_source_bundle(
        output_root=output_root,
        job_manifest_path=job_path,
        source_bundle_name="result_source_bundle_repack_R001",
    )
    result = json.loads(
        (output_root / "result_source_bundle_repack_R001" / "result.json").read_text(encoding="utf-8")
    )

    assert report["source_bundle"].endswith("result_source_bundle_repack_R001")
    assert result["prediction_return"]["applicability"] == "required"
    assert set(result["large_artifacts"][0]) == {
        "artifact_id", "kind", "server_path", "size_bytes", "sha256", "retention",
    }
    assert result["large_artifacts"][0]["kind"] == "model_checkpoint"
