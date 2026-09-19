from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analyze_local import _aggregate, analyze_batch, analyze_batches, compute_metrics
from data import Normalization
from errors import IdentityMismatchError


def _normalization() -> Normalization:
    names = tuple(f"p{i}" for i in range(30))
    return Normalization(
        names,
        np.arange(30, dtype=np.float64),
        np.linspace(1, 2, 30),
        "train",
        1,
        (-100, 100),
        9472,
        "synthetic",
    )


def _write_internal(
    path: Path,
    model: str,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 42,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_z=prediction,
        target_z=target,
        patients=np.asarray(["A", "A", "B", "B"], dtype=str),
        source_groups=np.asarray(["sA", "sA", "sB", "sB"], dtype=str),
        spots=np.asarray(["x0", "x1", "x0", "x1"], dtype=str),
        pathways=np.asarray([f"p{i}" for i in range(30)], dtype=str),
        arm=np.asarray("point"),
        model=np.asarray(model),
        seed=np.asarray(seed),
    )


def test_metric_report_keeps_patient_macro_and_flattened_pcc_distinct():
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    prediction = target.copy()
    metrics = compute_metrics(
        prediction,
        target,
        ["A", "A", "B", "B"],
        _normalization(),
    )
    assert metrics["patient_macro_pathway_pcc_measured"] == 1.0
    assert metrics["pooled_flattened_pcc"] == 1.0
    assert metrics["pooled_z_rmse"] == 0.0
    assert metrics["pooled_z_mae"] == 0.0
    assert metrics["pooled_spearman"] == 1.0
    assert metrics["pooled_ccc"] == 1.0
    assert metrics["train_std_normalized_rmse_mean"] == 0.0
    assert metrics["raw_truth_scope"] == "inverse_of_clipped_z_target_not_original_raw_truth"


def test_original_raw_truth_difference_from_inverse_z_is_reported_not_hidden():
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    normalization = _normalization()
    supplied_raw = normalization.inverse_transform(target) + 0.5
    metrics = compute_metrics(
        target,
        target,
        ["A", "A", "B", "B"],
        normalization,
        raw_target=supplied_raw,
    )
    assert metrics["raw_truth_scope"] == "identity_aligned_original_raw_truth"
    assert metrics["raw_target_vs_inverse_z_mae"] == pytest.approx(0.5)
    assert metrics["raw_target_vs_inverse_z_max_abs"] == pytest.approx(0.5)


def test_single_seed_analysis_never_fabricates_three_seed_sd_or_changes_raw(tmp_path: Path):
    batch = tmp_path / "batch"
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    _write_internal(batch / "reference_42_uni2h" / "raw" / "internal_best.npz", "uni2h", target, target)
    _write_internal(batch / "train_42_uni" / "raw" / "internal_best.npz", "uni", target + 0.1, target)
    manifest = {
        "status": "succeeded",
        "scope": "seed42",
        "tasks": [
            {"kind": "reference", "model": "uni2h", "seed": 42, "run_id": "reference_42_uni2h", "status": "succeeded"},
            {"kind": "train", "model": "uni", "seed": 42, "run_id": "train_42_uni", "status": "succeeded"},
        ],
    }
    (batch / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw_path = batch / "train_42_uni" / "raw" / "internal_best.npz"
    before = raw_path.read_bytes()
    output = tmp_path / "analysis"
    report = analyze_batch(batch, output, _normalization())
    assert raw_path.read_bytes() == before
    assert report["scope"] == "single_seed_42"
    assert report["aggregate"]["uni"]["n_seeds"] == 1
    assert "sample_sd" not in report["aggregate"]["uni"]
    assert (output / "首批种子42审核报告.md").is_file()
    markdown = (output / "首批种子42审核报告.md").read_text(encoding="utf-8")
    assert "预测头参数量" in markdown
    assert "270110" in markdown
    assert (output / "metric_details.json").is_file()
    assert report["spatial_metrics"]["status"] == "not_computed_not_predefined"
    assert report["top_k_overlap"]["status"] == "not_computed_not_predefined"


def test_three_seed_statistics_are_computed_per_split_not_from_averaged_predictions():
    metric_names = (
        "patient_macro_pathway_pcc_measured",
        "pooled_flattened_pcc",
        "pooled_pathway_pcc",
        "pooled_z_mse",
        "pooled_z_rmse",
        "pooled_z_mae",
        "pooled_spearman",
        "pooled_ccc",
        "train_std_normalized_rmse_mean",
        "raw_mae",
        "raw_r2",
    )
    rows = []
    for seed, value in zip((42, 43, 44), (1.0, 2.0, 3.0)):
        rows.append(
            {
                "model": "uni",
                "seed": seed,
                "split": "external_test",
                **{name: value for name in metric_names},
            }
        )
    rows[0]["raw_r2"] = float("nan")
    aggregate = _aggregate(rows, split="external_test")
    assert aggregate["uni"]["mean"]["pooled_z_rmse"] == 2.0
    assert aggregate["uni"]["sample_sd"]["pooled_z_rmse"] == 1.0
    assert aggregate["uni"]["mean"]["raw_r2"] == 2.5
    assert aggregate["uni"]["three_seed_scope_complete"] is True


def test_seed42_and_remaining_batches_can_be_combined_into_one_three_seed_report(tmp_path: Path):
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    batches = [tmp_path / "seed42_batch", tmp_path / "remaining_batch"]
    tasks_by_batch = [[], []]
    for model in ("uni2h", "uni", "virchow2"):
        kind = "reference" if model == "uni2h" else "train"
        for seed in (42, 43, 44):
            batch_index = 0 if seed == 42 else 1
            run_id = f"{kind}_{seed}_{model}"
            _write_internal(
                batches[batch_index] / run_id / "raw" / "internal_best.npz",
                model,
                target + (seed - 42) * 0.01,
                target,
                seed=seed,
            )
            tasks_by_batch[batch_index].append(
                {
                    "kind": kind,
                    "model": model,
                    "seed": seed,
                    "run_id": run_id,
                    "status": "succeeded",
                }
            )
    for batch, tasks in zip(batches, tasks_by_batch):
        batch.mkdir(parents=True, exist_ok=True)
        (batch / "batch_manifest.json").write_text(
            json.dumps({"status": "succeeded", "tasks": tasks}), encoding="utf-8"
        )
    output = tmp_path / "combined_analysis"
    report = analyze_batches(batches, output, _normalization())
    assert report["scope"] == "three_seed_complete"
    assert report["aggregate"]["virchow2"]["three_seed_scope_complete"] is True
    assert "sample_sd" in report["aggregate"]["uni2h"]
    assert (output / "三种子统一分析报告.md").is_file()
    markdown = (output / "三种子统一分析报告.md").read_text(encoding="utf-8")
    assert "三种子均值 ± 样本标准差" in markdown
    assert "0.010000 ± 0.010000" in markdown


def test_local_analysis_rejects_prediction_metadata_from_a_different_seed(tmp_path: Path):
    batch = tmp_path / "batch"
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    run_id = "train_43_uni"
    _write_internal(batch / run_id / "raw" / "internal_best.npz", "uni", target, target)
    (batch / "batch_manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "kind": "train",
                        "model": "uni",
                        "seed": 43,
                        "run_id": run_id,
                        "status": "succeeded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(IdentityMismatchError, "seed"):
        analyze_batch(batch, tmp_path / "analysis", _normalization())


def test_local_analysis_rejects_a_batch_from_another_protocol(tmp_path: Path):
    batch = tmp_path / "batch"
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    run_id = "train_42_uni"
    _write_internal(batch / run_id / "raw" / "internal_best.npz", "uni", target, target)
    (batch / "batch_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "different_protocol",
                "tasks": [
                    {
                        "kind": "train",
                        "model": "uni",
                        "seed": 42,
                        "run_id": run_id,
                        "status": "succeeded",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IdentityMismatchError, match="protocol_version"):
        analyze_batch(batch, tmp_path / "analysis", _normalization())


def test_external_truth_is_joined_by_identity_and_never_read_from_server_prediction(tmp_path: Path):
    batch = tmp_path / "batch"
    normalization = _normalization()
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    train_id = "train_42_uni"
    external_id = "external_42_uni"
    _write_internal(batch / train_id / "raw" / "internal_best.npz", "uni", target, target)
    external_path = batch / external_id / "raw" / "external_predictions.npz"
    external_path.parent.mkdir(parents=True)
    spots = np.asarray(["x0", "x1", "x2", "x3"], dtype=str)
    np.savez_compressed(
        external_path,
        pred_z=target,
        pred_raw=normalization.inverse_transform(target),
        patients=np.asarray(["XZY"] * 4, dtype=str),
        source_groups=np.asarray(["MPP2_XZY_source01"] * 4, dtype=str),
        spots=spots,
        pathways=np.asarray([f"p{i}" for i in range(30)], dtype=str),
        checkpoint_kind=np.asarray("formal"),
        arm=np.asarray("point"),
        model=np.asarray("uni"),
        seed=np.asarray(42),
    )
    (batch / "batch_manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"kind": "train", "model": "uni", "seed": 42, "run_id": train_id, "status": "succeeded"},
                    {"kind": "external", "model": "uni", "seed": 42, "run_id": external_id, "status": "succeeded"},
                ]
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "xzy_truth.npz"
    reverse = np.asarray([3, 2, 1, 0])
    np.savez_compressed(
        truth_path,
        target_z=target[reverse],
        target_raw=normalization.inverse_transform(target)[reverse],
        patients=np.asarray(["XZY"] * 4, dtype=str)[reverse],
        source_groups=np.asarray(["MPP2_XZY_source01"] * 4, dtype=str)[reverse],
        spots=spots[reverse],
        pathways=np.asarray([f"p{i}" for i in range(30)], dtype=str),
    )
    report = analyze_batch(
        batch,
        tmp_path / "analysis",
        normalization,
        external_targets=truth_path,
    )
    external_row = next(row for row in report["rows"] if row["split"] == "external_test")
    assert external_row["pooled_flattened_pcc"] == 1.0
    assert external_row["raw_target_vs_inverse_z_mae"] == 0.0
    assert report["external_metrics_status"] == "computed"


def test_cli_count_contract_can_reject_a_truncated_internal_prediction(tmp_path: Path):
    batch = tmp_path / "batch"
    target = np.stack(
        [np.arange(30), np.arange(30) + 1, np.arange(30), np.arange(30) + 2]
    ).astype(np.float64)
    run_id = "train_42_uni"
    _write_internal(batch / run_id / "raw" / "internal_best.npz", "uni", target, target)
    (batch / "batch_manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"kind": "train", "model": "uni", "seed": 42, "run_id": run_id, "status": "succeeded"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IdentityMismatchError, match="internal_val"):
        analyze_batch(
            batch,
            tmp_path / "analysis",
            _normalization(),
            expected_counts={"internal_val": 1078, "external_test": 1039},
        )
