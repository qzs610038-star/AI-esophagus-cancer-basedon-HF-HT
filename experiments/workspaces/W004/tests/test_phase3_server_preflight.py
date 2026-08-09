from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.workspaces.W004.code.phase3_pipeline.server_preflight import run_preflight


PATHWAYS = [f"p{i:02d}" for i in range(30)]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, include_excluded_asset: bool = False) -> tuple[Path, Path, Path]:
    exp = tmp_path / "experiments" / "exp"
    package = exp / "slide_seed_package_v001"
    source = [
        {"case_id": "c1", "slide_id": "s1"},
        {"case_id": "c2", "slide_id": "s2"},
        {"case_id": "c3", "slide_id": "bad"},
    ]
    effective = source[:2]
    excluded = [{"slide_id": "bad", "reason": "manual"}]
    _write_csv(package / "cohort_manifest.csv", ["case_id", "slide_id"], source)
    _write_csv(package / "effective_cohort.csv", ["case_id", "slide_id"], effective)
    _write_csv(package / "excluded_bad_slides.csv", ["slide_id", "reason"], excluded)
    assignments = [dict(row, label="x", split=split) for row, split in zip(effective, ("train", "val"))]
    assignments.append({"case_id": "c1", "slide_id": "s1-test", "label": "x", "split": "test"})
    # Keep exact effective coverage while all required split labels exist by using a three-slide test fixture.
    source.insert(2, {"case_id": "c1", "slide_id": "s3"})
    effective.append(source[2])
    _write_csv(package / "cohort_manifest.csv", ["case_id", "slide_id"], source)
    _write_csv(package / "effective_cohort.csv", ["case_id", "slide_id"], effective)
    assignments = [dict(row, label="x", split=split) for row, split in zip(effective, ("train", "val", "test"))]
    for task in ("pCR", "MPR"):
        _write_csv(package / "seed_1" / task / "slide_assignments_0.csv", ["case_id", "slide_id", "label", "split"], assignments)

    canonical = tmp_path / "mpp_standard_splits" / "group_2"
    canonical.mkdir(parents=True)
    (canonical / "zscore_manifest.json").write_text(json.dumps({"pathway_names": PATHWAYS}), encoding="utf-8")
    config = {
        "source_slide_count": 4,
        "excluded_bad_slide_count": 1,
        "selected_effective_slide_count": 3,
        "expected_patient_count": 2,
        "seeds": [1],
        "tasks": ["pCR", "MPR"],
        "folds_per_seed": 1,
        "training_fit_policy": {
            "fit_scope": "train_only",
            "validation_mode": "transform_only",
            "test_mode": "transform_only_after_checkpoint_freeze",
            "external_mode": "transform_only_no_fit",
        },
    }
    config_path = exp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    records = []
    asset_rows = effective if not include_excluded_asset else effective[:-1] + [{"case_id": "c3", "slide_id": "bad"}]
    for row in asset_rows:
        slide = row["slide_id"]
        pathology = exp / f"{slide}-pathology.npy"
        pathway = exp / f"{slide}-pathway.npy"
        coordinates = exp / f"{slide}-coordinates.npy"
        np.save(pathology, np.ones((2, 4), dtype=np.float32))
        np.save(pathway, np.ones((2, 30), dtype=np.float32))
        np.save(coordinates, np.ones((2, 2), dtype=np.float32))
        records.append({
            "case_id": row["case_id"], "slide_id": slide,
            "pathology_path": str(pathology), "pathology_sha256": _sha(pathology),
            "pathway_path": str(pathway), "pathway_sha256": _sha(pathway),
            "coordinates_path": str(coordinates), "coordinates_sha256": _sha(coordinates),
        })
    asset_path = exp / "assets.json"
    fit_scope_records = [
        {"seed": 1, "task": task, "fold": 0, "fit_slide_ids": ["s1"]}
        for task in ("pCR", "MPR")
    ]
    asset_path.write_text(json.dumps({"schema_version": "1.0", "pathway_names": PATHWAYS, "records": records, "fit_scope_records": fit_scope_records}), encoding="utf-8")
    return exp, config_path, asset_path


def test_full_preflight_passes_and_patient_overlap_is_warn(tmp_path: Path) -> None:
    exp, config, assets = _fixture(tmp_path)
    report = run_preflight(exp, config, assets)
    assert report.failed is False
    assert {item.name: item.status for item in report.results}["slide_split_patient_overlap"] == "WARN"
    assert report.payload()["verdict"] == "GO"


def test_excluded_slide_reentry_is_hard_failure(tmp_path: Path) -> None:
    exp, config, assets = _fixture(tmp_path, include_excluded_asset=True)
    report = run_preflight(exp, config, assets)
    by_name = {item.name: item for item in report.results}
    assert report.failed is True
    assert by_name["feature_mpp2_coverage"].status == "FAIL"


def test_missing_asset_manifest_aggregates_all_asset_dependent_hard_failures(tmp_path: Path) -> None:
    exp, config, _ = _fixture(tmp_path)
    report = run_preflight(exp, config, None)
    failed = {item.name for item in report.results if item.status == "FAIL"}
    assert {"feature_mpp2_coverage", "pathway_30_order", "patch_coordinate_alignment", "train_only_fit_isolation"} <= failed


def test_validation_slide_in_fit_scope_is_hard_failure(tmp_path: Path) -> None:
    exp, config, assets = _fixture(tmp_path)
    payload = json.loads(assets.read_text(encoding="utf-8"))
    payload["fit_scope_records"][0]["fit_slide_ids"].append("s2")
    assets.write_text(json.dumps(payload), encoding="utf-8")
    report = run_preflight(exp, config, assets)
    by_name = {item.name: item for item in report.results}
    assert by_name["train_only_fit_isolation"].status == "FAIL"


def test_legacy_zscore_compatibility_route_is_explicit_and_not_raw(tmp_path: Path) -> None:
    exp, config, assets = _fixture(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload.update(
        {
            "pathway_input_mode": "legacy_zscore_compatibility",
            "accepted_pathway_input": "legacy_zscore_compatibility",
            "compatibility_only": True,
            "raw_mpp2_claim": False,
            "pathway_evidence_status": "pending_review",
        }
    )
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    asset_payload = json.loads(assets.read_text(encoding="utf-8"))
    asset_payload.update(
        {
            "pathway_input_mode": "legacy_zscore_compatibility",
            "compatibility_annotation": {
                "source_kind": "legacy_zscore",
                "raw_mpp2_claim": False,
                "evidence_status": "pending_review",
            },
        }
    )
    assets.write_text(json.dumps(asset_payload), encoding="utf-8")

    report = run_preflight(exp, config, assets)

    assert report.failed is False
    payload = report.payload()
    assert payload["verdict"] == "COMPATIBILITY_ONLY"
    assert payload["pathway_input_mode"] == "legacy_zscore_compatibility"
    assert payload["compatibility_only"] is True
    assert payload["training_authorized"] is False
    by_name = {item.name: item for item in report.results}
    assert by_name["pathway_input_route"].status == "WARN"


def test_legacy_zscore_route_rejects_missing_compatibility_annotation(tmp_path: Path) -> None:
    exp, config, assets = _fixture(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload.update(
        {
            "pathway_input_mode": "legacy_zscore_compatibility",
            "accepted_pathway_input": "legacy_zscore_compatibility",
            "compatibility_only": True,
            "raw_mpp2_claim": False,
            "pathway_evidence_status": "pending_review",
        }
    )
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    report = run_preflight(exp, config, assets)

    by_name = {item.name: item for item in report.results}
    assert report.failed is True
    assert by_name["feature_mpp2_coverage"].status == "FAIL"
