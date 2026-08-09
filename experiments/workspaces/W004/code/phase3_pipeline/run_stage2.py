"""Governed Stage 2 runner and result-source finalizer for W004.

The runner consumes only the explicit server asset manifest and the committed
slide assignment package.  It never discovers sibling files, fits a transform
on validation/test data, or writes into the source worktree.  The actual
server invocation is still guarded by the v2 job runner; this module only
becomes a training entrypoint after that boundary is crossed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contracts import validate_fold_assignments
from .engine import FoldTrainingResult, SlideExample, annotate_fold_predictions, train_single_fold
from .evaluation import binary_metrics, patient_cluster_bootstrap_auc, validate_repeated_holdout_predictions
from .io import load_verified_patch_bag, sha256_file
from .models import MILBagClassifier
from .transforms import slice_percentile_rank, spatial_smooth


EXPERIMENT_ID = "phase3_dual_baseline_spatial_pathway_transfer_v001_20260804"
WORKSPACE_ID = "W004"
TASKS = ("pCR", "MPR")
SEEDS = (1, 2, 42, 61)
FOLDS = tuple(range(5))
REQUIRED_PREDICTION_COLUMNS = (
    "sample_id",
    "spatial_cluster_id",
    "pathway_id",
    "y_true",
    "y_pred",
)

CARD_SPECS: dict[str, dict[str, Any]] = {
    "S2-ABS-001": {
        "transform_name": "absolute_identity",
        "transform_definition": "legacy z-score pathway matrix unchanged; spatial_lambda=0",
    },
    "S2-ORD-SLIDE-001": {
        "transform_name": "slice_ordinal",
        "transform_definition": "within-slide percentile rank independently for each of 30 pathways",
    },
    "S2-SPATIAL-IDENTITY-001": {
        "transform_name": "spatial_smoothed_k5_lambda05",
        "transform_definition": "coordinate kNN smoothing with k=5 and lambda=0.5",
    },
    "S2-SPATIAL-SHUFFLE-001": {
        "transform_name": "spatial_coordinate_permutation_control",
        "transform_definition": "features unchanged; deterministic coordinate permutation before kNN smoothing",
    },
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_json(value: Any) -> Any:
    """Convert NumPy/Python scalar values and NaN metrics to JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_safe_json(item) for item in value.tolist()]
    return value


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _resolve_manifest_record(record: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    resolved = dict(record)
    for role in ("pathology", "pathway", "coordinates"):
        raw = Path(str(record.get(f"{role}_path", "")))
        if not raw.is_absolute():
            raw = (manifest_path.parent / raw).resolve()
        resolved[f"{role}_path"] = str(raw)
    return resolved


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "workspace_id",
        "execution_mode",
        "selected_split_unit",
        "source_slide_count",
        "excluded_bad_slide_count",
        "selected_effective_slide_count",
        "expected_patient_count",
        "seeds",
        "folds_per_seed",
        "tasks",
        "pathology_dim",
        "pathway_dim",
        "pathway_input_mode",
        "accepted_pathway_input",
        "pathway_evidence_status",
        "raw_mpp2_claim",
        "compatibility_only",
        "training_fit_policy",
        "model",
        "stage2_cards",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Stage 2 config missing fields: {missing}")
    if config["experiment_id"] != EXPERIMENT_ID or config["workspace_id"] != WORKSPACE_ID:
        raise ValueError("Stage 2 config is not bound to W004 experiment")
    if config["execution_mode"] != "server_training_after_explicit_approval":
        raise ValueError("Stage 2 config is not a formal server-training config")
    if config["selected_split_unit"] != "slide":
        raise ValueError("Stage 2 runner only accepts the registered slide split")
    if list(config["seeds"]) != list(SEEDS) or int(config["folds_per_seed"]) != len(FOLDS):
        raise ValueError("Stage 2 seed/fold grid is not 4 seeds x 5 folds")
    if list(config["tasks"]) != list(TASKS):
        raise ValueError("Stage 2 runner requires pCR and MPR in one batch")
    if config["pathway_input_mode"] != "legacy_zscore_compatibility":
        raise ValueError("Stage 2 W004 runner must remain on legacy compatibility input")
    if config["accepted_pathway_input"] != config["pathway_input_mode"]:
        raise ValueError("accepted pathway input does not match pathway input mode")
    if config["pathway_evidence_status"] not in {"diagnostic_only", "pending_review"}:
        raise ValueError("compatibility pathway evidence must remain pending_review/diagnostic_only")
    if config["raw_mpp2_claim"] is not False or config["compatibility_only"] is not True:
        raise ValueError("legacy pathway route must be compatibility-only and non-raw")
    policy = config["training_fit_policy"]
    expected_policy = {
        "fit_scope": "train_only",
        "validation_mode": "transform_only",
        "test_mode": "transform_only_after_checkpoint_freeze",
        "external_mode": "transform_only_no_fit",
    }
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("Stage 2 training fit policy is not train-only")
    model = config["model"]
    if model.get("image_dim") != int(config["pathology_dim"]) or model.get("pathway_dim") != int(config["pathway_dim"]):
        raise ValueError("model dimensions do not match the Stage 2 config")
    cards = config["stage2_cards"]
    if not isinstance(cards, list) or not cards:
        raise ValueError("stage2_cards must be a non-empty list")
    unknown = sorted(set(str(item) for item in cards) - set(CARD_SPECS))
    if unknown:
        raise ValueError(f"Stage 2 config contains unsupported cards: {unknown}")
    return config


def load_asset_records(asset_manifest_path: Path, *, expected_count: int = 307) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load explicit records and retain the manifest's compatibility annotation."""

    payload = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if payload.get("pathway_input_mode") != "legacy_zscore_compatibility":
        raise ValueError("server asset manifest pathway route is not legacy compatibility")
    annotation = payload.get("compatibility_annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("server asset manifest lacks compatibility annotation")
    if annotation.get("source_kind") != "legacy_zscore" or annotation.get("raw_mpp2_claim") is not False:
        raise ValueError("server asset manifest has an invalid raw-MPP2 compatibility claim")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"server asset manifest must contain exactly {expected_count} records")
    resolved_records = [_resolve_manifest_record(item, asset_manifest_path) for item in records]
    slide_ids = [str(item.get("slide_id", "")).strip() for item in resolved_records]
    if any(not item for item in slide_ids) or len(set(slide_ids)) != expected_count:
        raise ValueError("server asset manifest slide IDs must be unique and non-empty")
    record_map = {slide_id: record for slide_id, record in zip(slide_ids, resolved_records)}
    return payload, record_map


def _binary_label(raw: object, task: str) -> int:
    """Map an explicit split label to 0/1 without using row position."""

    text = str(raw).strip().lower()
    positive = task.lower()
    if text in {"1", "1.0", "true", positive, f"{positive}_positive"}:
        return 1
    if text in {"0", "0.0", "false", f"non_{positive}", f"non-{positive}", f"{positive}_negative"}:
        return 0
    raise ValueError(f"cannot map {task} label {raw!r} to binary")


def load_assignments(
    split_package_root: Path,
    *,
    task: str,
    seed: int,
    fold: int,
    expected_slide_ids: set[str],
) -> pd.DataFrame:
    path = split_package_root / f"seed_{seed}" / task / f"slide_assignments_{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing committed assignment file: {path}")
    assignments = pd.read_csv(path, dtype=str)
    required = {"slide_id", "case_id", "label", "split"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"assignment missing columns: {sorted(required - set(assignments.columns))}")
    audit = validate_fold_assignments(
        assignments[["slide_id", "case_id", "split"]],
        split_unit="slide",
        expected_slide_ids=expected_slide_ids,
    )
    assignments = assignments.copy()
    assignments["slide_id"] = assignments["slide_id"].astype(str)
    assignments["case_id"] = assignments["case_id"].astype(str)
    assignments["split"] = assignments["split"].astype(str)
    assignments["_audit_overlap_count"] = int(audit["patient_overlap_count"])
    return assignments


def _transform_pathway(
    pathway: np.ndarray,
    coordinates: np.ndarray,
    *,
    card_id: str,
    slide_id: str,
    seed: int,
    fold: int,
) -> np.ndarray:
    if card_id == "S2-ABS-001":
        transformed = pathway.copy()
    elif card_id == "S2-ORD-SLIDE-001":
        transformed = np.asarray(slice_percentile_rank(pathway, [slide_id] * len(pathway)))
    elif card_id == "S2-SPATIAL-IDENTITY-001":
        transformed = np.asarray(spatial_smooth(pathway, coordinates, k=5, lam=0.5))
    elif card_id == "S2-SPATIAL-SHUFFLE-001":
        permutation = np.random.default_rng(
            _stable_seed(card_id, slide_id, seed, fold)
        ).permutation(len(pathway))
        transformed = np.asarray(
            spatial_smooth(
                pathway,
                coordinates,
                k=5,
                lam=0.5,
                coordinate_permutation=permutation,
            )
        )
    else:
        raise ValueError(f"unsupported Stage 2 card: {card_id}")
    return np.asarray(transformed, dtype=np.float32)


def build_examples(
    assignments: pd.DataFrame,
    *,
    split: str,
    task: str,
    card_id: str,
    seed: int,
    fold: int,
    record_map: Mapping[str, Mapping[str, Any]],
    bag_cache: dict[str, Any],
) -> list[SlideExample]:
    rows = assignments[assignments["split"] == split]
    if rows.empty:
        raise ValueError(f"{task} seed={seed} fold={fold} has empty {split} split")
    examples: list[SlideExample] = []
    for row in rows.itertuples(index=False):
        slide_id = str(row.slide_id)
        record = record_map.get(slide_id)
        if record is None:
            raise ValueError(f"assignment references an unbound slide: {slide_id}")
        if str(row.case_id) != str(record["case_id"]):
            raise ValueError(f"case/slide identity mismatch for {slide_id}")
        if slide_id not in bag_cache:
            bag_cache[slide_id] = load_verified_patch_bag(record, pathway_dim=30)
        bag = bag_cache[slide_id]
        pathway = _transform_pathway(
            np.asarray(bag.pathway),
            np.asarray(bag.coordinates),
            card_id=card_id,
            slide_id=slide_id,
            seed=seed,
            fold=fold,
        )
        examples.append(
            SlideExample(
                case_id=str(row.case_id),
                slide_id=slide_id,
                pathology=torch.as_tensor(np.asarray(bag.pathology), dtype=torch.float32),
                pathway=torch.as_tensor(pathway, dtype=torch.float32),
                label=_binary_label(row.label, task),
            )
        )
    return examples


def _new_model(config: Mapping[str, Any]) -> MILBagClassifier:
    model = config["model"]
    return MILBagClassifier(
        image_dim=int(model["image_dim"]),
        pathway_dim=int(model["pathway_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        fusion=str(model["fusion"]),
        pooling=str(model["pooling"]),
        low_rank_dim=int(model.get("low_rank_dim", 16)),
        num_outputs=1,
    )


def _decorate_predictions(
    predictions: pd.DataFrame,
    *,
    task: str,
    seed: int,
    fold: int,
    card_id: str,
    split: str,
    evidence_scope: str,
    warnings: Sequence[str],
    prediction_kind: str,
) -> pd.DataFrame:
    table = annotate_fold_predictions(
        predictions,
        task=task,
        seed=seed,
        fold=fold,
        model_name="stage2_mil_adaptive_gated_attention",
        prediction_kind="repeated_holdout_test" if prediction_kind == "repeated_holdout_test" else "partition_oof",
        evidence_scope=evidence_scope,
        warnings=warnings,
    )
    if prediction_kind != "repeated_holdout_test":
        table["prediction_kind"] = prediction_kind
    table["split"] = split
    table["card_id"] = card_id
    table["sample_id"] = table["slide_id"].astype(str)
    table["spatial_cluster_id"] = "NA"
    table["pathway_id"] = "all_30"
    ordered = [
        "sample_id",
        "case_id",
        "slide_id",
        "spatial_cluster_id",
        "pathway_id",
        "y_true",
        "y_pred",
        "task",
        "seed",
        "fold",
        "model",
        "card_id",
        "split",
        "prediction_kind",
        "evidence_scope",
        "warnings",
    ]
    return table[ordered]


def _fit_metric(table: pd.DataFrame) -> dict[str, Any]:
    metrics = binary_metrics(
        table["y_true"].to_numpy(),
        table["y_pred"].to_numpy(),
    )
    try:
        metrics["patient_cluster_bootstrap"] = patient_cluster_bootstrap_auc(
            table.rename(columns={"model": "model"}),
            n_bootstrap=200,
            seed=42,
        )
    except ValueError as exc:
        metrics["patient_cluster_bootstrap"] = None
        metrics["warning"] = f"patient_cluster_bootstrap_unavailable:{exc}"
    return _safe_json(metrics)


def _save_checkpoint(model: nn.Module, path: Path, metadata: Mapping[str, Any]) -> str:
    state = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return sha256_file(path)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_card(
    *,
    card_id: str,
    config_path: Path,
    asset_manifest_path: Path,
    split_package_root: Path,
    output_root: Path,
    device: str | None = None,
    trainer: Callable[..., FoldTrainingResult] = train_single_fold,
    fit_limit: int | None = None,
) -> dict[str, Any]:
    """Run exactly one registered card (40 independent endpoint/fold fits)."""

    if card_id not in CARD_SPECS:
        raise ValueError(f"card is not draft-ready or unsupported: {card_id}")
    config = _load_config(config_path)
    if card_id not in {str(item) for item in config["stage2_cards"]}:
        raise ValueError(f"card is not enabled by the Stage 2 config: {card_id}")
    if output_root.exists():
        existing = {
            item.name
            for item in output_root.iterdir()
            if item.name not in {"attempt_started.json"}
        }
        if existing:
            raise ValueError(f"output root is not empty; implicit retry is forbidden: {sorted(existing)}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, record_map = load_asset_records(
        asset_manifest_path,
        expected_count=int(config["selected_effective_slide_count"]),
    )
    expected_slide_ids = set(record_map)
    if int(config["pathway_dim"]) != 30:
        raise ValueError("W004 Stage 2 requires 30 pathway columns")
    actual_device = str(device or config.get("device", "cuda"))
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable; no CPU fallback is implicit")

    # Cache only verified bags; transformed pathway views are rebuilt per fold.
    bag_cache: dict[str, Any] = {}
    fit_records: list[dict[str, Any]] = []
    test_tables: list[pd.DataFrame] = []
    validation_tables: list[pd.DataFrame] = []
    checkpoint_records: list[dict[str, Any]] = []
    log_lines = [
        f"experiment_id={EXPERIMENT_ID}",
        f"workspace_id={WORKSPACE_ID}",
        f"card_id={card_id}",
        f"asset_manifest_sha256={sha256_file(asset_manifest_path)}",
        f"device={actual_device}",
        "pathway_input_mode=legacy_zscore_compatibility",
        "raw_mpp2_claim=false",
        "evidence_status=COMPATIBILITY_ONLY/pending_review",
    ]
    fit_counter = 0
    for task in TASKS:
        for seed in SEEDS:
            for fold in FOLDS:
                if fit_limit is not None and fit_counter >= fit_limit:
                    break
                fit_counter += 1
                assignments = load_assignments(
                    split_package_root,
                    task=task,
                    seed=seed,
                    fold=fold,
                    expected_slide_ids=expected_slide_ids,
                )
                train_examples = build_examples(
                    assignments,
                    split="train",
                    task=task,
                    card_id=card_id,
                    seed=seed,
                    fold=fold,
                    record_map=record_map,
                    bag_cache=bag_cache,
                )
                validation_examples = build_examples(
                    assignments,
                    split="val",
                    task=task,
                    card_id=card_id,
                    seed=seed,
                    fold=fold,
                    record_map=record_map,
                    bag_cache=bag_cache,
                )
                test_examples = build_examples(
                    assignments,
                    split="test",
                    task=task,
                    card_id=card_id,
                    seed=seed,
                    fold=fold,
                    record_map=record_map,
                    bag_cache=bag_cache,
                )
                model = _new_model(config)
                trained = trainer(
                    model,
                    train_examples,
                    validation_examples,
                    test_examples,
                    seed=seed,
                    epochs=int(config["epochs"]),
                    learning_rate=float(config["learning_rate"]),
                    patience=int(config["patience"]),
                    patient_weighting=True,
                    patient_overlap_policy="allow_with_warning",
                    device=actual_device,
                )
                warnings = list(trained.warnings)
                warnings.append("COMPATIBILITY_ONLY: legacy z-score pathway input is not raw MPP2")
                validation_table = _decorate_predictions(
                    trained.validation_predictions,
                    task=task,
                    seed=seed,
                    fold=fold,
                    card_id=card_id,
                    split="validation",
                    evidence_scope=trained.evidence_scope,
                    warnings=warnings,
                    prediction_kind="validation_selection",
                )
                test_table = _decorate_predictions(
                    trained.test_predictions,
                    task=task,
                    seed=seed,
                    fold=fold,
                    card_id=card_id,
                    split="test",
                    evidence_scope=trained.evidence_scope,
                    warnings=warnings,
                    prediction_kind="repeated_holdout_test",
                )
                validation_tables.append(validation_table)
                test_tables.append(test_table)
                fit_name = f"{task.lower()}_seed{seed}_fold{fold}"
                validation_path = output_root / "predictions" / task / f"seed_{seed}" / f"fold_{fold}" / "validation.csv"
                test_path = output_root / "predictions" / task / f"seed_{seed}" / f"fold_{fold}" / "test.csv"
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                validation_table.to_csv(validation_path, index=False, lineterminator="\n")
                test_table.to_csv(test_path, index=False, lineterminator="\n")
                history_path = output_root / "training_history" / f"{fit_name}.json"
                _write_json(history_path, {"task": task, "seed": seed, "fold": fold, "history": _safe_json(trained.history)})
                best_epoch = min(
                    trained.history,
                    key=lambda item: float(item["validation_loss"]),
                )["epoch"] if trained.history else None
                checkpoint_path = output_root / "checkpoints" / f"{fit_name}.pt"
                checkpoint_sha = _save_checkpoint(
                    trained.model,
                    checkpoint_path,
                    {
                        "task": task,
                        "seed": seed,
                        "fold": fold,
                        "card_id": card_id,
                        "best_epoch": best_epoch,
                        "selection_signal": "validation_bce_minimum",
                        "test_evaluated_after_checkpoint_freeze": True,
                    },
                )
                fit_metric = {
                    "task": task,
                    "seed": seed,
                    "fold": fold,
                    "card_id": card_id,
                    "validation": _fit_metric(validation_table),
                    "test": _fit_metric(test_table),
                    "evidence_scope": trained.evidence_scope,
                    "warnings": warnings,
                    "best_epoch": best_epoch,
                    "checkpoint_path": _relative(checkpoint_path, output_root),
                    "checkpoint_sha256": checkpoint_sha,
                    "validation_prediction_path": _relative(validation_path, output_root),
                    "test_prediction_path": _relative(test_path, output_root),
                    "training_history_path": _relative(history_path, output_root),
                }
                fit_records.append(_safe_json(fit_metric))
                checkpoint_records.append(
                    {
                        "artifact_id": f"checkpoint_{fit_name}",
                        "server_path": str(checkpoint_path.resolve()),
                        "size_bytes": checkpoint_path.stat().st_size,
                        "sha256": checkpoint_sha,
                        "retention": "retain_on_server_until_explicit_user_disposition",
                        "card_id": card_id,
                        "task": task,
                        "seed": seed,
                        "fold": fold,
                    }
                )
                log_lines.append(f"completed {fit_counter}: {fit_name} best_epoch={best_epoch}")
            if fit_limit is not None and fit_counter >= fit_limit:
                break
        if fit_limit is not None and fit_counter >= fit_limit:
            break
    expected_fits = len(TASKS) * len(SEEDS) * len(FOLDS)
    if fit_limit is None and fit_counter != expected_fits:
        raise AssertionError(f"Stage 2 card fit count mismatch: {fit_counter} != {expected_fits}")
    if not test_tables or not validation_tables:
        raise ValueError("Stage 2 card produced no predictions")

    test_all = pd.concat(test_tables, ignore_index=True)
    validation_all = pd.concat(validation_tables, ignore_index=True)
    repeated_holdout_audit = _safe_json(validate_repeated_holdout_predictions(test_all))
    metrics: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "workspace_id": WORKSPACE_ID,
        "card_id": card_id,
        "run_units": fit_counter,
        "expected_run_units": expected_fits,
        "pathway_input_mode": "legacy_zscore_compatibility",
        "raw_mpp2_claim": False,
        "evidence_status": "COMPATIBILITY_ONLY/pending_review",
        "repeated_holdout_audit": repeated_holdout_audit,
        "delta_auc_synergy": None,
        "delta_auc_warning": "single-modality same-seed/fold controls are not included in this 40-unit Stage 2 card; no delta was fabricated",
        "per_fit": fit_records,
        "task_test_metrics": {
            task: (
                _safe_json(
                    binary_metrics(
                        test_all.loc[test_all["task"] == task, "y_true"].to_numpy(),
                        test_all.loc[test_all["task"] == task, "y_pred"].to_numpy(),
                    )
                )
                if not test_all.loc[test_all["task"] == task].empty
                else None
            )
            for task in TASKS
        },
    }
    _write_json(output_root / "metrics.json", metrics)
    selection_proof = {
        "selection_rule": "minimum validation BCE; early stop after patience=5; restore best validation checkpoint; test only after freeze",
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "patience": int(config["patience"]),
        "patient_overlap_policy": "allow_with_warning",
        "prediction_kind": "repeated_holdout_test",
        "fit_records": fit_records,
    }
    _write_json(output_root / "selection_proof.json", selection_proof)
    _write_json(
        output_root / "resolved_config.json",
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "asset_manifest_path": str(asset_manifest_path),
            "asset_manifest_sha256": sha256_file(asset_manifest_path),
            "split_package_manifest_sha256": sha256_file(split_package_root / "package_manifest.json"),
            "card": CARD_SPECS[card_id],
            "config": config,
        },
    )
    prediction_index = {
        "required_columns": list(REQUIRED_PREDICTION_COLUMNS),
        "validation_prediction_count": len(validation_tables),
        "test_prediction_count": len(test_tables),
        "validation_rows": int(len(validation_all)),
        "test_rows": int(len(test_all)),
        "files": [
            {
                "path": _relative(path, output_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted((output_root / "predictions").rglob("*.csv"))
        ],
    }
    _write_json(output_root / "prediction_index.json", prediction_index)
    (output_root / "runner.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    metadata = {
        "status": "training_payload_ready_for_terminal_packaging",
        "training_started": True,
        "result_imported": False,
        "evidence_accepted": False,
        "experiment_id": EXPERIMENT_ID,
        "workspace_id": WORKSPACE_ID,
        "card_id": card_id,
        "run_units": fit_counter,
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": sha256_file(asset_manifest_path),
        "output_root": str(output_root.resolve()),
        "metrics_path": "metrics.json",
        "selection_proof_path": "selection_proof.json",
        "prediction_index_path": "prediction_index.json",
        "checkpoint_records": checkpoint_records,
        "compatibility_only": True,
        "raw_mpp2_claim": False,
    }
    _write_json(output_root / "runner_metadata.json", metadata)
    return _safe_json(metadata)


def _artifact(path: Path, *, artifact_id: str, kind: str, evidence_role: str, attempt_id: str, root: Path, evaluation_split: str | None = None, retention: str = "retain_with_immutable_result_bundle") -> dict[str, Any]:
    item: dict[str, Any] = {
        "artifact_id": artifact_id,
        "path": _relative(path, root),
        "kind": kind,
        "evidence_role": evidence_role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_attempt_id": attempt_id,
        "retention": retention,
    }
    if evaluation_split is not None:
        item["evaluation_split"] = evaluation_split
    return item


def finalize_result_source_bundle(
    *,
    output_root: Path,
    job_manifest_path: Path,
    result_id: str | None = None,
    source_bundle_name: str = "result_source_bundle",
) -> dict[str, Any]:
    """Create the immutable v2 source result bundle after terminal evidence exists."""

    job = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    attempt_id = str(job["attempt_id"])
    terminal_path = output_root / "attempt_terminal.json"
    started_path = output_root / "attempt_started.json"
    if not started_path.is_file() or not terminal_path.is_file():
        raise FileNotFoundError("attempt_started.json and attempt_terminal.json are required before packaging")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("job_id") != job.get("job_id") or terminal.get("attempt_id") != attempt_id:
        raise ValueError("terminal event is not bound to the job manifest")
    if not source_bundle_name or Path(source_bundle_name).name != source_bundle_name:
        raise ValueError("source_bundle_name must be a single safe directory name")
    source_bundle = output_root / source_bundle_name
    if source_bundle.exists() and any(source_bundle.iterdir()):
        raise ValueError("result_source_bundle is not empty; immutable packaging retry is forbidden")
    artifacts_root = source_bundle / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    copies: list[tuple[Path, Path]] = [
        (job_manifest_path, artifacts_root / "job.json"),
        (started_path, artifacts_root / "attempt_started.json"),
        (terminal_path, artifacts_root / "attempt_terminal.json"),
    ]
    for source_name in (
        "metrics.json",
        "selection_proof.json",
        "prediction_index.json",
        "resolved_config.json",
        "runner_metadata.json",
        "runner.log",
    ):
        source = output_root / source_name
        if source.is_file():
            copies.append((source, artifacts_root / source_name))
    prediction_files = sorted((output_root / "predictions").rglob("*.csv"))
    for source in prediction_files:
        copies.append((source, artifacts_root / "predictions" / source.relative_to(output_root / "predictions")))
    history_files = sorted((output_root / "training_history").glob("*.json"))
    for source in history_files:
        copies.append((source, artifacts_root / "training_history" / source.name))
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    result_artifacts: list[dict[str, Any]] = []
    result_artifacts.append(_artifact(artifacts_root / "job.json", artifact_id="job_manifest", kind="job_manifest", evidence_role="critical", attempt_id=attempt_id, root=source_bundle))
    result_artifacts.append(_artifact(artifacts_root / "attempt_started.json", artifact_id="attempt_started", kind="attempt_event", evidence_role="critical", attempt_id=attempt_id, root=source_bundle))
    result_artifacts.append(_artifact(artifacts_root / "attempt_terminal.json", artifact_id="attempt_terminal", kind="attempt_event", evidence_role="critical", attempt_id=attempt_id, root=source_bundle))
    for filename, artifact_id, kind, role in (
        ("metrics.json", "metrics", "metrics", "critical"),
        ("selection_proof.json", "selection_proof", "selection_proof", "critical"),
        ("prediction_index.json", "prediction_index", "prediction_index", "critical"),
        ("resolved_config.json", "resolved_config", "resolved_config", "critical"),
        ("runner_metadata.json", "runner_metadata", "run_metadata", "supporting"),
        ("runner.log", "runner_log", "log", "diagnostic"),
    ):
        path = artifacts_root / filename
        if path.is_file():
            result_artifacts.append(_artifact(path, artifact_id=artifact_id, kind=kind, evidence_role=role, attempt_id=attempt_id, root=source_bundle))
    evaluated_splits: list[dict[str, str]] = []
    for source in prediction_files:
        relative = PurePosixPath("artifacts") / "predictions" / source.relative_to(output_root / "predictions").as_posix()
        copied = source_bundle / relative
        split_id = source.relative_to(output_root / "predictions").with_suffix("").as_posix().replace("/", "-")
        artifact_id = "prediction_" + hashlib.sha256(split_id.encode("utf-8")).hexdigest()[:20]
        result_artifacts.append(
            _artifact(
                copied,
                artifact_id=artifact_id,
                kind="raw_prediction_table",
                evidence_role="critical",
                attempt_id=attempt_id,
                root=source_bundle,
                evaluation_split=split_id,
            )
        )
        evaluated_splits.append({"split_id": split_id, "artifact_id": artifact_id})
    for source in history_files:
        copied = artifacts_root / "training_history" / source.name
        result_artifacts.append(
            _artifact(
                copied,
                artifact_id="history_" + source.stem,
                kind="training_history",
                evidence_role="supporting",
                attempt_id=attempt_id,
                root=source_bundle,
            )
        )

    metadata = json.loads((output_root / "runner_metadata.json").read_text(encoding="utf-8"))
    # Checkpoint records carry fit-level diagnostics (card/task/seed/fold), but
    # result-envelope large_artifacts deliberately permits only portable asset
    # identity and retention fields.  Keep the detailed fit provenance in the
    # copied training history and runner metadata, while registering the binary
    # checkpoint itself without leaking disallowed fields into the envelope.
    large_artifacts = [
        {
            "artifact_id": str(record["artifact_id"]),
            "kind": "model_checkpoint",
            "server_path": str(record["server_path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
            "retention": str(record["retention"]),
        }
        for record in metadata.get("checkpoint_records", [])
    ]
    result_status = "success" if terminal.get("status") == "completed" and int(terminal.get("returncode", -1)) == 0 else "failed"
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    from scripts.pfmval_governance import build_result_v2

    job_for_result = dict(job)
    result = build_result_v2(
        job_for_result,
        result_id=result_id or f"{job['job_id']}-result-R001",
        status=result_status,
        artifacts=result_artifacts,
        metrics=metrics,
        metric_artifact_ids=["metrics"],
        large_artifacts=large_artifacts,
        prediction_return={
            "applicability": "required",
            "evaluated_splits": evaluated_splits,
        } if result_status == "success" else None,
    )
    _write_json(source_bundle / "result.json", result)
    _write_json(source_bundle / "artifacts.json", result_artifacts)
    _write_json(source_bundle / "large_artifacts.json", large_artifacts)
    _write_json(source_bundle / "metrics.json", metrics)
    return {
        "source_bundle": str(source_bundle),
        "result_id": result["result_id"],
        "status": result["status"],
        "artifact_count": len(result_artifacts),
        "prediction_artifact_count": len(evaluated_splits),
        "large_artifact_count": len(large_artifacts),
        "result_sha256": sha256_file(source_bundle / "result.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--split-package", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--job-manifest", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--finalize-result", action="store_true")
    parser.add_argument("--result-id")
    parser.add_argument("--source-bundle-name", default="result_source_bundle")
    args = parser.parse_args(argv)
    if args.finalize_result:
        if not args.output_root or not args.job_manifest:
            parser.error("--finalize-result requires --output-root and --job-manifest")
        print(json.dumps(finalize_result_source_bundle(
            output_root=args.output_root.resolve(),
            job_manifest_path=args.job_manifest.resolve(),
            result_id=args.result_id,
            source_bundle_name=args.source_bundle_name,
        ), ensure_ascii=False, indent=2))
        return 0
    required = {
        "card_id": args.card_id,
        "config": args.config,
        "asset_manifest": args.asset_manifest,
        "split_package": args.split_package,
        "output_root": args.output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"training mode missing: {missing}")
    metadata = run_card(
        card_id=str(args.card_id),
        config_path=args.config.resolve(),
        asset_manifest_path=args.asset_manifest.resolve(),
        split_package_root=args.split_package.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
