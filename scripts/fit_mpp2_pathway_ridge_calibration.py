#!/usr/bin/env python3
"""Run the approved MPP2 train-only pathway Ridge calibration protocol.

The control flow is deliberately linear: internal predictions -> nested LOPO ->
frozen calibrator artifacts -> one external XZY evaluation.  Do not reorder it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_mpp_manifest import ManifestMPPDataset, build_manifest_external, parse_xy
from scripts.mpp2_pathway_ridge_calibration import (
    CalibratedMPP2,
    choose_lambda_one_se,
    lambda_stability,
    nested_lopo,
    fit_all_pathways,
    frozen_calibrator_pcc_invariance,
    pathway_decisions,
    per_pathway_metrics,
    patient_balanced_mse,
)
from scripts.pfmval_state import active_mpp_repair, sha256_file, validate_state
from train_mpp_uni2h_mlp import MPPMLPHead, resolve_manifest_data_roots


EXPERIMENT_ID = "mpp2_pathway_ridge_calibration_v001_20260717"
PATIENTS = ("HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ")
EXPECTED_CHECKPOINT_SHA = "c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Approved MPP2 pathway Ridge calibration")
    result.add_argument("--splits_root", required=True)
    result.add_argument("--manifest_labels_root", required=True)
    result.add_argument("--cache_root", required=True)
    result.add_argument("--flat_cache_root", required=True)
    result.add_argument("--head_checkpoint", required=True)
    result.add_argument("--output_dir", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch_size", type=int, default=64)
    # deploy/run_experiment.ps1 appends this launcher-wide resource limit.
    result.add_argument("--num_threads", type=int, default=8)
    return result


def read_registry() -> dict[str, Any]:
    registry = json.loads((PROJECT_ROOT / "experiments" / "experiment_registry.json").read_text(encoding="utf-8"))
    entry = next((item for item in registry["experiments"] if item.get("id") == EXPERIMENT_ID), None)
    if entry is None:
        raise RuntimeError("approved calibration experiment is absent from registry")
    if not entry.get("execution_approved"):
        raise RuntimeError("registry does not authorize calibration execution")
    if entry.get("execution_directive_id") != "DIR-20260717-003":
        raise RuntimeError("registry execution directive binding is unexpected")
    return entry


def load_head(path: Path, device: torch.device) -> tuple[MPPMLPHead, dict[str, Any]]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f"accepted checkpoint SHA mismatch: expected={EXPECTED_CHECKPOINT_SHA} actual={actual_sha}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    first = state.get("head.0.weight")
    last = state.get("head.3.weight")
    if first is None or last is None:
        raise RuntimeError("accepted checkpoint does not have the exact MPPMLPHead layout")
    model = MPPMLPHead(in_dim=int(first.shape[1]), hidden=int(first.shape[0]), out_dim=int(last.shape[0]))
    model.load_state_dict(state, strict=True)
    if int(checkpoint.get("epoch", -1)) != 15:
        raise RuntimeError(f"accepted checkpoint epoch mismatch: {checkpoint.get('epoch')}")
    model.to(device).eval()
    return model, checkpoint


def predict_dataset(model: MPPMLPHead, dataset: ManifestMPPDataset, patient: str,
                    device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            predictions.append(model(features.to(device)).cpu().numpy().astype(np.float64))
            truth.append(labels.numpy().astype(np.float64))
    rows = []
    for path, _ in dataset.samples:
        patch_id = Path(path).stem
        x, y = parse_xy(patch_id)
        if x is None or y is None:
            raise RuntimeError(f"patch coordinate is required for calibration provenance: {patch_id}")
        rows.append({"patient_id": patient, "patch_id": patch_id, "x": x, "y": y})
    return np.vstack(predictions), np.vstack(truth), rows


def legacy_pooled_pcc(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.corrcoef(np.ravel(truth), np.ravel(prediction))[0, 1])


def summary_metrics(truth: np.ndarray, prediction: np.ndarray, means: np.ndarray, stds: np.ndarray,
                    patients: Sequence[str]) -> dict[str, Any]:
    per_pathway = per_pathway_metrics(truth, prediction, means, stds)
    return {
        "legacy_pooled_pcc": legacy_pooled_pcc(truth, prediction),
        "mean_per_pathway_pcc": float(np.nanmean([row["pcc"] for row in per_pathway])),
        "mean_per_pathway_raw_r2": float(np.nanmean([row["raw_r2"] for row in per_pathway])),
        "mean_raw_mae": float(np.nanmean([row["raw_mae"] for row in per_pathway])),
        "mean_ccc": float(np.nanmean([row["ccc"] for row in per_pathway])),
        "mean_z_mse": float(np.mean((truth - prediction) ** 2)),
        "patient_balanced_z_mse": patient_balanced_mse(truth, prediction, patients),
        "per_pathway": per_pathway,
    }


def spatial_bootstrap(truth: np.ndarray, base: np.ndarray, calibrated: np.ndarray,
                      means: np.ndarray, stds: np.ndarray, metadata: Sequence[dict[str, Any]], grid: int) -> dict[str, Any]:
    labels: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(metadata):
        labels.setdefault((int(row["x"]) // grid, int(row["y"]) // grid), []).append(index)
    clusters = list(labels.values())
    if len(clusters) < 2:
        raise RuntimeError(f"grid={grid} produced fewer than two spatial clusters")
    rng = np.random.default_rng(42)
    delta_r2 = np.empty(5000, dtype=np.float64)
    delta_mae = np.empty(5000, dtype=np.float64)
    for replicate in range(5000):
        indices = np.concatenate([clusters[item] for item in rng.integers(0, len(clusters), len(clusters))])
        baseline = summary_metrics(truth[indices], base[indices], means, stds, ["XZY"] * len(indices))
        candidate = summary_metrics(truth[indices], calibrated[indices], means, stds, ["XZY"] * len(indices))
        delta_r2[replicate] = candidate["mean_per_pathway_raw_r2"] - baseline["mean_per_pathway_raw_r2"]
        delta_mae[replicate] = candidate["mean_raw_mae"] - baseline["mean_raw_mae"]
    return {
        "grid_pixels": grid,
        "cluster_count": len(clusters),
        "bootstrap_replicates": 5000,
        "seed": 42,
        "delta_mean_raw_r2_ci95": [float(np.quantile(delta_r2, 0.025)), float(np.quantile(delta_r2, 0.975))],
        "delta_mean_raw_mae_ci95": [float(np.quantile(delta_mae, 0.025)), float(np.quantile(delta_mae, 0.975))],
    }


def main() -> int:
    args = parser().parse_args()
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    torch.set_num_threads(args.num_threads)
    entry = read_registry()
    report = validate_state(PROJECT_ROOT, strict=True, task="training", host_scope="server")
    report.emit()
    if not report.ok:
        raise RuntimeError("authoritative state blocks calibration")
    repair = active_mpp_repair(PROJECT_ROOT)
    if repair is None or repair.get("data_manifest_id") != entry["data_manifest_id"]:
        raise RuntimeError("active repaired data manifest does not match the approved experiment")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    roots = resolve_manifest_data_roots(args.splits_root, args.manifest_labels_root, 2)
    zscore_params = json.loads(roots["zscore_params"].read_text(encoding="utf-8"))
    manifest = pd.read_csv(roots["split_manifest"])
    model, checkpoint = load_head(Path(args.head_checkpoint), device)

    # Stage 1: only six internal-val patients.  No external Dataset is constructed above this line.
    all_predictions: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_rows: list[dict[str, Any]] = []
    pathway_order: list[str] | None = None
    for patient in PATIENTS:
        labels = roots["labels"] / "val" / patient / f"{patient}_ssGSEA_zscore.csv"
        dataset = ManifestMPPDataset(manifest, args.cache_root, 2, patient, "internal_val", str(labels),
                                     flat_cache_root=args.flat_cache_root, allow_missing=False)
        prediction, truth, rows = predict_dataset(model, dataset, patient, device, args.batch_size)
        if pathway_order is None:
            pathway_order = list(dataset.target_cols)
        elif pathway_order != list(dataset.target_cols):
            raise RuntimeError("internal-val pathway order differs between patients")
        all_predictions.append(prediction)
        all_truth.append(truth)
        all_rows.extend(rows)
    assert pathway_order is not None
    if len(all_rows) != int(entry["calibration_samples"]):
        raise RuntimeError(f"internal-val sample count mismatch: expected={entry['calibration_samples']} actual={len(all_rows)}")
    keys = [(row["patient_id"], row["patch_id"]) for row in all_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate patient+patch key in internal-val predictions")
    truth = np.vstack(all_truth)
    base_prediction = np.vstack(all_predictions)
    if truth.shape != (len(all_rows), 30) or not np.isfinite(truth).all() or not np.isfinite(base_prediction).all():
        raise RuntimeError("internal predictions must be finite [1078,30]")
    pathway_params = zscore_params.get("pathways", {})
    if set(pathway_order) != set(pathway_params) or len(pathway_order) != 30:
        raise RuntimeError("active z-score parameter pathways do not match prediction columns")
    means = np.asarray([pathway_params[item]["mean"] for item in pathway_order], dtype=np.float64)
    stds = np.asarray([pathway_params[item]["std"] for item in pathway_order], dtype=np.float64)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise RuntimeError("z-score parameters are invalid")
    patients = np.asarray([row["patient_id"] for row in all_rows], dtype=str)
    baseline = summary_metrics(truth, base_prediction, means, stds, patients)
    expected = entry["baseline_internal_validation"]
    if abs(baseline["mean_z_mse"] - float(expected["reference_val_loss"])) > float(expected["loss_tolerance"]):
        raise RuntimeError("baseline internal-val loss does not reproduce accepted checkpoint evidence")
    if abs(baseline["legacy_pooled_pcc"] - float(expected["reference_val_pcc"])) > float(expected["pcc_tolerance"]):
        raise RuntimeError("baseline internal-val pooled PCC does not reproduce accepted checkpoint evidence")
    frame = pd.DataFrame(all_rows)
    for index, name in enumerate(pathway_order):
        frame[f"true_{name}"] = truth[:, index]
        frame[f"pred_{name}"] = base_prediction[:, index]
    frame.to_csv(output / "predictions_internal_val_base.csv", index=False)

    # Stage 2: nested LOPO and final train-only refit.  XZY remains untouched.
    nested = nested_lopo(truth, base_prediction, patients)
    mask, decisions = pathway_decisions(truth, base_prediction, nested["oof_prediction"], patients, means, stds, nested["slopes"])
    oof = nested["oof_prediction"].copy()
    oof[:, ~mask] = base_prediction[:, ~mask]
    oof_metrics = summary_metrics(truth, oof, means, stds, patients)
    stability = lambda_stability(nested["folds"])
    oof_pcc_delta = np.asarray([a["pcc"] - b["pcc"] for a, b in zip(oof_metrics["per_pathway"], baseline["per_pathway"])])
    internal_gate = {
        "lambda_stability": stability,
        "patient_balanced_z_mse_nonworse": oof_metrics["patient_balanced_z_mse"] <= baseline["patient_balanced_z_mse"] + 1e-12,
        "mean_per_pathway_raw_r2_improved": oof_metrics["mean_per_pathway_raw_r2"] > baseline["mean_per_pathway_raw_r2"] + 1e-12,
        "oof_fold_heterogeneous_pcc_max_abs_delta": float(np.nanmax(np.abs(oof_pcc_delta))),
        "enabled_pathways": int(mask.sum()),
    }
    internal_gate["passed"] = bool(
        stability["passed"] and internal_gate["patient_balanced_z_mse_nonworse"] and
        internal_gate["mean_per_pathway_raw_r2_improved"]
    )
    nested_payload = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_internal": baseline,
        "calibrated_nested_oof": oof_metrics,
        "outer_folds": nested["folds"],
        "internal_gate": internal_gate,
    }
    write_json(output / "nested_lopo_metrics.json", nested_payload)
    decisions_frame = pd.DataFrame([{**{"pathway": pathway_order[i]}, **row} for i, row in enumerate(decisions)])
    decisions_frame.to_csv(output / "pathway_decisions.csv", index=False)
    if not internal_gate["passed"]:
        raise RuntimeError("internal nested-LOPO stability gate failed; calibrator was not frozen and XZY was not read")
    final_lambda, final_selection = choose_lambda_one_se(truth, base_prediction, patients)
    final_k, final_b, final_reasons = fit_all_pathways(truth, base_prediction, patients, final_lambda)
    final_k[~mask] = 1.0
    final_b[~mask] = 0.0
    final_internal_prediction = base_prediction * final_k + final_b
    final_pcc_invariance = frozen_calibrator_pcc_invariance(truth, base_prediction, final_internal_prediction)
    internal_gate["final_frozen_pcc_max_abs_delta"] = final_pcc_invariance["max_abs_delta"]
    internal_gate["final_frozen_pcc_invariant"] = final_pcc_invariance["invariant"]
    write_json(output / "nested_lopo_metrics.json", nested_payload)
    if not final_pcc_invariance["invariant"]:
        raise RuntimeError("final frozen calibrator violated the per-pathway PCC invariance check")
    pathway_order_sha = canonical_sha256(pathway_order)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "execution_directive_id": entry["execution_directive_id"],
        "base_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "base_checkpoint_epoch": 15,
        "data_manifest_id": entry["data_manifest_id"],
        "zscore_params_sha256": sha256_file(roots["zscore_params"]),
        "pathway_order_sha256": pathway_order_sha,
        "internal_predictions_sha256": sha256_file(output / "predictions_internal_val_base.csv"),
    }
    calibrator = {
        "schema_version": "1.0",
        "method": entry["method"],
        "final_lambda": final_lambda,
        "final_lambda_selection": final_selection,
        "pathway_order": pathway_order,
        "k": final_k.tolist(),
        "b": final_b.tolist(),
        "pathway_enabled_mask": mask.tolist(),
        "final_fit_fallback_reasons": final_reasons,
        "provenance": provenance,
    }
    write_json(output / "calibrator.json", calibrator)
    provenance["calibrator_sha256"] = sha256_file(output / "calibrator.json")
    write_json(output / "pathway_order.json", pathway_order)
    write_json(output / "model_provenance.json", provenance)

    # Stage 3: this is the first XZY access, after the calibrator hash exists on disk.
    external_labels = roots["labels"] / "external" / "XZY" / "XZY_ssGSEA_zscore_by_group_2_train.csv"
    external = build_manifest_external(args.flat_cache_root, 2, "XZY", str(external_labels), target_cols=pathway_order)
    external_prediction, external_truth, external_rows = predict_dataset(model, external, "XZY", device, args.batch_size)
    if len(external_rows) != int(entry["external_samples"]):
        raise RuntimeError(f"XZY sample count mismatch: expected={entry['external_samples']} actual={len(external_rows)}")
    calibrated_external = external_prediction * final_k + final_b
    external_base = summary_metrics(external_truth, external_prediction, means, stds, ["XZY"] * len(external_rows))
    external_cal = summary_metrics(external_truth, calibrated_external, means, stds, ["XZY"] * len(external_rows))
    per_pathway = []
    for index, pathway in enumerate(pathway_order):
        per_pathway.append({
            "pathway": pathway,
            "baseline_raw_r2": external_base["per_pathway"][index]["raw_r2"],
            "calibrated_raw_r2": external_cal["per_pathway"][index]["raw_r2"],
            "delta_raw_r2": external_cal["per_pathway"][index]["raw_r2"] - external_base["per_pathway"][index]["raw_r2"],
            "baseline_raw_mae": external_base["per_pathway"][index]["raw_mae"],
            "calibrated_raw_mae": external_cal["per_pathway"][index]["raw_mae"],
            "delta_raw_mae": external_cal["per_pathway"][index]["raw_mae"] - external_base["per_pathway"][index]["raw_mae"],
            "baseline_pcc": external_base["per_pathway"][index]["pcc"],
            "calibrated_pcc": external_cal["per_pathway"][index]["pcc"],
        })
    pd.DataFrame(per_pathway).to_csv(output / "per_pathway_metrics.csv", index=False)
    external_frame = pd.DataFrame(external_rows)
    for index, name in enumerate(pathway_order):
        external_frame[f"true_{name}"] = external_truth[:, index]
        external_frame[f"pred_base_{name}"] = external_prediction[:, index]
        external_frame[f"pred_calibrated_{name}"] = calibrated_external[:, index]
    external_frame.to_csv(output / "predictions_external_xzy.csv", index=False)
    spaces = [spatial_bootstrap(external_truth, external_prediction, calibrated_external, means, stds, external_rows, grid)
              for grid in (1024, 2048, 4096)]
    write_json(output / "spatial_sensitivity.json", {"grids": spaces})
    pcc_external_delta = np.asarray([row["calibrated_pcc"] - row["baseline_pcc"] for row in per_pathway])
    r2_nonworse = int(sum(row["delta_raw_r2"] >= -1e-12 for row in per_pathway))
    external_gate = {
        "mean_raw_r2_gt_minus_0_088": external_cal["mean_per_pathway_raw_r2"] > -0.088,
        "raw_mae_lt_1176_2114": external_cal["mean_raw_mae"] < 1176.2114,
        "mean_per_pathway_pcc_invariant": float(np.nanmax(np.abs(pcc_external_delta))) < 1e-10,
        "raw_r2_nonworse_pathways": r2_nonworse,
        "at_least_18_r2_nonworse": r2_nonworse >= 18,
        "recommended_delta_mean_raw_r2": external_cal["mean_per_pathway_raw_r2"] - external_base["mean_per_pathway_raw_r2"],
        "spatial_robust_strong_evidence": sum(grid["delta_mean_raw_r2_ci95"][0] > 0 for grid in spaces) >= 2,
    }
    external_gate["passed"] = bool(
        external_gate["mean_raw_r2_gt_minus_0_088"] and external_gate["raw_mae_lt_1176_2114"] and
        external_gate["mean_per_pathway_pcc_invariant"] and external_gate["at_least_18_r2_nonworse"]
    )
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "calibrator_sha256": provenance["calibrator_sha256"],
        "internal_gate": internal_gate,
        "external_baseline": external_base,
        "external_calibrated": external_cal,
        "external_gate": external_gate,
        "evidence_interpretation": "spatial_robust_strong_evidence only describes XZY spatial stability, not cross-patient generalization",
    }
    write_json(output / "metrics.json", metrics)
    if external_gate["passed"]:
        package = output / "mpp2_calibrated_raw_v001"
        package.mkdir()
        shutil.copy2(args.head_checkpoint, package / "best_checkpoint.pth")
        shutil.copy2(output / "calibrator.json", package / "calibrator.json")
        shutil.copy2(roots["zscore_params"], package / "zscore_params_from_train.json")
        shutil.copy2(output / "pathway_order.json", package / "pathway_order.json")
        shutil.copy2(output / "model_provenance.json", package / "model_provenance.json")
        shutil.copy2(PROJECT_ROOT / "scripts" / "mpp2_pathway_ridge_calibration.py", package / "calibrated_mpp2.py")
        (package / "README_PHASE3_INPUT_CONTRACT.md").write_text(
            "# Phase 3 input contract\n\nThe 30 numeric columns emitted by CalibratedMPP2 are final calibrated raw ssGSEA pathway scores. "
            "Phase 3 may align identifiers but must not inverse-z-score, recalibrate, within-patient z-score, rank-normalize or quantile-normalize them.\n",
            encoding="utf-8",
        )
        manifest_entries = {str(path.relative_to(package)).replace("\\", "/"): sha256_file(path)
                            for path in package.rglob("*") if path.is_file()}
        write_json(package / "sha256_manifest.json", manifest_entries)
    (output / "training_summary.txt").write_text(
        f"experiment_id={EXPERIMENT_ID}\ninternal_gate_passed={internal_gate['passed']}\n"
        f"external_gate_passed={external_gate['passed']}\ncalibrator_sha256={provenance['calibrator_sha256']}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
