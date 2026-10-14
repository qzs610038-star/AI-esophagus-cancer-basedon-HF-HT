"""四臂 internal-only 训练、checkpoint 冻结与一次性 XZY 评估。"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_mpp_manifest import build_manifest_external, merge_manifest_patients

from .batch_plan import (
    FixedBatchSampler,
    PlannedBatch,
    SampleIdentity,
    build_patient_balanced_plan,
    write_batch_plan_csv,
)
from .losses import ArmLoss, fit_train_fold_tau_y, patient_balanced_z_mse
from .model import (
    ARM_ORDER,
    FrozenBaseline,
    ResidualContrastiveModel,
    build_shared_initial_states,
    load_accepted_baseline,
)
from .result_contract import write_json, write_prediction_csv


class ContractError(RuntimeError):
    """数据、身份、防泄漏或冻结合同的 HARD 失败。"""


@dataclass(frozen=True)
class FeatureRecord:
    feature_path: Path
    target: torch.Tensor
    patient: str
    spot_id: str
    split: str


class MetadataFeatureDataset(Dataset):
    def __init__(self, records: Sequence[FeatureRecord], pathway_names: Sequence[str]):
        self.records = list(records)
        self.pathway_names = list(pathway_names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        feature = torch.load(record.feature_path, map_location="cpu")
        if feature.ndim == 2:
            feature = feature[0, :]
        return {
            "feature": feature.to(dtype=torch.float32),
            "target": record.target.clone().to(dtype=torch.float32),
            "patient": record.patient,
            "spot_id": record.spot_id,
            "split": record.split,
        }


def collate_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "feature": torch.stack([row["feature"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "patient": [str(row["patient"]) for row in rows],
        "spot_id": [str(row["spot_id"]) for row in rows],
        "split": [str(row["split"]) for row in rows],
    }


class DataAccessBoundary:
    """internal_only 模式从接口层禁止解析 XZY。"""

    def __init__(self, mode: str):
        if mode not in {"internal_only", "external_benchmark"}:
            raise ValueError("unsupported data access mode")
        self.mode = mode

    def require_internal(self) -> None:
        if self.mode != "internal_only":
            raise ContractError("internal dataset requested outside internal_only mode")

    def require_external(self) -> None:
        if self.mode != "external_benchmark":
            raise ContractError("XZY is inaccessible before external_benchmark mode")


@dataclass
class TrainingContext:
    protocol: dict[str, Any]
    server_paths: dict[str, Any]
    train_dataset: MetadataFeatureDataset
    val_dataset: MetadataFeatureDataset
    batch_plan: list[PlannedBatch]
    tau_y: float
    shared_initial_states: dict[str, dict[str, torch.Tensor]]
    pathway_names: list[str]
    zscore_mean: np.ndarray
    zscore_std: np.ndarray
    repair_group_root: Path


def read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"YAML root must be an object: {path}")
    return payload


def resolve_registered_path(server_paths: Mapping[str, Any], path_id: str) -> Path:
    entry = server_paths.get("paths", {}).get(path_id)
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ContractError(f"registered path id is missing: {path_id}")
    configured = Path(str(entry["path"]))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def resolve_active_repair_group(
    repair_registry: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Path:
    """解析冻结 data manifest 对应的服务器修复标签目录。"""

    expected_manifest = str(protocol["data_manifest_id"])
    if repair_registry.get("active_data_manifest_id") != expected_manifest:
        raise ContractError("active repaired data manifest differs from the frozen protocol")
    matches = [
        repair
        for repair in repair_registry.get("repairs", [])
        if repair.get("data_manifest_id") == expected_manifest
        and repair.get("status") == "active"
    ]
    if len(matches) != 1:
        raise ContractError("frozen data manifest must resolve to exactly one active repair")
    stage_root = Path(str(matches[0].get("server_stage_path", "")))
    if not stage_root.is_absolute():
        raise ContractError("active repair server_stage_path must be absolute")
    return stage_root / f"group_{protocol['data']['train_mpp_id']}"


def fairness_signature(protocol: Mapping[str, Any]) -> dict[str, Any]:
    training = protocol["training"]
    return {
        "arms": list(protocol["arms"]),
        "seed": protocol["seed"],
        "patients": list(training["patients"]),
        "batch_size": training["batch_size"],
        "spots_per_patient": training["spots_per_patient"],
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "max_epochs": training["max_epochs"],
        "patience": training["patience"],
        "min_delta": training["min_delta"],
        "checkpoint_metric": training["checkpoint_metric"],
        "candidate_set": protocol["loss"]["candidate_set"],
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if list(protocol.get("arms", [])) != list(ARM_ORDER):
        raise ContractError("four-arm order must be FBR/RCC/HCR/CPGCR")
    training = protocol.get("training", {})
    if (
        training.get("patients_per_batch") != 6
        or training.get("spots_per_patient") != 5
        or training.get("batch_size") != 30
    ):
        raise ContractError("patient-balanced batch fairness config has drifted")
    if protocol.get("loss", {}).get("candidate_set") != "all_valid_in_batch":
        raise ContractError("contrastive candidate set is not complete")
    selection = protocol.get("selection_policy", {})
    if selection.get("external_may_select_checkpoint") is not False:
        raise ContractError("external benchmark may not select checkpoints")
    if protocol.get("approval", {}).get("create_jobs_in_this_revision") is not False:
        raise ContractError("this implementation revision may not create jobs")
    external = protocol.get("external_policy", {})
    if (
        external.get("timing") != "once_after_each_internal_checkpoint_freeze"
        or external.get("fixed_order") != list(ARM_ORDER)
        or external.get("freeze_four_arm_configuration_before_first_external") is not True
    ):
        raise ContractError("external benchmark timing/order/freeze contract has drifted")
    replay = protocol.get("baseline_replay_contract", {})
    if set(replay) != {"reference_val_loss", "reference_val_pcc", "loss_tolerance", "pcc_tolerance"}:
        raise ContractError("baseline replay contract is incomplete")


def _flatten_concat_dataset(
    dataset: Any,
    *,
    split: str,
    expected_feature_dim: int,
) -> MetadataFeatureDataset:
    records: list[FeatureRecord] = []
    pathway_names: list[str] | None = None
    for patient_dataset in dataset.datasets:
        if int(patient_dataset.feat_dim) != expected_feature_dim:
            raise ContractError(
                f"{split} feature dimension mismatch for {patient_dataset.patient}: "
                f"{patient_dataset.feat_dim} != {expected_feature_dim}"
            )
        names = list(patient_dataset.target_cols)
        if pathway_names is None:
            pathway_names = names
        elif pathway_names != names:
            raise ContractError("30-pathway column order differs between patients")
        for feature_path, target in patient_dataset.samples:
            if target.ndim != 1 or target.shape[0] != 30 or not torch.isfinite(target).all():
                raise ContractError(f"{split} target tensor is invalid for {patient_dataset.patient}")
            records.append(FeatureRecord(
                feature_path=Path(feature_path),
                target=target,
                patient=str(patient_dataset.patient),
                spot_id=Path(feature_path).stem,
                split=split,
            ))
    if pathway_names is None or len(pathway_names) != 30:
        raise ContractError("dataset must expose the accepted 30-pathway order")
    identities = {(record.patient, record.spot_id) for record in records}
    if len(identities) != len(records):
        raise ContractError("patient+spot identity is not one-to-one")
    return MetadataFeatureDataset(records, pathway_names)


def _load_zscore_parameters(group_root: Path, pathway_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((group_root / "zscore_params_from_train.json").read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train":
        raise ContractError("z-score parameters were not fitted on train")
    parameters = payload.get("pathways") or payload.get("params") or payload
    means = []
    stds = []
    for name in pathway_names:
        entry = parameters[name]
        means.append(float(entry["mean"]))
        stds.append(float(entry.get("std", entry.get("scale"))))
    mean_array = np.asarray(means)
    std_array = np.asarray(stds)
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all() or np.any(std_array <= 0):
        raise ContractError("train-only z-score parameters are non-finite or non-positive")
    return mean_array, std_array


def _read_unique_label_stems(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ContractError(f"label table is empty: {path}")
    stems = frame.iloc[:, 0].astype(str).map(lambda value: Path(value).stem)
    duplicated = stems[stems.duplicated(keep=False)]
    if not duplicated.empty:
        raise ContractError(f"duplicate label identity in {path}: {duplicated.iloc[0]}")
    return set(stems)


def _validate_internal_identity_contract(
    manifest: pd.DataFrame,
    *,
    labels_root: Path,
    patients: Sequence[str],
) -> None:
    required = {"patient", "split", "patch_stem"}
    if not required.issubset(manifest.columns):
        raise ContractError(f"split manifest columns are incomplete: {sorted(required - set(manifest.columns))}")
    identities = manifest[["patient", "patch_stem"]].astype(str)
    if identities.duplicated(keep=False).any():
        row = identities[identities.duplicated(keep=False)].iloc[0]
        raise ContractError(f"split manifest identity is duplicated: {row['patient']}::{row['patch_stem']}")
    expected_patients = set(patients)
    for split, label_subdir in (("train", "train"), ("internal_val", "val")):
        subset = manifest[manifest["split"] == split]
        if set(subset["patient"].astype(str)) != expected_patients:
            raise ContractError(f"{split} patient identity set differs from the fixed six patients")
        for patient in patients:
            expected = set(
                subset[subset["patient"] == patient]["patch_stem"].astype(str)
            )
            label_path = labels_root / label_subdir / patient / f"{patient}_ssGSEA_zscore.csv"
            observed = _read_unique_label_stems(label_path)
            if expected != observed:
                raise ContractError(f"{patient}/{split} manifest-to-label identity mismatch")


def prepare_training_context(
    protocol: dict[str, Any],
    server_paths: dict[str, Any],
    *,
    batch_plan_output: str | Path,
) -> TrainingContext:
    boundary = DataAccessBoundary("internal_only")
    boundary.require_internal()
    validate_protocol(protocol)
    data = protocol["data"]
    patients = list(protocol["training"]["patients"])
    repair_registry = json.loads(
        (PROJECT_ROOT / "project_state" / "mpp_repair_registry.json").read_text(encoding="utf-8")
    )
    splits_root = resolve_registered_path(server_paths, data["splits_path_id"])
    source_group_root = splits_root / f"group_{data['train_mpp_id']}"
    repair_group_root = resolve_active_repair_group(repair_registry, protocol)
    manifest = pd.read_csv(source_group_root / "split_manifest.csv")
    split_info = json.loads((source_group_root / "split_info.json").read_text(encoding="utf-8"))
    if int(split_info.get("leakage_pairs", 0)) != 0:
        raise ContractError("split leakage_pairs is not zero")
    zscore_manifest = json.loads(
        (repair_group_root / "zscore_manifest.json").read_text(encoding="utf-8")
    )
    if set(zscore_manifest.get("fit_patients", [])) != set(patients):
        raise ContractError("z-score fit patients differ from the six training patients")
    if data["external_patient"] in zscore_manifest.get("fit_patients", []):
        raise ContractError("XZY appears in z-score fitting patients")
    cache_root = resolve_registered_path(server_paths, data["cache_path_id"])
    flat_cache_root = resolve_registered_path(server_paths, data["flat_cache_path_id"])
    labels_root = repair_group_root / "labels"
    _validate_internal_identity_contract(manifest, labels_root=labels_root, patients=patients)
    train_concat = merge_manifest_patients(
        manifest,
        str(cache_root),
        data["train_mpp_id"],
        patients,
        "train",
        str(labels_root),
        flat_cache_root=str(flat_cache_root),
    )
    val_concat = merge_manifest_patients(
        manifest,
        str(cache_root),
        data["train_mpp_id"],
        patients,
        "internal_val",
        str(labels_root),
        flat_cache_root=str(flat_cache_root),
    )
    expected_feature_dim = int(protocol["model"]["feature_dim"])
    train_dataset = _flatten_concat_dataset(
        train_concat,
        split="train",
        expected_feature_dim=expected_feature_dim,
    )
    val_dataset = _flatten_concat_dataset(
        val_concat,
        split="internal_val",
        expected_feature_dim=expected_feature_dim,
    )
    if train_dataset.pathway_names != val_dataset.pathway_names:
        raise ContractError("train/internal-val pathway order differs")
    train_identities = {(record.patient, record.spot_id) for record in train_dataset.records}
    val_identities = {(record.patient, record.spot_id) for record in val_dataset.records}
    if train_identities & val_identities:
        raise ContractError("actual train/internal-val identities overlap")
    if {record.patient for record in train_dataset.records} != set(patients):
        raise ContractError("loaded training patient set differs from the fixed six patients")
    if {record.patient for record in val_dataset.records} != set(patients):
        raise ContractError("loaded internal-val patient set differs from the fixed six patients")
    identities = [
        SampleIdentity(index, record.patient, record.spot_id)
        for index, record in enumerate(train_dataset.records)
    ]
    batch_plan = build_patient_balanced_plan(
        identities,
        patients=patients,
        spots_per_patient=protocol["training"]["spots_per_patient"],
        epochs=protocol["training"]["max_epochs"],
        seed=protocol["random_streams"]["batch_plan"],
    )
    write_batch_plan_csv(batch_plan_output, batch_plan, identities)
    train_targets = torch.stack([record.target for record in train_dataset.records])
    train_patients = [record.patient for record in train_dataset.records]
    tau_y = fit_train_fold_tau_y(
        train_targets,
        train_patients,
        split_name="train",
        seed=protocol["seed"],
        max_pairs=protocol["loss"]["tau_y_max_pairs"],
    )
    model_config = protocol["model"]
    shared_states = build_shared_initial_states(
        seed=protocol["random_streams"]["model_initialization"],
        input_dim=model_config["feature_dim"],
        pathway_dim=model_config["pathway_dim"],
        projection_dim=model_config["projection_dim"],
    )
    means, stds = _load_zscore_parameters(repair_group_root, train_dataset.pathway_names)
    return TrainingContext(
        protocol=protocol,
        server_paths=server_paths,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_plan=batch_plan,
        tau_y=tau_y,
        shared_initial_states=shared_states,
        pathway_names=train_dataset.pathway_names,
        zscore_mean=means,
        zscore_std=stds,
        repair_group_root=repair_group_root,
    )


def _snapshot_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_state_equal(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> None:
    if before.keys() != after.keys() or any(not torch.equal(before[key], after[key]) for key in before):
        raise ContractError("external benchmark modified model state")


def _predict_arrays(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    predictions = []
    targets = []
    patients: list[str] = []
    spot_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            if isinstance(model, ResidualContrastiveModel):
                output = model.inference(features)
            else:
                output = model(features)
            predictions.append(output.cpu())
            targets.append(batch["target"])
            patients.extend(batch["patient"])
            spot_ids.extend(batch["spot_id"])
    return (
        torch.cat(targets).numpy(),
        torch.cat(predictions).numpy(),
        patients,
        spot_ids,
    )


def _metrics(
    truth_z: np.ndarray,
    prediction_z: np.ndarray,
    truth_raw: np.ndarray,
    prediction_raw: np.ndarray,
    patients: Sequence[str],
) -> dict[str, float]:
    pb_mse = float(patient_balanced_z_mse(
        torch.from_numpy(prediction_z),
        torch.from_numpy(truth_z),
        patients,
    ).item())
    pooled_pcc = float(np.corrcoef(truth_z.ravel(), prediction_z.ravel())[0, 1])
    raw_mae = float(np.mean(np.abs(truth_raw - prediction_raw)))
    raw_r2 = []
    for column in range(truth_raw.shape[1]):
        residual = np.square(truth_raw[:, column] - prediction_raw[:, column]).sum()
        total = np.square(truth_raw[:, column] - truth_raw[:, column].mean()).sum()
        raw_r2.append(1.0 - residual / total)
    return {
        "patient_balanced_z_mse": pb_mse,
        "pooled_z_mse": float(np.mean(np.square(truth_z - prediction_z))),
        "z_mae": float(np.mean(np.abs(truth_z - prediction_z))),
        "pooled_pcc": pooled_pcc,
        "raw_mae": raw_mae,
        "mean_pathway_raw_r2": float(np.mean(raw_r2)),
    }


def evaluate_and_write(
    model: nn.Module,
    dataset: MetadataFeatureDataset,
    *,
    context: TrainingContext,
    device: torch.device,
    split: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_records)
    truth_z, prediction_z, patients, spot_ids = _predict_arrays(model, loader, device=device)
    truth_raw = truth_z * context.zscore_std + context.zscore_mean
    prediction_raw = prediction_z * context.zscore_std + context.zscore_mean
    output_root = Path(output_dir)
    predictions_path = write_prediction_csv(
        output_root / f"{split}_predictions.csv",
        patients=patients,
        spot_ids=spot_ids,
        split=split,
        pathway_names=context.pathway_names,
        truth_z=truth_z,
        prediction_z=prediction_z,
        truth_raw=truth_raw,
        prediction_raw=prediction_raw,
    )
    metrics_path = write_json(
        output_root / f"{split}_metrics.json",
        _metrics(truth_z, prediction_z, truth_raw, prediction_raw, patients),
    )
    return {"predictions": predictions_path, "metrics": metrics_path}


def _build_baseline(context: TrainingContext) -> nn.Module:
    protocol = context.protocol
    checkpoint_path = resolve_registered_path(
        context.server_paths,
        protocol["data"]["baseline_checkpoint_path_id"],
    )
    return load_accepted_baseline(
        checkpoint_path,
        input_dim=protocol["model"]["feature_dim"],
        hidden_dim=protocol["model"]["baseline_hidden_dim"],
        pathway_dim=protocol["model"]["pathway_dim"],
        dropout=protocol["model"]["baseline_dropout"],
    )


def train_arm(
    context: TrainingContext,
    *,
    arm: str,
    device: torch.device,
    output_dir: str | Path,
) -> tuple[nn.Module, dict[str, Path]]:
    if arm not in {"RCC", "HCR", "CPGCR"}:
        raise ValueError("train_arm only accepts trainable arms")
    protocol = context.protocol
    baseline = _build_baseline(context)
    model = ResidualContrastiveModel(
        baseline,
        arm=arm,
        input_dim=protocol["model"]["feature_dim"],
        pathway_dim=protocol["model"]["pathway_dim"],
        projection_dim=protocol["model"]["projection_dim"],
        initial_states=context.shared_initial_states,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.optimizer_parameters()),
        lr=protocol["training"]["learning_rate"],
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) in optimizer_ids for parameter in model.baseline.parameters()):
        raise ContractError("B0 entered the optimizer")
    criterion = ArmLoss(
        arm,
        contrastive_weight=protocol["loss"]["contrastive_weight"],
        tau_z=protocol["loss"]["tau_z"],
        tau_y=context.tau_y if arm == "CPGCR" else None,
    )
    val_loader = DataLoader(
        context.val_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_records,
    )
    best_value = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, protocol["training"]["max_epochs"] + 1):
        train_loader = DataLoader(
            context.train_dataset,
            batch_sampler=FixedBatchSampler(context.batch_plan, epoch=epoch),
            collate_fn=collate_records,
        )
        model.train()
        total_absolute = 0.0
        total_contrastive = 0.0
        batches = 0
        for batch in train_loader:
            features = batch["feature"].to(device)
            targets = batch["target"].to(device)
            output = model(features, targets if arm in {"HCR", "CPGCR"} else None)
            loss, parts = criterion(
                prediction=output.prediction,
                target=targets,
                patients=batch["patient"],
                image_embedding=output.image_embedding,
                pathway_embedding=output.pathway_embedding,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_absolute += float(parts["absolute"].detach().cpu())
            total_contrastive += float(parts["contrastive"].detach().cpu())
            batches += 1
        truth, prediction, val_patients, _ = _predict_arrays(model, val_loader, device=device)
        val_value = float(patient_balanced_z_mse(
            torch.from_numpy(prediction),
            torch.from_numpy(truth),
            val_patients,
        ).item())
        history.append({
            "epoch": epoch,
            "train_absolute_loss": total_absolute / batches,
            "train_contrastive_loss": total_contrastive / batches,
            "internal_val_patient_balanced_z_mse": val_value,
        })
        if val_value < best_value - protocol["training"]["min_delta"]:
            best_value = val_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= protocol["training"]["patience"]:
            break
    if best_state is None:
        raise ContractError(f"{arm} produced no valid internal checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint_path = output_root / "best_checkpoint.pth"
    torch.save({
        "arm": arm,
        "best_epoch": best_epoch,
        "internal_val_patient_balanced_z_mse": best_value,
        "model_state_dict": best_state,
    }, checkpoint_path)
    internal = evaluate_and_write(
        model,
        context.val_dataset,
        context=context,
        device=device,
        split="internal_val",
        output_dir=output_root,
    )
    return model, {
        "training_history": history_path,
        "checkpoint": checkpoint_path,
        "internal_predictions": internal["predictions"],
        "internal_metrics": internal["metrics"],
    }


def replay_fbr(
    context: TrainingContext,
    *,
    device: torch.device,
    output_dir: str | Path,
) -> tuple[nn.Module, dict[str, Path]]:
    baseline = FrozenBaseline(_build_baseline(context)).to(device)
    internal = evaluate_and_write(
        baseline,
        context.val_dataset,
        context=context,
        device=device,
        split="internal_val",
        output_dir=output_dir,
    )
    checkpoint_meta = write_json(Path(output_dir) / "checkpoint.json", {
        "arm": "FBR",
        "type": "accepted_baseline_reference",
        "selection": "historical accepted checkpoint; not reselected",
    })
    internal_metrics = json.loads(Path(internal["metrics"]).read_text(encoding="utf-8"))
    replay_contract = context.protocol["baseline_replay_contract"]
    observed_loss = float(internal_metrics["pooled_z_mse"])
    observed_pcc = float(internal_metrics["pooled_pcc"])
    if not np.isfinite(observed_loss) or not np.isfinite(observed_pcc):
        raise ContractError("FBR replay metrics are non-finite")
    loss_delta = abs(observed_loss - float(replay_contract["reference_val_loss"]))
    pcc_delta = abs(observed_pcc - float(replay_contract["reference_val_pcc"]))
    replay_check = write_json(Path(output_dir) / "baseline_replay_check.json", {
        "status": "PASS" if (
            loss_delta <= float(replay_contract["loss_tolerance"])
            and pcc_delta <= float(replay_contract["pcc_tolerance"])
        ) else "HARD_FAIL",
        "reference_val_loss": replay_contract["reference_val_loss"],
        "observed_val_loss": observed_loss,
        "loss_absolute_difference": loss_delta,
        "reference_val_pcc": replay_contract["reference_val_pcc"],
        "observed_val_pcc": observed_pcc,
        "pcc_absolute_difference": pcc_delta,
    })
    if (
        loss_delta > float(replay_contract["loss_tolerance"])
        or pcc_delta > float(replay_contract["pcc_tolerance"])
    ):
        raise ContractError("FBR does not reproduce the accepted internal benchmark")
    return baseline, {
        "checkpoint": checkpoint_meta,
        "baseline_replay_check": replay_check,
        "internal_predictions": internal["predictions"],
        "internal_metrics": internal["metrics"],
    }


def load_external_dataset(context: TrainingContext) -> MetadataFeatureDataset:
    boundary = DataAccessBoundary("external_benchmark")
    boundary.require_external()
    data = context.protocol["data"]
    labels_root = context.repair_group_root / "labels"
    flat_cache_root = resolve_registered_path(context.server_paths, data["flat_cache_path_id"])
    label_path = (
        labels_root
        / "external"
        / data["external_patient"]
        / f"{data['external_patient']}_ssGSEA_zscore_by_group_{data['train_mpp_id']}_train.csv"
    )
    candidate_cache_dirs = [
        flat_cache_root / str(data["external_mpp_id"]) / data["external_patient"],
        flat_cache_root / f"MPP{data['external_mpp_id']}_UNI" / data["external_patient"],
    ]
    external_cache = next((path for path in candidate_cache_dirs if path.is_dir()), None)
    if external_cache is None:
        raise ContractError(f"external feature cache is missing: {candidate_cache_dirs}")
    feature_stems = {path.stem for path in external_cache.glob("*.pt")}
    label_stems = _read_unique_label_stems(label_path)
    if not feature_stems or feature_stems != label_stems:
        raise ContractError("XZY feature-to-label identity set is not one-to-one")
    external = build_manifest_external(
        str(flat_cache_root),
        data["external_mpp_id"],
        data["external_patient"],
        str(label_path),
        target_cols=context.pathway_names,
    )
    if int(external.feat_dim) != int(context.protocol["model"]["feature_dim"]):
        raise ContractError("XZY feature dimension differs from the frozen model contract")
    if list(external.target_cols) != context.pathway_names:
        raise ContractError("XZY pathway order differs from the frozen training order")
    records = [
        FeatureRecord(Path(feature_path), target, data["external_patient"], Path(feature_path).stem, "XZY")
        for feature_path, target in external.samples
    ]
    return MetadataFeatureDataset(records, context.pathway_names)


def external_benchmark_once(
    context: TrainingContext,
    *,
    model: nn.Module,
    device: torch.device,
    output_dir: str | Path,
    frozen_config: Mapping[str, Any],
) -> dict[str, Path]:
    """冻结后只读评估；返回前逐 tensor 核对模型未改变。"""

    state_before = _snapshot_state(model)
    config_before = copy.deepcopy(dict(frozen_config))
    external_dataset = load_external_dataset(context)
    outputs = evaluate_and_write(
        model,
        external_dataset,
        context=context,
        device=device,
        split="XZY",
        output_dir=output_dir,
    )
    _assert_state_equal(state_before, _snapshot_state(model))
    if config_before != dict(frozen_config):
        raise ContractError("external benchmark modified frozen configuration")
    return {
        "external_predictions": outputs["predictions"],
        "external_metrics": outputs["metrics"],
    }
