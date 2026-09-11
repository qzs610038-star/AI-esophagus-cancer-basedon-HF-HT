"""Update-counted full-batch training with exact warm-start and resumability."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import DataBundle, PartitionData
from .metrics import SelectionCandidate, compute_regression_metrics, select_best_candidate
from .model import SpatialWarmstartModel, load_stage1_hc
from .predict import save_model_bundle
from .spatial import SpatialGraph, SpatialPoint, build_spatial_graph


ARM_ORDER = ("spatial_residual_only", "point_continue", "spatial_joint")


@dataclass(frozen=True)
class ArmPolicy:
    arm: str
    train_h: bool
    train_c: bool
    train_b: bool
    use_spatial: bool
    lr_hc: float | None
    lr_b: float | None


def arm_policy(config: Mapping[str, Any], arm: str) -> ArmPolicy:
    rows = {row["id"]: row for row in config["parameters"]["arms"]}
    if arm not in rows:
        raise KeyError(f"unknown experiment arm: {arm}")
    row = rows[arm]
    return ArmPolicy(
        arm=arm, train_h=bool(row["train_H"]), train_c=bool(row["train_C"]),
        train_b=bool(row["train_B"]), use_spatial=arm != "point_continue",
        lr_hc=float(row["lr_HC"]) if row["lr_HC"] is not None else None,
        lr_b=float(row["lr_B"]) if row["lr_B"] is not None else None,
    )


def configure_arm(model: SpatialWarmstartModel, policy: ArmPolicy) -> None:
    for parameter in model.shared.parameters():
        parameter.requires_grad_(policy.train_h)
    for parameter in model.regression.parameters():
        parameter.requires_grad_(policy.train_c)
    model.B.requires_grad_(policy.train_b)
    if not policy.train_h:
        model.shared.eval()
    if not policy.train_c:
        model.regression.eval()


def build_optimizer(
    model: SpatialWarmstartModel, policy: ArmPolicy, weight_decay: float,
) -> torch.optim.AdamW:
    groups: list[dict[str, Any]] = []
    hc = [
        parameter for module in (model.shared, model.regression)
        for parameter in module.parameters() if parameter.requires_grad
    ]
    if hc:
        if policy.lr_hc is None:
            raise ValueError(f"{policy.arm}: lr_HC is missing")
        groups.append({"params": hc, "lr": policy.lr_hc, "name": "H_C"})
    if model.B.requires_grad:
        if policy.lr_b is None:
            raise ValueError(f"{policy.arm}: lr_B is missing")
        groups.append({"params": [model.B], "lr": policy.lr_b, "name": "B"})
    if not groups:
        raise ValueError(f"{policy.arm}: no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def _points(partition: PartitionData) -> list[SpatialPoint]:
    return [
        SpatialPoint(
            identity=str(spot), patient=str(patient), partition=partition.name,
            x=float(xy[0]), y=float(xy[1]),
        )
        for patient, spot, xy in zip(
            partition.patient_ids.tolist(), partition.spot_ids.tolist(), partition.coordinates.tolist()
        )
    ]


def build_graph(partition: PartitionData, graph_config: Mapping[str, Any]) -> SpatialGraph:
    native_steps: dict[str, float] = {}
    for patient, step in zip(partition.patient_ids.tolist(), partition.native_steps.tolist()):
        previous = native_steps.setdefault(str(patient), float(step))
        if not np.isclose(previous, float(step)):
            raise ValueError(f"native step differs within patient={patient}")
    return build_spatial_graph(
        _points(partition), partition.graph_features, native_steps=native_steps,
        radius_in_native_steps=float(graph_config["radius_in_native_steps"]),
        max_neighbors=int(graph_config["max_neighbors"]),
        distance_sigma=float(graph_config["distance_sigma"]),
        image_temperature=float(graph_config["image_temperature"]),
        self_raw_weight=float(graph_config["self_raw_weight"]),
    )


def _dropout_hidden(
    hidden: torch.Tensor, probability: float, generator: torch.Generator | None,
) -> torch.Tensor:
    if probability <= 0:
        return hidden
    if generator is None:
        raise ValueError("training dropout requires the dedicated paired generator")
    # Draw on CPU so point and joint trajectories use the same stream regardless
    # of device or graph work.  The mask is moved without consuming CUDA RNG.
    mask = torch.rand(hidden.shape, generator=generator, device="cpu") >= probability
    return hidden * mask.to(device=hidden.device, dtype=hidden.dtype) / (1.0 - probability)


def _forward_train(
    model: SpatialWarmstartModel, features: torch.Tensor, graph: SpatialGraph | None,
    policy: ArmPolicy, dropout_probability: float, dropout_generator: torch.Generator,
    cached: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None,
) -> torch.Tensor:
    if cached is not None:
        hidden, delta, point = cached
    else:
        hidden = model.shared(features)
        delta = graph.spatial_delta(hidden) if graph is not None else None
        dropped = _dropout_hidden(hidden, dropout_probability, dropout_generator)
        point = model.regression.linear(dropped)
    if not policy.use_spatial:
        return point
    if delta is None:
        raise ValueError("spatial arm requires a graph delta")
    return point + F.linear(delta, model.B)


def _predict(
    model: SpatialWarmstartModel, partition: PartitionData, graph: SpatialGraph | None,
    use_spatial: bool, device: torch.device,
) -> np.ndarray:
    model.eval()
    features = torch.as_tensor(partition.features, dtype=torch.float32, device=device)
    with torch.no_grad():
        hidden = model.shared(features)
        point = model.regression.linear(hidden)
        if use_spatial:
            if graph is None:
                raise ValueError("spatial prediction requires graph")
            point = point + F.linear(graph.spatial_delta(hidden), model.B)
    return point.float().cpu().numpy()


def _candidate(update: int, report: Mapping[str, Any], arm: str) -> SelectionCandidate:
    return SelectionCandidate(
        update=int(update),
        patient_macro_pathway_pcc=float(report["patient_macro_pathway_pcc"]),
        zMSE=float(report["zMSE"]), arm=arm,
    )


def _state_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _rng_state(dropout_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dropout": dropout_generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], dropout_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    dropout_generator.set_state(state["dropout"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_payload(
    *, model: SpatialWarmstartModel, optimizer: torch.optim.Optimizer, arm: str,
    seed: int, update: int, best: SelectionCandidate, best_state: Mapping[str, torch.Tensor],
    early_reference: float, no_improvement: int, history: list[dict[str, Any]],
    dropout_generator: torch.Generator, config_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "arm": arm, "seed": int(seed), "update": int(update),
        "model_state": _state_cpu(model.state_dict()), "optimizer_state": optimizer.state_dict(),
        "best": asdict(best), "best_model_state": _state_cpu(best_state),
        "early_reference": float(early_reference), "no_improvement": int(no_improvement),
        "history": deepcopy(history), "rng_state": _rng_state(dropout_generator),
        "config_fingerprint": dict(config_fingerprint),
    }


def _fingerprint(config: Mapping[str, Any], arm: str, seed: int) -> dict[str, Any]:
    p = config["parameters"]
    return {
        "experiment_id": config["experiment_id"], "plan_version": config["plan_version"],
        "arm": arm, "seed": int(seed), "input_dim": int(p["input_dim"]),
        "hidden_dim": int(p["hidden_dim"]), "output_dim": int(p["output_dim"]),
        "dropout": float(p["dropout"]), "optimizer": p["optimizer"],
        "weight_decay": float(p["weight_decay"]), "lr_schedule": p["lr_schedule"],
        "arm_parameters": next(dict(row) for row in p["arms"] if row["id"] == arm),
        "initial_update_budget": int(p["initial_update_budget"]),
        "hard_max_updates": int(p["hard_max_updates"]),
        "validation_every_updates": int(p["validation_every_updates"]),
        "min_updates_before_early_stop": int(p["min_updates_before_early_stop"]),
        "patience_validation_checks": int(p["patience_validation_checks"]),
        "min_delta_patient_macro_pcc": float(p["min_delta_patient_macro_pcc"]),
        "graph": dict(p["graph"]),
    }


def train_arm(
    config: Mapping[str, Any], bundle: DataBundle, arm: str, run_dir: str | Path,
    weights_dir: str | Path, *, resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None, stop_after_updates: int | None = None,
) -> dict[str, Any]:
    """Train one seed-42 arm.  This function never opens external-test labels."""
    seed = int(config["parameters"]["primary_seed"])
    if seed != 42:
        raise ValueError("this package is restricted to seed 42")
    policy = arm_policy(config, arm)
    p = config["parameters"]
    required_contract = {
        "batch_mode": "full_batch", "loss": "all_point_pathway_mean_zMSE",
        "optimizer": "AdamW", "lr_schedule": "constant",
    }
    for key, expected in required_contract.items():
        if p.get(key) != expected:
            raise ValueError(f"this implementation requires parameters.{key}={expected!r}")
    if not bool(p.get("extend_at_1000_if_not_early_stopped", False)):
        raise ValueError("the predeclared 1000-to-2000 extension must remain enabled")
    if bool(p.get("backbone_trainable")) or bool(p.get("lora_enabled")) or bool(p.get("contrastive_enabled")):
        raise ValueError("backbone training, LoRA, and contrastive learning are outside this package")
    run_dir, weights_dir = Path(run_dir).resolve(), Path(weights_dir).resolve()
    raw_dir = run_dir / "raw" / arm / f"seed_{seed}"
    checkpoint_dir = weights_dir / arm / f"seed_{seed}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    setup_started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SpatialWarmstartModel(
        input_dim=int(p["input_dim"]), hidden_dim=int(p["hidden_dim"]),
        output_dim=int(p["output_dim"]), dropout=float(p["dropout"]),
    )
    source_path = Path(config["inputs"]["stage1_checkpoint"])
    source_state = torch.load(source_path, map_location="cpu", weights_only=True)
    source_report = load_stage1_hc(model, source_state)
    configure_arm(model, policy)
    model.to(device=device, dtype=torch.float32)
    optimizer = build_optimizer(model, policy, float(p["weight_decay"]))
    model_setup_seconds = float(time.perf_counter() - setup_started)
    dropout_generator = torch.Generator(device="cpu").manual_seed(seed + 910_247)
    train, val = bundle.partitions["train"], bundle.partitions["internal_val"]
    graph_cfg = p["graph"]
    graph_started = time.perf_counter()
    train_graph = build_graph(train, graph_cfg) if policy.use_spatial else None
    val_graph = build_graph(val, graph_cfg) if policy.use_spatial else None
    graph_build_seconds = float(time.perf_counter() - graph_started)
    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    targets = torch.as_tensor(train.targets, dtype=torch.float32, device=device)
    cached = None
    if arm == "spatial_residual_only":
        model.shared.eval(); model.regression.eval()
        with torch.no_grad():
            hidden = model.shared(features)
            delta = train_graph.spatial_delta(hidden)
            point = model.regression.linear(hidden)
        cached = (hidden, delta, point)

    start = 0
    history: list[dict[str, Any]] = []
    best: SelectionCandidate
    best_state: dict[str, torch.Tensor]
    early_reference: float
    no_improvement = 0
    fingerprint = _fingerprint(config, arm, seed)
    base_warmup_state = _state_cpu(model.state_dict())
    torch.save(
        {"schema_version": "1.0", "kind": "warmup_step0", "arm": arm, "seed": seed,
         "update": 0, "model_state": base_warmup_state,
         "source_checkpoint": str(source_path), "source_load_report": asdict(source_report)},
        checkpoint_dir / "warmup.pt",
    )
    warmup_bundle_path = save_model_bundle(
        checkpoint_dir / "warmup_bundle.pt", model, arm=arm,
        spatial_enabled=policy.use_spatial, graph_parameters=dict(graph_cfg),
        pathway_names=bundle.pathway_names, normalization=bundle.normalization,
        feature_contract={"dimension": int(p["input_dim"]), "dtype": "float32",
                          "source": "frozen_UNI2-h_CLS_cache", "augmentation": False},
        native_step_contract={"source": "inputs/mpp2/split_info.json or explicit external cache",
                              "unit": "native_split_manifest_xy"},
        coordinate_contract={"identity": "patient_id+patch_stem", "columns": ["x", "y"],
                             "graph_scope": "within_patient_and_partition"},
        provenance={"kind": "warmup_step0", "source_checkpoint": str(source_path),
                    "arm": arm, "seed": seed, "selected_update": 0},
    )
    source_tolerance = float(p.get(
        "source_cache_prediction_max_abs_tolerance",
        p["initial_prediction_max_abs_tolerance"],
    ))
    source_precision = str(config["inputs"].get("source_prediction_precision", "unspecified"))
    if resume_checkpoint is not None:
        consistency: dict[str, Any] = {
            "tolerance": source_tolerance, "source_precision": source_precision,
        }
        step0_arrays: dict[str, np.ndarray] = {"pathway_names": np.asarray(bundle.pathway_names, dtype=str)}
        for name, partition, graph in (("train", train, train_graph), ("internal_val", val, val_graph)):
            predicted = _predict(model, partition, graph, policy.use_spatial, device)
            if partition.source_predictions is None:
                raise KeyError(f"{name}: source predictions are required for resume inheritance check")
            maximum = float(np.max(np.abs(predicted - partition.source_predictions)))
            consistency[name] = {"max_abs": maximum, "passed": maximum <= consistency["tolerance"]}
            step0_arrays[f"prediction_{name}"] = predicted
            step0_arrays[f"target_{name}"] = partition.targets
            step0_arrays[f"patient_{name}"] = partition.patient_ids
            step0_arrays[f"patch_stem_{name}"] = partition.spot_ids
            if maximum > consistency["tolerance"]:
                raise AssertionError(f"{name}: resume source prediction mismatch max_abs={maximum}")
        _write_json(raw_dir / "step0_consistency.json", consistency)
        np.savez_compressed(raw_dir / "step0_predictions.npz", **step0_arrays)
        resume_path = Path(resume_checkpoint).resolve()
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("arm") != arm or int(payload.get("seed", -1)) != seed:
            raise ValueError("resume checkpoint arm/seed does not match requested trajectory")
        if payload.get("config_fingerprint") != fingerprint:
            raise ValueError("resume checkpoint configuration does not match this trajectory")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start = int(payload["update"])
        best = SelectionCandidate(**payload["best"])
        best_state = _state_cpu(payload["best_model_state"])
        early_reference = float(payload["early_reference"])
        no_improvement = int(payload["no_improvement"])
        history = list(payload["history"])
        _restore_rng(payload["rng_state"], dropout_generator)
    else:
        step0_pred = _predict(model, val, val_graph, policy.use_spatial, device)
        step0_report = compute_regression_metrics(step0_pred, val.targets, val.patient_ids)
        step0_report.update({"update": 0, "arm": arm, "elapsed_seconds": 0.0})
        history.append(step0_report)
        best = _candidate(0, step0_report, arm)
        best_state = _state_cpu(model.state_dict())
        early_reference = best.patient_macro_pathway_pcc
        consistency: dict[str, Any] = {
            "tolerance": source_tolerance, "source_precision": source_precision,
        }
        step0_arrays: dict[str, np.ndarray] = {"pathway_names": np.asarray(bundle.pathway_names, dtype=str)}
        for name, partition, graph in (("train", train, train_graph), ("internal_val", val, val_graph)):
            predicted = _predict(model, partition, graph, policy.use_spatial, device)
            if partition.source_predictions is None:
                raise KeyError(f"{name}: source predictions are required for step0 inheritance check")
            maximum = float(np.max(np.abs(predicted - partition.source_predictions)))
            consistency[name] = {"max_abs": maximum, "passed": maximum <= consistency["tolerance"]}
            step0_arrays[f"prediction_{name}"] = predicted
            step0_arrays[f"target_{name}"] = partition.targets
            step0_arrays[f"patient_{name}"] = partition.patient_ids
            step0_arrays[f"patch_stem_{name}"] = partition.spot_ids
            if maximum > consistency["tolerance"]:
                raise AssertionError(f"{name}: step0/source prediction mismatch max_abs={maximum}")
        _write_json(raw_dir / "step0_consistency.json", consistency)
        np.savez_compressed(raw_dir / "step0_predictions.npz", **step0_arrays)
        warmup_payload = torch.load(checkpoint_dir / "warmup.pt", map_location="cpu", weights_only=False)
        warmup_payload["metrics"] = step0_report
        torch.save(warmup_payload, checkpoint_dir / "warmup.pt")

    validation_every = int(p["validation_every_updates"])
    minimum_updates = int(p["min_updates_before_early_stop"])
    patience = int(p["patience_validation_checks"])
    min_delta = float(p["min_delta_patient_macro_pcc"])
    initial_budget = int(p["initial_update_budget"])
    hard_max = int(p["hard_max_updates"])
    if not (0 < validation_every <= initial_budget <= hard_max):
        raise ValueError("invalid update budget/validation schedule")
    setup_seconds = float(time.perf_counter() - setup_started)
    began = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopped_reason = "budget_exhausted"
    last_update = start
    loop_max = hard_max if stop_after_updates is None else min(hard_max, int(stop_after_updates))
    if loop_max <= start:
        raise ValueError("stop_after_updates must be greater than the resume update")
    for update in range(start + 1, loop_max + 1):
        last_update = update
        update_started = time.perf_counter()
        if policy.train_h or policy.train_c:
            model.shared.train(policy.train_h)
            model.regression.train(policy.train_c)
        else:
            model.shared.eval(); model.regression.eval()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward_train(
            model, features, train_graph, policy, float(p["dropout"]), dropout_generator, cached
        )
        loss = F.mse_loss(prediction.float(), targets.float())
        loss.backward()
        optimizer.step()
        update_seconds = float(time.perf_counter() - update_started)
        if update % validation_every != 0 and update != loop_max:
            continue
        validation_started = time.perf_counter()
        val_prediction = _predict(model, val, val_graph, policy.use_spatial, device)
        report = compute_regression_metrics(val_prediction, val.targets, val.patient_ids)
        validation_seconds = float(time.perf_counter() - validation_started)
        report.update({
            "update": update, "arm": arm, "train_loss": float(loss.detach().cpu()),
            "elapsed_seconds": float(time.perf_counter() - began),
            "budget_phase": "initial" if update <= initial_budget else "extended",
            "update_seconds": update_seconds, "validation_seconds": validation_seconds,
            "milestone": update if update in {50, 500, 1000, 2000} else None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        })
        history.append(report)
        candidate = _candidate(update, report, arm)
        if select_best_candidate([best, candidate]).update == candidate.update:
            best = candidate
            best_state = _state_cpu(model.state_dict())
        report["best_update_after_validation"] = best.update
        report["recent_primary_metric"] = [
            float(row["patient_macro_pathway_pcc"]) for row in history[-5:]
        ]
        if candidate.patient_macro_pathway_pcc > early_reference + min_delta:
            early_reference = candidate.patient_macro_pathway_pcc
            no_improvement = 0
        else:
            no_improvement += 1
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, arm=arm, seed=seed, update=update,
            best=best, best_state=best_state, early_reference=early_reference,
            no_improvement=no_improvement, history=history,
            dropout_generator=dropout_generator, config_fingerprint=fingerprint,
        )
        torch.save(payload, checkpoint_dir / "last.pt")
        _write_json(raw_dir / "history.json", history)
        print(json.dumps({
            "arm": arm, "seed": seed, "update": update,
            "patient_macro_pathway_pcc": report["patient_macro_pathway_pcc"],
            "flattened_pooled_pcc": report["flattened_pooled_pcc"],
            "zMSE": report["zMSE"], "best_update": best.update,
            "elapsed_seconds": report["elapsed_seconds"],
            "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        }, ensure_ascii=False), flush=True)
        if update >= minimum_updates and no_improvement >= patience:
            stopped_reason = "early_stopped"
            break
    if stop_after_updates is not None and last_update < hard_max and stopped_reason == "budget_exhausted":
        stopped_reason = "interrupted"
    model.load_state_dict(best_state, strict=True)
    best_report = next(row for row in history if int(row["update"]) == int(best.update))
    step0_report = next(row for row in history if int(row["update"]) == 0)
    formal_payload = {
        "schema_version": "1.0", "kind": "internal_best", "arm": arm, "seed": seed,
        "update": best.update, "model_state": best_state, "metrics": best_report,
        "source_checkpoint": str(source_path), "config_fingerprint": fingerprint,
    }
    torch.save(formal_payload, checkpoint_dir / "formal.pt")
    formal_train = _predict(model, train, train_graph, policy.use_spatial, device)
    formal_val = _predict(model, val, val_graph, policy.use_spatial, device)
    bundle_path = save_model_bundle(
        checkpoint_dir / "model_bundle.pt", model, arm=arm,
        spatial_enabled=policy.use_spatial, graph_parameters=dict(graph_cfg),
        pathway_names=bundle.pathway_names, normalization=bundle.normalization,
        feature_contract={
            "dimension": int(p["input_dim"]), "dtype": "float32",
            "source": "frozen_UNI2-h_CLS_cache", "augmentation": False,
        },
        native_step_contract={
            "source": "inputs/mpp2/split_info.json or explicit external cache",
            "unit": "native_split_manifest_xy",
        },
        coordinate_contract={
            "identity": "patient_id+patch_stem", "columns": ["x", "y"],
            "graph_scope": "within_patient_and_partition",
        },
        provenance={
            "source_checkpoint": str(source_path), "arm": arm, "seed": seed,
            "selected_update": best.update,
        },
    )
    np.savez_compressed(
        raw_dir / "predictions.npz",
        prediction_train=formal_train, target_train=train.targets,
        patient_train=train.patient_ids, patch_stem_train=train.spot_ids,
        prediction_internal_val=formal_val, target_internal_val=val.targets,
        patient_internal_val=val.patient_ids, patch_stem_internal_val=val.spot_ids,
        pathway_names=np.asarray(bundle.pathway_names, dtype=str),
    )
    if train_graph is not None:
        np.savez_compressed(
            raw_dir / "graph_train.npz", neighbor_index=train_graph.neighbor_index,
            neighbor_weight=train_graph.neighbor_weight, neighbor_mask=train_graph.neighbor_mask,
            self_weight=train_graph.self_weight, degree=train_graph.degree,
            neighbor_distance_native_steps=train_graph.neighbor_distance_native_steps,
            neighbor_cosine_similarity=train_graph.neighbor_cosine_similarity,
            patient=train.patient_ids, patch_stem=train.spot_ids,
            coordinates=train.coordinates, native_step=train.native_steps,
        )
        np.savez_compressed(
            raw_dir / "graph_internal_val.npz", neighbor_index=val_graph.neighbor_index,
            neighbor_weight=val_graph.neighbor_weight, neighbor_mask=val_graph.neighbor_mask,
            self_weight=val_graph.self_weight, degree=val_graph.degree,
            neighbor_distance_native_steps=val_graph.neighbor_distance_native_steps,
            neighbor_cosine_similarity=val_graph.neighbor_cosine_similarity,
            patient=val.patient_ids, patch_stem=val.spot_ids,
            coordinates=val.coordinates, native_step=val.native_steps,
        )
    _write_json(raw_dir / "history.json", history)
    summary = {
        "arm": arm, "seed": seed, "best_update": best.update,
        "best_metrics": asdict(best), "last_update": last_update,
        "best_validation_metrics": best_report, "step0_metrics": step0_report,
        "stop_reason": stopped_reason, "extended_after_initial_budget": last_update > initial_budget,
        "warmup": str((checkpoint_dir / "warmup.pt").resolve()),
        "formal": str((checkpoint_dir / "formal.pt").resolve()),
        "last": str((checkpoint_dir / "last.pt").resolve()),
        "model_bundle": str(bundle_path.resolve()),
        "warmup_bundle": str(warmup_bundle_path.resolve()),
        "external_evaluation": "not_run_by_training",
        "timing": {
            "model_and_source_setup_seconds": model_setup_seconds,
            "graph_build_seconds": graph_build_seconds,
            "setup_total_seconds": setup_seconds,
            "training_and_validation_seconds": float(time.perf_counter() - began),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
    }
    _write_json(raw_dir / "summary.json", summary)
    return summary
