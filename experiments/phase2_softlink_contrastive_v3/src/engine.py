"""Executable, resumable Phase2-v4 training orchestration.

This module owns scheduling, isolation, persistence and selection.  Costly
tensor/image operations live behind a small runtime protocol so production can
use :class:`TorchRuntime`, while CPU smoke tests inject a tiny offline runtime.
Unlike ``orchestrator.build_task_dag``, this module invokes every training
stage and records completed work for resume.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import data, protocols, training
from . import backbone as backbone_api
from . import diagnostics, losses, metrics, models, spatial
from .io_utils import task_directory, weight_registry_entry


class TrainingRuntime(Protocol):
    """The concrete work performed at each non-plan-only pipeline boundary."""

    def materialize(self, split: protocols.ProtocolSplit, config: Mapping[str, Any]) -> dict[str, Any]: ...
    def fit_training_artifacts(self, partitions: Mapping[str, Any], split: protocols.ProtocolSplit) -> dict[str, Any]: ...
    def resource_short_test(self, context: Mapping[str, Any], candidates: Sequence[int]) -> dict[str, Any]: ...
    def warmup(self, context: Mapping[str, Any], seed: int, checkpoint_dir: Path) -> dict[str, Any]: ...
    def pilot(self, context: Mapping[str, Any], seed: int, lr: float, warmup: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, Any]: ...
    def stage1(self, context: Mapping[str, Any], seed: int, cell: str, lr: float, warmup: Mapping[str, Any], checkpoint_dir: Path, reused_pilot: Mapping[str, Any] | None = None) -> dict[str, Any]: ...
    def cache_cls(self, context: Mapping[str, Any], seed: int, cell: str, endpoint: int, stage1_result: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]: ...
    def stage2(self, context: Mapping[str, Any], seed: int, task: training.Stage2Task, stage1_result: Mapping[str, Any], cache: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, Any]: ...
    def evaluate(self, context: Mapping[str, Any], seed: int, cell: str, stage1_result: Mapping[str, Any], destination: Path) -> dict[str, Any]: ...


class ResourceShortTestFailure(RuntimeError):
    """Hard resource failure carrying measurements that must still be persisted."""

    def __init__(self, message: str, result_payload: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.result_payload = dict(result_payload)


def _config(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items() if str(key) not in {"model", "optimizer", "state"}}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _load_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(config.get("manifest"), pd.DataFrame):
        return config["manifest"].copy()
    data_config = config.get("data", {})
    location = data_config.get("split_manifest_file") or config.get("split_manifest_file")
    if not location:
        raise ValueError("config must provide data.split_manifest_file")
    path = Path(location)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return data.read_split_manifest(path)


def _splits(manifest: pd.DataFrame, protocol: str, folds: Sequence[str] | None) -> list[protocols.ProtocolSplit]:
    if protocol == "original":
        values = [protocols.build_original_split(manifest)]
    elif protocol == "lopo6":
        values = protocols.build_lopo6_splits(manifest)
    else:
        raise ValueError("protocol must be 'original' or 'lopo6'")
    requested = None if not folds else {str(fold) for fold in folds}
    if requested:
        unknown = requested - {split.fold for split in values}
        if unknown:
            raise ValueError(f"unknown folds for {protocol}: {sorted(unknown)}")
        values = [split for split in values if split.fold in requested]
    return values


def _cell_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    # The training module's names are stable state/checkpoint keys.  Config
    # display names may carry the more verbose lora_r8 spelling.
    return tuple(training.MAIN_CELLS)


def _state_path(run_dir: Path) -> Path:
    return run_dir / "raw" / "engine_state.json"


def _state_load(path: Path, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("engine_state exists; pass resume=True to continue")
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": "v4", "tasks": {}}


def _state_save(path: Path, state: Mapping[str, Any]) -> None:
    _write_json(path, state)


def _task_key(split: protocols.ProtocolSplit, seed: int, kind: str, extra: str = "") -> str:
    suffix = f"/{extra}" if extra else ""
    return f"{split.protocol}/{split.fold}/seed_{seed}/{kind}{suffix}"


def _task(
    state: dict[str, Any], state_path: Path, key: str, resume: bool, action, result_path: Path,
) -> tuple[Any | None, bool]:
    entry = state["tasks"].get(key)
    if resume and entry and entry.get("status") == "completed":
        return (json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else entry.get("result")), True
    state["tasks"][key] = {"status": "running", "result_path": str(result_path)}
    _state_save(state_path, state)
    try:
        result = action()
        _write_json(result_path, result)
        state["tasks"][key] = {"status": "completed", "result_path": str(result_path), "result": _jsonable(result)}
        _state_save(state_path, state)
        return result, False
    except Exception as error:
        failure_payload = getattr(error, "result_payload", None)
        if failure_payload is not None:
            _write_json(result_path, failure_payload)
        state["tasks"][key] = {"status": "failed", "error": str(error), "result_path": str(result_path)}
        _state_save(state_path, state)
        raise
    finally:
        # At a resumable task boundary every production model is disposable.
        # Release dead Python references and return cached blocks to WDDM so
        # completed cells do not create a staircase of reserved GPU memory.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _partition_summary(partitions: Mapping[str, Any]) -> dict[str, Any]:
    """Persist identities/counts only; OnlineRecord objects are rebuilt on resume."""
    summary: dict[str, Any] = {}
    for name, records in partitions.items():
        identities = [list(record.identity) for record in records] if all(hasattr(record, "identity") for record in records) else []
        patients = sorted({str(record.patient) for record in records}) if all(hasattr(record, "patient") for record in records) else []
        summary[name] = {
            "count": len(records),
            "identities": identities,
            "patients": patients,
        }
    return summary


def _registered_weight_entry(*, protocol: str, fold: str, seed: int, cell: str,
                             endpoint: int, directory: Path, warmup: str | Path,
                             formal: str | Path, last: str | Path,
                             base_checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Register only paths that the concrete runtime actually writes."""
    def item(value: str | Path) -> dict[str, str]:
        path=Path(value).resolve()
        return {"path":str(path),"status":"present" if path.is_file() else "missing"}
    return {
        "protocol":protocol,"fold":fold,"seed":int(seed),"cell":cell,"endpoint":int(endpoint),
        "weight_directory":str(directory.resolve()),"warmup":item(warmup),"formal":item(formal),"last":item(last),
        "base_checkpoint_reference":str(base_checkpoint) if base_checkpoint is not None else None,
    }


class TorchRuntime:
    """Default raw-PNG/ssGSEA adapter.

    It deliberately materialises raw data and training-only statistics before
    requesting the concrete trainer.  The package's deployment launcher may
    supply a richer runtime object (for accelerator-specific AMP/loader code)
    through ``dependencies``; this adapter fails closed rather than silently
    substituting cached/held-out labels for real online training.
    """

    def __init__(self) -> None:
        self._base_backbones: dict[int, nn.Module] = {}
        self._warmup_states: dict[tuple[int, int], dict[str, Any]] = {}
        self._stage2_c_initial: dict[tuple[int, int, str, int], dict[str, torch.Tensor]] = {}
        self._spatial_graphs: dict[tuple[int, str, str, str], spatial.SpatialGraph] = {}

    def _device(self, context: Mapping[str, Any]) -> torch.device:
        configured = context["config"].get("device")
        return torch.device(configured or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _stage1_precision(self, context: Mapping[str, Any], device: torch.device | None = None) -> str:
        device = self._device(context) if device is None else device
        requested = str(
            context["config"].get("training", {}).get("precision", {}).get("stage1", "float32")
        ).lower()
        if requested in {"server_verified_bf16_or_float32", "auto"}:
            requested = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        if requested not in {"bf16", "float32"}:
            raise ValueError("training.precision.stage1 must be 'bf16' or 'float32'")
        if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured BF16 stage1 precision is unsupported by this CUDA device")
        return "bf16" if requested == "bf16" and device.type == "cuda" else "float32"

    def _autocast(self, context: Mapping[str, Any], device: torch.device | None = None):
        device = self._device(context) if device is None else device
        if self._stage1_precision(context, device) == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _configure_gradient_checkpointing(
        self, context: Mapping[str, Any], backbone: nn.Module,
    ) -> tuple[bool, str | None]:
        """Enable activation checkpointing when supported, without blocking training.

        UNI2-h/timm normally exposes ``set_grad_checkpointing``.  The optional
        fallback is intentional: a packaging/version mismatch is recorded but
        must not replace a runnable non-checkpointed training path.
        """
        cfg = context["config"].get("training", {}).get("gradient_checkpointing", {})
        # Default true is required for resumes created before v3.0.3, because
        # those runs intentionally retain their immutable older config snapshot.
        enabled = bool(cfg.get("enabled", True)) if isinstance(cfg, Mapping) else bool(cfg)
        if not enabled:
            return False, None
        try:
            backbone_api.set_gradient_checkpointing(backbone, True)
        except RuntimeError as error:
            message = f"gradient checkpointing unavailable; continuing without it: {error}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return False, message
        return True, None

    def _normalization(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        z = context["artifacts"]["zscore"]
        return {"mean": np.asarray(z["mean"], dtype=np.float32), "std": np.asarray(z["std"], dtype=np.float32)}

    def _regression_report(self, context, prediction, target, patient_ids):
        z=self._normalization(context)
        raw_prediction=np.asarray(prediction)*z["std"]+z["mean"]
        raw_target=np.asarray(target)*z["std"]+z["mean"]
        return metrics.regression_metrics(prediction,target,patient_ids,raw_pred=raw_prediction,raw_target=raw_target)

    def _native_steps(self, context: Mapping[str, Any]) -> dict[str, float]:
        configured = context["config"].get("graph", {}).get("native_steps")
        if configured:
            return {str(patient): float(value) for patient, value in configured.items()}
        location = context["config"].get("data", {}).get("slide_geometry_file")
        if not location:
            raise ValueError("graph.native_steps or data.slide_geometry_file must be configured for spatial stage2")
        path = Path(location)
        if not path.is_absolute(): path = Path(__file__).resolve().parents[1] / path
        geometry = pd.read_csv(path)
        if not {"patient_id", "s"}.issubset(geometry.columns):
            raise ValueError("slide_geometry_file must contain patient_id and s")
        steps = {str(row.patient_id): float(row.s) for row in geometry.itertuples(index=False) if pd.notna(row.s) and float(row.s) > 0}
        needed = {record.patient for record in context["partitions"]["train"]}
        missing = needed - set(steps)
        if missing: raise ValueError(f"slide geometry lacks native step for patients: {sorted(missing)}")
        return steps

    def _spatial_graph(self, context, partition: str, features, native_steps):
        split=context["split"]; key=(id(context["config"]),split.protocol,split.fold,partition)
        if key not in self._spatial_graphs:
            self._spatial_graphs[key]=spatial.build_spatial_graph(context["partitions"][partition],features,native_steps=native_steps)
        return self._spatial_graphs[key]

    def _base(self, context: Mapping[str, Any]) -> nn.Module:
        key = id(context["config"])
        if key not in self._base_backbones:
            checkpoint = context["config"].get("data", {}).get("uni2_h_weights")
            if not checkpoint:
                raise ValueError("data.uni2_h_weights is required for online training")
            # Keep exactly one read-only CPU master.  Every online task gets a
            # disposable device copy, so the GPU never retains both a frozen
            # reference backbone and one model per completed experiment cell.
            self._base_backbones[key], _ = backbone_api.load_uni2_h(checkpoint, device="cpu")
        return self._base_backbones[key]

    def _fresh_backbone(self, context: Mapping[str, Any]) -> nn.Module:
        return deepcopy(self._base(context)).to(self._device(context))

    def _dataset(self, context: Mapping[str, Any], records, *, seed: int, augment: bool):
        split = context["split"]
        return data.OnlinePatchDataset(
            records, self._normalization(context), data.build_image_transform(augment=augment),
            protocol=split.protocol, fold=split.fold, seed=seed, augment=augment,
        )

    def _batch(self, context: Mapping[str, Any], records, indices, *, seed: int, epoch: int = 0, update: int = 0, augment: bool = False):
        dataset = self._dataset(context, records, seed=seed, augment=augment)
        samples = [dataset[data.SampleRequest(int(index), epoch, update, slot)] for slot, index in enumerate(indices)]
        device = self._device(context)
        return (
            torch.stack([sample["image"] for sample in samples]).to(device),
            torch.stack([sample["target"] for sample in samples]).to(device),
            [str(sample["patient"]) for sample in samples],
        )

    @torch.no_grad()
    def _predict(self, context: Mapping[str, Any], model: models.Stage1Model, records, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        model.eval(); predictions, targets, patients = [], [], []
        batch_size = int(context["resource_short_test"]["batch_size"])
        for start in range(0, len(records), batch_size):
            images, target, ids = self._batch(context, records, range(start, min(start + batch_size, len(records))), seed=seed)
            with self._autocast(context):
                prediction = model(images)["pred"]
            predictions.append(prediction.float().cpu().numpy()); targets.append(target.float().cpu().numpy()); patients.extend(ids)
        return np.concatenate(predictions), np.concatenate(targets), patients

    def materialize(self, split: protocols.ProtocolSplit, config: Mapping[str, Any]) -> dict[str, Any]:
        cfg = config.get("data", {})
        # Explicit templates win.  The fallback is the server's documented
        # group/patient layout and remains entirely local (no downloads).
        label_template = cfg.get("raw_label_template") or str(Path(str(cfg["labels_root"])) / "{group}" / "{patient}" / "{patient}_ssGSEA.csv")
        image_template = cfg.get("image_template") or str(Path(str(cfg["patch_images_root"])) / "{group}" / "{patient}" / "{patch_stem}.png")
        def make(rows: pd.DataFrame):
            return data.materialize_records(rows, raw_label_template=str(label_template), image_template=str(image_template))[0] if len(rows) else []
        return {"train": make(split.train_rows), "val": make(split.internal_val_rows), "held": make(split.held_out_rows)}

    def fit_training_artifacts(self, partitions: Mapping[str, Any], split: protocols.ProtocolSplit) -> dict[str, Any]:
        train = list(partitions["train"])
        values = np.stack([record.raw_target for record in train])
        # These helpers record train identities and use ddof=1 by default.
        zscore = protocols.fit_zscore(split.train_rows, values=values, ddof=1)
        means = protocols.fit_patient_means(split.train_rows, values=values)
        z_patient_means = {
            patient: (mean - zscore.mean) / zscore.std
            for patient, mean in means.patient_means.items()
        }
        return {
            "zscore": {"mean": zscore.mean, "std": zscore.std, "ddof": zscore.ddof, "fit_identity": zscore.fit_identity},
            "patient_means": z_patient_means, "patient_mean_fit_identity": means.fit_identity, "z_ddof": zscore.ddof,
        }

    def resource_short_test(self, context: Mapping[str, Any], candidates: Sequence[int]) -> dict[str, Any]:
        records = context["partitions"]["train"]
        device, requested = self._device(context), [int(value) for value in candidates]
        if not requested or 64 not in requested or 128 not in requested:
            raise ValueError("v4 resource test must evaluate batch sizes 64 and 128")
        measurements = []
        resource_warnings: list[str] = []
        precision = self._stage1_precision(context, device)
        margin = int(context["config"].get("resource_short_test", {}).get("minimum_memory_margin_bytes", 2 * 1024**3))
        for batch_size in sorted(requested, reverse=True):
            optimizer = probe = backbone = last_prediction = eval_prediction = eval_images = update = None
            checkpointing_enabled = False
            try:
                # Measure real r=8 regression updates, including backward and
                # AdamW.  The disposable model is never used for selection.
                torch.manual_seed(20260908)
                backbone = self._fresh_backbone(context)
                models.inject_independent_qv_lora(backbone, rank=8, alpha=16, dropout=0.05)
                checkpointing_enabled, checkpointing_warning = self._configure_gradient_checkpointing(
                    context, backbone,
                )
                if checkpointing_warning and checkpointing_warning not in resource_warnings:
                    resource_warnings.append(checkpointing_warning)
                probe = models.Stage1Model(backbone).to(device).train()
                for parameter in probe.teacher.parameters():
                    parameter.requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in probe.parameters() if parameter.requires_grad],
                    lr=1e-5, weight_decay=1e-4,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                last_prediction = None
                def update(iteration: int) -> None:
                    nonlocal last_prediction
                    indices = [index % len(records) for index in range(iteration * batch_size, (iteration + 1) * batch_size)]
                    images, target, _ = self._batch(context, records, indices, seed=20260908, update=iteration, augment=True)
                    optimizer.zero_grad(set_to_none=True)
                    with self._autocast(context, device):
                        last_prediction = probe(images)["pred"]
                        loss = torch.nn.functional.mse_loss(last_prediction.float(), target.float())
                    loss.backward()
                    optimizer.step()
                # Warmup is excluded from throughput timing.
                for iteration in range(10):
                    update(iteration)
                if device.type == "cuda": torch.cuda.synchronize(device)
                start = time.perf_counter()
                for iteration in range(30):
                    update(iteration + 10)
                if device.type == "cuda": torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                probe.eval()
                eval_indices = [index % len(records) for index in range(batch_size)]
                eval_images, _, _ = self._batch(
                    context, records, eval_indices, seed=20260908, augment=False,
                )
                eval_start = time.perf_counter()
                with torch.no_grad():
                    with self._autocast(context, device):
                        eval_prediction = probe(eval_images)["pred"].float().cpu()
                if device.type == "cuda": torch.cuda.synchronize(device)
                evaluation_time = time.perf_counter() - eval_start
                if device.type == "cuda":
                    allocated_memory = int(torch.cuda.max_memory_allocated(device))
                    reserved_memory = int(torch.cuda.max_memory_reserved(device))
                    free_memory, total_memory = (int(value) for value in torch.cuda.mem_get_info(device))
                else:
                    allocated_memory = reserved_memory = free_memory = total_memory = 0
                measurements.append({
                    "batch_size": batch_size, "warmup_updates": 10, "timed_updates": 30,
                    "updates_per_second": 30 / max(elapsed, 1e-12),
                    "graphs_per_second": 30 * batch_size / max(elapsed, 1e-12),
                    "evaluation_time": evaluation_time,
                    "allocated_memory": allocated_memory,
                    "reserved_memory": reserved_memory,
                    "free_memory_after_measurement": free_memory,
                    "device_total_memory": total_memory,
                    "precision": precision,
                    "gradient_checkpointing": checkpointing_enabled,
                    "trainable_parameters": models.count_trainable_parameters(probe),
                    "prediction_sample": eval_prediction[:2].tolist(),
                    "status": "passed",
                })
            except torch.cuda.OutOfMemoryError:
                measurements.append({
                    "batch_size": batch_size, "warmup_updates": 10, "timed_updates": 30,
                    "status": "oom", "precision": precision,
                    "gradient_checkpointing": checkpointing_enabled,
                    "allocated_memory": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                    "reserved_memory": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
                    "free_memory_after_measurement": int(torch.cuda.mem_get_info(device)[0]) if device.type == "cuda" else 0,
                    "device_total_memory": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0,
                })
            finally:
                optimizer = probe = backbone = last_prediction = eval_prediction = eval_images = update = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        by_size={row["batch_size"]:row for row in measurements}
        if 64 not in by_size or by_size[64].get("status") == "oom":
            raise ResourceShortTestFailure(
                "required fallback batch 64 failed the 10+30 resource test",
                {
                    "status": "failed",
                    "batch_size": None,
                    "measurements": measurements,
                    "fallback_used": True,
                    "memory_margin_bytes": margin,
                    "memory_risk": True,
                    "warning": "batch64 OOM; formal training cannot start",
                    "warnings": ["batch64 OOM; formal training cannot start"],
                    "precision": precision,
                    "selection_reason": "batch64_oom",
                },
            )
        batch64_memory_risk = bool(device.type == "cuda" and (
            int(by_size[64]["reserved_memory"]) + margin >= int(by_size[64]["device_total_memory"])
            or int(by_size[64]["free_memory_after_measurement"]) < margin
        ))
        if batch64_memory_risk:
            message = (
                f"batch64 completed but lacks the configured {margin} byte GPU memory margin; "
                "continuing formal training with batch64 as configured, with possible WDDM shared-memory slowdown"
            )
            resource_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        row128 = by_size.get(128)
        if row128 is None:
            usable_128, reason = False, "batch128_unavailable"
        elif row128.get("status") == "oom":
            usable_128, reason = False, "batch128_oom"
        elif device.type == "cuda" and (
            int(row128["reserved_memory"]) + margin >= int(row128["device_total_memory"])
            or int(row128["free_memory_after_measurement"]) < margin
        ):
            usable_128, reason = False, "batch128_insufficient_memory_margin"
        else:
            usable_128, reason = True, "batch128_selected"
        choose_128 = bool(
            usable_128
            and float(row128["graphs_per_second"]) >= float(by_size[64]["graphs_per_second"])
        )
        if usable_128 and not choose_128:
            reason = "batch64_higher_or_equal_image_throughput"
        if batch64_memory_risk:
            reason = "batch64_insufficient_memory_margin"
        warning = "; ".join(resource_warnings) if resource_warnings else None
        return {
            "batch_size": 128 if choose_128 else 64,
            "measurements": measurements,
            "fallback_used": not choose_128,
            "memory_margin_bytes": margin,
            "memory_risk": batch64_memory_risk,
            "warning": warning,
            "warnings": resource_warnings,
            "precision": precision,
            "selection_reason": reason,
        }

    def warmup(self, context, seed, checkpoint_dir):
        device, base = self._device(context), None
        train, val = context["partitions"]["train"], context["partitions"]["val"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Frozen CLS extraction is real; cache each isolated partition before
        # H/C training so a failed warmup can resume without repeating UNI2-h.
        def cls(records, partition):
            nonlocal base
            cache_path = checkpoint_dir / f"frozen_cls_{partition}.pt"
            identities = [[str(value) for value in record.identity] for record in records]
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                features = cached.get("features") if isinstance(cached, Mapping) else None
                if (
                    isinstance(features, torch.Tensor)
                    and cached.get("identities") == identities
                    and tuple(features.shape) == (len(records), backbone_api.UNI2_FEATURE_DIM)
                ):
                    return features.to(device=device, dtype=torch.float32)
                warnings.warn(
                    f"ignoring incompatible frozen CLS cache: {cache_path}",
                    RuntimeWarning, stacklevel=2,
                )
            if base is None:
                base = self._fresh_backbone(context)
            values = []
            for start in range(0, len(records), int(context["resource_short_test"]["batch_size"])):
                images, _, _ = self._batch(context, records, range(start, min(start + int(context["resource_short_test"]["batch_size"]), len(records))), seed=seed)
                with torch.no_grad():
                    with self._autocast(context, device):
                        values.append(backbone_api.forward_cls(base, images).float().cpu())
            features = torch.cat(values)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save({"identities": identities, "features": features}, temporary)
            temporary.replace(cache_path)
            return features.to(device=device, dtype=torch.float32)
        train_cls, val_cls = cls(train, "train"), cls(val, "internal_val")
        if base is not None:
            del base
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        train_y = torch.as_tensor(np.stack([record.raw_target for record in train]), device=device, dtype=torch.float32)
        z = self._normalization(context); train_y = (train_y - torch.as_tensor(z["mean"], device=device)) / torch.as_tensor(z["std"], device=device)
        val_y = torch.as_tensor(np.stack([record.raw_target for record in val]), device=device, dtype=torch.float32); val_y = (val_y-torch.as_tensor(z["mean"],device=device))/torch.as_tensor(z["std"],device=device)
        shared, head = models.SharedProjection().to(device), models.RegressionHead().to(device)
        bundle = nn.ModuleDict({"shared": shared, "head": head})
        optimizer = torch.optim.AdamW(bundle.parameters(), lr=3e-4, weight_decay=1e-4)
        sampler=training.PatientBalancedBatchSampler([record.patient for record in train],int(context["resource_short_test"]["batch_size"]),protocol=context["split"].protocol,fold=context["split"].fold,seed=seed)
        updates_per_epoch=len(sampler.batches(1))
        planned_epoch, planned_batches = None, None
        def batch_at(epoch, update):
            nonlocal planned_epoch, planned_batches
            if planned_epoch != epoch:
                planned_epoch, planned_batches = epoch, sampler.batches(epoch)
            return planned_batches[update]
        warmup_draw_counts=np.zeros(len(train),dtype=np.int64)
        def step(epoch, update, lam):
            choice_array=batch_at(epoch, update); np.add.at(warmup_draw_counts,choice_array,1); choice=torch.as_tensor(choice_array,device=device)
            bundle.train(); optimizer.zero_grad(); loss=torch.nn.functional.mse_loss(head(shared(train_cls[choice])), train_y[choice]); loss.backward(); optimizer.step(); return {"mse":float(loss.detach()),"actual_batch":int(choice.numel())}
        def validate(epoch):
            bundle.eval()
            with torch.no_grad(): pred=head(shared(val_cls)).cpu().numpy()
            return self._regression_report(context,pred,val_y.cpu().numpy(),[record.patient for record in val])
        def compact(state):
            return {
                "shared": {name.removeprefix("shared."): value.detach().cpu() for name, value in state.items() if name.startswith("shared.")},
                "head": {name.removeprefix("head."): value.detach().cpu() for name, value in state.items() if name.startswith("head.")},
            }
        def save(kind, epoch, state, row): torch.save(compact(state), checkpoint_dir / f"{kind}.pt")
        result = training.run_common_warmup(step, validate, model=bundle, updates_per_epoch=updates_per_epoch, checkpoint_callback=save)
        compact_state = compact(result["formal_state"])
        # warmup.pt is the selected common start copied by every formal cell;
        # formal.pt is retained as its explicit model-selection checkpoint.
        torch.save(compact_state, checkpoint_dir / "warmup.pt")
        np.savez(checkpoint_dir/"draw_counts.npz",mpp_id=np.asarray([record.mpp_id for record in train]),patient=np.asarray([record.patient for record in train],dtype=str),patch_stem=np.asarray([record.patch_stem for record in train],dtype=str),count=warmup_draw_counts)
        self._warmup_states[(id(context["config"]), int(seed))] = compact_state
        return {"history": result["history"], "checkpoint": str((checkpoint_dir / "formal.pt").resolve()), "warmup_checkpoint":str((checkpoint_dir/"warmup.pt").resolve()),"formal_checkpoint":str((checkpoint_dir/"formal.pt").resolve()),"last_checkpoint":str((checkpoint_dir/"last.pt").resolve()),"draw_counts":str((checkpoint_dir/"draw_counts.npz").resolve()),"frozen_cls_cache":{"train":str((checkpoint_dir/"frozen_cls_train.pt").resolve()),"internal_val":str((checkpoint_dir/"frozen_cls_internal_val.pt").resolve())},"checkpoint_directory":str(checkpoint_dir.resolve())}

    def _stage_model(self, context, seed, cell, warmup):
        # Every same-rank cell receives the same LoRA/T initialization for the
        # protocol/fold/seed; H/C are then replaced by the selected warmup.
        torch.manual_seed(104729 + int(seed))
        device, backbone = self._device(context), self._fresh_backbone(context)
        if cell.startswith("r8_"): models.inject_independent_qv_lora(backbone, rank=8, alpha=16)
        if cell.startswith("r2_"): models.inject_independent_qv_lora(backbone, rank=2, alpha=4)
        if cell.startswith(("r8_","r2_")): models.assert_lora_contract(backbone,(20,21,22,23))
        self._configure_gradient_checkpointing(context, backbone)
        model = models.Stage1Model(backbone).to(device)
        state = self._warmup_states.get((id(context["config"]), int(seed)))
        if state is None:
            state = torch.load(warmup["checkpoint"], map_location=device, weights_only=True)
        model.shared.load_state_dict(state["shared"]); model.regression.load_state_dict(state["head"])
        return model

    def _online_model(self, context, seed, cell, stage1_result):
        # Reconstruct on demand from the CPU master/checkpoint.  Returning a
        # disposable model avoids one full UNI2-h GPU allocation per grid cell.
        warmup = {"checkpoint": stage1_result["warmup_checkpoint"]}
        model = self._stage_model(context, seed, cell, warmup)
        model.load_state_dict(torch.load(stage1_result["checkpoint"], map_location=self._device(context), weights_only=True), strict=False)
        return model

    def pilot(self, context, seed, lr, warmup, checkpoint_dir):
        result = self.stage1(context, seed, "r8_regression", lr, warmup, checkpoint_dir)
        epoch5 = next((row for row in result["history"] if int(row["epoch"]) == 5), None)
        if epoch5 is None:
            raise RuntimeError("LR pilot did not complete its required fifth epoch")
        return {**result, "epoch": 5, "lr": float(lr), "patient_macro_pathway_pcc": epoch5["patient_macro_pathway_pcc"], "zMSE": epoch5["zMSE"]}

    def stage1(self, context, seed, cell, lr, warmup, checkpoint_dir, reused_pilot=None):
        if reused_pilot is not None and "_runtime_key" in reused_pilot:
            return dict(reused_pilot)
        model, train, val = self._stage_model(context, seed, cell, warmup), context["partitions"]["train"], context["partitions"]["val"]
        sampler = training.PatientBalancedBatchSampler([record.patient for record in train], int(context["resource_short_test"]["batch_size"]), protocol=context["split"].protocol, fold=context["split"].fold, seed=seed)
        contrast_mode = "patient_centered" if "centered_contrastive" in cell else ("global" if "global_contrastive" in cell else None)
        if contrast_mode is None:
            for parameter in model.teacher.parameters():
                parameter.requires_grad_(False)
        lora_parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and ".backbone." in f".{name}" and name.split(".")[-1] in {"q_A", "q_B", "v_A", "v_B"}]
        h_c_parameters = list(model.shared.parameters()) + list(model.regression.parameters())
        teacher_parameters = list(model.teacher.parameters()) if contrast_mode else []
        optimizer = torch.optim.AdamW(
            [{"params": lora_parameters, "lr": float(lr)}, {"params": h_c_parameters, "lr": 3e-5}, {"params": teacher_parameters, "lr": 3e-4}],
            weight_decay=1e-4,
        )
        batches = sampler.batches(1)
        planned_epoch, planned_batches = None, None
        def batch_at(epoch, update):
            nonlocal planned_epoch, planned_batches
            if planned_epoch != epoch:
                planned_epoch, planned_batches = epoch, sampler.batches(epoch)
            return planned_batches[update]
        draw_counts=np.zeros(len(train),dtype=np.int64)
        tau_y = None
        tau_details = None
        if contrast_mode:
            teacher_batches = []
            temperature_sampler=training.PatientBalancedBatchSampler([record.patient for record in train], int(context["resource_short_test"]["batch_size"]), protocol=context["split"].protocol, fold=context["split"].fold, seed=20260908)
            fixed_batches=temperature_sampler.batches(1)
            for index in range(20):
                choice = fixed_batches[index % len(fixed_batches)]
                teacher_target = self._batch(context, train, choice, seed=20260908, epoch=1, update=index, augment=False)[1]
                teacher_batches.append((teacher_target, [train[int(i)].patient for i in choice]))
            tau_details = losses.select_teacher_temperature(
                teacher_batches, mode=contrast_mode,
                patient_means=context["artifacts"]["patient_means"], return_details=True,
            )
            tau_y = float(tau_details["tau_y"])

        # Fixed, training-only diagnostic points: first 32 stable manifest
        # identities per training patient, reused at initialization and epochs.
        diagnostic_records = []
        for patient in sorted({record.patient for record in train}):
            patient_records = sorted(
                (record for record in train if record.patient == patient),
                key=lambda record: record.identity,
            )
            diagnostic_records.extend(patient_records[:32])
        warmup_shared=deepcopy(model.shared).eval()
        for parameter in warmup_shared.parameters(): parameter.requires_grad_(False)

        def diagnostic_snapshot():
            model.eval(); cls_rows=[]; frozen_rows=[]; h_rows=[]; v_rows=[]
            batch_size=int(context["resource_short_test"]["batch_size"])
            for start in range(0,len(diagnostic_records),batch_size):
                stop=min(start+batch_size,len(diagnostic_records))
                images,targets,ids=self._batch(context,diagnostic_records,range(start,stop),seed=seed,augment=False)
                teacher_input=targets
                if contrast_mode=="patient_centered":
                    means=torch.stack([torch.as_tensor(context["artifacts"]["patient_means"][patient],device=targets.device,dtype=targets.dtype) for patient in ids])
                    teacher_input=targets-means
                with torch.no_grad():
                    with self._autocast(context, self._device(context)):
                        output=model(images,teacher_inputs=teacher_input if contrast_mode else None)
                        with models.lora_disabled(model.backbone):
                            frozen=backbone_api.forward_cls(model.backbone,images)
                cls_rows.append(output["cls"].float()); frozen_rows.append(frozen.float()); h_rows.append(output["h"].float())
                if contrast_mode: v_rows.append(output["teacher_embedding"].float())
            cls=torch.cat(cls_rows); frozen=torch.cat(frozen_rows); h=torch.cat(h_rows)
            cosine=torch.nn.functional.cosine_similarity(cls,frozen,dim=1)
            cls_relative=(torch.linalg.vector_norm(cls-frozen,dim=1)/torch.linalg.vector_norm(frozen,dim=1).clamp_min(1e-12))
            warm_h=warmup_shared(frozen)
            h_relative=(torch.linalg.vector_norm(h-warm_h,dim=1)/torch.linalg.vector_norm(warm_h,dim=1).clamp_min(1e-12))
            result={
                "points":len(diagnostic_records), "cls_cosine_to_frozen_mean":float(cosine.mean()),
                "cls_max_abs_to_frozen":float(torch.max(torch.abs(cls-frozen))),
                "cls_matches_frozen_rtol1e5_atol1e6":bool(torch.allclose(cls,frozen,rtol=1e-5,atol=1e-6)),
                "cls_relative_l2_mean":float(cls_relative.mean()), "h_relative_to_common_warmup_mean":float(h_relative.mean()),
                "image_embedding_distribution":diagnostics.representation_statistics(h),
            }
            if v_rows: result["teacher_embedding_distribution"]=diagnostics.representation_statistics(torch.cat(v_rows))
            if cell.startswith(("r8_","r2_")): result["delta_w_ratio"]=models.lora_delta_report(model.backbone)
            return result

        initial_diagnostics=diagnostic_snapshot()
        if cell.startswith(("r8_","r2_")) and not initial_diagnostics["cls_matches_frozen_rtol1e5_atol1e6"]:
            raise AssertionError(f"LoRA zero-increment forward mismatch: {initial_diagnostics['cls_max_abs_to_frozen']}")

        def gradient_summary(gradients):
            present=[gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
            if not present: return {"norm":0.0,"vector":None}
            vector=torch.cat(present); return {"norm":float(torch.linalg.vector_norm(vector)),"vector":vector}
        def step(epoch, update, lam):
            model.train()
            choice = batch_at(epoch, update); images, targets, ids = self._batch(context, train, choice, seed=seed, epoch=epoch, update=update, augment=True)
            np.add.at(draw_counts,choice,1)
            teacher_input = targets
            if contrast_mode == "patient_centered":
                means=torch.stack([torch.as_tensor(context["artifacts"]["patient_means"][patient],device=targets.device,dtype=targets.dtype) for patient in ids])
                teacher_input=targets-means
            optimizer.zero_grad()
            with self._autocast(context, self._device(context)):
                output=model(images, teacher_inputs=teacher_input if contrast_mode else None)
                mse_value=torch.nn.functional.mse_loss(output["pred"].float(),targets.float())
            value=mse_value
            contrast_value=None; teacher_stats=None
            if contrast_mode:
                qi, qy=losses.build_teacher_targets(targets, ids, mode=contrast_mode, tau_y=float(tau_y), patient_means=context["artifacts"]["patient_means"])
                contrast_value=losses.bidirectional_soft_contrastive_loss(output["h"],output["teacher_embedding"],qi,qy)
                value=value+float(lam)*contrast_value
                ids_array=np.asarray(ids,dtype=object); cross=torch.as_tensor(ids_array[:,None]!=ids_array[None,:],device=qi.device)
                entropy=-(qi*qi.clamp_min(torch.finfo(qi.dtype).tiny).log()).sum(dim=1)
                teacher_stats={"entropy":float(entropy.mean()),"effective_candidates":float(torch.exp(entropy).mean()),"diagonal_mass":float(torch.diagonal(qi).mean()),"cross_patient_mass":float((qi*cross).sum(dim=1).mean())}
            gradient_diagnostic=None
            if update < 2 and epoch == 1:
                named_h=list(model.shared.parameters()); named_lora=lora_parameters
                mse_h=gradient_summary(torch.autograd.grad(mse_value,named_h,retain_graph=True,allow_unused=True))
                mse_lora=gradient_summary(torch.autograd.grad(mse_value,named_lora,retain_graph=True,allow_unused=True)) if named_lora else {"norm":0.0,"vector":None}
                gradient_diagnostic={"after_update_index":update,"mse_h_norm":mse_h["norm"],"mse_lora_norm":mse_lora["norm"]}
                if contrast_value is not None:
                    con_h=gradient_summary(torch.autograd.grad(float(lam)*contrast_value,named_h,retain_graph=True,allow_unused=True))
                    con_lora=gradient_summary(torch.autograd.grad(float(lam)*contrast_value,named_lora,retain_graph=True,allow_unused=True)) if named_lora else {"norm":0.0,"vector":None}
                    gradient_diagnostic.update({"contrast_h_norm":con_h["norm"],"contrast_lora_norm":con_lora["norm"],"mse_contrast_h_cosine":diagnostics.gradient_cosine(mse_h["vector"],con_h["vector"]) if mse_h["vector"] is not None and con_h["vector"] is not None else None,"mse_contrast_lora_cosine":diagnostics.gradient_cosine(mse_lora["vector"],con_lora["vector"]) if mse_lora["vector"] is not None and con_lora["vector"] is not None else None})
            value.backward(); optimizer.step(); return {"loss":float(value.detach()),"mse":float(mse_value.detach()),"contrastive":float(contrast_value.detach()) if contrast_value is not None else None,"actual_batch":len(choice),"lambda":float(lam),"teacher":teacher_stats,"gradient":gradient_diagnostic}
        def trainable_state():
            return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
        def validate(epoch):
            pred, target, patients=self._predict(context,model,val,seed); report=self._regression_report(context,pred,target,patients); report["diagnostics"]=diagnostic_snapshot(); return report
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        def save(kind, epoch, state, row): torch.save(state, checkpoint_dir / f"{kind}.pt")
        original_validate=validate
        def validate(epoch):
            row=original_validate(epoch); torch.save(trainable_state(),checkpoint_dir/f"epoch_{epoch}.pt"); return row
        result=training.run_stage1(
            step, validate, model=model, state_getter=trainable_state,
            updates_per_epoch=len(batches), checkpoint_callback=save,
        )
        np.savez(checkpoint_dir/"draw_counts.npz",mpp_id=np.asarray([record.mpp_id for record in train]),patient=np.asarray([record.patient for record in train],dtype=str),patch_stem=np.asarray([record.patch_stem for record in train],dtype=str),count=draw_counts)
        diagnostic = {"trainable_parameters": models.count_trainable_parameters(model)}
        if cell.startswith(("r8_", "r2_")):
            diagnostic["delta_w_ratio"] = models.lora_delta_report(model.backbone)
        return {"history":result["history"],"checkpoint":str((checkpoint_dir/"formal.pt").resolve()),"formal_checkpoint":str((checkpoint_dir/"formal.pt").resolve()),"last_checkpoint":str((checkpoint_dir/"last.pt").resolve()),"checkpoint_directory":str(checkpoint_dir.resolve()),"warmup_checkpoint":str(warmup["checkpoint"]),"draw_counts":str((checkpoint_dir/"draw_counts.npz").resolve()),"_runtime_key":f"{seed}/{cell}","tau_y":tau_y,"tau_selection":tau_details,"diagnostics":{"initial":initial_diagnostics,"final":diagnostic}}

    def cache_cls(self, context, seed, cell, endpoint, stage1_result, cache_dir):
        model=self._online_model(context, seed, cell, stage1_result)
        endpoint_path=Path(stage1_result["checkpoint_directory"])/f"epoch_{int(endpoint)}.pt"
        if not endpoint_path.is_file(): raise FileNotFoundError(f"missing requested stage1 endpoint checkpoint: {endpoint_path}")
        model.load_state_dict(torch.load(endpoint_path,map_location=self._device(context),weights_only=True),strict=False)
        model.eval()
        cache_dir.mkdir(parents=True,exist_ok=True); payload={}; graph_payload={}
        graph_path=cache_dir.parent.parent/"frozen_uni_graph.npz"
        existing_graph=np.load(graph_path,allow_pickle=False) if graph_path.is_file() else None
        for name in ("train","val","held"):
            records=context["partitions"][name]
            if records:
                cls_values, frozen_cls_values, hidden_values, pred_values, target_values, patients = [], [], [], [], [], []
                for start in range(0, len(records), int(context["resource_short_test"]["batch_size"])):
                    images, target, ids = self._batch(context, records, range(start, min(start + int(context["resource_short_test"]["batch_size"]), len(records))), seed=seed)
                    with torch.no_grad():
                        with self._autocast(context, self._device(context)):
                            output=model(images)
                            if existing_graph is None:
                                with models.lora_disabled(model.backbone):
                                    frozen_cls=backbone_api.forward_cls(model.backbone,images)
                    cls_values.append(output["cls"].float().cpu().numpy())
                    if existing_graph is None: frozen_cls_values.append(frozen_cls.float().cpu().numpy())
                    hidden_values.append(output["h"].float().cpu().numpy()); pred_values.append(output["pred"].float().cpu().numpy()); target_values.append(target.float().cpu().numpy()); patients.extend(ids)
                patient_array=np.asarray(patients,dtype=str); stem_array=np.asarray([record.patch_stem for record in records],dtype=str)
                payload[f"cls_{name}"]=np.concatenate(cls_values); payload[f"h_{name}"]=np.concatenate(hidden_values); payload[f"pred_{name}"]=np.concatenate(pred_values); payload[f"target_{name}"]=np.concatenate(target_values); payload[f"patient_{name}"]=patient_array
                if existing_graph is None:
                    graph_payload[f"cls_{name}"]=np.concatenate(frozen_cls_values); graph_payload[f"patient_{name}"]=patient_array; graph_payload[f"patch_stem_{name}"]=stem_array
                elif not np.array_equal(existing_graph[f"patient_{name}"],patient_array) or not np.array_equal(existing_graph[f"patch_stem_{name}"],stem_array):
                    raise AssertionError(f"frozen graph cache identity mismatch for {name}")
        if existing_graph is None: np.savez(graph_path,**graph_payload)
        graph_arrays=np.load(graph_path,allow_pickle=False)
        if cell.startswith("frozen_"):
            maximum=max(float(np.max(np.abs(payload[f"cls_{name}"]-graph_arrays[f"cls_{name}"]))) for name in ("train","val","held") if f"cls_{name}" in payload)
            if not all(np.allclose(payload[f"cls_{name}"],graph_arrays[f"cls_{name}"],rtol=1e-5,atol=1e-6) for name in ("train","val","held") if f"cls_{name}" in payload):
                raise AssertionError(f"frozen online/cache CLS mismatch: max_abs={maximum}")
        else: maximum=None
        path=cache_dir/"stage1_predictions.npz"; np.savez(path,**payload)
        return {"cache":str(path.resolve()),"graph_cache":str(graph_path.resolve()),"endpoint":int(endpoint),"dtype":"float32","augmentation":False,"frozen_cls_cache_contract":"shared graph morphology only","frozen_online_cache_max_abs":maximum}

    def stage2(self, context, seed, task, stage1_result, cache, checkpoint_dir):
        arrays=np.load(cache["cache"],allow_pickle=False); graph_arrays=np.load(cache["graph_cache"],allow_pickle=False); train_y=torch.as_tensor(arrays["target_train"],device=self._device(context)); val_y=torch.as_tensor(arrays["target_val"],device=self._device(context))
        torch.manual_seed(int(seed) * 1000 + int(task.endpoint))
        endpoint_path=Path(stage1_result["checkpoint_directory"])/f"epoch_{int(task.endpoint)}.pt"
        if not endpoint_path.is_file(): raise FileNotFoundError(f"missing requested stage1 endpoint checkpoint: {endpoint_path}")
        endpoint_state=torch.load(endpoint_path,map_location="cpu",weights_only=True)
        shared_state={name.removeprefix("shared."):value for name,value in endpoint_state.items() if name.startswith("shared.")}
        if not shared_state: raise RuntimeError(f"stage1 endpoint checkpoint lacks shared H state: {endpoint_path}")
        shared=models.SharedProjection(); shared.load_state_dict(shared_state)
        mode="spatial" if task.head=="spatial" else "point"; head=models.Stage2Head(shared,mode=mode)
        head.set_shared_mode({"inherit":"freeze","reset":"reset","continue":"continue"}[task.h_mode]); head.to(self._device(context))
        c_key=(id(context["config"]),int(seed),task.cell,int(task.endpoint))
        if c_key not in self._stage2_c_initial:
            self._stage2_c_initial[c_key]=deepcopy(head.regression.state_dict())
        else:
            head.regression.load_state_dict(self._stage2_c_initial[c_key])
        features=torch.as_tensor(arrays["cls_train"],device=train_y.device,dtype=torch.float32); val_features=torch.as_tensor(arrays["cls_val"],device=train_y.device,dtype=torch.float32)
        native_steps=self._native_steps(context) if mode=="spatial" else None
        train_graph=self._spatial_graph(context,"train",graph_arrays["cls_train"],native_steps) if mode=="spatial" else None
        val_graph=self._spatial_graph(context,"val",graph_arrays["cls_val"],native_steps) if mode=="spatial" else None
        shared_parameters=[p for p in head.shared.parameters() if p.requires_grad]
        cb_parameters=list(head.regression.parameters()) + ([head.B] if head.B is not None else [])
        groups=[{"params":cb_parameters,"lr":3e-4}]
        if task.h_mode=="continue": groups.insert(0,{"params":shared_parameters,"lr":3e-5})
        elif task.h_mode=="reset": groups.insert(0,{"params":shared_parameters,"lr":3e-4})
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        def step(epoch,update,lam):
            head.train()
            optimizer.zero_grad()
            delta=train_graph.spatial_delta(head.shared(features)) if train_graph is not None else None
            pred=head(features,delta) if mode=="spatial" else head(features)
            loss=torch.nn.functional.mse_loss(pred,train_y); loss.backward(); optimizer.step(); return {"loss":float(loss.detach())}
        def validate(epoch):
            head.eval()
            with torch.no_grad():
                delta=val_graph.spatial_delta(head.shared(val_features)) if val_graph is not None else None
                pred=(head(val_features,delta) if mode=="spatial" else head(val_features)).cpu().numpy()
            report=self._regression_report(context,pred,val_y.cpu().numpy(),arrays["patient_val"].tolist())
            if val_graph is not None: report["spatial_residual_metrics"]=metrics.spatial_residual_metrics(pred,val_y.cpu().numpy(),val_graph)
            return report
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        def save(kind,epoch,state,row): torch.save(state,checkpoint_dir/f"stage2_{task.head}_{task.h_mode}_{kind}.pt")
        result=training.run_stage2(step,validate,model=head,checkpoint_callback=save)
        if result["formal_state"] is None:
            raise RuntimeError("stage2 produced no internally selected checkpoint")
        head.load_state_dict(result["formal_state"])
        head.eval()
        held_report=None
        if "cls_held" in arrays:
            held_features=torch.as_tensor(arrays["cls_held"],device=train_y.device,dtype=torch.float32)
            held_y=torch.as_tensor(arrays["target_held"],device=train_y.device,dtype=torch.float32)
            with torch.no_grad():
                held_graph=self._spatial_graph(context,"held",graph_arrays["cls_held"],native_steps) if mode=="spatial" else None
                held_delta=held_graph.spatial_delta(head.shared(held_features)) if held_graph is not None else None
                held_prediction=(head(held_features,held_delta) if mode=="spatial" else head(held_features)).cpu().numpy()
            held_report=self._regression_report(context,held_prediction,held_y.cpu().numpy(),arrays["patient_held"].tolist())
            if held_graph is not None: held_report["spatial_residual_metrics"]=metrics.spatial_residual_metrics(held_prediction,held_y.cpu().numpy(),held_graph)
            np.savez(Path(cache["cache"]).parent/f"stage2_{task.head}_{task.h_mode}_held.npz",prediction=held_prediction,target=held_y.cpu().numpy(),patient=arrays["patient_held"])
        stem=f"stage2_{task.head}_{task.h_mode}"
        return {"history":result["history"],"h_mode":task.h_mode,"head":task.head,"checkpoint":str((checkpoint_dir/f"{stem}_formal.pt").resolve()),"formal_checkpoint":str((checkpoint_dir/f"{stem}_formal.pt").resolve()),"last_checkpoint":str((checkpoint_dir/f"{stem}_last.pt").resolve()),"warmup_checkpoint":str((Path(stage1_result["checkpoint_directory"])/f"epoch_{task.endpoint}.pt").resolve()),"checkpoint_directory":str(checkpoint_dir.resolve()),"held_evaluation":{"selection_used":False,"metrics":held_report} if held_report is not None else None}

    def evaluate(self, context, seed, cell, stage1_result, destination):
        records=context["partitions"]["held"]
        destination.mkdir(parents=True,exist_ok=True)
        if not records: return {"selection_used":False,"status":"no_held_out_partition"}
        model=self._online_model(context, seed, cell, stage1_result)
        endpoint=Path(stage1_result["checkpoint_directory"])/"epoch_5.pt"
        model.load_state_dict(torch.load(endpoint,map_location=self._device(context),weights_only=True),strict=False)
        pred,target,patients=self._predict(context,model,records,seed)
        report=self._regression_report(context,pred,target,patients); np.savez(destination/"held_predictions.npz",prediction=pred,target=target,patient=np.asarray(patients,dtype=str))
        return {"selection_used":False,"metrics":report}


def execute_experiment(
    config: str | Path | Mapping[str, Any] | None = None,
    run_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
    protocol: str | None = None,
    seeds: Sequence[int] | None = None,
    folds: Sequence[str] | None = None,
    resume: bool = False,
    plan_only: bool = False,
    *,
    config_path: str | Path | None = None,
    dependencies: TrainingRuntime | None = None,
) -> dict[str, Any]:
    """Execute the v4 schedule; return a non-zero result on any failed stage.

    Held-out and XZY partitions are handed solely to ``evaluate`` after all
    train/internal selection steps.  Fixed-e5 and best-epoch sensitivity
    outputs are persisted under separate raw directories.
    """
    try:
        if plan_only:
            raise ValueError("execute_experiment is the non-plan-only backend")
        if config is not None and config_path is not None:
            raise ValueError("supply either config or config_path, not both")
        if run_dir is None or weights_dir is None or protocol is None or seeds is None:
            raise ValueError("run_dir, weights_dir, protocol, and seeds are required")
        run_root, weights_root = Path(run_dir), Path(weights_dir)
        if run_root.resolve() == weights_root.resolve():
            raise ValueError("run_dir and weights_dir must be distinct")
        if not seeds:
            raise ValueError("seeds must not be empty")
        if config is None and config_path is None:
            raise ValueError("config or config_path is required")
        cfg, runtime = _config(config if config is not None else config_path), dependencies or TorchRuntime()
        manifest = _load_manifest(cfg)
        selected_splits = _splits(manifest, str(protocol), folds)
        raw = run_root / "raw"; raw.mkdir(parents=True, exist_ok=True)
        state_path, state = _state_path(run_root), _state_load(_state_path(run_root), bool(resume))
        weights_entries: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"protocol": protocol, "folds": [], "status": "executed"}

        for split in selected_splits:
            # Records contain tensors/paths and are deliberately never restored
            # from JSON.  Rebuild them from the raw templates on every process.
            partitions = runtime.materialize(split, cfg)
            _write_json(raw / split.protocol / split.fold / "materialized.json", _partition_summary(partitions))
            artifacts, _ = _task(
                state, state_path, _task_key(split, 0, "fit_training_only"), bool(resume),
                lambda: runtime.fit_training_artifacts(partitions, split), raw / split.protocol / split.fold / "training_artifacts.json",
            )
            if int(artifacts.get("z_ddof", 1)) != 1:
                raise AssertionError("Phase2 v4 requires train-only ddof=1 z-score fitting")
            context = {"config": cfg, "split": split, "partitions": partitions, "artifacts": artifacts}
            short, _ = _task(
                state, state_path, _task_key(split, 0, "resource_short_test"), bool(resume),
                lambda: runtime.resource_short_test(context, cfg.get("training", {}).get("batch_size_candidates", (64, 128))),
                raw / split.protocol / split.fold / "resource_short_test.json",
            )
            context["resource_short_test"] = short
            fold_summary = {"fold": split.fold, "seeds": []}

            for seed in (int(value) for value in seeds):
                seed_base = raw / split.protocol / split.fold / f"seed_{seed}"
                warmup_dir = task_directory(weights_root, protocol=split.protocol, fold=split.fold, seed=seed, cell="common_warmup", endpoint=0)
                warmup, _ = _task(
                    state, state_path, _task_key(split, seed, "warmup"), bool(resume),
                    lambda: runtime.warmup(context, seed, warmup_dir), seed_base / "warmup.json",
                )
                weights_entries.append(_registered_weight_entry(
                    protocol=split.protocol,fold=split.fold,seed=seed,cell="common_warmup",endpoint=0,
                    directory=warmup_dir,warmup=warmup.get("warmup_checkpoint",warmup["checkpoint"]),formal=warmup.get("formal_checkpoint",warmup["checkpoint"]),last=warmup.get("last_checkpoint",warmup["checkpoint"]),
                    base_checkpoint=cfg.get("data",{}).get("uni2_h_weights"),
                ))
                _write_json(run_root/"model_weights.json",{"schema_version":"v4","weights_separate_from_runs":True,"entries":weights_entries})
                pilots: dict[float, Mapping[str, Any]] = {}
                selected_lr: float | None = None
                if seed == 42:
                    for lr in cfg.get("training", {}).get("learning_rates", {}).get("lora_candidates", (1e-5, 3e-5, 1e-4)):
                        value = float(lr)
                        result, _ = _task(
                            state, state_path, _task_key(split, seed, "lr_pilot", f"lr_{value:g}"), bool(resume),
                            lambda value=value: runtime.pilot(context, seed, value, warmup, task_directory(weights_root, protocol=split.protocol, fold=split.fold, seed=seed, cell=f"r8_regression_lr_{value:g}", endpoint=5)),
                            seed_base / "pilots" / f"lr_{value:g}.json",
                        )
                        pilots[value] = result
                    chosen = training.select_learning_rate(list(pilots.values()))
                    selected_lr = float(chosen["lr"])
                    _write_json(seed_base / "lr_selection.json", {"selected": chosen, "scope": "internal_validation_only"})
                else:
                    selection_file = raw / split.protocol / split.fold / "seed_42" / "lr_selection.json"
                    if not selection_file.is_file():
                        raise RuntimeError("seed 43/44 requires completed seed-42 LR selection")
                    selected_lr = float(json.loads(selection_file.read_text(encoding="utf-8"))["selected"]["lr"])

                stage1_results: dict[str, Mapping[str, Any]] = {}
                for cell in _cell_names(cfg):
                    reused = pilots.get(selected_lr) if seed == 42 and cell == "r8_regression" else None
                    result, _ = _task(
                        state, state_path, _task_key(split, seed, "stage1", cell), bool(resume),
                        lambda cell=cell, reused=reused: runtime.stage1(context, seed, cell, float(selected_lr), warmup, task_directory(weights_root, protocol=split.protocol, fold=split.fold, seed=seed, cell=cell, endpoint=5), reused),
                        seed_base / "stage1" / f"{cell}.json",
                    )
                    stage1_results[cell] = result
                    weights_entries.append(_registered_weight_entry(
                        protocol=split.protocol,fold=split.fold,seed=seed,cell=cell,endpoint=5,
                        directory=Path(result.get("checkpoint_directory",Path(result["checkpoint"]).parent)),warmup=result.get("warmup_checkpoint",warmup.get("checkpoint")),formal=result.get("formal_checkpoint",result["checkpoint"]),last=result.get("last_checkpoint",result["checkpoint"]),
                        base_checkpoint=cfg.get("data",{}).get("uni2_h_weights"),
                    ))
                    _write_json(run_root/"model_weights.json",{"schema_version":"v4","weights_separate_from_runs":True,"entries":weights_entries})

                fixed_results, best_results = [], []
                for cell, stage1_result in stage1_results.items():
                    history = stage1_result.get("history", [])
                    best_epoch = int(training.select_best_epoch(history)["epoch"])
                    cache5, _ = _task(
                        state, state_path, _task_key(split, seed, "cache", f"{cell}/e5"), bool(resume),
                        lambda cell=cell, stage1_result=stage1_result: runtime.cache_cls(context, seed, cell, 5, stage1_result, seed_base / "cache" / cell / "e5"),
                        seed_base / "cache" / cell / "e5.json",
                    )
                    for task in [task for task in training.stage2_task_matrix() if task.cell == cell]:
                        stage2_dir=task_directory(weights_root, protocol=split.protocol, fold=split.fold, seed=seed, cell=task.cell, endpoint=5)
                        result, _ = _task(
                            state, state_path, _task_key(split, seed, "stage2_fixed", f"{task.cell}/{task.h_mode}/{task.head}"), bool(resume),
                            lambda task=task,stage2_dir=stage2_dir: runtime.stage2(context, seed, task, stage1_result, cache5, stage2_dir),
                            seed_base / "stage2" / "fixed_e5" / f"{task.cell}_{task.h_mode}_{task.head}.json",
                        )
                        fixed_results.append({"task": asdict(task), "result": result})
                        weights_entries.append(_registered_weight_entry(
                            protocol=split.protocol,fold=split.fold,seed=seed,cell=f"{task.cell}/stage2_{task.head}_{task.h_mode}",endpoint=task.endpoint,
                            directory=Path(result.get("checkpoint_directory",stage2_dir)),warmup=result.get("warmup_checkpoint",stage1_result.get("checkpoint")),formal=result.get("formal_checkpoint",stage2_dir/f"stage2_{task.head}_{task.h_mode}_formal.pt"),last=result.get("last_checkpoint",stage2_dir/f"stage2_{task.head}_{task.h_mode}_last.pt"),
                            base_checkpoint=cfg.get("data",{}).get("uni2_h_weights"),
                        ))
                        _write_json(run_root/"model_weights.json",{"schema_version":"v4","weights_separate_from_runs":True,"entries":weights_entries})
                    if best_epoch != 5:
                        cache_best, _ = _task(
                            state, state_path, _task_key(split, seed, "cache", f"{cell}/e{best_epoch}"), bool(resume),
                            lambda cell=cell, stage1_result=stage1_result, best_epoch=best_epoch: runtime.cache_cls(context, seed, cell, best_epoch, stage1_result, seed_base / "cache" / cell / f"e{best_epoch}"),
                            seed_base / "cache" / cell / f"e{best_epoch}.json",
                        )
                        for task in [item for item in training.stage2_task_matrix({cell: best_epoch}) if item.cell == cell and item.endpoint == best_epoch]:
                            stage2_dir=task_directory(weights_root, protocol=split.protocol, fold=split.fold, seed=seed, cell=task.cell, endpoint=best_epoch)
                            result, _ = _task(
                                state, state_path, _task_key(split, seed, "stage2_best", f"{task.cell}/{task.h_mode}/{task.head}/e{best_epoch}"), bool(resume),
                                lambda task=task,stage2_dir=stage2_dir: runtime.stage2(context, seed, task, stage1_result, cache_best, stage2_dir),
                                seed_base / "stage2" / "best_epoch_sensitivity" / f"{task.cell}_{task.h_mode}_{task.head}_e{best_epoch}.json",
                            )
                            best_results.append({"task": asdict(task), "result": result})
                            weights_entries.append(_registered_weight_entry(
                                protocol=split.protocol,fold=split.fold,seed=seed,cell=f"{task.cell}/stage2_{task.head}_{task.h_mode}_sensitivity",endpoint=task.endpoint,
                                directory=Path(result.get("checkpoint_directory",stage2_dir)),warmup=result.get("warmup_checkpoint",stage1_result.get("checkpoint")),formal=result.get("formal_checkpoint",stage2_dir/f"stage2_{task.head}_{task.h_mode}_formal.pt"),last=result.get("last_checkpoint",stage2_dir/f"stage2_{task.head}_{task.h_mode}_last.pt"),
                                base_checkpoint=cfg.get("data",{}).get("uni2_h_weights"),
                            ))
                            _write_json(run_root/"model_weights.json",{"schema_version":"v4","weights_separate_from_runs":True,"entries":weights_entries})
                    # Evaluation follows all selection and does not feed state/metrics choices.
                    _task(
                        state, state_path, _task_key(split, seed, "evaluate", cell), bool(resume),
                        lambda cell=cell, stage1_result=stage1_result: runtime.evaluate(context, seed, cell, stage1_result, seed_base / "evaluation" / cell),
                        seed_base / "evaluation" / f"{cell}.json",
                    )
                    if context["partitions"].get("xzy"):
                        xzy_context = {**context, "partitions": {**context["partitions"], "held": context["partitions"]["xzy"]}}
                        _task(
                            state, state_path, _task_key(split, seed, "evaluate_xzy", cell), bool(resume),
                            lambda cell=cell, stage1_result=stage1_result: runtime.evaluate(xzy_context, seed, cell, stage1_result, seed_base / "evaluation" / "xzy" / cell),
                            seed_base / "evaluation" / f"xzy_{cell}.json",
                        )
                _write_json(seed_base / "fixed_e5" / "stage2_results.json", fixed_results)
                _write_json(seed_base / "best_epoch_sensitivity" / "stage2_results.json", best_results)
                fold_summary["seeds"].append({"seed": seed, "selected_lr": selected_lr, "fixed_stage2_tasks": len(fixed_results), "best_epoch_extra_tasks": len(best_results)})
            summary["folds"].append(fold_summary)

        _write_json(run_root / "model_weights.json", {"schema_version": "v4", "weights_separate_from_runs": True, "entries": weights_entries})
        _write_json(raw / "engine_summary.json", summary)
        return {"exit_code": 0, **summary, "state_path": str(state_path)}
    except Exception as error:
        return {"exit_code": 1, "status": "failed", "error": str(error)}
