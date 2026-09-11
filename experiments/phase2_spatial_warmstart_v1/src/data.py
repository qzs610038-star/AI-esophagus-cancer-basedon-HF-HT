"""Strict readers for the frozen Phase-1 feature caches.

The training package never instantiates UNI2-h.  Rows from the prediction and
graph caches are joined by ``(patient_id, spot_id)`` and then checked against
the packaged split manifest; array length is never treated as identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


PARTITION_KEYS = {"train": "train", "internal_val": "val", "external_test": "held"}
_COORD_RE = re.compile(r"(?:patch_)?x(-?\d+)_y(-?\d+)$")


@dataclass(frozen=True)
class PartitionData:
    name: str
    features: np.ndarray
    graph_features: np.ndarray
    patient_ids: np.ndarray
    spot_ids: np.ndarray
    coordinates: np.ndarray
    native_steps: np.ndarray
    targets: np.ndarray | None = None
    source_predictions: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.features.shape[0])
        fields = {
            "graph_features": self.graph_features,
            "patient_ids": self.patient_ids,
            "spot_ids": self.spot_ids,
            "coordinates": self.coordinates,
            "native_steps": self.native_steps,
        }
        if self.targets is not None:
            fields["targets"] = self.targets
        if self.source_predictions is not None:
            fields["source_predictions"] = self.source_predictions
        for label, value in fields.items():
            if len(value) != n:
                raise ValueError(f"{self.name}: {label} has {len(value)} rows, expected {n}")
        if self.features.ndim != 2 or self.graph_features.ndim != 2:
            raise ValueError(f"{self.name}: feature arrays must be two-dimensional")
        if self.coordinates.shape != (n, 2):
            raise ValueError(f"{self.name}: coordinates must have shape ({n}, 2)")
        if len(set(zip(self.patient_ids.tolist(), self.spot_ids.tolist()))) != n:
            raise ValueError(f"{self.name}: duplicate patient/spot identity")
        for label, value in fields.items():
            if np.issubdtype(np.asarray(value).dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"{self.name}: {label} contains non-finite values")


@dataclass(frozen=True)
class DataBundle:
    partitions: Mapping[str, PartitionData]
    pathway_names: tuple[str, ...]
    normalization: Mapping[str, Any]
    sources: Mapping[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _npz_required(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"cache is missing required array: {key}")
    return np.asarray(arrays[key])


def _decode_strings(values: np.ndarray, label: str) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    decoded = np.asarray([
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in values.tolist()
    ], dtype=str)
    if np.any(decoded == ""):
        raise ValueError(f"{label} contains empty values")
    return decoded


def _coordinates_from_spots(spot_ids: np.ndarray) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for spot_id in spot_ids.tolist():
        match = _COORD_RE.search(str(spot_id))
        if match is None:
            raise ValueError(f"cannot derive coordinates from spot identity: {spot_id}")
        rows.append((float(match.group(1)), float(match.group(2))))
    return np.asarray(rows, dtype=np.float32)


def _manifest_rows(package_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = package_dir / "inputs" / "mpp2" / "split_manifest.csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["patient"], row["patch_stem"])
            if key in rows:
                raise ValueError(f"duplicate identity in split manifest: {key}")
            rows[key] = row
    return rows


def _native_steps(package_dir: Path) -> dict[str, float]:
    payload = _read_json(package_dir / "inputs" / "mpp2" / "split_info.json")
    result: dict[str, float] = {}
    for patient, row in payload["per_patient_main_stride"].items():
        dx, dy = float(row["main_dx"]), float(row["main_dy"])
        if dx <= 0 or dy <= 0 or not np.isclose(dx, dy):
            raise ValueError(f"native step must be a positive square stride for patient={patient}")
        result[str(patient)] = dx
    return result


def _identity_arrays(
    graph_arrays: Mapping[str, np.ndarray], suffix: str, package_dir: Path,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patients = _decode_strings(_npz_required(graph_arrays, f"patient_{suffix}"), f"patient_{suffix}")
    spots = _decode_strings(_npz_required(graph_arrays, f"patch_stem_{suffix}"), f"patch_stem_{suffix}")
    manifest = _manifest_rows(package_dir)
    step_map = _native_steps(package_dir)
    coords: list[tuple[float, float]] = []
    steps: list[float] = []
    for patient, spot in zip(patients.tolist(), spots.tolist()):
        row = manifest.get((patient, spot))
        if row is None:
            if partition != "external_test":
                raise KeyError(f"{partition}: identity absent from packaged manifest: {(patient, spot)}")
            x, y = _coordinates_from_spots(np.asarray([spot]))[0]
        else:
            if row["split"] != partition:
                raise ValueError(
                    f"{partition}: identity {(patient, spot)} belongs to split={row['split']}"
                )
            x, y = float(row["x"]), float(row["y"])
        if patient not in step_map:
            if partition == "external_test":
                # A server-supplied cache may carry explicit native-step arrays.
                explicit = graph_arrays.get(f"native_step_{suffix}")
                if explicit is None:
                    raise KeyError(f"external native step is unavailable for patient={patient}")
                step = float(np.asarray(explicit)[len(steps)])
            else:
                raise KeyError(f"native step is unavailable for patient={patient}")
        else:
            step = step_map[patient]
        coords.append((x, y))
        steps.append(step)
    return patients, spots, np.asarray(coords, dtype=np.float32), np.asarray(steps, dtype=np.float32)


def _partition(
    prediction_arrays: Mapping[str, np.ndarray], graph_arrays: Mapping[str, np.ndarray],
    package_dir: Path, name: str, *, require_targets: bool,
) -> PartitionData:
    suffix = PARTITION_KEYS[name]
    features = _npz_required(prediction_arrays, f"cls_{suffix}").astype(np.float32, copy=False)
    graph_features = _npz_required(graph_arrays, f"cls_{suffix}").astype(np.float32, copy=False)
    pred_patients = _decode_strings(
        _npz_required(prediction_arrays, f"patient_{suffix}"), f"prediction.patient_{suffix}"
    )
    patients, spots, coords, native_steps = _identity_arrays(
        graph_arrays, suffix, package_dir, name
    )
    if not np.array_equal(pred_patients, patients):
        raise ValueError(f"{name}: prediction and graph cache patient order differs")
    targets = None
    target_key = f"target_{suffix}"
    if target_key in prediction_arrays:
        targets = np.asarray(prediction_arrays[target_key], dtype=np.float32)
    elif require_targets:
        raise KeyError(f"{name}: cache is missing required labels {target_key}")
    source_prediction = None
    pred_key = f"pred_{suffix}"
    if pred_key in prediction_arrays:
        source_prediction = np.asarray(prediction_arrays[pred_key], dtype=np.float32)
    return PartitionData(
        name=name, features=features, graph_features=graph_features,
        patient_ids=patients, spot_ids=spots, coordinates=coords,
        native_steps=native_steps, targets=targets, source_predictions=source_prediction,
    )


def load_bundle(
    config: Mapping[str, Any], package_dir: str | Path, *,
    include_external: bool = False, require_external_targets: bool = False,
) -> DataBundle:
    """Load immutable caches and enforce the identity/dimension contract."""
    package_dir = Path(package_dir).resolve()
    inputs = config["inputs"]
    prediction_path = Path(inputs["feature_cache"])
    graph_path = Path(inputs["graph_feature_cache"])
    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing Phase-1 feature cache: {prediction_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"missing frozen graph feature cache: {graph_path}")
    with np.load(prediction_path, allow_pickle=False) as prediction_arrays, np.load(
        graph_path, allow_pickle=False
    ) as graph_arrays:
        partitions: dict[str, PartitionData] = {
            "train": _partition(prediction_arrays, graph_arrays, package_dir, "train", require_targets=True),
            "internal_val": _partition(
                prediction_arrays, graph_arrays, package_dir, "internal_val", require_targets=True
            ),
        }
        if include_external:
            suffix = PARTITION_KEYS["external_test"]
            if f"cls_{suffix}" not in prediction_arrays or f"cls_{suffix}" not in graph_arrays:
                raise KeyError("external cache arrays cls_held are unavailable")
            partitions["external_test"] = _partition(
                prediction_arrays, graph_arrays, package_dir, "external_test",
                require_targets=require_external_targets,
            )
    expected_in = int(config["parameters"]["input_dim"])
    expected_out = int(config["parameters"]["output_dim"])
    for name, part in partitions.items():
        if part.features.shape[1] != expected_in or part.graph_features.shape[1] != expected_in:
            raise ValueError(f"{name}: expected {expected_in}-dimensional frozen features")
        if part.targets is not None and part.targets.shape != (len(part.features), expected_out):
            raise ValueError(f"{name}: labels must have {expected_out} pathways")
        if part.source_predictions is not None and part.source_predictions.shape != (
            len(part.features), expected_out
        ):
            raise ValueError(f"{name}: source predictions must have {expected_out} pathways")
        if not np.allclose(part.features, part.graph_features, rtol=1e-5, atol=1e-6):
            maximum = float(np.max(np.abs(part.features - part.graph_features)))
            raise ValueError(
                f"{name}: frozen prediction and graph CLS order/value mismatch; max_abs={maximum}"
            )
    z_manifest = _read_json(package_dir / "inputs" / "mpp2" / "zscore_manifest.json")
    pathways = tuple(str(item) for item in z_manifest["pathway_names"])
    if len(pathways) != expected_out or len(set(pathways)) != expected_out:
        raise ValueError("packaged pathway order is invalid")
    normalization = {
        "scale": "training_zscore",
        "ddof": int(z_manifest["ddof"]),
        "fit_split": z_manifest["fit_split"],
        "excluded_splits": z_manifest["excluded_splits"],
        "parameters_embedded": False,
        "note": "cache targets are already standardized; raw-label transforms require source statistics",
    }
    return DataBundle(
        partitions=partitions, pathway_names=pathways, normalization=normalization,
        sources={"feature_cache": str(prediction_path), "graph_feature_cache": str(graph_path)},
    )


def preflight(config: Mapping[str, Any], package_dir: str | Path) -> dict[str, Any]:
    """Run source checks without reading external-test labels."""
    package_dir = Path(package_dir).resolve()
    bundle = load_bundle(config, package_dir, include_external=False)
    checkpoint = Path(config["inputs"]["stage1_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Phase-1 endpoint checkpoint: {checkpoint}")
    from .model import SpatialWarmstartModel, load_stage1_hc
    parameters = config["parameters"]
    source_model = SpatialWarmstartModel(
        input_dim=int(parameters["input_dim"]), hidden_dim=int(parameters["hidden_dim"]),
        output_dim=int(parameters["output_dim"]), dropout=float(parameters.get("dropout", 0.3)),
    )
    source_load = load_stage1_hc(source_model, checkpoint)
    import torch
    source_model.eval()
    tolerance = float(parameters.get(
        "source_cache_prediction_max_abs_tolerance",
        parameters.get("initial_prediction_max_abs_tolerance", 1e-6),
    ))
    source_precision = str(config["inputs"].get("source_prediction_precision", "unspecified"))
    consistency: dict[str, dict[str, float | bool]] = {}
    for name, partition in bundle.partitions.items():
        if partition.source_predictions is None:
            raise KeyError(f"{name}: source predictions are required for inheritance preflight")
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(partition.features), 1024):
                features = torch.as_tensor(partition.features[start:start + 1024], dtype=torch.float32)
                rows.append(source_model(features, use_spatial=False).prediction.numpy())
        predicted = np.concatenate(rows, axis=0)
        maximum = float(np.max(np.abs(predicted - partition.source_predictions)))
        consistency[name] = {"max_abs": maximum, "passed": maximum <= tolerance}
        if maximum > tolerance:
            raise AssertionError(f"{name}: source H/C prediction mismatch max_abs={maximum}")
    warnings: list[dict[str, str]] = []
    slide_status_path = package_dir / "inputs" / "slide_mapping_status.json"
    if slide_status_path.is_file():
        slide_status = _read_json(slide_status_path)
        if slide_status.get("status") != "verified":
            warnings.append({
                "code": str(slide_status.get("error_code", "SLIDE_MAPPING_UNVERIFIED")),
                "message": (
                    "slide mapping is not verified; current v1.1 graph contract groups by "
                    "patient and partition, so this is recorded rather than silently treated "
                    "as proof that a patient has one slide"
                ),
            })
    return {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "source_hc_load": asdict(source_load),
        "source_prediction_consistency": {
            "tolerance": tolerance,
            "source_precision": source_precision,
            "partitions": consistency,
        },
        "partitions": {name: len(part.features) for name, part in bundle.partitions.items()},
        "pathways": len(bundle.pathway_names),
        "external_labels_read": False,
        "normalization": dict(bundle.normalization),
        "warnings": warnings,
    }
