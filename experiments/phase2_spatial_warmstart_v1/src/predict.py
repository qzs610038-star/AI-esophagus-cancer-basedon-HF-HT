"""Strict model bundles, label-free inference, and isolated external evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

try:  # Package imports for tests and ``python -m`` execution.
    from .metrics import compute_regression_metrics
    from .model import SpatialWarmstartModel
    from .spatial import SpatialGraph
except ImportError:  # Direct ``python src/main.py`` execution.
    from metrics import compute_regression_metrics
    from model import SpatialWarmstartModel
    from spatial import SpatialGraph


BUNDLE_SCHEMA_VERSION = "phase2_spatial_warmstart_model_bundle_v1"
_ARMS = {
    "point_continue": False,
    "spatial_residual_only": True,
    "spatial_joint": True,
}
_REQUIRED_BUNDLE_KEYS = {
    "schema_version",
    "model_config",
    "model_state_dict",
    "arm",
    "spatial_enabled",
    "graph_parameters",
    "pathway_names",
    "normalization",
    "feature_contract",
    "native_step_contract",
    "coordinate_contract",
}


def _plain_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    return dict(value)


def _validate_metadata(
    *,
    model: SpatialWarmstartModel,
    arm: str,
    spatial_enabled: bool,
    graph_parameters: Mapping[str, Any],
    pathway_names: Sequence[str],
    normalization: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    native_step_contract: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    arm = str(arm)
    if arm not in _ARMS:
        raise ValueError(f"unknown warm-start arm: {arm!r}")
    if bool(spatial_enabled) is not _ARMS[arm]:
        raise ValueError(f"arm={arm!r} is incompatible with spatial_enabled={spatial_enabled!r}")
    pathways = [str(name) for name in pathway_names]
    if len(pathways) != model.output_dim or len(set(pathways)) != len(pathways):
        raise ValueError("pathway_names must be unique and match model.output_dim")
    if any(not name for name in pathways):
        raise ValueError("pathway_names cannot contain empty values")
    feature = _plain_mapping(feature_contract, "feature_contract")
    if int(feature.get("dimension", -1)) != model.input_dim:
        raise ValueError("feature_contract.dimension must match model.input_dim")
    return {
        "arm": arm,
        "spatial_enabled": bool(spatial_enabled),
        "graph_parameters": _plain_mapping(graph_parameters, "graph_parameters"),
        "pathway_names": pathways,
        "normalization": _plain_mapping(normalization, "normalization"),
        "feature_contract": feature,
        "native_step_contract": _plain_mapping(native_step_contract, "native_step_contract"),
        "coordinate_contract": _plain_mapping(coordinate_contract, "coordinate_contract"),
    }


def save_model_bundle(
    path: str | Path,
    model: SpatialWarmstartModel,
    *,
    arm: str,
    spatial_enabled: bool,
    graph_parameters: Mapping[str, Any],
    pathway_names: Sequence[str],
    normalization: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    native_step_contract: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save the complete H/C/B predictor and every inference-time contract.

    Point bundles retain an explicit zero B so all arms share one strict state
    schema.  This prevents a future loader from silently inventing parameters.
    """

    if not isinstance(model, SpatialWarmstartModel):
        raise TypeError("model must be a SpatialWarmstartModel")
    metadata = _validate_metadata(
        model=model,
        arm=arm,
        spatial_enabled=spatial_enabled,
        graph_parameters=graph_parameters,
        pathway_names=pathway_names,
        normalization=normalization,
        feature_contract=feature_contract,
        native_step_contract=native_step_contract,
        coordinate_contract=coordinate_contract,
    )
    if not spatial_enabled and torch.count_nonzero(model.B.detach()).item() != 0:
        raise ValueError("point arm bundle must contain an explicit zero B")
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
            "dropout": float(model.regression.dropout.p),
        },
        "model_state_dict": state,
        **metadata,
    }
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        bundle["provenance"] = dict(provenance)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    return destination


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"model bundle does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("model bundle must be a dictionary")
    missing = sorted(_REQUIRED_BUNDLE_KEYS - set(payload))
    if missing:
        raise KeyError(f"model bundle is missing required fields: {missing}")
    if payload["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported model bundle schema: {payload['schema_version']!r}")
    if not isinstance(payload["model_config"], Mapping):
        raise TypeError("model_config must be a mapping")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise TypeError("model_state_dict must be a mapping")
    return payload


def load_model_bundle(
    path: str | Path, *, device: str | torch.device | None = None,
) -> tuple[SpatialWarmstartModel, dict[str, Any]]:
    """Load a complete bundle with exact state-dict matching."""

    payload = _load_payload(Path(path))
    config = dict(payload["model_config"])
    try:
        model = SpatialWarmstartModel(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            output_dim=int(config["output_dim"]),
            dropout=float(config["dropout"]),
        )
    except KeyError as error:
        raise KeyError(f"model_config is missing {error.args[0]!r}") from error
    _validate_metadata(
        model=model,
        arm=str(payload["arm"]),
        spatial_enabled=bool(payload["spatial_enabled"]),
        graph_parameters=payload["graph_parameters"],
        pathway_names=payload["pathway_names"],
        normalization=payload["normalization"],
        feature_contract=payload["feature_contract"],
        native_step_contract=payload["native_step_contract"],
        coordinate_contract=payload["coordinate_contract"],
    )
    # Deliberately strict: H, C, and B must all be present with exact shapes,
    # and no teacher/LoRA tensors belong in a final inference bundle.
    model.load_state_dict(dict(payload["model_state_dict"]), strict=True)
    if not bool(payload["spatial_enabled"]) and torch.count_nonzero(model.B).item() != 0:
        raise ValueError("point arm bundle contains a non-zero B")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model.to(target_device).eval()
    return model, payload


def _model_device(model: SpatialWarmstartModel) -> torch.device:
    return next(model.parameters()).device


def predict_features(
    model: SpatialWarmstartModel,
    features: np.ndarray | Tensor,
    *,
    spatial_enabled: bool,
    graph: SpatialGraph | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict from frozen features and an optional fixed graph, without labels."""

    if not isinstance(model, SpatialWarmstartModel):
        raise TypeError("model must be a SpatialWarmstartModel")
    if spatial_enabled and graph is None:
        raise ValueError("spatial prediction requires a fixed graph")
    if not spatial_enabled and graph is not None:
        raise ValueError("point prediction does not accept a graph")
    target_device = torch.device(device) if device is not None else _model_device(model)
    model.to(target_device)
    values = torch.as_tensor(features, dtype=torch.float32, device=target_device)
    if values.ndim != 2 or values.shape[1] != model.input_dim:
        raise ValueError(f"features must have shape [points, {model.input_dim}]")
    if not torch.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(values, graph=graph, use_spatial=bool(spatial_enabled))
        return output.prediction.detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        model.train(was_training)


def fixed_beta_one_smoothing(
    prediction: np.ndarray | Tensor, graph: SpatialGraph,
) -> np.ndarray:
    """Apply the prespecified beta=1 diagnostic, including graph self weight."""

    values = torch.as_tensor(prediction, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] != len(graph.point_ids):
        raise ValueError("prediction must have shape [graph_points, pathways]")
    if not torch.isfinite(values).all():
        raise ValueError("prediction contains non-finite values")
    with torch.no_grad():
        smoothed = graph.smooth_predictions(values, beta=1.0)
    return smoothed.cpu().numpy().astype(np.float64, copy=False)


def evaluate_external(
    model: SpatialWarmstartModel,
    features: np.ndarray | Tensor,
    *,
    targets: np.ndarray,
    patient_ids: Sequence[str],
    spatial_enabled: bool,
    graph: SpatialGraph | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen endpoint on external labels without selecting a model."""

    prediction = predict_features(
        model,
        features,
        spatial_enabled=spatial_enabled,
        graph=graph,
        device=device,
    ).astype(np.float64, copy=False)
    target = np.asarray(targets, dtype=np.float64)
    metrics = compute_regression_metrics(prediction, target, patient_ids)
    return {
        "split": "external_test",
        "selection_used": False,
        "training_performed": False,
        "prediction": prediction,
        "target": target,
        "patient_ids": np.asarray(patient_ids, dtype=str),
        "metrics": metrics,
    }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "evaluate_external",
    "fixed_beta_one_smoothing",
    "load_model_bundle",
    "predict_features",
    "save_model_bundle",
]
