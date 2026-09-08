"""Load and validate the single editable config.json."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from errors import ConfigError, UnsupportedAlgorithmError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_ARMS = ("point", "relation", "spatial", "joint")
SUPPORTED_SPLITS = ("train", "internal_val", "external")

_SUPPORTED = {
    "relation.distance": frozenset({"mean_squared_standardized_pathways"}),
    "relation.support": frozenset({"truncated"}),
    "relation.quantile_method": frozenset({"linear"}),
    "relation.far_patient_scope": frozenset({"all", "same_patient"}),
    "model.activation": frozenset({"gelu_exact"}),
    "model.spatial_initialization": frozenset({"zeros"}),
    "training.optimizer": frozenset({"AdamW"}),
    "training.precision": frozenset({"float32"}),
}

_UNIMPLEMENTED_HINTS = {
    "relation.support": "首轮只实现 truncated Q/P，不提供稠密Q或RKD。",
    "relation.distance": "只实现30维标准化标签均方距离。",
    "graph.hops": "只实现一跳局部图，不提供多跳/动态图/超图。",
}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def package_root() -> Path:
    return PACKAGE_ROOT


def resolve_path(value: str | Path | None, root: Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or PACKAGE_ROOT) / path


def _require(mapping: dict, key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"配置缺少 {ctx}.{key}")
    return mapping[key]


def _reject_if_unsupported(field: str, value: Any, allowed: frozenset) -> None:
    if value not in allowed:
        hint = _UNIMPLEMENTED_HINTS.get(field, "该开关当前未实现。")
        raise UnsupportedAlgorithmError(
            f"未实现或不允许的配置 {field}={value!r}。允许值: {sorted(allowed)}。{hint}"
        )


def validate_config(config: dict) -> None:
    if config.get("plan_version") != "v2.1":
        raise ConfigError(f"plan_version 必须为 v2.1，当前={config.get('plan_version')!r}")

    data = _require(config, "data", "root")
    model = _require(config, "model", "root")
    relation = _require(config, "relation", "root")
    graph = _require(config, "graph", "root")
    training = _require(config, "training", "root")
    precheck = _require(config, "precheck", "root")
    analysis = _require(config, "analysis", "root")
    runtime = _require(config, "runtime", "root")
    _require(runtime, "runs_root", "runtime")
    _require(runtime, "weights_root", "runtime")

    for key in ("input_dim", "output_dim", "expected_train_points", "expected_internal_val_points"):
        _require(data, key, "data")
    if int(data["input_dim"]) != 1536:
        raise ConfigError(f"data.input_dim 必须为1536，当前={data['input_dim']}")
    if int(data["output_dim"]) != 30:
        raise ConfigError(f"data.output_dim 必须为30，当前={data['output_dim']}")
    if int(data["expected_train_points"]) != 9472:
        raise ConfigError("data.expected_train_points 必须为9472")
    if int(data["expected_internal_val_points"]) != 1078:
        raise ConfigError("data.expected_internal_val_points 必须为1078")

    hidden = int(_require(model, "hidden_dim", "model"))
    rel_dim = int(_require(model, "relation_dim", "model"))
    if hidden < 1 or rel_dim < 1:
        raise ConfigError("model.hidden_dim 与 model.relation_dim 必须为正整数")
    _reject_if_unsupported("model.activation", model.get("activation"), _SUPPORTED["model.activation"])
    _reject_if_unsupported(
        "model.spatial_initialization",
        model.get("spatial_initialization"),
        _SUPPORTED["model.spatial_initialization"],
    )
    if model.get("relation_bias") is not False:
        raise ConfigError("model.relation_bias 必须为 false")
    if model.get("spatial_bias") is not False:
        raise ConfigError("model.spatial_bias 必须为 false")

    _reject_if_unsupported("relation.distance", relation.get("distance"), _SUPPORTED["relation.distance"])
    _reject_if_unsupported("relation.support", relation.get("support"), _SUPPORTED["relation.support"])
    _reject_if_unsupported(
        "relation.quantile_method", relation.get("quantile_method"), _SUPPORTED["relation.quantile_method"]
    )
    _reject_if_unsupported(
        "relation.far_patient_scope",
        relation.get("far_patient_scope"),
        _SUPPORTED["relation.far_patient_scope"],
    )
    for name in ("max_near", "max_far", "threshold_pairs"):
        if int(relation[name]) < 1:
            raise ConfigError(f"relation.{name} 必须为正整数")
    if not (0.0 <= float(relation["near_quantile"]) <= 1.0):
        raise ConfigError("relation.near_quantile 必须在[0,1]")
    if not (0.0 <= float(relation["far_quantile"]) <= 1.0):
        raise ConfigError("relation.far_quantile 必须在[0,1]")

    hops = graph.get("hops")
    if hops != 1:
        raise UnsupportedAlgorithmError(
            f"未实现或不允许的配置 graph.hops={hops!r}。允许值: [1]。{_UNIMPLEMENTED_HINTS['graph.hops']}"
        )
    if graph.get("directed") is not True:
        raise UnsupportedAlgorithmError("graph.directed 必须为 true；不实现对称化。")
    if float(graph["radius_in_native_steps"]) <= 0:
        raise ConfigError("graph.radius_in_native_steps 必须为正")
    if int(graph["max_neighbors"]) < 1:
        raise ConfigError("graph.max_neighbors 必须为正整数")
    if not isinstance(graph.get("use_image_similarity"), bool):
        raise ConfigError("graph.use_image_similarity 必须为 bool")

    _reject_if_unsupported("training.optimizer", training.get("optimizer"), _SUPPORTED["training.optimizer"])
    _reject_if_unsupported("training.precision", training.get("precision"), _SUPPORTED["training.precision"])
    if training.get("amp") is not False:
        raise UnsupportedAlgorithmError("training.amp 必须为 false；本包不实现 AMP。")
    if training.get("tf32") is not False:
        raise UnsupportedAlgorithmError("training.tf32 必须为 false；本包不实现 TF32。")
    if int(training["batch_size"]) < 1:
        raise ConfigError("training.batch_size 必须为正整数")
    if training.get("keep_last_batch") is not True:
        raise ConfigError("training.keep_last_batch 必须为 true")
    arms = list(training.get("arms") or [])
    if arms != list(SUPPORTED_ARMS):
        raise ConfigError(f"training.arms 必须恰好为 {list(SUPPORTED_ARMS)}，当前={arms}")
    for seed in training.get("seeds") or []:
        if int(seed) < 0:
            raise ConfigError("training.seeds 必须为非负整数")

    if precheck.get("automatic_parameter_changes") is not False:
        raise ConfigError("precheck.automatic_parameter_changes 必须为 false")
    if int(precheck["relation_complete_shuffles"]) < 1:
        raise ConfigError("precheck.relation_complete_shuffles 必须为正整数")
    for split in precheck.get("graph_splits") or []:
        if split not in ("train", "internal_val"):
            raise ConfigError(f"precheck.graph_splits 仅允许 train/internal_val，当前含 {split}")

    smoothing = analysis.get("fixed_prediction_smoothing") or {}
    if smoothing.get("fit_beta") is True:
        raise UnsupportedAlgorithmError("analysis.fixed_prediction_smoothing.fit_beta 必须为 false；不拟合β。")
    if smoothing.get("used_for_selection") is True:
        raise UnsupportedAlgorithmError(
            "analysis.fixed_prediction_smoothing.used_for_selection 必须为 false；平滑不参与选点。"
        )


def resolve_config_paths(config: dict, root: Path | None = None) -> dict:
    cfg = deepcopy(config)
    base = root or PACKAGE_ROOT
    cfg["_package_root"] = str(base)
    data = cfg["data"]
    relative_keys = (
        "split_directory",
        "split_manifest_file",
        "split_info_file",
        "zscore_manifest_file",
        "slide_mapping_file",
        "slide_mapping_status_file",
        "slide_geometry_file",
        "normalization_file",
        "feature_source_manifest",
        "input_status_file",
    )
    for key in relative_keys:
        if data.get(key):
            data[key] = str(resolve_path(data[key], base))
    if data.get("labels_root"):
        data["labels_root"] = str(resolve_path(data["labels_root"], base))
    if data.get("external_point_table"):
        data["external_point_table"] = str(resolve_path(data["external_point_table"], base))
    return cfg


def load_config(path: str | Path | None = None, *, package_dir: Path | None = None) -> dict:
    root = Path(package_dir) if package_dir is not None else PACKAGE_ROOT
    config_path = Path(path) if path is not None else root / "config.json"
    config = read_json(config_path)
    validate_config(config)
    resolved = resolve_config_paths(config, root)
    resolved["_config_path"] = str(Path(config_path).resolve())
    return resolved
