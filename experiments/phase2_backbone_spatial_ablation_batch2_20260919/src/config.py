"""Frozen spatial-11 contract and package-relative input resolution."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from errors import ConfigError


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "fullfov-spatial-backbone-batch2-v1"
EXPERIMENT_ID = "phase2_backbone_spatial_ablation_batch2_20260919"
SUPPORTED_ARMS = ("spatial",)
TRAIN_MODELS = ("hoptimus0", "hoptimus1", "phikonv2")
BASELINE_MODELS = ("uni2h", "uni", "virchow2")
GEOMETRY_PROTOCOL = "full_fov_224_bicubic_v1"
SEEDS = (45, 46, 47)
MODEL_DIMS = {
    "hoptimus0": 1536,
    "hoptimus1": 1536,
    "phikonv2": 1024,
    "uni2h": 1536,
    "uni": 1024,
    "virchow2": 1280,
}
NORMALIZATION_BY_MODEL = {
    "hoptimus0": "hoptimus_rgb_v1",
    "hoptimus1": "hoptimus_rgb_v1",
    "phikonv2": "imagenet_rgb_v1",
    "uni2h": "imagenet_rgb_v1",
    "uni": "imagenet_rgb_v1",
    "virchow2": "imagenet_rgb_v1",
}
EXPECTED_COUNTS = {"train": 9472, "internal_val": 1078, "external_test": 1039}
FROZEN_SPATIAL11 = {
    "training.optimizer": "Adam",
    "training.learning_rate": 1e-4,
    "training.weight_decay": 0.0,
    "training.batch_size": 32,
    "model.hidden_dim": 256,
    "model.dropout": 0.3,
    "training.lr_schedule": "constant",
    "training.b_lr_multiplier": 1.0,
    "graph.radius_in_native_steps": 2.0,
    "graph.max_neighbors": 12,
    "graph.distance_sigma": 0.5,
    "graph.image_temperature": 0.7716396069760084,
    "graph.self_raw_weight": 0.5,
}


def package_root() -> Path:
    return PACKAGE_ROOT


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def spatial_head_parameter_count(feature_dim: int) -> int:
    """H/C/B count: 256*d + 15,646 with H bias, C bias, B no-bias."""
    return 256 * int(feature_dim) + 15_646


def preprocess_profile(model_name: str, *, geometry: str = GEOMETRY_PROTOCOL) -> str:
    return f"{geometry}__{NORMALIZATION_BY_MODEL[model_name]}"


def validate_config(config: dict) -> None:
    _require(isinstance(config, dict), "config 必须为 JSON object")
    _require(config.get("schema_version") == "1.0", "schema_version 必须为 1.0")
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "protocol_version 不匹配")
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment_id 不匹配")
    _require("search" not in config, "本批没有 search 阶段，禁止携带搜索配置")
    data = config.get("data") or {}
    _require(data.get("split_id") == "MPP2/group_2", "只实现 MPP2/group_2")
    _require(data.get("label_version") == "barcode-repair-v003", "只实现原 30 通路目标")
    _require(data.get("expected_counts") == EXPECTED_COUNTS, "划分点数与冻结协议不一致")
    _require(int(data.get("output_dim", 0)) == 30, "通路数必须为 30")
    _require(data.get("external_patient") == "XZY", "外部患者必须为 XZY")
    _require(list(data.get("development_patients") or []) == ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"], "内部开发患者必须为六名冻结名单")
    paths = config.get("paths") or {}
    for name in ("runs_root", "weights_root", "feature_caches_root", "image_root", "labels_root", "hf_home"):
        _require(bool(paths.get(name)), f"缺少 paths.{name}")
    _require(len({str(paths[name]).casefold() for name in ("runs_root", "weights_root", "feature_caches_root")}) == 3,
             "runs、weights、feature_caches 必须为不同目录")
    inputs = config.get("inputs") or {}
    for name in ("split_manifest", "split_info", "zscore_manifest", "normalization", "model_manifest",
                 "baseline_reference_manifest", "slide_mapping", "slide_geometry"):
        _require(bool(inputs.get(name)), f"缺少 inputs.{name}")
    preprocess = config.get("preprocessing") or {}
    _require(preprocess.get("geometry_protocol") == GEOMETRY_PROTOCOL, "几何协议必须为 full_fov_224_bicubic_v1")
    _require(preprocess.get("interpolation") == "bicubic" and preprocess.get("antialias") is True,
             "预处理必须显式采用 bicubic/antialias")
    _require(preprocess.get("random_augmentation") is False, "禁止随机图像增强")
    _require(set((preprocess.get("normalization_profiles") or {})) == {"imagenet_rgb_v1", "hoptimus_rgb_v1"},
             "必须显式登记 imagenet_rgb_v1 与 hoptimus_rgb_v1")
    extraction = config.get("feature_extraction") or {}
    _require(extraction.get("dtype") == "float32" and extraction.get("amp") is False and extraction.get("tf32") is False,
             "特征提取必须为 FP32 且关闭 AMP/TF32")
    _require(extraction.get("overwrite_existing") is False, "禁止覆盖既有特征缓存")
    _require(extraction.get("save_mean_patch") is False, "本批主缓存只保存选定单 patch 特征")
    model = config.get("model") or {}
    _require(int(model.get("hidden_dim", 0)) == 256 and int(model.get("relation_dim", 0)) == 256, "hidden/relation 必须为 256")
    _require(float(model.get("dropout", -1)) == 0.3, "dropout 必须为 0.3")
    _require(model.get("activation") == "gelu_exact" and model.get("spatial_initialization") == "zeros",
             "模型激活或 B 初始化与约定不符")
    _require(model.get("shared_bias") is True and model.get("readout_bias") is True, "H/C 必须含 bias")
    _require(model.get("spatial_bias") is False, "B 必须无 bias")
    graph = config.get("graph") or {}
    _require(graph.get("hops") == 1 and graph.get("directed") is True, "只实现单跳有向图")
    _require(graph.get("use_image_similarity") is True, "空间头要求形态边权开启")
    _require(float(graph.get("radius_in_native_steps")) == 2.0, "radius_in_native_steps 必须为 2.0")
    _require(int(graph.get("max_neighbors")) == 12, "max_neighbors 必须为 12")
    _require(float(graph.get("distance_sigma")) == 0.5, "distance_sigma 必须为 0.5")
    _require(float(graph.get("self_raw_weight")) == 0.5, "self_raw_weight 必须为 0.5")
    _require(abs(float(graph.get("image_temperature")) - 0.7716396069760084) < 1e-12, "image_temperature 必须为冻结 spatial-11 值")
    training = config.get("training") or {}
    _require(training.get("optimizer") == "Adam", "冻结配方优化器必须为 Adam")
    _require(training.get("lr_schedule") == "constant", "冻结配方学习率日程必须为 constant")
    _require(float(training.get("learning_rate")) == 1e-4, "learning_rate 必须为 1e-4")
    _require(float(training.get("weight_decay")) == 0.0, "Adam 臂 weight_decay 必须为 0")
    _require(float(training.get("b_lr_multiplier")) == 1.0, "B 学习率倍数必须为 1.0")
    _require(int(training.get("batch_size")) == 32 and training.get("keep_last_batch") is True, "batch_size=32 且保留末批")
    _require(training.get("precision") == "float32" and training.get("amp") is False and training.get("tf32") is False,
             "训练必须为 FP32 且关闭 AMP/TF32")
    _require(int(training.get("max_epochs")) == 120, "max_epochs 必须为 120")
    _require(list(training.get("betas")) == [0.9, 0.999], "Adam betas 必须为 0.9, 0.999")
    selection = config.get("selection") or {}
    _require(selection.get("metric") == "patient_macro_pathway_pcc", "必须按内部患者—通路 PCC 选模")
    _require(selection.get("tie_metric") == "patient_macro_z_mse_selection", "破同分指标不匹配")
    _require(int(selection.get("formal_start_epoch")) == 1, "formal_start_epoch 必须为 1")
    _require(int(selection.get("early_stop_count_start_epoch")) == 41, "早停计数必须从第 41 轮开始")
    _require(int(selection.get("early_stop_patience")) == 20, "patience 必须为 20")
    execution = config.get("execution") or {}
    _require(execution.get("default_action") == "check-inputs", "默认动作必须为 check-inputs")
    _require(execution.get("final_seeds") == [45, 46, 47], "种子必须为 45/46/47")
    _require(execution.get("train_models") == list(TRAIN_MODELS), "新训练编码器必须为三个第二批模型")
    _require(execution.get("baseline_models") == list(BASELINE_MODELS), "基线必须为 UNI2-h/UNI/Virchow2")
    _require(execution.get("arm") == "spatial", "本批只训练空间臂")
    _require(execution.get("external_after_formal_lock_only") is True, "外部评估只能在正式检查点之后")


def load_config(path: str | Path | None = None, *, package_dir: str | Path | None = None) -> dict:
    root = Path(package_dir or PACKAGE_ROOT).resolve()
    config_path = Path(path or root / "config.json").resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取配置 {config_path}: {exc}") from exc
    validate_config(config)
    result = copy.deepcopy(config)
    result["_package_root"] = str(root)
    result["_config_path"] = str(config_path)
    for name, value in result["inputs"].items():
        candidate = Path(value)
        result["inputs"][name] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    for name in ("slide_geometry_file", "slide_mapping_file", "normalization_file"):
        if result["data"].get(name):
            candidate = Path(result["data"][name])
            result["data"][name] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    return result


def model_config(base: dict, model_name: str) -> dict:
    _require(model_name in MODEL_DIMS, f"未知编码器 {model_name}")
    cfg = copy.deepcopy(base)
    cfg["data"]["input_dim"] = MODEL_DIMS[model_name]
    cfg["model_name"] = model_name
    cfg["input_protocol"] = GEOMETRY_PROTOCOL
    cfg["normalization_profile"] = NORMALIZATION_BY_MODEL[model_name]
    cfg["preprocess_profile"] = preprocess_profile(model_name)
    cfg["task_recipe"] = "frozen_spatial"
    return cfg
