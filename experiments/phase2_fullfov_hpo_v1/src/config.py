"""Single editable experiment contract and package-relative input resolution."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from errors import ConfigError


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_ARMS = ("point", "spatial", "no_b")
SUPPORTED_MODELS = ("uni2h", "uni", "virchow2")
SUPPORTED_PROTOCOLS = ("legacy_crop_0875_v1", "full_fov_224_bicubic_v1")
MODEL_DIMS = {"uni2h": 1536, "uni": 1024, "virchow2": 1280}
EXPECTED_COUNTS = {"train": 9472, "internal_val": 1078, "external_test": 1039}


def package_root() -> Path:
    return PACKAGE_ROOT


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(config: dict) -> None:
    """Validate the research protocol without probing server-only assets."""
    _require(isinstance(config, dict), "config 必须为 JSON object")
    _require(config.get("schema_version") == "1.0", "schema_version 必须为 1.0")
    _require(config.get("protocol_version") == "fullfov-hpo-v1", "protocol_version 不匹配")
    _require(config.get("experiment_id") == "phase2_fullfov_hpo_v1", "experiment_id 不匹配")
    data = config.get("data") or {}
    _require(data.get("split_id") == "MPP2/group_2", "只实现 MPP2/group_2")
    _require(data.get("label_version") == "barcode-repair-v003", "只实现原 30 通路目标")
    _require(data.get("expected_counts") == EXPECTED_COUNTS, "划分点数与冻结协议不一致")
    _require(int(data.get("output_dim", 0)) == 30, "通路数必须为 30")
    _require(data.get("external_patient") == "XZY", "外部患者必须为 XZY")
    _require(len(data.get("development_patients") or []) == 6, "内部开发患者必须为六名")
    _require(int(data.get("input_dim", 0)) in MODEL_DIMS.values(), "输入维度不属于指定编码器")
    paths = config.get("paths") or {}
    for name in ("runs_root", "weights_root", "feature_caches_root", "image_root", "labels_root"):
        _require(bool(paths.get(name)), f"缺少 paths.{name}")
    _require(len({str(paths[name]).casefold() for name in ("runs_root", "weights_root", "feature_caches_root")}) == 3,
             "runs、weights、feature_caches 必须为不同目录")
    inputs = config.get("inputs") or {}
    for name in ("split_manifest", "split_info", "zscore_manifest", "normalization", "model_manifest", "slide_mapping", "slide_geometry"):
        _require(bool(inputs.get(name)), f"缺少 inputs.{name}")
    preprocess = config.get("preprocessing") or {}
    _require(set((preprocess.get("protocols") or {})) == set(SUPPORTED_PROTOCOLS), "必须声明历史与全视野两种预处理")
    _require(preprocess.get("interpolation") == "bicubic" and preprocess.get("antialias") is True,
             "预处理必须显式采用 bicubic/antialias")
    _require(preprocess.get("random_augmentation") is False, "首轮不使用随机图像增强")
    extraction = config.get("feature_extraction") or {}
    _require(extraction.get("dtype") == "float32" and extraction.get("amp") is False and extraction.get("tf32") is False,
             "特征提取必须为 FP32 且关闭 AMP/TF32")
    _require(extraction.get("overwrite_existing") is False, "禁止覆盖既有特征缓存")
    model = config.get("model") or {}
    _require(int(model.get("hidden_dim", 0)) > 0, "hidden_dim 必须为正")
    _require(0 <= float(model.get("dropout", -1)) < 1, "dropout 必须在 [0,1)")
    _require(model.get("activation") == "gelu_exact" and model.get("spatial_initialization") == "zeros",
             "模型激活或 B 初始化与约定不符")
    graph = config.get("graph") or {}
    _require(graph.get("hops") == 1 and graph.get("directed") is True, "只实现单跳有向图")
    _require(graph.get("use_image_similarity") is True, "主实验要求形态边权开启")
    for name in ("radius_in_native_steps", "distance_sigma", "image_temperature", "self_raw_weight"):
        _require(float(graph.get(name, 0)) > 0, f"graph.{name} 必须为正")
    _require(int(graph.get("max_neighbors", 0)) > 0, "max_neighbors 必须为正")
    training = config.get("training") or {}
    _require(training.get("optimizer") in ("Adam", "AdamW"), "优化器只支持 Adam/AdamW")
    _require(training.get("lr_schedule") in ("constant", "warmup_cosine"), "未知学习率日程")
    _require(float(training.get("learning_rate", 0)) > 0, "学习率必须为正")
    _require(float(training.get("weight_decay", -1)) >= 0, "权重衰减不能为负")
    _require(float(training.get("b_lr_multiplier", 0)) > 0, "B 学习率倍数必须为正")
    _require(int(training.get("batch_size", 0)) > 0 and training.get("keep_last_batch") is True,
             "批大小必须为正且保留最后批次")
    _require(training.get("precision") == "float32" and training.get("amp") is False and training.get("tf32") is False,
             "训练必须为 FP32 且关闭 AMP/TF32")
    _require(int(training.get("max_epochs", 0)) > 0, "max_epochs 必须为正")
    selection = config.get("selection") or {}
    _require(selection.get("metric") == "patient_macro_pathway_pcc", "必须按内部患者—通路 PCC 选模")
    _require(selection.get("tie_metric") == "patient_macro_z_mse_selection", "破同分指标不匹配")
    search = config.get("search") or {}
    _require(int(search.get("anchor_trials", 0)) + int(search.get("random_trials", 0)) + int(search.get("tpe_trials", 0))
             == int(search.get("initial_trials", -1)) == 24, "初搜名额必须为 2+6+16=24")
    _require(int(search.get("spatial_restricted_trials", 0)) + int(search.get("spatial_joint_trials", 0)) == 24,
             "空间受限/联合名额总数必须为 24")
    _require(search.get("replication_seeds") == [42, 43, 44] and int(search.get("top_k", 0)) == 3,
             "搜索复核必须为前三配置的种子42/43/44")
    execution = config.get("execution") or {}
    _require(execution.get("paired_seeds") == [42, 43, 44], "配对种子必须为42/43/44")
    _require(execution.get("final_seeds") == [45, 46, 47], "最终复核种子必须为45/46/47")
    _require(execution.get("models") == list(SUPPORTED_MODELS), "编码器清单不匹配")
    _require(execution.get("phase3_export_seed") == 45, "Phase3 导出固定种子45")


def load_config(path: str | Path | None = None, *, package_dir: str | Path | None = None) -> dict:
    """Resolve small packaged inputs against code, not the current working directory."""
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


def trial_config(base: dict, *, model_name: str, parameters: dict | None = None, selection_mode: str = "paired") -> dict:
    """Create an isolated, effective per-trial config without touching the frozen base."""
    _require(model_name in MODEL_DIMS, f"未知编码器 {model_name}")
    _require(selection_mode in ("paired", "search"), "selection_mode 必须为 paired/search")
    cfg = copy.deepcopy(base)
    cfg["data"]["input_dim"] = MODEL_DIMS[model_name]
    cfg["model_name"] = model_name
    cfg["selection_mode"] = selection_mode
    if parameters:
        for dotted, value in parameters.items():
            section, key = dotted.split(".", 1)
            _require(section in ("training", "model", "graph") and key in cfg[section],
                     f"不支持参数覆盖 {dotted}")
            cfg[section][key] = value
    if selection_mode == "search":
        search = cfg["search"]
        cfg["training"]["max_epochs"] = int(search["max_epochs"])
        cfg["selection"].update(
            formal_start_epoch=int(search["formal_start_epoch"]),
            early_stop_count_start_epoch=int(search["early_stop_count_start_epoch"]),
            early_stop_patience=int(search["early_stop_patience"]),
            early_stop_min_delta=float(search["early_stop_min_delta"]),
            checkpoint_tolerance=float(search["checkpoint_tolerance"]),
        )
    validate_config(cfg)
    return cfg
