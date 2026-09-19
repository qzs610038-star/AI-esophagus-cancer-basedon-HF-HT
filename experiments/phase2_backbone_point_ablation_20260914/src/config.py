"""Load and fail closed on the frozen v1.1 ablation contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from errors import ConfigError


EXPECTED_COUNTS = {"train": 9472, "internal_val": 1078, "external_test": 1039}
EXPECTED_STEPS = [
    "rgb",
    "resize_shorter_side_256_bicubic_antialias",
    "center_crop_224",
    "to_float_tensor_0_1",
    "imagenet_normalize",
]


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ConfigError(f"{field} 必须为 {expected!r}，实际={actual!r}")


def validate_config(config: dict) -> None:
    _require_equal(config.get("schema_version"), "1.0", "schema_version")
    _require_equal(config.get("protocol_version"), "v1.1-20260915", "protocol_version")
    _require_equal(
        config.get("experiment_id"),
        "phase2_backbone_point_ablation_20260914",
        "experiment_id",
    )
    _require_equal(
        config.get("python_interpreter"),
        r"C:\Users\AIPatho1\pfmval_env\Scripts\python.exe",
        "python_interpreter",
    )

    paths = config.get("paths") or {}
    for key, expected in {
        "code_root": r"D:\AIPatho\qzs\code",
        "runs_root": r"D:\AIPatho\qzs\runs",
        "weights_root": r"D:\AIPatho\qzs\weights",
        "feature_caches_root": r"D:\AIPatho\qzs\feature_caches",
        "hf_home": r"D:\AIPatho\shared\.cache\huggingface",
        "image_root": r"D:\AIPatho\Patch\visiumhd_patch\2",
        "labels_root": r"D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003\group_2\labels",
        "historical_reference_root": r"D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10",
        "uni2h_partner_cache_root": r"D:\AIPatho\qzs\pfmval_deploy_git\uni2h_cache",
        "uni2h_flat_cache_root": r"D:\AIPatho\qzs\pfmval_deploy_git\mpp_uni2h_cache",
    }.items():
        _require_equal(paths.get(key), expected, f"paths.{key}")

    package_root_value = config.get("_package_root")
    if not package_root_value:
        raise ConfigError("配置缺少已解析的 _package_root")
    package_root = Path(str(package_root_value)).resolve()
    expected_inputs = {
        "split_manifest": package_root / "inputs" / "mpp2" / "split_manifest.csv",
        "split_info": package_root / "inputs" / "mpp2" / "split_info.json",
        "zscore_manifest": package_root / "inputs" / "mpp2" / "zscore_manifest.json",
        "normalization": package_root / "inputs" / "mpp2" / "zscore_params_from_train.json",
        "model_manifest": package_root / "inputs" / "model_manifest.json",
        "baseline_reference_manifest": package_root / "inputs" / "baseline_reference_manifest.json",
    }
    inputs = config.get("inputs") or {}
    _require_equal(set(inputs), set(expected_inputs), "inputs keys")
    for key, expected in expected_inputs.items():
        actual = Path(str(inputs.get(key))).resolve()
        if str(actual).casefold() != str(expected.resolve()).casefold():
            raise ConfigError(f"inputs.{key} 必须使用包内冻结文件，实际={actual}")

    data = config.get("data") or {}
    _require_equal(data.get("split_id"), "MPP2/group_2", "data.split_id")
    _require_equal(data.get("label_version"), "barcode-repair-v003", "data.label_version")
    _require_equal(data.get("mpp_id"), 2, "data.mpp_id")
    _require_equal(data.get("expected_counts"), EXPECTED_COUNTS, "data.expected_counts.external_test")
    _require_equal(data.get("output_dim"), 30, "data.output_dim")
    _require_equal(data.get("external_patient"), "XZY", "data.external_patient")
    _require_equal(
        data.get("development_patients"),
        ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"],
        "data.development_patients",
    )
    _require_equal(data.get("source_group_template"), "MPP2_{patient}_source01", "data.source_group_template")

    preprocessing = config.get("preprocessing") or {}
    _require_equal(preprocessing.get("steps"), EXPECTED_STEPS, "preprocessing.steps")
    _require_equal(preprocessing.get("resize_shorter_side"), 256, "preprocessing.resize_shorter_side")
    _require_equal(preprocessing.get("crop_size"), 224, "preprocessing.crop_size")
    _require_equal(preprocessing.get("interpolation"), "bicubic", "preprocessing.interpolation")
    _require_equal(preprocessing.get("antialias"), True, "preprocessing.antialias")
    _require_equal(preprocessing.get("mean"), [0.485, 0.456, 0.406], "preprocessing.mean")
    _require_equal(preprocessing.get("std"), [0.229, 0.224, 0.225], "preprocessing.std")
    _require_equal(preprocessing.get("random_augmentation"), False, "preprocessing.random_augmentation")

    model = config.get("model") or {}
    for key, expected in {
        "hidden_dim": 256,
        "output_dim": 30,
        "dropout": 0.3,
        "activation": "gelu_exact",
        "projection_bias": True,
        "readout_bias": True,
    }.items():
        _require_equal(model.get(key), expected, f"model.{key}")

    training = config.get("training") or {}
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "batch_size": 256,
        "keep_last_batch": True,
        "max_epochs": 60,
        "precision": "float32",
        "amp": False,
        "tf32": False,
        "num_workers": 0,
        "cpu_threads": 8,
        "resume_supported": False,
    }
    for key, expected in expected_training.items():
        value = training.get(key)
        if key == "amp" and value is not False:
            raise ConfigError("AMP 必须关闭")
        _require_equal(value, expected, f"training.{key}")

    selection = config.get("selection") or {}
    expected_selection = {
        "metric": "patient_macro_pathway_pcc",
        "tie_metric": "patient_macro_z_mse_selection",
        "formal_start_epoch": 6,
        "checkpoint_tolerance": 1e-6,
        "early_stop_count_start_epoch": 16,
        "early_stop_min_delta": 1e-4,
        "early_stop_patience": 10,
        "constant_prediction_selection_penalty": -1.0,
    }
    for key, expected in expected_selection.items():
        _require_equal(selection.get(key), expected, f"selection.{key}")

    execution = config.get("execution") or {}
    _require_equal(execution.get("default_scope"), "seed42", "execution.default_scope")
    _require_equal(execution.get("baseline_model"), "uni2h", "execution.baseline_model")
    if "uni2h" in list(execution.get("train_models") or []):
        raise ConfigError("UNI2-h 只能作为历史匹配参照，禁止在本包重训")
    _require_equal(execution.get("train_models"), ["uni", "virchow2"], "execution.train_models")
    _require_equal(execution.get("scopes"), {"seed42": [42], "remaining": [43, 44]}, "execution.scopes")
    _require_equal(execution.get("external_after_formal_lock_only"), True, "execution.external_after_formal_lock_only")
    _require_equal(execution.get("stop_on_first_failure"), True, "execution.stop_on_first_failure")

    randomness = config.get("randomness") or {}
    _require_equal(
        randomness,
        {"model_seed_offset": 0, "center_seed_offset": 100000, "dropout_seed_offset": 200000},
        "randomness",
    )

    extraction = config.get("feature_extraction") or {}
    _require_equal(
        extraction.get("feature_version"),
        "cls_meanpatch_fp32_resize256_centercrop224_v1",
        "feature_extraction.feature_version",
    )
    _require_equal(extraction.get("initial_batch_size"), 16, "feature_extraction.initial_batch_size")
    _require_equal(extraction.get("fallback_batch_sizes"), [8, 4, 2, 1], "feature_extraction.fallback_batch_sizes")
    _require_equal(extraction.get("dtype"), "float32", "feature_extraction.dtype")
    _require_equal(extraction.get("amp"), False, "feature_extraction.amp")
    _require_equal(extraction.get("tf32"), False, "feature_extraction.tf32")
    _require_equal(extraction.get("save_mean_patch"), True, "feature_extraction.save_mean_patch")
    _require_equal(extraction.get("overwrite_existing"), False, "feature_extraction.overwrite_existing")


def _resolve_package_inputs(config: dict, package_root: Path) -> None:
    inputs = config.get("inputs") or {}
    for key, value in list(inputs.items()):
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            inputs[key] = str((package_root / path).resolve())


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取配置 {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("配置根必须是 JSON object")
    package_root = config_path.parent
    config["_package_root"] = str(package_root)
    config["_config_path"] = str(config_path)
    _resolve_package_inputs(config, package_root)
    validate_config(config)
    return config
