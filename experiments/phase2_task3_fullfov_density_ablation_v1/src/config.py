"""Configuration loading and protocol assertions."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from errors import ConfigError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _resolve_local(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PACKAGE_ROOT / path).resolve())


def load_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path else PACKAGE_ROOT / "config.json"
    cfg = json.loads(source.read_text(encoding="utf-8-sig"))
    cfg = copy.deepcopy(cfg)
    for key in ("split_manifest", "zscore_manifest", "dense_zscore_params", "slide_mapping", "slide_geometry", "model_manifest"):
        cfg["inputs"][key] = _resolve_local(cfg["inputs"][key])
    cfg["data"]["slide_geometry_file"] = cfg["inputs"]["slide_geometry"]
    audit_root = Path(cfg["label_audit"]["source_root"])
    if not audit_root.is_absolute():
        audit_root = (REPO_ROOT / audit_root).resolve()
    cfg["label_audit"]["source_root"] = str(audit_root)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["experiment_id"] != "phase2_task3_fullfov_density_ablation_v1":
        raise ConfigError("experiment_id 不符合本包冻结合同")
    data, model, graph = cfg["data"], cfg["model"], cfg["graph"]
    if (data["input_dim"], model["hidden_dim"], data["output_dim"]) != (1536, 256, 30):
        raise ConfigError("模型必须为 1536→256→30")
    if data["full_fov_protocol"] != "full_fov_224_bicubic_v1" or data["crop"] is not None or data["augmentation"]:
        raise ConfigError("必须保持 full-FOV 224 bicubic、无裁剪/增强")
    if model["activation"] != "gelu_exact" or float(model["dropout"]) != 0.3 or not model["spatial_zero_init"] or model["spatial_bias"] is not False:
        raise ConfigError("GELU/dropout/空间零初始化或无空间偏置合同被改变")
    expected_graph = (2.0, 12, 0.5, 0.7716396069760084, 0.5)
    actual_graph = (float(graph["radius_in_native_steps"]), int(graph["max_neighbors"]), float(graph["distance_sigma"]), float(graph["image_temperature"]), float(graph["self_raw_weight"]))
    if actual_graph != expected_graph or graph["directed"] is not True or int(graph["hops"]) != 1:
        raise ConfigError("空间图合同被改变")
    opt, train = cfg["optimizer"], cfg["training"]
    if (opt["name"], float(opt["learning_rate"]), float(opt["weight_decay"]), int(opt["batch_size"]), opt["schedule"]) != ("Adam", 1e-4, 0.0, 32, "constant"):
        raise ConfigError("优化器合同被改变")
    if int(train["max_updates"]) != 14800 or train["early_stopping"] is not False:
        raise ConfigError("正式训练必须固定 14,800 更新并关闭早停")
    if cfg["selection"]["primary"]["every_updates"] * cfg["selection"]["primary"]["opportunities"] != 14800:
        raise ConfigError("等更新选择器必须正好提供50次机会")
    if cfg["selection"]["sensitivity"]["epochs"] != 50 or cfg["selection"]["sensitivity"]["opportunities"] != 50:
        raise ConfigError("等50轮选择器必须正好提供50次机会")
    if cfg["selection"]["external_used_for_selection"] is not False:
        raise ConfigError("外部XZY不得参与选模")
