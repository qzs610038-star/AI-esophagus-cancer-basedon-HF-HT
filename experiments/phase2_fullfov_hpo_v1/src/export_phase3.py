"""Phase3 hand-off contract for the frozen UNI2-h spatial model.

The exporter deliberately writes metadata and an executable Python interface, not
model weights.  The formal checkpoint remains in the configured weights root and
is always re-checked before inference.
"""

from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data import PointTable, load_normalization, load_pathway_names
from errors import ConfigError
from predict import load_formal_model, predict_table


FULL_PROTOCOL = "full_fov_224_bicubic_v1"
PRIMARY_MODEL = "uni2h"
PRIMARY_ARM = "spatial"
PRIMARY_SEED = 45
CONTRACT_VERSION = "phase3_export_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON 根节点必须为对象: {source}")
    return payload


def _plain_config(config: dict) -> dict:
    """Remove runtime-only keys before placing a frozen config in JSON."""
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def _checkpoint_payload(path: str | Path) -> dict:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ConfigError("checkpoint 不是 Phase2 字典载荷")
    return payload


def _require_primary_target(model_name: str, arm: str, seed: int) -> None:
    if str(model_name) != PRIMARY_MODEL or str(arm) != PRIMARY_ARM or int(seed) != PRIMARY_SEED:
        raise ConfigError(
            "Phase3 导出固定为 UNI2-h 的 spatial 臂、种子 45；"
            f"当前 model={model_name!r}, arm={arm!r}, seed={seed!r}"
        )


def _validate_checkpoint_contract(
    config: dict, checkpoint_path: str | Path, *, model_name: str, arm: str, seed: int
) -> tuple[dict, dict]:
    _require_primary_target(model_name, arm, seed)
    payload = _checkpoint_payload(checkpoint_path)
    selection = payload.get("selection") or {}
    if payload.get("kind") != "formal" or selection.get("kind") != "formal":
        raise ConfigError("Phase3 仅接收内部验证选出的 formal checkpoint")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ConfigError("checkpoint 的实验臂或种子与导出目标不一致")
    frozen = payload.get("config")
    if not isinstance(frozen, dict):
        raise ConfigError("formal checkpoint 缺少冻结训练配置")
    for section in ("data", "model", "graph", "preprocessing"):
        if section not in frozen:
            raise ConfigError(f"formal checkpoint 缺少 {section} 配置")
    for field in ("input_dim", "output_dim"):
        if int(payload.get(field, -1)) != int(frozen["data"].get(field, -1)):
            raise ConfigError(f"checkpoint {field} 与其冻结配置不一致")
    if int(payload.get("hidden_dim", -1)) != int(frozen["model"].get("hidden_dim", -1)):
        raise ConfigError("checkpoint hidden_dim 与其冻结配置不一致")
    # The caller's config identifies the packaged inputs.  Dynamic HPO values
    # must come from the checkpoint, rather than silently reverting to defaults.
    if int(config["data"]["input_dim"]) != int(frozen["data"]["input_dim"]):
        raise ConfigError("当前编码器 CLS 维度与 formal checkpoint 不一致")
    if frozen.get("model_name") != model_name or frozen.get("selection_mode") != "search":
        raise ConfigError("formal checkpoint 未标记为 UNI2-h 冻结搜索配方")
    return payload, _plain_config(frozen)


def build_phase3_contract(
    config: dict,
    checkpoint_path: str | Path,
    *,
    model_name: str = PRIMARY_MODEL,
    arm: str = PRIMARY_ARM,
    seed: int = PRIMARY_SEED,
    frozen_selection_path: str | Path | None = None,
) -> dict:
    """Build a serialisable Phase3 contract without writing files or weights.

    ``config`` supplies package input locations; H/C/B and graph hyperparameters
    are taken from the formal checkpoint's frozen training config.
    """
    payload, frozen_config = _validate_checkpoint_contract(
        config, checkpoint_path, model_name=model_name, arm=arm, seed=seed
    )
    inputs = config.get("inputs") or {}
    manifest = _read_json(inputs["model_manifest"])
    model_spec = ((manifest.get("models") or {}).get(model_name))
    if not isinstance(model_spec, dict):
        raise ConfigError(f"模型清单缺少 {model_name}")
    input_dim = int(frozen_config["data"]["input_dim"])
    if int(model_spec.get("cls_dim", -1)) != input_dim:
        raise ConfigError("UNI2-h 模型清单 CLS 维度与 formal checkpoint 不一致")

    pathway_names = load_pathway_names(inputs["zscore_manifest"])
    if len(pathway_names) != int(frozen_config["data"]["output_dim"]):
        raise ConfigError("通路顺序与 checkpoint 输出维度不一致")
    normalization = load_normalization(inputs["normalization"], pathway_names)
    if normalization.fit_split != "train" or normalization.n_train_samples != int(config["data"]["expected_counts"]["train"]):
        raise ConfigError("标准化参数并非登记的训练集拟合结果")
    if frozen_selection_path is not None:
        selection_record = _read_json(frozen_selection_path)
        if selection_record.get("external_test_used_for_selection") is not False:
            raise ConfigError("冻结选择记录未确认仅使用内部验证集")
        frozen_parameters = (selection_record.get("spatial") or {}).get("parameters") or {}
        if not frozen_parameters:
            raise ConfigError("冻结选择记录缺少空间配方")
        for dotted, expected in frozen_parameters.items():
            section, key = dotted.split(".", 1)
            if frozen_config.get(section, {}).get(key) != expected:
                raise ConfigError(f"正式检查点配置与冻结空间配方不一致: {dotted}")
    protocol = FULL_PROTOCOL
    protocols = (frozen_config.get("preprocessing") or {}).get("protocols") or {}
    if protocol not in protocols:
        raise ConfigError("冻结配置缺少全视野预处理协议")
    geometry_path = Path(inputs.get("slide_geometry") or config["data"]["slide_geometry_file"])
    with geometry_path.open("r", newline="", encoding="utf-8-sig") as handle:
        geometry_rows = list(csv.DictReader(handle))
    if not geometry_rows or any(not row.get("coordinate_unit") or not row.get("s") for row in geometry_rows):
        raise ConfigError("导出缺少原生图坐标单位或步长")
    checkpoint = Path(checkpoint_path).resolve()
    return {
        "schema_version": CONTRACT_VERSION,
        "created_at": _utc_now(),
        "status": "prepared_not_evaluated",
        "accepted_conclusion": False,
        "purpose": "Phase3 UNI2-h 全视野空间预测交接；不修改现有消费端。",
        "model": {
            "name": model_name,
            "architecture": model_spec.get("architecture"),
            "repo_id": model_spec.get("repo_id"),
            "revision": model_spec.get("revision"),
            "snapshot_path": model_spec.get("snapshot_path"),
            "cls_dim": input_dim,
            "cls_token_index": model_spec.get("cls_index", 0),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "kind": "formal",
            "arm": arm,
            "seed": int(seed),
            "epoch": int((payload.get("selection") or {}).get("epoch")),
            "selection": payload.get("selection"),
            "weights_copied": False,
        },
        "dynamic_dimensions": {
            "input_dim": input_dim,
            "hidden_dim": int(frozen_config["model"]["hidden_dim"]),
            "output_dim": int(frozen_config["data"]["output_dim"]),
        },
        "preprocessing": {
            "protocol": protocol,
            "steps": protocols[protocol],
            "mean": list((frozen_config.get("preprocessing") or {}).get("mean") or []),
            "std": list((frozen_config.get("preprocessing") or {}).get("std") or []),
            "requires_square_input": True,
        },
        "graph": frozen_config["graph"],
        "geometry": {
            "source_path": str(geometry_path.resolve()),
            "groups": [{"slide_id": row["slide_id"], "coordinate_unit": row["coordinate_unit"],
                        "native_step": float(row["s"]),
                        "patch_coverage_size": float(row["patch_coverage_size"]) if row.get("patch_coverage_size") else None,
                        "status": row.get("status")} for row in geometry_rows],
            "physical_patch_coverage_verified": False,
        },
        "pathway_order": pathway_names,
        "normalization": {
            "fit_split": normalization.fit_split,
            "ddof": normalization.ddof,
            "n_train_samples": normalization.n_train_samples,
            "clip_applied_after_transform": normalization.clip_applied_after_transform,
            "clip_range": list(normalization.clip_range),
            "mean": normalization.mean.tolist(),
            "std": normalization.std.tolist(),
            "source_path": normalization.source_path,
            "inverse_transform": "pred_raw = pred_z * std + mean; exactly once",
        },
        "frozen_selection_path": None if frozen_selection_path is None else str(Path(frozen_selection_path).resolve()),
        "inference_config": frozen_config,
        "inference_interface": {
            "python_function": "export_phase3.predict_from_phase3_contract(contract_path, table, device='cuda')",
            "input": "PointTable: label-free, full external_test split, identity-aligned UNI2-h CLS features",
            "output": "float64 z-score predictions [N, output_dim] in PointTable identity order",
            "external_labels_used": False,
        },
    }


def write_phase3_contract(contract: dict, output_dir: str | Path) -> Path:
    """Write a new contract once.  Existing exports are never overwritten."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "phase3_contract.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    return path


def export_phase3_bundle(
    config: dict,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    frozen_selection_path: str | Path | None = None,
    seed_sources: list[dict] | None = None,
) -> Path:
    """Create the Phase3 hand-off JSON and return its path.

    The function performs no prediction, feature extraction, training, or weight
    copy.  It validates the selected checkpoint contract before exporting.
    """
    contract = build_phase3_contract(
        config, checkpoint_path, frozen_selection_path=frozen_selection_path
    )
    contract["seed_sources"] = list(seed_sources or [])
    return write_phase3_contract(contract, output_dir)


def load_phase3_contract(contract_path: str | Path) -> dict:
    contract = _read_json(contract_path)
    if contract.get("schema_version") != CONTRACT_VERSION:
        raise ConfigError("不支持的 Phase3 导出合同版本")
    checkpoint = contract.get("checkpoint") or {}
    _require_primary_target(
        str((contract.get("model") or {}).get("name")),
        str(checkpoint.get("arm")),
        int(checkpoint.get("seed", -1)),
    )
    if checkpoint.get("kind") != "formal" or checkpoint.get("weights_copied") is not False:
        raise ConfigError("Phase3 合同不是 formal checkpoint 的无权重引用")
    return contract


def predict_from_phase3_contract(
    contract_path: str | Path, table: PointTable, *, device: str | None = "cuda"
) -> np.ndarray:
    """The callable Phase3 inference entrypoint.

    It re-loads the referenced formal checkpoint with strict state-dict checks and
    calls the same prediction implementation used by Phase2 external evaluation.
    """
    contract = load_phase3_contract(contract_path)
    checkpoint = contract["checkpoint"]
    inference_config = contract.get("inference_config")
    if not isinstance(inference_config, dict):
        raise ConfigError("Phase3 合同缺少冻结 inference_config")
    model, metadata = load_formal_model(
        inference_config, checkpoint["path"], arm=checkpoint["arm"], seed=int(checkpoint["seed"]), device=device
    )
    expected = contract.get("dynamic_dimensions") or {}
    if int(metadata["input_dim"]) != int(expected.get("input_dim", -1)):
        raise ConfigError("Phase3 合同输入维度与 checkpoint 不一致")
    return predict_table(inference_config, arm=checkpoint["arm"], model=model, table=table, device=device)
