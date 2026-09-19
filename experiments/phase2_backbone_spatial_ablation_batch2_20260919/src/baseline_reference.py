"""Read-only accepted full-FOV spatial baselines. Never retrain them."""

from __future__ import annotations

import json
import math
from pathlib import Path

from errors import BaselineReferenceError


EXPECTED_MODELS = ("uni2h", "uni", "virchow2")
EXPECTED_SEEDS = (45, 46, 47)
EXPECTED_ARM = "spatial"
EXPECTED_PROTOCOL = "full_fov_224_bicubic_v1"
EXPECTED_SOURCE_EXPERIMENT = "phase2_fullfov_hpo_v1"
EXPECTED_SOURCE_BATCH = "20260916_231813_101_7b1b9a79"
EXPECTED_RESULT_ID = "phase2-fullfov-hpo-v1-20260916-231813-result"
METRIC_FIELDS = ("patient_macro_pathway_pcc", "pooled_pcc", "pooled_z_mse")


def planned_baseline_task_ids() -> list[str]:
    return [
        f"final-ablation__{model}__{EXPECTED_PROTOCOL}__spatial__{seed}__frozen_spatial"
        for model in EXPECTED_MODELS
        for seed in EXPECTED_SEEDS
    ]


def _read(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineReferenceError(f"无法读取基线清单 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineReferenceError("基线清单根必须是 JSON object")
    return payload


def _status(path: str | None) -> dict:
    if not path:
        return {"path": None, "exists": False, "kind": "missing"}
    item = Path(path)
    return {
        "path": str(item),
        "exists": item.exists(),
        "kind": "directory" if item.is_dir() else "file" if item.is_file() else "missing",
    }


def _validate_embedded_metrics(task_id: str, item: dict, field: str, split: str) -> None:
    metrics = item.get(field)
    if not isinstance(metrics, dict):
        raise BaselineReferenceError(f"{task_id} 缺少包内已接纳指标: {field}")
    if metrics.get("split") != split:
        raise BaselineReferenceError(f"{task_id} {field}.split 不匹配: {metrics.get('split')}")
    for name in METRIC_FIELDS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise BaselineReferenceError(f"{task_id} {field}.{name} 不是有限数值")
    n_points = metrics.get("n_points")
    if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 1:
        raise BaselineReferenceError(f"{task_id} {field}.n_points 必须是正整数")


def load_baseline_reference(
    manifest_path: str | Path,
    *,
    require_prediction_files: bool = False,
    require_source_files: bool = False,
) -> dict:
    path = Path(manifest_path).resolve()
    payload = _read(path)
    if payload.get("schema_version") != "1.0":
        raise BaselineReferenceError("基线清单 schema_version 必须为 1.0")
    if payload.get("source_experiment") != EXPECTED_SOURCE_EXPERIMENT:
        raise BaselineReferenceError("基线必须来自 phase2_fullfov_hpo_v1")
    if payload.get("source_batch") != EXPECTED_SOURCE_BATCH:
        raise BaselineReferenceError("基线批次必须为已接纳的 20260916_231813_101_7b1b9a79")
    if payload.get("accepted_result_id") != EXPECTED_RESULT_ID:
        raise BaselineReferenceError("基线 accepted result ID 不匹配")
    if payload.get("evidence_status") != "accepted":
        raise BaselineReferenceError("基线 evidence_status 必须为 accepted")
    tasks = payload.get("tasks") or []
    planned = planned_baseline_task_ids()
    ids = [str(item.get("task_id")) for item in tasks]
    if len(ids) != len(set(ids)):
        raise BaselineReferenceError(f"基线任务 ID 重复: {sorted({i for i in ids if ids.count(i) > 1})}")
    if sorted(ids) != sorted(planned):
        missing = [item for item in planned if item not in ids]
        extra = [item for item in ids if item not in planned]
        raise BaselineReferenceError(f"基线任务集合不完整或含额外项: missing={missing}, extra={extra}")
    seeds_by_model: dict[str, set[int]] = {name: set() for name in EXPECTED_MODELS}
    resolved = []
    missing_files: list[str] = []
    for item in tasks:
        task_id = str(item["task_id"])
        model = str(item.get("model"))
        arm = str(item.get("arm"))
        seed = int(item.get("seed"))
        protocol = str(item.get("protocol"))
        if arm != EXPECTED_ARM:
            raise BaselineReferenceError(f"{task_id} 模型臂不是 spatial: {arm}")
        if protocol != EXPECTED_PROTOCOL:
            raise BaselineReferenceError(f"{task_id} 协议不是全视野: {protocol}")
        if model not in EXPECTED_MODELS:
            raise BaselineReferenceError(f"{task_id} 不是三个已接纳基线编码器之一")
        seeds_by_model[model].add(seed)
        _validate_embedded_metrics(task_id, item, "accepted_internal_metrics", "internal_val")
        _validate_embedded_metrics(task_id, item, "accepted_external_metrics", "external_test")
        files = {
            "task_record": _status(item.get("task_record")),
            "internal_metrics": _status(item.get("internal_metrics")),
            "internal_predictions": _status(item.get("internal_predictions")),
            "external_predictions": _status(item.get("external_predictions")),
            "external_metrics": _status(item.get("external_metrics")),
            "formal_checkpoint_metadata": _status(item.get("formal_checkpoint_metadata")),
        }
        required = ["task_record", "internal_metrics", "formal_checkpoint_metadata"] if require_source_files else []
        if require_prediction_files:
            required.extend(["internal_predictions", "external_predictions"])
        for name in required:
            if not files[name]["exists"]:
                missing_files.append(f"{task_id}:{name}={files[name]['path']}")
        if item.get("evidence_status") != "accepted":
            raise BaselineReferenceError(f"{task_id} evidence_status 不是 accepted")
        resolved.append({**item, "files": files})
    for model, seeds in seeds_by_model.items():
        if seeds != set(EXPECTED_SEEDS):
            raise BaselineReferenceError(f"{model} 种子不全: {sorted(seeds)}")
    if missing_files:
        raise BaselineReferenceError("基线路径缺失，拒绝静默跳过: " + "; ".join(missing_files))
    return {
        "source_experiment": payload["source_experiment"],
        "source_batch": payload["source_batch"],
        "accepted_result_id": payload["accepted_result_id"],
        "evidence_status": "accepted",
        "n_tasks": len(resolved),
        "tasks": resolved,
        "do_not_retrain": True,
        "embedded_metrics_valid": True,
        "source_files_required": bool(require_source_files or require_prediction_files),
    }
