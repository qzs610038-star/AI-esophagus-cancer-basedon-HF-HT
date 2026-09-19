"""Local-only summaries for returned Phase2 prediction NPZ artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from errors import ConfigError
from selection import patient_macro_metrics, pooled_z_mse


REPORT_VERSION = "phase2_local_prediction_report_v1"


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _pooled_pcc(pred: np.ndarray, target: np.ndarray) -> float | None:
    x = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return None if denom == 0.0 else float(np.sum(x * y) / denom)


def _scalar_bool(value: Any, *, name: str) -> bool:
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ConfigError(f"{name} 必须是一个标量")
    return bool(flat[0])


def _load_prediction(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=True) as artifact:
        required = {"pred_z", "target_z", "target_available", "patient_id", "slide_id", "spot_id", "pathway_names", "mean", "std"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ConfigError(f"预测文件缺少字段 {missing}: {source}")
        values = {name: artifact[name] for name in artifact.files}
    pred = np.asarray(values["pred_z"], dtype=np.float64)
    if pred.ndim != 2 or not pred.size or not np.isfinite(pred).all():
        raise ConfigError(f"pred_z 必须为非空有限二维数组: {source}")
    n, p = pred.shape
    for name in ("patient_id", "slide_id", "spot_id"):
        if np.asarray(values[name]).reshape(-1).size != n:
            raise ConfigError(f"{name} 与 pred_z 行数不一致: {source}")
    pathways = [str(v) for v in np.asarray(values["pathway_names"]).reshape(-1).tolist()]
    if len(pathways) != p or len(set(pathways)) != p:
        raise ConfigError(f"通路顺序非法或重复: {source}")
    for name in ("mean", "std"):
        vector = np.asarray(values[name], dtype=np.float64).reshape(-1)
        if vector.size != p or not np.isfinite(vector).all():
            raise ConfigError(f"{name} 与通路数不一致或非有限: {source}")
    if np.any(np.asarray(values["std"], dtype=np.float64) <= 0):
        raise ConfigError(f"std 必须为正数: {source}")
    return {"source": str(source.resolve()), "values": values, "pred_z": pred, "n": n, "p": p, "pathway_names": pathways}


def evaluate_prediction_file(path: str | Path) -> dict[str, Any]:
    """Compute report metrics for one returned prediction artifact.

    ``patient_macro_pathway_pcc`` is the measured patient--pathway equal-weight
    PCC.  ``pooled_pcc`` is one Pearson correlation after flattening all points
    and pathways; the two values are intentionally reported separately.
    """
    loaded = _load_prediction(path)
    values, pred = loaded["values"], loaded["pred_z"]
    available = _scalar_bool(values["target_available"], name="target_available")
    base: dict[str, Any] = {
        "source": loaded["source"], "status": "unscored" if not available else "scored",
        "n_points": loaded["n"], "n_pathways": loaded["p"], "pathway_order": loaded["pathway_names"],
        "target_available": available,
        "split": str(np.asarray(values["split"]).reshape(-1)[0]) if "split" in values else None,
        "checkpoint_metadata": None,
    }
    metadata = values.get("checkpoint_metadata")
    if metadata is not None:
        flat = np.asarray(metadata, dtype=object).reshape(-1)
        if flat.size == 1 and isinstance(flat[0], dict):
            base["checkpoint_metadata"] = flat[0]
    if not available:
        return base
    target = np.asarray(values["target_z"], dtype=np.float64)
    if target.shape != pred.shape or not np.isfinite(target).all():
        raise ConfigError("target_available=true 时 target_z 必须与 pred_z 对齐且有限")
    patient_ids = [str(v) for v in np.asarray(values["patient_id"]).reshape(-1).tolist()]
    metrics = patient_macro_metrics(pred, target, patient_ids, pathway_names=loaded["pathway_names"])
    patient_mse = [record["z_mse_mean"] for record in metrics.per_patient.values() if np.isfinite(record["z_mse_mean"])]
    base.update({
        "patient_macro_pathway_pcc": _finite_or_none(metrics.patient_macro_pathway_pcc_measured),
        "patient_macro_pathway_pcc_selection_penalized": _finite_or_none(metrics.patient_macro_pathway_pcc),
        "pooled_pcc": _pooled_pcc(pred, target),
        "pooled_z_mse": _finite_or_none(pooled_z_mse(pred, target)),
        "patient_macro_z_mse": _finite_or_none(float(np.mean(patient_mse))) if patient_mse else None,
        "n_patients": metrics.n_patients,
        "n_valid_patient_pathway": metrics.n_valid_patient_pathway,
        "n_excluded_patient_pathway": metrics.n_excluded_patient_pathway,
        "n_constant_prediction_penalized": metrics.n_constant_prediction_penalized,
        "computable": metrics.computable,
        "incomputable_reason": metrics.incomputable_reason,
        "exclusions": metrics.exclusions,
    })
    return base


def _mean_sd(records: Iterable[dict], metric: str) -> dict[str, float | int | None]:
    values = [float(item[metric]) for item in records if item.get(metric) is not None and np.isfinite(item[metric])]
    if not values:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(values), "mean": float(np.mean(values)), "sd": None if len(values) < 2 else float(np.std(values, ddof=1))}


def _task_fields(report: dict) -> dict[str, Any]:
    """Normalise task identity recorded with a prediction artifact.

    ``task_id`` remains attached to every run.  The aggregate key omits its seed
    segment so the three planned seeds form one recipe group instead of being
    folded together with other models or experiment arms.
    """
    metadata = report.get("checkpoint_metadata") or {}
    task_id = str(metadata.get("task_id") or "")
    parts = task_id.split("__") if task_id else []
    parsed = {
        "stage": parts[0] if len(parts) >= 1 else None,
        "model": parts[1] if len(parts) >= 2 else None,
        "protocol": parts[2] if len(parts) >= 3 else None,
        "arm": parts[3] if len(parts) >= 4 else None,
        "seed": parts[4] if len(parts) >= 5 else None,
        "recipe": parts[5] if len(parts) >= 6 else None,
        "candidate_id": "__".join(parts[6:]) if len(parts) >= 7 else None,
    }
    def field(name: str) -> Any:
        return metadata.get(name, parsed[name])
    seed = field("seed")
    try:
        seed = None if seed is None or seed == "" else int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"预测 checkpoint_metadata.seed 非法: {seed!r}") from exc
    fields = {
        "task_id": task_id or None,
        "stage": field("stage"), "model": field("model"), "protocol": field("protocol"),
        "arm": field("arm"), "seed": seed, "recipe": field("recipe"),
        "candidate_id": field("candidate_id"),
        "split": metadata.get("split", report.get("split")),
    }
    report["task"] = fields
    return fields


def _recipe_group_key(fields: dict[str, Any], source: str) -> tuple[str, ...]:
    # Unknown task metadata cannot safely be pooled with a different artifact.
    required = ("stage", "model", "protocol", "arm", "recipe", "split")
    if any(fields.get(name) in (None, "") for name in required):
        return ("unclassified", source)
    return tuple(str(fields.get(name)) for name in required) + (str(fields.get("candidate_id") or ""),)


def _group_reports(reports: list[dict]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for report in reports:
        fields = _task_fields(report)
        groups.setdefault(_recipe_group_key(fields, report["source"]), []).append(report)
    output = []
    for key, items in sorted(groups.items()):
        first = items[0]["task"]
        scored = [item for item in items if item["status"] == "scored"]
        seeds = [item["task"]["seed"] for item in items if item["task"]["seed"] is not None]
        duplicate_seeds = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        output.append({
            "group_id": "__".join(key),
            "stage": first["stage"], "model": first["model"], "protocol": first["protocol"],
            "arm": first["arm"], "recipe": first["recipe"], "split": first["split"],
            "candidate_id": first["candidate_id"],
            "n_files": len(items), "n_scored_files": len(scored),
            "seeds": sorted(set(seeds)), "duplicate_seeds": duplicate_seeds,
            "seed_summary": {
                "patient_macro_pathway_pcc": _mean_sd(scored, "patient_macro_pathway_pcc"),
                "pooled_pcc": _mean_sd(scored, "pooled_pcc"),
                "pooled_z_mse": _mean_sd(scored, "pooled_z_mse"),
                "patient_macro_z_mse": _mean_sd(scored, "patient_macro_z_mse"),
            },
            "task_ids": [item["task"]["task_id"] for item in items],
        })
    return output


def summarize_prediction_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Aggregate each recipe separately; never pool the planned 51 predictors."""
    reports = [evaluate_prediction_file(path) for path in paths]
    if not reports:
        raise ConfigError("至少需要一个预测 NPZ 文件")
    reference_order = reports[0]["pathway_order"]
    for report in reports[1:]:
        if report["pathway_order"] != reference_order:
            raise ConfigError("预测文件通路顺序不一致，不能汇总")
    return {
        "schema_version": REPORT_VERSION,
        "computed_locally": True,
        "server_metric_generation": False,
        "accepted_conclusion": False,
        "n_files": len(reports),
        "n_scored_files": sum(report["status"] == "scored" for report in reports),
        "pathway_order": reference_order,
        "runs": reports,
        "recipe_groups": _group_reports(reports),
        "definitions": {
            "patient_macro_pathway_pcc": "逐患者逐通路 PCC 后等权平均；常量预测以测得 PCC 缺失报告。",
            "pooled_pcc": "全部点位×通路 z 分数展平后计算一次 PCC。",
            "pooled_z_mse": "全部点位×通路 z 分数的均方误差。",
        },
    }


def write_local_report(paths: Iterable[str | Path], output_path: str | Path, *, skipped_files: list[dict] | None = None) -> Path:
    """Write one derived local report.  The caller chooses a fresh output path."""
    report = summarize_prediction_files(paths)
    report["skipped_npz_files"] = list(skipped_files or [])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return output


def discover_prediction_files(prediction_dir: str | Path) -> tuple[list[Path], list[dict]]:
    """Recursively locate valid prediction artifacts, retaining skipped NPZ reasons."""
    root = Path(prediction_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    valid: list[Path] = []
    skipped: list[dict] = []
    for path in sorted(root.rglob("*.npz")):
        try:
            _load_prediction(path)
        except Exception as exc:
            skipped.append({"source": str(path.resolve()), "reason": f"{type(exc).__name__}: {exc}"})
        else:
            valid.append(path)
    if not valid:
        raise ConfigError(f"{root} 及其 raw 子目录中没有有效的预测 NPZ 文件")
    return valid, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总回传的 Phase2 预测 NPZ；只在本地计算派生指标")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="回传运行目录，可递归扫描 raw/")
    parser.add_argument("--output", type=Path, required=True, help="新的本地分析 JSON；拒绝覆盖")
    args = parser.parse_args(argv)
    try:
        paths, skipped = discover_prediction_files(args.prediction_dir)
        output = write_local_report(paths, args.output, skipped_files=skipped)
        print(json.dumps({"status": "completed", "output": str(output), "n_prediction_files": len(paths), "n_skipped_npz": len(skipped)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"local_report failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
