"""Merge new spatial runs with accepted baselines. Never call this a confirmed result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baseline_reference import load_baseline_reference
from dispatch import planned_external_keys
from errors import ConfigError
from local_report import _task_fields, evaluate_prediction_file, write_local_report
from run_io import write_json


PAIRED_COMPARISONS = (
    ("hoptimus0", "uni2h", "same_dim_1536_primary"),
    ("hoptimus1", "uni2h", "same_dim_1536_primary"),
    ("phikonv2", "uni", "same_dim_1024"),
    ("hoptimus0", "virchow2", "vs_strong_pathology_baseline"),
    ("hoptimus1", "virchow2", "vs_strong_pathology_baseline"),
    ("phikonv2", "virchow2", "vs_strong_pathology_baseline"),
    ("hoptimus0", "uni2h", "vs_current_deployed_baseline"),
    ("hoptimus1", "uni2h", "vs_current_deployed_baseline"),
    ("phikonv2", "uni2h", "vs_current_deployed_baseline"),
)
METRICS = ("patient_macro_pathway_pcc", "pooled_pcc", "pooled_z_mse")


def coverage_records(expected: list[tuple[str, int, str]], actual: list[dict]) -> dict:
    """Standalone copy of the audit helper so src never imports tests."""
    expected_set = list(expected)
    indexed: dict[tuple[str, int, str], list[dict]] = {}
    for item in actual:
        key = (str(item.get("model")), int(item.get("seed")), str(item.get("split")))
        indexed.setdefault(key, []).append(item)
    missing, extras, failed, duplicate = [], [], [], []
    for key in expected_set:
        rows = indexed.get(key) or []
        if not rows:
            missing.append({"model": key[0], "seed": key[1], "split": key[2]})
        elif len(rows) > 1:
            duplicate.append({"model": key[0], "seed": key[1], "split": key[2], "n": len(rows)})
        elif rows[0].get("status") != "succeeded":
            failed.append({**{"model": key[0], "seed": key[1], "split": key[2]}, "status": rows[0].get("status")})
    extra = [key for key in indexed if key not in set(expected_set)]
    extras.extend({"model": k[0], "seed": k[1], "split": k[2]} for k in extra)
    complete = not (missing or failed or duplicate)
    return {
        "complete": complete,
        "status": "PASS" if complete and not extras else "WARN",
        "missing": missing,
        "failed": failed,
        "duplicate": duplicate,
        "extra": extras,
    }


def _mean_sd(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(values), "mean": float(np.mean(values)), "sd": None if len(values) < 2 else float(np.std(values, ddof=1))}


def _group_model_seeds(records: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = {}
    for item in records:
        if item.get("model") is None or item.get("seed") is None:
            continue
        by_model.setdefault(str(item["model"]), []).append(item)
    for model, items in by_model.items():
        groups[model] = {
            "seeds": {int(item["seed"]): item for item in items},
            "summary": {metric: _mean_sd([float(item[metric]) for item in items if item.get(metric) is not None]) for metric in METRICS},
        }
    return groups


def _paired_deltas(new_groups: dict, baseline_groups: dict) -> list[dict]:
    rows = []
    seen = set()
    for new_model, base_model, reason in PAIRED_COMPARISONS:
        key = (new_model, base_model, reason)
        if key in seen:
            continue
        seen.add(key)
        per_seed = []
        for seed in (45, 46, 47):
            left = (new_groups.get(new_model) or {}).get("seeds", {}).get(seed)
            right = (baseline_groups.get(base_model) or {}).get("seeds", {}).get(seed)
            if not left or not right:
                per_seed.append({"seed": seed, "status": "missing"})
                continue
            deltas = {}
            for metric in METRICS:
                lv, rv = left.get(metric), right.get(metric)
                deltas[metric] = None if lv is None or rv is None else float(lv) - float(rv)
            per_seed.append({"seed": seed, "status": "compared", "delta": deltas})
        compared = [row for row in per_seed if row["status"] == "compared"]
        summary = {}
        for metric in METRICS:
            values = [row["delta"][metric] for row in compared if row.get("delta", {}).get(metric) is not None]
            summary[metric] = _mean_sd(values)
        rows.append({
            "new_model": new_model,
            "baseline_model": base_model,
            "reason": reason,
            "per_seed": per_seed,
            "mean_delta": summary,
            "complete": len(compared) == 3,
        })
    return rows


def _baseline_split_records(baseline: dict, split: str) -> list[dict]:
    records = []
    key = "accepted_internal_metrics" if split == "internal_val" else "accepted_external_metrics"
    for task in baseline["tasks"]:
        metrics = task.get(key) or {}
        records.append({
            "task_id": task["task_id"],
            "model": task["model"],
            "seed": int(task["seed"]),
            "split": split,
            "status": "succeeded" if metrics.get("patient_macro_pathway_pcc") is not None else "missing",
            "patient_macro_pathway_pcc": metrics.get("patient_macro_pathway_pcc"),
            "pooled_pcc": metrics.get("pooled_pcc"),
            "pooled_z_mse": metrics.get("pooled_z_mse"),
            "source": metrics.get("source") or task.get("internal_metrics" if split == "internal_val" else "external_metrics"),
            "origin": "accepted_baseline_not_retrained",
        })
    return records


def _discover_split_files(batch: Path, pattern: str) -> list[Path]:
    return sorted(batch.glob(pattern))


def analyze_local(config: dict, *, run_dir: Path) -> dict:
    batch = run_dir.parent
    baseline = load_baseline_reference(config["inputs"]["baseline_reference_manifest"], require_prediction_files=False)
    internal_files = [path for run in batch.glob("train-spatial_*") if run.is_dir() for path in run.rglob("internal_best.npz")]
    external_files = [path for run in batch.glob("external-eval_*") if run.is_dir() for path in run.glob("raw/*.npz")]
    new_internal = []
    skipped = []
    for path in internal_files:
        try:
            report = evaluate_prediction_file(path)
            _task_fields(report)
        except Exception as exc:
            skipped.append({"source": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        task = report.get("task") or {}
        new_internal.append({
            "task_id": task.get("task_id") or path.parent.parent.name,
            "model": task.get("model"),
            "seed": task.get("seed"),
            "split": "internal_val",
            "status": "succeeded" if report.get("status") == "scored" else report.get("status"),
            "patient_macro_pathway_pcc": report.get("patient_macro_pathway_pcc"),
            "pooled_pcc": report.get("pooled_pcc"),
            "pooled_z_mse": report.get("pooled_z_mse"),
            "source": report.get("source"),
            "origin": "new_run",
        })
    new_external = []
    for path in external_files:
        try:
            report = evaluate_prediction_file(path)
            _task_fields(report)
        except Exception as exc:
            skipped.append({"source": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        task = report.get("task") or {}
        new_external.append({
            "task_id": task.get("task_id") or path.stem,
            "model": task.get("model"),
            "seed": task.get("seed"),
            "split": "external_test",
            "status": "succeeded" if report.get("status") == "scored" else report.get("status"),
            "patient_macro_pathway_pcc": report.get("patient_macro_pathway_pcc"),
            "pooled_pcc": report.get("pooled_pcc"),
            "pooled_z_mse": report.get("pooled_z_mse"),
            "source": report.get("source"),
            "origin": "new_run",
        })
    expected_new = [(model, seed, split) for model in ("hoptimus0", "hoptimus1", "phikonv2") for seed in (45, 46, 47) for split in ("internal_val", "external_test")]
    expected_base = [(model, seed, split) for model in ("uni2h", "uni", "virchow2") for seed in (45, 46, 47) for split in ("internal_val", "external_test")]
    baseline_internal = _baseline_split_records(baseline, "internal_val")
    baseline_external = _baseline_split_records(baseline, "external_test")
    new_cov = coverage_records(expected_new, new_internal + new_external)
    base_cov = coverage_records(expected_base, baseline_internal + baseline_external)
    complete = bool(new_cov["complete"] and base_cov["complete"])
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if external_files:
        try:
            write_local_report(external_files, analysis_dir / "new_external_metrics.json", skipped_files=skipped)
        except FileExistsError:
            pass
    if internal_files:
        try:
            write_local_report(internal_files, analysis_dir / "new_internal_metrics.json", skipped_files=skipped)
        except FileExistsError:
            pass
    payload = {
        "status": "completed" if complete else "partial",
        "accepted_conclusion": False,
        "complete_batch2_comparison": complete,
        "title_must_say_incomplete": not complete,
        "new_task_ids_expected": planned_external_keys(),
        "coverage": {"new": new_cov, "baseline": base_cov},
        "internal_val": {
            "new": _group_model_seeds(new_internal),
            "baseline_accepted": _group_model_seeds(baseline_internal),
            "paired_deltas": _paired_deltas(_group_model_seeds(new_internal), _group_model_seeds(baseline_internal)),
        },
        "external_xzy": {
            "new": _group_model_seeds(new_external),
            "baseline_accepted": _group_model_seeds(baseline_external),
            "paired_deltas": _paired_deltas(_group_model_seeds(new_external), _group_model_seeds(baseline_external)),
            "single_historical_patient": True,
        },
        "metric_definitions": {
            "patient_macro_pathway_pcc": "逐患者逐通路 PCC 后等权平均",
            "pooled_pcc": "全部点位×通路 z 分数展平后计算一次 PCC",
            "pooled_z_mse": "全部点位×通路 z 分数的均方误差",
            "not_interchangeable": True,
        },
        "interpretation_limits": [
            "official_expected_mpp=0.5 与当前 patch 物理尺度关系尚未核实",
            "比较对象是编码器及其配套输入归一化，不是强制统一像素归一化后的纯权重比较",
            "特征维数不同导致头部参数量不同",
            "图像特征参与图边权，替换编码器也会改变图权重",
            "H-optimus-1 有人工访问审批和更严格非商业许可",
            "外部 XZY 只有一个历史患者",
            "所有新模型共享冻结 spatial-11，不代表各编码器独立调优上限",
            "结果未经 Registry 接纳前只能称为待登记或探索结果",
        ],
        "skipped": skipped,
        "n_new_internal_files": len(internal_files),
        "n_new_external_files": len(external_files),
    }
    write_json(analysis_dir / "batch2_comparison.json", payload)
    return {
        "status": payload["status"],
        "complete_batch2_comparison": complete,
        "analysis_path": str((analysis_dir / "batch2_comparison.json").resolve()),
        "missing_new": new_cov["missing"],
        "missing_baseline": base_cov["missing"],
        "failed_new": new_cov["failed"],
        "accepted_conclusion": False,
        "computed_locally": True,
    }


def main() -> int:
    import argparse
    from config import load_config
    parser = argparse.ArgumentParser(description="合并新运行与已接纳基线；不训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_local(load_config(args.config), run_dir=args.run_dir.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
