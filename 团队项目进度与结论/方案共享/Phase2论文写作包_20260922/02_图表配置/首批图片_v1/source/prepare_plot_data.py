"""生成图P2-3、图P2-4的制图数据；仅读取既有登记与运行汇总，不重算预测指标。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


TOLERANCE = 1e-9
MAIN_ID = "phase2_fullfov_hpo_v1"
DENSITY_ID = "phase2_spatial_density_comparison_v001_20260904"
MAIN_METRICS = {
    "patient_macro_pathway_pcc": "macro_pcc",
    "pooled_pcc": "pooled_pcc",
}
DENSITY_METRICS = {
    "patient_macro_pathway_pcc": "macro_pcc",
    "pooled_flattened_pcc": "pooled_pcc",
}
MODELS = ("uni", "uni2h", "virchow2")
MAIN_ARMS = ("point", "no_b", "spatial")
DENSITY_ARMS = ("dense", "stride_2x2", "random_equal")
SPLITS = ("internal_val", "external_test")
SELECTORS = ("equal_updates", "equal_50_epochs")
SEEDS = (45, 46, 47)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def experiment(registry: dict, experiment_id: str) -> dict:
    matches = [item for item in registry["experiments"] if item["id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Registry 未唯一包含 {experiment_id}")
    return matches[0]


def check_accepted(item: dict) -> None:
    if item["status"] != "done" or item["evidence_status"] != "accepted":
        raise ValueError(f"{item['id']} 不是当前已接纳结果")


def main_rows(item: dict) -> tuple[list[dict], list[dict]]:
    summary, per_seed = [], []
    rows = [row for row in item["results_summary"] if row["stage"] == "final-ablation"]
    if len(rows) != 9:
        raise ValueError(f"主实验最终消融组合数应为9，实际为{len(rows)}")
    for row in rows:
        if row["model"] not in MODELS or row["arm"] not in MAIN_ARMS:
            raise ValueError(f"主实验出现非目标组合: {row['model']}/{row['arm']}")
        for split in SPLITS:
            block = row[split]
            if tuple(block["seeds"]) != SEEDS or block["n_seeds"] != 3:
                raise ValueError(f"主实验种子不符合45/46/47: {row['model']}/{row['arm']}/{split}")
            values_by_seed = {entry["seed"]: entry for entry in block["per_seed"]}
            for registry_metric, plot_metric in MAIN_METRICS.items():
                summary.append({
                    "model": row["model"], "arm": row["arm"], "split": split,
                    "metric": plot_metric, "mean": block["mean"][registry_metric],
                    "sample_sd": block["sample_sd"][registry_metric], "n_seeds": block["n_seeds"],
                })
                for seed in SEEDS:
                    per_seed.append({
                        "model": row["model"], "arm": row["arm"], "split": split,
                        "metric": plot_metric, "seed": seed,
                        "value": values_by_seed[seed][registry_metric],
                    })
    return summary, per_seed


def density_rows(item: dict, result_root: Path) -> tuple[list[dict], list[dict]]:
    points = {"dense": 9472, "stride_2x2": 2383, "random_equal": 2383}
    values: dict[tuple[str, str, str, str], list[float]] = {}
    per_seed = []
    for arm in DENSITY_ARMS:
        for seed in SEEDS:
            source = result_root / "runs" / f"{arm}__seed{seed}" / "run_summary.json"
            run = read_json(source)
            if run["status"] != "complete" or run["mode"] != "formal":
                raise ValueError(f"运行不是正式完成状态: {source}")
            for selector in SELECTORS:
                for split in SPLITS:
                    evaluation = run["evaluations"][selector][split]
                    for source_metric, plot_metric in DENSITY_METRICS.items():
                        value = evaluation[source_metric]
                        key = (arm, selector, split, plot_metric)
                        values.setdefault(key, []).append(value)
                        per_seed.append({
                            "arm": arm, "selector": selector, "split": split,
                            "metric": plot_metric, "seed": seed, "value": value,
                        })
    summary = []
    for arm in DENSITY_ARMS:
        for selector in SELECTORS:
            for split in SPLITS:
                for metric in DENSITY_METRICS.values():
                    series = values[(arm, selector, split, metric)]
                    if len(series) != 3:
                        raise ValueError(f"密度消融种子数不是3: {arm}/{selector}/{split}/{metric}")
                    summary.append({
                        "arm": arm, "selector": selector, "split": split,
                        "metric": metric, "mean": mean(series),
                        "sample_sd": stdev(series), "n_seeds": len(series),
                        "train_points": points[arm],
                    })
    return summary, per_seed


def graph_rows(result_root: Path) -> list[dict]:
    source = result_root / "analysis" / "review_training_graph_stats.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for arm in DENSITY_ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if {int(row["seed"]) for row in selected} != set(SEEDS):
            raise ValueError(f"图统计缺少种子: {arm}")
        output.append({
            "arm": arm, "n_seeds": len(selected),
            "train_points": int(selected[0]["n_nodes"]),
            "mean_degree": mean(float(row["mean_degree"]) for row in selected),
            "isolated_rate": mean(float(row["isolated_rate"]) for row in selected),
            "a_self_mean": mean(float(row["a_self_mean"]) for row in selected),
            "neighbor_weight_mean_b": mean(float(row["neighbor_weight_mean_b"]) for row in selected),
        })
    return output


def index_rows(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, dict]:
    return {tuple(row[field] for field in fields): row for row in rows}


def compare_main_registry_to_runtime(item: dict, result_root: Path) -> dict:
    runtime_rows = []
    for filename in ("local_metrics.json", "external_metrics.json"):
        report = read_json(result_root / "analysis" / filename)
        runtime_rows.extend(row for row in report["recipe_groups"] if row["stage"] == "final-ablation")
    runtime = index_rows(runtime_rows, ("model", "arm", "split"))
    differences = []
    for row in item["results_summary"]:
        if row["stage"] != "final-ablation":
            continue
        for split in SPLITS:
            source = runtime[(row["model"], row["arm"], split)]["seed_summary"]
            for metric in MAIN_METRICS:
                differences.append(abs(row[split]["mean"][metric] - source[metric]["mean"]))
                differences.append(abs(row[split]["sample_sd"][metric] - source[metric]["sd"]))
    maximum = max(differences, default=0.0)
    if maximum > TOLERANCE:
        raise ValueError(f"主实验 Registry 与运行汇总最大差异 {maximum} 超过 {TOLERANCE}")
    return {"comparison": "Registry results_summary vs local_metrics/external_metrics recipe_groups", "max_abs_difference": maximum, "within_tolerance": True}


def compare_density_registry_to_runs(item: dict, summary: list[dict]) -> dict:
    lookup = index_rows(summary, ("arm", "selector", "split", "metric"))
    differences = []
    for row in item["results_summary"]:
        if row["selector"] != "equal_updates":
            continue
        for registry_metric, plot_metric in DENSITY_METRICS.items():
            observed = lookup[(row["arm"], "equal_updates", row["split"], plot_metric)]
            differences.append(abs(row["mean"][registry_metric] - observed["mean"]))
            differences.append(abs(row["sample_sd"][registry_metric] - observed["sample_sd"]))
    maximum = max(differences, default=0.0)
    if maximum > TOLERANCE:
        raise ValueError(f"密度消融 Registry 与运行汇总最大差异 {maximum} 超过 {TOLERANCE}")
    return {"comparison": "Registry equal_updates results_summary vs nine run_summary.json files", "max_abs_difference": maximum, "within_tolerance": True}


def compare_evidence_snapshot(registry: dict, evidence: dict) -> dict:
    differences = []
    for experiment_id in (MAIN_ID, DENSITY_ID):
        current = experiment(registry, experiment_id)
        snapshot = experiment(evidence, experiment_id)
        for key in ("status", "evidence_status", "accepted_at"):
            if current.get(key) != snapshot.get(key):
                raise ValueError(f"证据摘录与当前 Registry 状态不同: {experiment_id}/{key}")
        differences.append({"experiment_id": experiment_id, "status_matches": True})
    return {"comparison": "当前 Registry vs 写作包已接纳结果摘录", "experiments": differences, "within_tolerance": True}


def run() -> None:
    package_root = Path(__file__).resolve().parents[3]
    project_root = package_root.parents[2]
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    registry_path = project_root / "experiments" / "experiment_registry.json"
    evidence_path = package_root / "04_证据与对应" / "已接纳结果摘录.json"
    registry, evidence = read_json(registry_path), read_json(evidence_path)
    main, density = experiment(registry, MAIN_ID), experiment(registry, DENSITY_ID)
    check_accepted(main)
    check_accepted(density)
    main_summary, main_per_seed = main_rows(main)
    density_root = project_root / "experiments" / "results" / "phase2_task3_fullfov_density_ablation_v1" / "formal_20260919_151714"
    density_summary, density_per_seed = density_rows(density, density_root)
    density_graph = graph_rows(density_root)
    main_root = project_root / "experiments" / "results" / "phase2_fullfov_hpo_v1" / "20260916_231813_101_7b1b9a79"
    validations = {
        "evidence_snapshot": compare_evidence_snapshot(registry, evidence),
        "main_registry_vs_runtime": compare_main_registry_to_runtime(main, main_root),
        "density_registry_vs_runtime": compare_density_registry_to_runs(density, density_summary),
    }
    write_csv(data_dir / "main_summary.csv", ["model", "arm", "split", "metric", "mean", "sample_sd", "n_seeds"], main_summary)
    write_csv(data_dir / "main_per_seed.csv", ["model", "arm", "split", "metric", "seed", "value"], main_per_seed)
    write_csv(data_dir / "density_summary.csv", ["arm", "selector", "split", "metric", "mean", "sample_sd", "n_seeds", "train_points"], density_summary)
    write_csv(data_dir / "density_per_seed.csv", ["arm", "selector", "split", "metric", "seed", "value"], density_per_seed)
    write_csv(data_dir / "density_graph.csv", ["arm", "n_seeds", "train_points", "mean_degree", "isolated_rate", "a_self_mean", "neighbor_weight_mean_b"], density_graph)
    manifest = {
        "purpose": "图P2-3与图P2-4制图辅助数据；不是正式实验包或新实验结果。",
        "generated_by": str(Path(__file__).resolve()),
        "hashes_computed": False,
        "accepted_experiments": {
            MAIN_ID: {"accepted_at": main["accepted_at"], "result_root": main["result_root"], "filter": "stage=final-ablation; models=uni,uni2h,virchow2; arms=point,no_b,spatial; seeds=45,46,47"},
            DENSITY_ID: {"accepted_at": density["accepted_at"], "result_root": str(density_root.relative_to(project_root)), "filter": "arms=dense,stride_2x2,random_equal; selectors=equal_updates,equal_50_epochs; seeds=45,46,47"},
        },
        "sources": {
            "registry": str(registry_path),
            "writing_package_evidence_snapshot": str(evidence_path),
            "main_runtime_summaries": [str(main_root / "analysis" / "local_metrics.json"), str(main_root / "analysis" / "external_metrics.json")],
            "density_run_summaries": str(density_root / "runs" / "<arm>__seed<45|46|47>" / "run_summary.json"),
            "density_graph_statistics": str(density_root / "analysis" / "review_training_graph_stats.csv"),
        },
        "statistics": {
            "mean": "45/46/47三种子的算术平均；直接读取已登记值或既有run_summary指标。",
            "sample_sd": "45/46/47三种子的样本标准差，ddof=1；不从预测文件复算。",
            "metric_mapping": {"macro_pcc": "patient_macro_pathway_pcc", "pooled_pcc": "pooled_pcc（主实验）或pooled_flattened_pcc（密度消融）"},
            "density_graph": "mean_degree为训练图平均邻居数；isolated_rate为无邻居节点比例；其余两列为既有图权重统计的三种子均值。",
        },
        "validations": validations,
        "outputs": {"main_summary.csv": len(main_summary), "main_per_seed.csv": len(main_per_seed), "density_summary.csv": len(density_summary), "density_per_seed.csv": len(density_per_seed), "density_graph.csv": len(density_graph)},
    }
    with (data_dir / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    run()
