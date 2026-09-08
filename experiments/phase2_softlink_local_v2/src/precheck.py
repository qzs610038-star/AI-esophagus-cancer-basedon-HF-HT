"""Training-free relation and graph coverage checks. No optimizer is created or stepped."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import load_config, package_root
from data import (
    attach_features,
    attach_labels,
    iter_index_batches,
    load_feature_source_manifest,
    load_normalization_from_config,
    load_pathway_names,
    load_split_point_table,
)
from errors import NonFiniteDataError, SlideMappingMissingError
from graph import build_split_graph, edges_as_records, graph_stats
from relations import compute_distance_thresholds, relation_loss, support_from_config

import torch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def simulate_relation_coverage(
    table,
    config: dict,
    *,
    output_dir: Path | None = None,
) -> dict:
    train = table.subset(np.where(table.split == "train")[0].tolist())
    if train.labels_z is None:
        raise NonFiniteDataError("关系预检需要训练标签")
    train.require_finite()
    rel = config["relation"]
    thresholds = compute_distance_thresholds(
        train.labels_z,
        n_pairs=int(rel["threshold_pairs"]),
        seed=int(rel["threshold_seed"]),
        near_quantile=float(rel["near_quantile"]),
        far_quantile=float(rel["far_quantile"]),
        quantile_method=str(rel["quantile_method"]),
    )
    n_train = len(train)
    patient_ids = np.array([ident.patient_id for ident in train.identities], dtype=object)
    counts = {pid: int(np.sum(patient_ids == pid)) for pid in np.unique(patient_ids)}
    rng = np.random.Generator(np.random.PCG64(int(config["precheck"]["relation_seed"])))
    n_shuffles = int(config["precheck"]["relation_complete_shuffles"])
    batch_size = int(config["training"]["batch_size"])
    keep_last = bool(config["training"]["keep_last_batch"])
    orders = [rng.permutation(n_train) for _ in range(n_shuffles)]

    n_centers = 0
    n_valid = 0
    n_no_near = 0
    n_no_far = 0
    near_counts: list[int] = []
    far_counts: list[int] = []
    near_spans: list[float] = []
    q_entropy: list[float] = []
    q_norm_entropy: list[float] = []
    near_same = 0
    near_pairs = 0
    far_same = 0
    far_pairs = 0
    baseline_near_num = 0.0
    baseline_far_num = 0.0
    n_single_near = 0
    n_batches = 0
    per_patient = {
        pid: {"centers": 0, "valid": 0, "near": 0, "far": 0} for pid in counts
    }

    dummy_z = torch.zeros((1, int(config["model"]["relation_dim"])), dtype=torch.float32)
    for shuffle_i, order in enumerate(orders, start=1):
        for batch_index, batch_pos in iter_index_batches(order, batch_size, keep_last=keep_last):
            n_batches += 1
            labels = torch.as_tensor(train.labels_z[batch_pos], dtype=torch.float32)
            patients = [str(patient_ids[int(i)]) for i in batch_pos]
            support = support_from_config(
                train.labels_z[batch_pos],
                patient_ids=patients,
                fixed_indices=[int(i) for i in batch_pos],
                config=config,
                thresholds=thresholds,
                run_seed=int(config["precheck"]["relation_seed"]),
                epoch=shuffle_i,
                batch_index=batch_index,
            )
            _, stats = relation_loss(
                dummy_z.expand(len(batch_pos), -1),
                labels,
                support,
                tau_y=float(rel["tau_y"]),
                tau_z=float(rel["tau_z"]),
            )
            n_centers += int(stats["n_centers"])
            n_valid += int(stats["n_valid"])
            n_no_near += int(stats["n_invalid_no_near"])
            n_no_far += int(stats["n_invalid_no_far"])
            n_single_near += int(stats["n_single_near"])
            for local_i, global_i in enumerate(batch_pos.tolist()):
                pid = str(patient_ids[global_i])
                per_patient[pid]["centers"] += 1
                near = support.near_indices[local_i]
                far = support.far_indices[local_i]
                near_counts.append(len(near))
                far_counts.append(len(far))
                if support.valid[local_i]:
                    per_patient[pid]["valid"] += 1
                per_patient[pid]["near"] += len(near)
                per_patient[pid]["far"] += len(far)
                if near:
                    d_near = [float(support.distances[local_i, j]) for j in near]
                    near_spans.append(max(d_near) - min(d_near))
                baseline = (counts[pid] - 1) / max(n_train - 1, 1)
                for j in near:
                    near_pairs += 1
                    near_same += int(patients[j] == pid)
                    baseline_near_num += baseline
                for j in far:
                    far_pairs += 1
                    far_same += int(patients[j] == pid)
                    baseline_far_num += baseline
            if stats["n_valid"] > 0 and np.isfinite(stats["mean_q_entropy"]):
                q_entropy.append(float(stats["mean_q_entropy"]))
            if stats["n_valid"] > 0 and np.isfinite(stats["mean_q_normalized_entropy"]):
                q_norm_entropy.append(float(stats["mean_q_normalized_entropy"]))

    report = {
        "kind": "relation_precheck",
        "seed": int(config["precheck"]["relation_seed"]),
        "n_shuffles": n_shuffles,
        "n_train": n_train,
        "n_batches": n_batches,
        "expected_batches": n_shuffles * int(np.ceil(n_train / batch_size)) if keep_last else None,
        "q10": thresholds.q10,
        "q50": thresholds.q50,
        "threshold_pairs": thresholds.n_pairs,
        "threshold_seed": thresholds.seed,
        "distance_summary": {
            "mean": thresholds.distance_mean,
            "std": thresholds.distance_std,
            "min": thresholds.distance_min,
            "max": thresholds.distance_max,
        },
        "n_center_visits": n_centers,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_centers if n_centers else float("nan"),
        "n_invalid_no_near": n_no_near,
        "n_invalid_no_far": n_no_far,
        "n_single_near": n_single_near,
        "mean_near": float(np.mean(near_counts)) if near_counts else float("nan"),
        "mean_far": float(np.mean(far_counts)) if far_counts else float("nan"),
        "mean_near_distance_span": float(np.mean(near_spans)) if near_spans else float("nan"),
        "mean_q_entropy": float(np.mean(q_entropy)) if q_entropy else float("nan"),
        "mean_q_normalized_entropy": float(np.mean(q_norm_entropy)) if q_norm_entropy else float("nan"),
        "near_same_patient_rate": near_same / near_pairs if near_pairs else float("nan"),
        "far_same_patient_rate": far_same / far_pairs if far_pairs else float("nan"),
        "random_identity_baseline_near": baseline_near_num / near_pairs if near_pairs else float("nan"),
        "random_identity_baseline_far": baseline_far_num / far_pairs if far_pairs else float("nan"),
        "per_patient": per_patient,
        "numerator_valid_centers": n_valid,
        "denominator_center_visits": n_centers,
        "optimizer_updated": False,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / "threshold_pairs.npz",
            pair_indices=thresholds.pair_indices,
            distances=thresholds.distances,
        )
        np.savez(output_dir / "center_orders.npz", **{f"shuffle_{i+1}": order for i, order in enumerate(orders)})
        _write_json(
            output_dir / "relations_train.json",
            {
                "q10": thresholds.q10,
                "q50": thresholds.q50,
                "seed": thresholds.seed,
                "n_pairs": thresholds.n_pairs,
                "quantile_method": thresholds.quantile_method,
                "pair_generation": "PCG64: integers i in [0,n), then t in [0,n-1); j=t+(t>=i)",
            },
        )
    return report


def simulate_graph_coverage(table, config: dict, *, output_dir: Path | None = None) -> dict:
    splits = list(config["precheck"]["graph_splits"])
    out = {"kind": "graph_precheck", "splits": {}}
    for split in splits:
        graph = build_split_graph(table, split, config)
        out["splits"][split] = graph_stats(graph, table)
        if output_dir is not None:
            rows = edges_as_records(graph)
            path = output_dir / f"graph_edges_{split}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            if rows:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    out["optimizer_updated"] = False
    return out


def run_precheck(
    config: dict,
    output_dir: str | Path,
    *,
    point_table=None,
    load_real_labels: bool = True,
    load_real_features: bool = True,
    artifact_kind: str = "official_precheck",
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    warnings: list[dict] = []
    table = point_table if point_table is not None else load_split_point_table(config)

    if table.labels_z is not None:
        table.require_finite()
    elif load_real_labels:
        try:
            names = load_pathway_names(config["data"]["zscore_manifest_file"])
            table = attach_labels(table, config["data"]["labels_root"], names, train_mpp_id=int(config["data"]["mpp_id"]))
        except Exception as exc:
            errors.append({"code": type(exc).__name__, "message": str(exc), "section": "labels"})

    relation_report = None
    if table.labels_z is not None and not any(item["section"] == "labels" for item in errors):
        try:
            relation_report = simulate_relation_coverage(table, config, output_dir=output)
            warn_rate = float(config["precheck"].get("low_valid_rate_warn") or 0.2)
            if relation_report["valid_rate"] < warn_rate:
                warnings.append(
                    {
                        "code": "LOW_RELATION_COVERAGE",
                        "message": f"有效中心比例={relation_report['valid_rate']:.4f}，低于提示阈值 {warn_rate}。不自动改参数。",
                    }
                )
        except Exception as exc:
            errors.append({"code": type(exc).__name__, "message": str(exc), "section": "relation"})
    elif load_real_labels and not any(item.get("section") == "labels" for item in errors):
        errors.append({"code": "LABELS_REQUIRED", "message": "关系预检缺少训练标签", "section": "labels"})

    if table.features is None and load_real_features:
        try:
            feature_manifest = load_feature_source_manifest(config["data"]["feature_source_manifest"])
            table = attach_features(table, feature_manifest=feature_manifest, expected_dim=int(config["data"]["input_dim"]))
        except Exception as exc:
            errors.append({"code": type(exc).__name__, "message": str(exc), "section": "features"})

    graph_report = None
    try:
        graph_report = simulate_graph_coverage(table, config, output_dir=output)
        for split, stats in graph_report["splits"].items():
            if stats["isolated_rate"] >= 0.5:
                warnings.append(
                    {
                        "code": "HIGH_ISOLATED_RATE",
                        "message": f"{split} 孤立比例={stats['isolated_rate']:.4f}。不自动改图参数。",
                    }
                )
    except SlideMappingMissingError as exc:
        errors.append({"code": exc.error_code, "message": str(exc), "section": "graph"})
    except Exception as exc:
        errors.append({"code": type(exc).__name__, "message": str(exc), "section": "graph"})

    status = "ok"
    if errors:
        status = "error"
    elif warnings:
        status = "ok_with_warnings"

    summary_lines = [
        "# Phase2 软连接 v2.1 训练前核对",
        "",
        f"状态: {status}",
        "本报告不更新优化器，不是正式训练结果。",
        "",
    ]
    if relation_report:
        summary_lines.extend(
            [
                "## 标签关系模拟",
                f"- 训练点数: {relation_report['n_train']}",
                f"- 打乱次数: {relation_report['n_shuffles']}，批次数: {relation_report['n_batches']}",
                f"- q10={relation_report['q10']:.6g}, q50={relation_report['q50']:.6g}",
                f"- 有效中心: {relation_report['n_valid']}/{relation_report['n_center_visits']} = {relation_report['valid_rate']:.4f}",
                f"- 近例均数 {relation_report['mean_near']:.3f}，远例均数 {relation_report['mean_far']:.3f}",
                f"- 近例同患者比例 {relation_report['near_same_patient_rate']:.4f}（随机基准 {relation_report['random_identity_baseline_near']:.4f}）",
                "",
            ]
        )
    if graph_report:
        summary_lines.append("## 图覆盖")
        for split, stats in graph_report["splits"].items():
            summary_lines.append(
                f"- {split}: 点数 {stats['n_nodes']}，边 {stats['n_edges']}，孤立比例 {stats['isolated_rate']:.4f}，a_self均值 {stats['a_self_mean']:.4f}"
            )
        summary_lines.append("")
    if errors:
        summary_lines.append("## 错误")
        for item in errors:
            summary_lines.append(f"- [{item.get('section')}] {item['code']}: {item['message']}")
        summary_lines.append("")
    if warnings:
        summary_lines.append("## 警告（不改参数）")
        for item in warnings:
            summary_lines.append(f"- {item['code']}: {item['message']}")
        summary_lines.append("")

    report = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "relation": relation_report,
        "graph": graph_report,
        "optimizer_updated": False,
        "artifact_kind": artifact_kind,
        "demo_or_test_artifact": artifact_kind != "official_precheck",
        "normalization": {
            "source": config["data"]["normalization_file"],
            "clip_note": load_normalization_from_config(config).inverse_note_zh,
        },
    }
    _write_json(output / "precheck_report.json", report)
    (output / "summary_zh.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 softlink v2.1 无训练预检")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.package_dir or package_root()
    config = load_config(args.config, package_dir=root)
    report = run_precheck(config, args.output_dir)
    print(json.dumps({"status": report["status"], "errors": report["errors"], "warnings": report["warnings"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
