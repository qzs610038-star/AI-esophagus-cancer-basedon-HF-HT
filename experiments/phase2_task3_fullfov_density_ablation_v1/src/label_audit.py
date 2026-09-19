"""Read-only audit of two saved *true-label* definitions; prediction columns are never loaded."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    return float(np.sum(xc * yc) / denom) if denom else float("nan")


def _load_true_only(path: Path, keys: Sequence[str], pathways: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    selected = list(keys) + [f"true_{name}" for name in pathways] + [f"true_raw_{name}" for name in pathways]
    missing = [name for name in selected if name not in header]
    if missing:
        raise ValueError(f"{path} 缺少真实标签审计列: {missing[:8]}")
    if any(name.startswith("pred_") or name.startswith("pred_raw_") for name in selected):
        raise RuntimeError("标签审计禁止选择任何 pred_* 列")
    frame = pd.read_csv(path, usecols=selected)
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{path} 的连接键不唯一")
    return frame


def _align_pair(original_path: Path, reconstructed_path: Path, keys: Sequence[str], pathways: Sequence[str], expected_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = _load_true_only(original_path, keys, pathways)
    reconstructed = _load_true_only(reconstructed_path, keys, pathways)
    left = original.merge(reconstructed[list(keys)], on=list(keys), how="left", indicator=True)
    right = reconstructed.merge(original[list(keys)], on=list(keys), how="left", indicator=True)
    if len(original) != expected_n or len(reconstructed) != expected_n or not (left["_merge"] == "both").all() or not (right["_merge"] == "both").all():
        raise ValueError(f"真实标签无法完整一对一匹配: original={len(original)}, reconstructed={len(reconstructed)}, expected={expected_n}")
    original = original.sort_values(list(keys)).reset_index(drop=True)
    reconstructed = reconstructed.sort_values(list(keys)).reset_index(drop=True)
    if not original[list(keys)].equals(reconstructed[list(keys)]):
        raise ValueError("排序后连接键仍不一致")
    return original, reconstructed


def _quartile_text(values: np.ndarray) -> str:
    q1, median, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return f"{median:.3g} [{q1:.3g}, {q3:.3g}]"


def _pathway_stats(pathway: str, internal_pair, external_pair, keys: Sequence[str]) -> dict:
    i_orig, i_recon = internal_pair; e_orig, e_recon = external_pair
    izo, izr = i_orig[f"true_{pathway}"].to_numpy(float), i_recon[f"true_{pathway}"].to_numpy(float)
    ezo, ezr = e_orig[f"true_{pathway}"].to_numpy(float), e_recon[f"true_{pathway}"].to_numpy(float)
    patient_pcc, patient_abs = [], []
    for patient in sorted(i_orig["patient_id"].astype(str).unique()):
        mask = i_orig["patient_id"].astype(str).to_numpy() == patient
        patient_pcc.append(_pearson(izo[mask], izr[mask]))
        patient_abs.append(float(np.mean(np.abs(izo[mask] - izr[mask]))))
    raw_io = i_orig[f"true_raw_{pathway}"].to_numpy(float); raw_ir = i_recon[f"true_raw_{pathway}"].to_numpy(float)
    raw_eo = e_orig[f"true_raw_{pathway}"].to_numpy(float); raw_er = e_recon[f"true_raw_{pathway}"].to_numpy(float)
    row = {
        "pathway": pathway,
        "original_internal_raw_median": float(np.median(raw_io)),
        "original_internal_raw_q1": float(np.quantile(raw_io, .25)),
        "original_internal_raw_q3": float(np.quantile(raw_io, .75)),
        "original_internal_raw_iqr": float(np.quantile(raw_io, .75) - np.quantile(raw_io, .25)),
        "reconstructed_internal_raw_median": float(np.median(raw_ir)),
        "reconstructed_internal_raw_q1": float(np.quantile(raw_ir, .25)),
        "reconstructed_internal_raw_q3": float(np.quantile(raw_ir, .75)),
        "reconstructed_internal_raw_iqr": float(np.quantile(raw_ir, .75) - np.quantile(raw_ir, .25)),
        "original_external_raw_median": float(np.median(raw_eo)),
        "original_external_raw_q1": float(np.quantile(raw_eo, .25)),
        "original_external_raw_q3": float(np.quantile(raw_eo, .75)),
        "original_external_raw_iqr": float(np.quantile(raw_eo, .75) - np.quantile(raw_eo, .25)),
        "reconstructed_external_raw_median": float(np.median(raw_er)),
        "reconstructed_external_raw_q1": float(np.quantile(raw_er, .25)),
        "reconstructed_external_raw_q3": float(np.quantile(raw_er, .75)),
        "reconstructed_external_raw_iqr": float(np.quantile(raw_er, .75) - np.quantile(raw_er, .25)),
        "internal_patient_equal_pcc": float(np.nanmean(patient_pcc)),
        "external_xzy_pcc": _pearson(ezo, ezr),
        "internal_patient_equal_mean_abs_delta_z": float(np.mean(patient_abs)),
        "external_mean_abs_delta_z": float(np.mean(np.abs(ezo - ezr))),
        "internal_sign_direction_agreement": float(np.mean(np.signbit(izo) == np.signbit(izr))),
        "external_sign_direction_agreement": float(np.mean(np.signbit(ezo) == np.signbit(ezr))),
        "n_internal": int(len(izo)),
        "n_external": int(len(ezo)),
    }
    row["original_internal_raw_summary"] = _quartile_text(raw_io)
    row["reconstructed_internal_raw_summary"] = _quartile_text(raw_ir)
    row["original_external_raw_summary"] = _quartile_text(raw_eo)
    row["reconstructed_external_raw_summary"] = _quartile_text(raw_er)
    return row


def _configure_font() -> None:
    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


def render_table(frame: pd.DataFrame, output_path: Path, summary: dict) -> None:
    _configure_font()
    rows = frame.reset_index(drop=True)
    fig = plt.figure(figsize=(22, 19), dpi=150, facecolor="white")
    ax = fig.add_axes([0.035, 0.115, 0.93, 0.79]); ax.set_xlim(0, 22); ax.set_ylim(-1.2, len(rows) + 1.8); ax.axis("off")
    widths = [3.4, 2.2, 2.2, 2.2, 2.2, 1.25, 1.25, 1.25, 1.25, 1.15, 1.15, 0.75, 0.75]
    starts = np.cumsum([0] + widths[:-1]).tolist()
    headers = ["通路", "原始raw\n内部 中位[Q1,Q3]", "319基因重构raw\n内部 中位[Q1,Q3]", "原始raw\n外部 中位[Q1,Q3]", "319基因重构raw\n外部 中位[Q1,Q3]", "内部\nPCC", "外部\nPCC", "内部\n|Δz|", "外部\n|Δz|", "内部\n同号率", "外部\n同号率", "n内", "n外"]
    for x, width, header in zip(starts, widths, headers):
        ax.add_patch(Rectangle((x, len(rows)), width, 1.5, facecolor="#20364f", edgecolor="white", linewidth=.8))
        ax.text(x + width / 2, len(rows) + .75, header, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    pcc_norm, dz_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), Normalize(vmin=0, vmax=max(1.0, float(rows[["internal_patient_equal_mean_abs_delta_z", "external_mean_abs_delta_z"]].to_numpy().max())))
    pcc_cmap, dz_cmap = mpl.colormaps["RdBu_r"], mpl.colormaps["YlOrRd"]
    for r, row in rows.iterrows():
        y = len(rows) - 1 - r
        base = "#f5f7fa" if r % 2 == 0 else "white"
        for x, width in zip(starts, widths):
            ax.add_patch(Rectangle((x, y), width, 1, facecolor=base, edgecolor="#d7dde5", linewidth=.45))
        text_values = [row["pathway"], row["original_internal_raw_summary"], row["reconstructed_internal_raw_summary"], row["original_external_raw_summary"], row["reconstructed_external_raw_summary"]]
        for c, value in enumerate(text_values):
            ax.text(starts[c] + (.10 if c == 0 else widths[c] / 2), y + .5, str(value), ha="left" if c == 0 else "center", va="center", fontsize=7.7 if c else 8.2)
        numeric = [row["internal_patient_equal_pcc"], row["external_xzy_pcc"], row["internal_patient_equal_mean_abs_delta_z"], row["external_mean_abs_delta_z"]]
        for offset, value in enumerate(numeric):
            c = 5 + offset; color = pcc_cmap(pcc_norm(value)) if offset < 2 else dz_cmap(dz_norm(value))
            ax.add_patch(Rectangle((starts[c], y), widths[c], 1, facecolor=color, edgecolor="#d7dde5", linewidth=.45))
            ax.text(starts[c] + widths[c] / 2, y + .5, f"{value:.3f}", ha="center", va="center", fontsize=8.2, color="black")
        for c, value in ((9, row["internal_sign_direction_agreement"]), (10, row["external_sign_direction_agreement"])):
            ax.text(starts[c] + widths[c] / 2, y + .5, f"{value:.1%}", ha="center", va="center", fontsize=8)
        for c, value in ((11, row["n_internal"]), (12, row["n_external"])):
            ax.text(starts[c] + widths[c] / 2, y + .5, str(int(value)), ha="center", va="center", fontsize=8)
    fig.text(.035, .958, "Phase2 任务二：原始30维真实通路分数 vs 319基因经 ssGSEA 重构30维真实通路分数", fontsize=17, fontweight="bold", color="#17283a")
    fig.text(.035, .925, "仅使用真实标签，不含预测值｜按内部患者等权PCC从低到高排序｜内部 1,078 点；外部XZY 1,039 点", fontsize=11, color="#3c536b")
    fig.text(.035, .074, f"总体患者—通路平均PCC：内部 {summary['overall_internal_patient_pathway_mean_pcc']:.6f}；外部 {summary['overall_external_pathway_mean_pcc']:.6f}。低一致性主要见 Apoptosis、mTOR Signaling、Unfolded Protein Response；高一致性主要见 mhc、Fibrosis、ifng。", fontsize=9.5, color="#263b50")
    fig.text(.035, .047, "说明：两套raw分数的计算单位和数量级不同；raw列只作各自分布文本展示，不把raw减法、比值或raw MAE解释为算法误差。PCC固定[-1,1]发散色阶，|Δz|使用顺序色阶。", fontsize=9.2, color="#5b3a29")
    fig.text(.035, .021, "这些是已观察差异，不直接证明原因。当前仅覆盖已保存的内部验证与外部XZY真实标签；本地无训练集原始标签源。", fontsize=9.2, color="#5b3a29")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_label_audit(config: dict, output_dir: str | Path | None = None) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    out = Path(output_dir) if output_dir else package_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    audit = config["label_audit"]; source_root = Path(audit["source_root"]); keys = list(audit["join_keys"])
    pathways = json.loads(Path(config["inputs"]["zscore_manifest"]).read_text(encoding="utf-8-sig"))["pathway_names"]
    if len(pathways) != int(audit["expected_pathways"]):
        raise ValueError("通路顺序不是冻结的30维顺序")
    paths = {
        "reconstructed_internal": source_root / audit["reconstructed_internal"],
        "reconstructed_external": source_root / audit["reconstructed_external"],
        "original_internal": source_root / audit["original_internal"],
        "original_external": source_root / audit["original_external"],
        "legacy_gene_branch_internal": source_root / audit["legacy_gene_branch_internal"],
        "legacy_gene_branch_external": source_root / audit["legacy_gene_branch_external"],
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError({name: str(path) for name, path in paths.items() if not path.is_file()})
    internal = _align_pair(paths["original_internal"], paths["reconstructed_internal"], keys, pathways, int(audit["expected_internal"]))
    external = _align_pair(paths["original_external"], paths["reconstructed_external"], keys, pathways, int(audit["expected_external"]))
    legacy_keys = [key for key in keys if key != "slide_id"]
    legacy_internal = _align_pair(paths["reconstructed_internal"], paths["legacy_gene_branch_internal"], legacy_keys, pathways, int(audit["expected_internal"]))
    legacy_external = _align_pair(paths["reconstructed_external"], paths["legacy_gene_branch_external"], legacy_keys, pathways, int(audit["expected_external"]))
    internal_original_z = internal[0][[f"true_{name}" for name in pathways]].to_numpy(float)
    internal_reconstructed_z = internal[1][[f"true_{name}" for name in pathways]].to_numpy(float)
    external_original_z = external[0][[f"true_{name}" for name in pathways]].to_numpy(float)
    external_reconstructed_z = external[1][[f"true_{name}" for name in pathways]].to_numpy(float)
    def branch_drift(pair):
        left, right = pair
        z_left = left[[f"true_{name}" for name in pathways]].to_numpy(float)
        z_right = right[[f"true_{name}" for name in pathways]].to_numpy(float)
        raw_left = left[[f"true_raw_{name}" for name in pathways]].to_numpy(float)
        raw_right = right[[f"true_raw_{name}" for name in pathways]].to_numpy(float)
        return {
            "n_points": int(len(left)),
            "z_mean_abs_difference": float(np.mean(np.abs(z_left - z_right))),
            "z_max_abs_difference": float(np.max(np.abs(z_left - z_right))),
            "raw_mean_abs_difference_descriptive_only": float(np.mean(np.abs(raw_left - raw_right))),
            "raw_max_abs_difference_descriptive_only": float(np.max(np.abs(raw_left - raw_right))),
            "n_nonzero_z_cells": int(np.sum(np.abs(z_left - z_right) > 0)),
        }
    frame = pd.DataFrame([_pathway_stats(pathway, internal, external, keys) for pathway in pathways]).sort_values("internal_patient_equal_pcc", ascending=True).reset_index(drop=True)
    csv_path = out / "task2_true_label_comparison_30_pathways.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "analysis_scope": "true_labels_only_no_predictions",
        "n_internal": int(len(internal[0])),
        "n_external": int(len(external[0])),
        "n_total_aligned": int(len(internal[0]) + len(external[0])),
        "n_pathways": int(len(pathways)),
        "pathway_order": pathways,
        "overall_internal_patient_pathway_mean_pcc": float(frame["internal_patient_equal_pcc"].mean()),
        "overall_external_pathway_mean_pcc": float(frame["external_xzy_pcc"].mean()),
        "overall_internal_patient_equal_mean_abs_delta_z": float(frame["internal_patient_equal_mean_abs_delta_z"].mean()),
        "overall_internal_point_weighted_mean_abs_delta_z": float(np.mean(np.abs(internal_original_z - internal_reconstructed_z))),
        "overall_external_mean_abs_delta_z": float(frame["external_mean_abs_delta_z"].mean()),
        "overall_internal_sign_direction_agreement": float(np.mean(np.signbit(internal_original_z) == np.signbit(internal_reconstructed_z))),
        "overall_external_sign_direction_agreement": float(np.mean(np.signbit(external_original_z) == np.signbit(external_reconstructed_z))),
        "weakest_internal_consistency": frame.head(3)[["pathway", "internal_patient_equal_pcc"]].to_dict("records"),
        "strongest_internal_consistency": frame.tail(3).sort_values("internal_patient_equal_pcc", ascending=False)[["pathway", "internal_patient_equal_pcc"]].to_dict("records"),
        "training_scope_available": False,
        "causal_interpretation": "not_established",
        "raw_scale_warning": "两套raw分数的计算单位和数量级不同，不将raw差、比值或raw MAE解释为算法误差。",
        "confirmed_legacy_common_truth_drift": {
            "description": "旧任务二两个分支原本声明共享重构真值，但保存的true_*并不相同，说明真值曾在分支内分别计算。",
            "internal": branch_drift(legacy_internal),
            "external": branch_drift(legacy_external),
        },
    }
    _write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write(out / "task2_true_label_comparison_summary.json", summary)
    manifest = {
        "read_only": True,
        "loaded_columns_policy": "仅join keys、true_*、true_raw_*；禁止pred_*与pred_raw_*",
        "join_keys": keys,
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "source_rows": {"internal_each": len(internal[0]), "external_each": len(external[0])},
        "hashes_computed": False,
        "original_files_modified": False,
    }
    _write(out / "task2_true_label_source_manifest.json", manifest)
    render_table(frame, out / "task2_true_label_comparison_30_pathways.png", summary)
    return {"summary": summary, "csv": str(csv_path), "png": str(out / "task2_true_label_comparison_30_pathways.png"), "source_manifest": str(out / "task2_true_label_source_manifest.json")}
