"""UNI2-h 调优阶段一：基于已接纳六编码器 spatial 臂结果的差距定位与内部诊断。

只读取已回传的结果文件，输出写入本脚本所在目录。外部 XZY 只做已接纳结果的拆解；
预测压缩、三种子平均等诊断只在内部验证集上计算，供后续按内部验证选模时参考。
从仓库根目录运行：
    & 'D:\\conda_envs\\pfmval_py310\\python.exe' -B '01_指南与解读/分析报告/UNI2h调优阶段一_差距定位_20261015/phase1_gap_analysis.py'
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BATCH2 = ROOT / "experiments/results/phase2_backbone_spatial_ablation_batch2_20260919/20260927_154318_475_2242f53b/analysis"
REF = "uni2h"
COMPETITORS = ["virchow2", "hoptimus0", "hoptimus1"]
MODELS = ["uni2h", "virchow2", "hoptimus0", "hoptimus1", "uni", "phikonv2"]
# 事后按通路名称归组，仅用于描述差距分布，不作预定假设。
IMMUNE_INFLAMMATORY = ["ifng", "mhc", "Interferon_Alpha", "icp", "Inflammatory_Response",
                       "IL6_JAK_STAT3", "toxic", "TNF_Signaling", "Complement", "tls"]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def macro_pcc(pred: np.ndarray, target: np.ndarray, patients: np.ndarray) -> float:
    values = [pearson(pred[patients == p, j], target[patients == p, j])
              for p in np.unique(patients) for j in range(target.shape[1])]
    return float(np.nanmean(values))


def pooled_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    return pearson(pred.ravel(), target.ravel())


def gap_tables(pp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    seed_mean = pp.groupby(["model", "split", "patient", "pathway"], as_index=False)["pcc"].mean()
    per_seed_patient = pp.groupby(["model", "split", "seed", "patient"], as_index=False)["pcc"].mean()

    patient_rows = []
    for (split, patient), grp in per_seed_patient.groupby(["split", "patient"]):
        ref = grp[grp.model == REF].set_index("seed")["pcc"]
        row = {"split": split, "patient": patient, f"{REF}_macro_pcc": ref.mean()}
        for comp in COMPETITORS:
            other = grp[grp.model == comp].set_index("seed")["pcc"]
            diff = (ref - other.reindex(ref.index)).dropna()
            row[f"{comp}_macro_pcc"] = other.mean()
            row[f"diff_vs_{comp}"] = diff.mean()
            row[f"diff_vs_{comp}_seed_min"] = diff.min()
            row[f"diff_vs_{comp}_seed_max"] = diff.max()
        patient_rows.append(row)
    patient_gap = pd.DataFrame(patient_rows).sort_values(["split", "patient"])

    wide = seed_mean.pivot_table(index=["split", "patient", "pathway"], columns="model", values="pcc").reset_index()
    pathway = wide.groupby(["split", "pathway"], as_index=False)[MODELS].mean()
    for comp in COMPETITORS:
        pathway[f"diff_vs_{comp}"] = pathway[REF] - pathway[comp]
    pathway["diff_vs_best_competitor"] = pathway[REF] - pathway[COMPETITORS].max(axis=1)
    pathway = pathway.sort_values(["split", "diff_vs_best_competitor"])

    combo = wide.copy()
    for comp in COMPETITORS:
        combo[f"diff_vs_{comp}"] = combo[REF] - combo[comp]
    combo["pathway_group"] = np.where(combo.pathway.isin(IMMUNE_INFLAMMATORY), "immune_inflammatory", "other")
    diff_cols = [f"diff_vs_{comp}" for comp in COMPETITORS]
    group = combo.groupby(["split", "pathway_group"], as_index=False)[MODELS[:4] + diff_cols].mean()
    group["n_pathways"] = combo.groupby(["split", "pathway_group"]).pathway.nunique().to_numpy()
    return {"patient_gap": patient_gap, "pathway_gap": pathway, "combo_gap": combo, "group_gap": group}


def win_summary(tables: dict[str, pd.DataFrame]) -> dict:
    out: dict = {}
    for split in ["internal_val", "external_test"]:
        pg = tables["patient_gap"][tables["patient_gap"].split == split]
        pw = tables["pathway_gap"][tables["pathway_gap"].split == split]
        cb = tables["combo_gap"][tables["combo_gap"].split == split]
        out[split] = {}
        for comp in COMPETITORS:
            col = f"diff_vs_{comp}"
            deficit = -cb[col]
            ordered = np.sort(deficit.to_numpy())[::-1]
            total = deficit.sum()
            out[split][comp] = {
                "mean_diff_macro_pcc": float(cb[col].mean()),
                "patients_uni2h_higher": f"{int((pg[col] > 0).sum())}/{len(pg)}",
                "pathways_uni2h_higher": f"{int((pw[col] > 0).sum())}/{len(pw)}",
                "patient_pathway_combos_uni2h_higher": f"{int((cb[col] > 0).sum())}/{len(cb)}",
                "share_of_net_deficit_from_worst_10pct_combos":
                    float(ordered[: max(1, len(ordered) // 10)].sum() / total) if total > 0 else None,
            }
    return out


def internal_diagnostics(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ensemble_rows = [], []
    internal = per_seed[per_seed.split == "internal_val"]
    for model, grp in internal.groupby("model"):
        preds, keys_ref, target_ref, patients_ref = [], None, None, None
        for _, rec in grp.sort_values("seed").iterrows():
            z = np.load(ROOT / rec["source"], allow_pickle=True)
            keys = np.char.add(np.char.add(z["patient_id"].astype(str), "|"),
                               np.char.add(np.char.add(z["slide_id"].astype(str), "|"), z["spot_id"].astype(str)))
            order = np.argsort(keys)
            keys, pred, target, patients = keys[order], z["pred_z"][order].astype(np.float64), \
                z["target_z"][order].astype(np.float64), z["patient_id"][order].astype(str)
            if keys_ref is None:
                keys_ref, target_ref, patients_ref = keys, target, patients
            elif not (np.array_equal(keys, keys_ref) and np.allclose(target, target_ref)):
                raise ValueError(f"{model} seed {rec['seed']} 身份或真值与首个种子不一致")
            ratios = [pred[patients == p, j].std() / target[patients == p, j].std()
                      for p in np.unique(patients) for j in range(target.shape[1])]
            macro = macro_pcc(pred, target, patients)
            if abs(macro - float(rec["pcc_macro"])) > 1e-8:
                raise ValueError(f"{model} seed {rec['seed']} 宏观 PCC 复算不一致: {macro} vs {rec['pcc_macro']}")
            rows.append({
                "model": model, "seed": int(rec["seed"]), "formal_epoch": int(z["formal_epoch"]),
                "macro_pcc": macro, "pooled_pcc": pooled_pcc(pred, target),
                "pooled_std_ratio_pred_over_true": float(pred.std() / target.std()),
                "median_patient_pathway_std_ratio": float(np.median(ratios)),
            })
            preds.append(pred)
        ens = np.mean(preds, axis=0)
        seed_rows = [r for r in rows if r["model"] == model]
        ensemble_rows.append({
            "model": model,
            "seed_mean_macro_pcc": float(np.mean([r["macro_pcc"] for r in seed_rows])),
            "ensemble3_macro_pcc": macro_pcc(ens, target_ref, patients_ref),
            "seed_mean_pooled_pcc": float(np.mean([r["pooled_pcc"] for r in seed_rows])),
            "ensemble3_pooled_pcc": pooled_pcc(ens, target_ref),
        })
    ens_df = pd.DataFrame(ensemble_rows)
    ens_df["ensemble_gain_macro"] = ens_df["ensemble3_macro_pcc"] - ens_df["seed_mean_macro_pcc"]
    ens_df["ensemble_gain_pooled"] = ens_df["ensemble3_pooled_pcc"] - ens_df["seed_mean_pooled_pcc"]
    return pd.DataFrame(rows).sort_values(["model", "seed"]), ens_df


def main() -> None:
    pp = pd.read_csv(BATCH2 / "six_model_patient_pathway.csv")
    per_seed = pd.read_csv(BATCH2 / "six_model_per_seed.csv")
    tables = gap_tables(pp)
    tables["patient_gap"].to_csv(OUT / "patient_gap.csv", index=False)
    tables["pathway_gap"].to_csv(OUT / "pathway_gap.csv", index=False)
    tables["combo_gap"].to_csv(OUT / "patient_pathway_gap.csv", index=False)
    tables["group_gap"].to_csv(OUT / "pathway_group_gap.csv", index=False)
    diag, ens = internal_diagnostics(per_seed)
    diag.to_csv(OUT / "internal_diagnostics_per_seed.csv", index=False)
    ens.to_csv(OUT / "internal_seed_ensemble.csv", index=False)
    summary = {
        "source_batch_analysis": str(BATCH2.relative_to(ROOT)).replace("\\", "/"),
        "arm": "full_fov spatial, seeds 45/46/47",
        "external_used_for_selection": False,
        "post_hoc_immune_inflammatory_group": IMMUNE_INFLAMMATORY,
        "wins": win_summary(tables),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
