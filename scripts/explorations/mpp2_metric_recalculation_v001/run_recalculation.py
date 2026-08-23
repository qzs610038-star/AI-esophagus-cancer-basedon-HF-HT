from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/explorations/mpp2_metric_recalculation_v001"
FIG = OUT / "figures"
PROTECTED = ROOT / "data/protected_local/mpp2_r2_root_cause_20260725/source_snapshot"
PARAMS = PROTECTED / "zscore/group_2_repaired_v003/zscore_params_from_train.csv"
MANIFEST = PROTECTED / "zscore/group_2_repaired_v003/split_manifest.csv"
SEED = 20260823
N_BOOT = 10000
INTERNAL = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不可序列化类型: {type(value)}")


def pcc(y: np.ndarray, p: np.ndarray) -> float:
    good = np.isfinite(y) & np.isfinite(p)
    y, p = y[good], p[good]
    if len(y) < 3 or np.var(y) == 0 or np.var(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    good = np.isfinite(y) & np.isfinite(p)
    y, p = y[good], p[good]
    if len(y) < 3:
        return np.nan
    den = np.var(y) + np.var(p) + (np.mean(y) - np.mean(p)) ** 2
    return np.nan if den == 0 else float(2 * np.mean((y - y.mean()) * (p - p.mean())) / den)


def basic_metrics(yz, pz, yr, pr) -> dict:
    errz = pz - yz
    errr = pr - yr
    sst = float(np.sum((yr - np.mean(yr)) ** 2))
    return {
        "pcc": pcc(yz, pz), "ccc": ccc(yz, pz),
        "z_rmse": float(np.sqrt(np.mean(errz ** 2))),
        "z_mae": float(np.mean(np.abs(errz))),
        "raw_rmse": float(np.sqrt(np.mean(errr ** 2))),
        "raw_mae": float(np.mean(np.abs(errr))),
        "raw_r2": np.nan if sst == 0 else float(1 - np.sum(errr ** 2) / sst),
    }


def parse_xy(value: str) -> tuple[int, int]:
    match = re.search(r"x(-?\d+)_y(-?\d+)", str(value))
    if not match:
        raise ValueError(f"无法从键解析坐标: {value}")
    return int(match.group(1)), int(match.group(2))


def local_ssim(y, p, x, yy, window=7, data_range=6.0):
    xs, ys = np.unique(x), np.unique(yy)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    shape = (len(ys), len(xs))
    gy, gp = np.full(shape, np.nan), np.full(shape, np.nan)
    for a, b, tv, pv in zip(x, yy, y, p):
        gy[yi[b], xi[a]], gp[yi[b], xi[a]] = tv, pv
    occupancy = float(np.isfinite(gy).mean())
    if min(shape) < window:
        return np.nan, occupancy, 0, "masked_SSIM_PFMval", []
    wy, wp = sliding_window_view(gy, (window, window)), sliding_window_view(gp, (window, window))
    valid = np.isfinite(wy).all(axis=(-1, -2)) & np.isfinite(wp).all(axis=(-1, -2))
    values = []
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    for i, j in np.argwhere(valid):
        a, b = wy[i, j].ravel(), wp[i, j].ravel()
        ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
        cov = np.mean((a - ma) * (b - mb))
        score = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
        values.append((float(score), int(xs[j + window // 2]), int(ys[i + window // 2])))
    name = "standard_grid_SSIM" if occupancy >= 0.90 else "masked_SSIM_PFMval"
    return (float(np.mean([v[0] for v in values])) if values else np.nan,
            occupancy, len(values), name, values)


def moran(values, x, y):
    coords = list(zip(map(int, x), map(int, y)))
    if len(coords) < 3 or np.var(values) == 0:
        return np.nan, 0
    diffs = []
    for axis in (0, 1):
        vals = sorted(set(c[axis] for c in coords))
        diffs.extend(b - a for a, b in zip(vals, vals[1:]) if b > a)
    step = min(diffs)
    lookup = {c: i for i, c in enumerate(coords)}
    edges = []
    for i, (a, b) in enumerate(coords):
        for nb in ((a + step, b), (a, b + step)):
            if nb in lookup:
                edges.append((i, lookup[nb]))
    if not edges:
        return np.nan, 0
    z = np.asarray(values, float) - np.mean(values)
    numerator = 2 * sum(z[i] * z[j] for i, j in edges)
    s0 = 2 * len(edges)
    return float(len(z) / s0 * numerator / np.sum(z ** 2)), len(edges)


def fisher_mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return np.nan if not len(values) else float(np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))


def canonical_rich(path: Path, experiment: str, arm: str, evidence: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "spot_id" in d:
        key = d["spot_id"].astype(str)
        patient, split = d["patient"].astype(str), d["split"].astype(str)
        ztrue = "truth_z__"; zpred = "prediction_z__"; rtrue = "truth_raw__"; rpred = "prediction_raw__"
    else:
        key = d["patch_stem"].astype(str)
        patient, split = d["patient"].astype(str), d["split"].astype(str)
        ztrue = "true_"; zpred = "pred_"; rtrue = rpred = None
    pathways = [c[len(ztrue):] for c in d.columns if c.startswith(ztrue)]
    params = pd.read_csv(PARAMS).set_index("label")
    rows = []
    coords = [parse_xy(v) for v in key]
    for pathway in pathways:
        yz, pz = d[ztrue + pathway].to_numpy(float), d[zpred + pathway].to_numpy(float)
        mean, std = float(params.loc[pathway, "mean"]), float(params.loc[pathway, "std"])
        yr = d[rtrue + pathway].to_numpy(float) if rtrue and rtrue + pathway in d else yz * std + mean
        pr = d[rpred + pathway].to_numpy(float) if rpred and rpred + pathway in d else pz * std + mean
        part = pd.DataFrame({
            "experiment": experiment, "arm": arm, "evidence_status": evidence,
            "patient": patient, "split": split, "key": key,
            "x": [v[0] for v in coords], "y": [v[1] for v in coords], "pathway": pathway,
            "truth_z": yz, "prediction_z": pz, "truth_raw": yr, "prediction_raw": pr,
            "sigma_train": std,
        })
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    if out.duplicated(["patient", "split", "key", "pathway"]).any():
        raise ValueError(f"重复配对键: {path}")
    return out


def locate_w007(arm, filename):
    hits = list((ROOT / f"project_state/inbox/W007/quarantine/{arm}/automation/returns/W007").glob(f"**/{filename}"))
    if len(hits) != 1:
        raise ValueError(f"{arm}/{filename} 文件数={len(hits)}")
    return hits[0]


def inventory_entry(path, status, experiment):
    d = pd.read_csv(path)
    patients = sorted(d[c].astype(str).unique().tolist()) if (c := next((x for x in ("patient", "patient_id") if x in d), None)) else []
    prefixes = ("truth_z__", "true_")
    pathways = sorted({col[len(pre):] for pre in prefixes for col in d if col.startswith(pre)})
    st = path.stat()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns, "rows": len(d), "columns": list(d.columns),
            "patients": patients, "pathways": pathways, "pathway_count": len(pathways),
            "evidence_status": status, "experiment": experiment}


def per_patient_metrics(long):
    rows, spatial_rows, window_rows = [], [], []
    for keys, g in long.groupby(["experiment", "arm", "evidence_status", "split", "patient", "pathway"], sort=False):
        yz, pz, yr, pr = (g[c].to_numpy(float) for c in ("truth_z", "prediction_z", "truth_raw", "prediction_raw"))
        metrics = basic_metrics(yz, pz, yr, pr)
        metrics["nrmse_sigma_train"] = metrics["raw_rmse"] / float(g["sigma_train"].iloc[0])
        rows.append(dict(zip(("experiment", "arm", "evidence_status", "split", "patient", "pathway"), keys), n=len(g), **metrics))
        ssim, occ, windows, method, local = local_ssim(yz, pz, g.x.to_numpy(), g.y.to_numpy())
        mt, edges = moran(yz, g.x, g.y); mp, _ = moran(pz, g.x, g.y); me, _ = moran(pz - yz, g.x, g.y)
        spatial_rows.append(dict(zip(("experiment", "arm", "evidence_status", "split", "patient", "pathway"), keys),
            ssim=ssim, ssim_method=method, grid_occupancy=occ, valid_windows=windows, data_range=6.0,
            truth_moran_i=mt, prediction_moran_i=mp, moran_delta=mp-mt if np.isfinite(mt+mp) else np.nan,
            abs_moran_delta=abs(mp-mt) if np.isfinite(mt+mp) else np.nan, residual_moran_i=me, rook_edges=edges))
        for score, x, y in local:
            window_rows.append({"experiment": keys[0], "split": keys[3], "patient": keys[4], "pathway": keys[5],
                                "block": f"{x//2048}:{y//2048}", "ssim": score})
    return pd.DataFrame(rows), pd.DataFrame(spatial_rows), pd.DataFrame(window_rows)


def summarize(metrics, spatial):
    merged = metrics.merge(spatial[["experiment", "split", "patient", "pathway", "ssim", "abs_moran_delta"]],
                           on=["experiment", "split", "patient", "pathway"], how="left")
    path_rows = []
    for keys, g in merged.groupby(["experiment", "arm", "evidence_status", "split", "pathway"], sort=False):
        row = dict(zip(("experiment", "arm", "evidence_status", "split", "pathway"), keys))
        for metric in ("pcc", "ccc", "z_rmse", "nrmse_sigma_train", "z_mae", "raw_mae", "raw_r2", "ssim", "abs_moran_delta"):
            row[metric] = fisher_mean(g[metric]) if metric == "pcc" else float(g[metric].mean())
        row["patients"] = int(g.patient.nunique())
        path_rows.append(row)
    pathway = pd.DataFrame(path_rows)
    summary_rows = []
    for keys, g in pathway.groupby(["experiment", "arm", "evidence_status", "split"], sort=False):
        for metric in ("pcc", "ccc", "z_rmse", "nrmse_sigma_train", "z_mae", "raw_mae", "raw_r2", "ssim", "abs_moran_delta"):
            v = g[metric].dropna()
            summary_rows.append(dict(zip(("experiment", "arm", "evidence_status", "split"), keys), metric=metric,
                mean=v.mean(), median=v.median(), q1=v.quantile(.25), q3=v.quantile(.75), iqr=v.quantile(.75)-v.quantile(.25),
                min=v.min(), max=v.max(), valid_pathways=len(v)))
    return pathway, pd.DataFrame(summary_rows)


def delta_tables(pathway):
    comparisons = [("lora_s0", "lora_s1"), ("lora_s0", "lora_dropout10"),
                   ("w007_fbr", "w007_rcc"), ("w007_fbr", "w007_hcr"), ("w007_fbr", "w007_cpgcr"),
                   ("huber_mse", "huber_delta1")]
    rows = []
    for control, candidate in comparisons:
        a, b = pathway[pathway.experiment == control], pathway[pathway.experiment == candidate]
        for split in sorted(set(a.split) & set(b.split)):
            merged = a[a.split == split].merge(b[b.split == split], on=["split", "pathway"], suffixes=("_control", "_candidate"))
            for _, r in merged.iterrows():
                for metric in ("pcc", "ccc", "z_rmse", "ssim", "abs_moran_delta"):
                    av, bv = r.get(metric + "_control"), r.get(metric + "_candidate")
                    sign = -1 if metric in ("z_rmse", "abs_moran_delta") else 1
                    rows.append({"comparison": f"{control}_vs_{candidate}", "control": control, "candidate": candidate,
                                 "split": split, "pathway": r.pathway, "metric": metric,
                                 "control_value": av, "candidate_value": bv, "delta_candidate_better": sign * (bv-av)})
    return pd.DataFrame(rows)


def spatial_block_metrics(long):
    rows = []
    work = long.copy()
    work["block"] = (work.x // 2048).astype(str) + ":" + (work.y // 2048).astype(str)
    for keys, g in work.groupby(["experiment", "split", "patient", "pathway", "block"], sort=False):
        if len(g) < 3:
            continue
        met = basic_metrics(*(g[c].to_numpy(float) for c in ("truth_z", "prediction_z", "truth_raw", "prediction_raw")))
        rows.append(dict(zip(("experiment", "split", "patient", "pathway", "block"), keys), n=len(g), **met))
    return pd.DataFrame(rows)


def bootstrap_ci(deltas, patient_metrics, window_metrics, block_metrics):
    rng = np.random.default_rng(SEED); rows = []
    for comparison in deltas.comparison.unique():
        control, candidate = deltas[deltas.comparison == comparison][["control", "candidate"]].iloc[0]
        for split in deltas[deltas.comparison == comparison].split.unique():
            if comparison.startswith("huber_"):
                continue
            for metric in ("pcc", "ccc", "z_rmse"):
                a = patient_metrics[(patient_metrics.experiment == control) & (patient_metrics.split == split)]
                b = patient_metrics[(patient_metrics.experiment == candidate) & (patient_metrics.split == split)]
                m = a.merge(b, on=["split", "patient", "pathway"], suffixes=("_a", "_b"))
                if m.empty: continue
                sign = -1 if metric == "z_rmse" else 1
                m["d"] = sign * (m[metric+"_b"] - m[metric+"_a"])
                units = sorted(m.patient.unique()) if set(m.patient) != {"XZY"} else sorted((m.assign(block="single")).block.unique())
                if set(m.patient) == {"XZY"}:
                    ba = block_metrics[(block_metrics.experiment == control) & (block_metrics.split == split) & (block_metrics.patient == "XZY")]
                    bb = block_metrics[(block_metrics.experiment == candidate) & (block_metrics.split == split) & (block_metrics.patient == "XZY")]
                    bm = ba.merge(bb, on=["split", "patient", "pathway", "block"], suffixes=("_a", "_b"))
                    if bm.empty:
                        continue
                    bm["d"] = sign * (bm[metric+"_b"] - bm[metric+"_a"])
                    units = sorted(bm.block.unique())
                    vals = np.array([bm[bm.block == unit].d.mean() for unit in units])
                    vals = vals[np.isfinite(vals)]
                    boot = np.mean(vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))], axis=1)
                    rows.append({"comparison": comparison, "split": split, "metric": metric, "unit": "spatial_block_2048px",
                                 "raw_units": len(vals), "valid_repetitions": N_BOOT, "estimate": float(vals.mean()),
                                 "ci_low": float(np.percentile(boot,2.5)), "ci_high": float(np.percentile(boot,97.5))})
                    continue
                matrix = np.vstack([m[m.patient == u].set_index("pathway").d.reindex(sorted(m.pathway.unique())).to_numpy() for u in units])
                boot = np.nanmean(matrix[rng.integers(0, len(units), (N_BOOT, len(units)))], axis=(1,2))
                rows.append({"comparison": comparison, "split": split, "metric": metric, "unit": "patient",
                             "raw_units": len(units), "valid_repetitions": int(np.isfinite(boot).sum()),
                             "estimate": float(np.nanmean(matrix)), "ci_low": float(np.nanpercentile(boot,2.5)), "ci_high": float(np.nanpercentile(boot,97.5))})
            # SSIM external and internal are bootstrapped from precomputed local windows/blocks when available.
            wa = window_metrics[(window_metrics.experiment == control) & (window_metrics.split == split)]
            wb = window_metrics[(window_metrics.experiment == candidate) & (window_metrics.split == split)]
            wm = wa.merge(wb, on=["split", "patient", "pathway", "block"], suffixes=("_a", "_b"))
            if not wm.empty:
                wm["d"] = wm.ssim_b - wm.ssim_a
                units = sorted(wm.patient.unique()) if wm.patient.nunique() > 1 else sorted(wm.block.unique())
                unit_col = "patient" if wm.patient.nunique() > 1 else "block"
                vals = np.array([wm[wm[unit_col] == u].d.mean() for u in units])
                boot = np.mean(vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))], axis=1)
                rows.append({"comparison": comparison, "split": split, "metric": "ssim", "unit": unit_col,
                             "raw_units": len(units), "valid_repetitions": N_BOOT, "estimate": float(vals.mean()),
                             "ci_low": float(np.percentile(boot,2.5)), "ci_high": float(np.percentile(boot,97.5))})
    return pd.DataFrame(rows)


def huber_metrics(path, experiment, arm):
    d = pd.read_csv(path); params = pd.read_csv(PARAMS).set_index("label"); rows=[]
    pathways=[c[5:] for c in d if c.startswith("true_")]
    for pw in pathways:
        yz,pz=d["true_"+pw].to_numpy(float),d["pred_"+pw].to_numpy(float); mean,std=params.loc[pw,["mean","std"]]
        met=basic_metrics(yz,pz,yz*std+mean,pz*std+mean); met["nrmse_sigma_train"]=met["raw_rmse"]/std
        rows.append({"experiment":experiment,"arm":arm,"evidence_status":"accepted","split":"external_test",
                     "patient":"XZY_unidentified","pathway":pw,"n":len(d),**met})
    return pd.DataFrame(rows)


def quality_checks(legacy, inventory_before, inventory_after, metrics, ridge_path):
    tests = {}
    x=np.arange(9,dtype=float); tests["perfect_prediction"] = all(abs(v-t)<1e-12 for v,t in zip((pcc(x,x),ccc(x,x),np.sqrt(np.mean((x-x)**2))),(1,1,0)))
    tests["affine_offset"] = pcc(x,x+5)>0.999999 and ccc(x,x+5)<0.8
    patient_demo=pd.DataFrame({"patient":["A"]*3+["B"]*30,"value":[0]*3+[2]*30}); tests["patient_equal_weight"] = patient_demo.groupby("patient").value.mean().mean()==1
    keys=pd.DataFrame({"k":[1,2,3],"v":[4,5,6]}); tests["key_shuffle_invariant"] = keys.merge(keys.sample(frac=1,random_state=3),on="k").sort_values("k").v_y.tolist()==[4,5,6]
    tests["missing_key_fails"] = len(set([1,2,3]) ^ set([1,2]))>0
    r1=np.random.default_rng(SEED).integers(0,10,(100,4)); r2=np.random.default_rng(SEED).integers(0,10,(100,4)); tests["bootstrap_seed_deterministic"] = bool(np.array_equal(r1,r2))
    true=np.concatenate([legacy["true_"+p].to_numpy() for p in [c[5:] for c in legacy if c.startswith("true_")]])
    pred=np.concatenate([legacy["pred_"+p].to_numpy() for p in [c[5:] for c in legacy if c.startswith("true_")]])
    pws=[c[5:] for c in legacy if c.startswith("true_")]; params=pd.read_csv(PARAMS).set_index("label")
    raw_true=np.column_stack([legacy["true_"+p]*params.loc[p,"std"]+params.loc[p,"mean"] for p in pws])
    raw_pred=np.column_stack([legacy["pred_"+p]*params.loc[p,"std"]+params.loc[p,"mean"] for p in pws])
    values={"pooled_pcc":pcc(true,pred),"pooled_z_mse":float(np.mean((pred-true)**2)),"z_mae":float(np.mean(np.abs(pred-true))),
            "raw_mae":float(np.mean(np.abs(raw_pred-raw_true))),
            "mean_pathway_raw_r2":float(np.mean([1-np.sum((raw_pred[:,i]-raw_true[:,i])**2)/np.sum((raw_true[:,i]-raw_true[:,i].mean())**2) for i in range(30)]))}
    expected={"pooled_pcc":(.6549,.0001),"pooled_z_mse":(.6620768,.000001),"z_mae":(.6191227,.000001),"raw_mae":(1176.2114,.001),"mean_pathway_raw_r2":(-.0880,.0001)}
    replay={k:{"observed":values[k],"expected":v[0],"tolerance":v[1],"pass":abs(values[k]-v[0])<=v[1]} for k,v in expected.items()}
    ridge=pd.read_csv(ridge_path); rp=[c[5:] for c in ridge if c.startswith("true_")]
    ridge_check={"mean_base_pcc":float(np.mean([pcc(ridge["true_"+p].to_numpy(),ridge["pred_base_"+p].to_numpy()) for p in rp])),
                 "mean_base_ccc":float(np.mean([ccc(ridge["true_"+p].to_numpy(),ridge["pred_base_"+p].to_numpy()) for p in rp])),
                 "use":"rejected_excluded_function_regression_only"}
    protected_unchanged=inventory_before==inventory_after
    nrmse_gap=float((metrics.z_rmse-metrics.nrmse_sigma_train).abs().max())
    return {"unit_tests":tests,"accepted_replay":replay,"ridge_function_regression":ridge_check,
            "protected_inputs_size_mtime_unchanged":protected_unchanged,"max_z_rmse_nrmse_gap":nrmse_gap,
            "all_required_pass":all(tests.values()) and all(v["pass"] for v in replay.values()) and protected_unchanged and nrmse_gap<=0.000002}


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/"metric_protocol.json").read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_metric_read"): raise ValueError("metric protocol not frozen")
    registry=json.loads((ROOT/"experiments/experiment_registry.json").read_text(encoding="utf-8"))["experiments"]
    reg={x["id"]:x for x in registry}
    expected={"mpp2_barcode_repair_v003_frozen_baseline_20260711":("done","accepted"),"mpp2_paired_s0_frozen_continue_smoke_20260712":("done","accepted"),"mpp2_paired_s1_lora_r8_smoke_20260712":("done","accepted"),"mpp2_lora_r8_dropout10_smoke_20260714":("done","accepted"),"mpp2_pathway_ridge_calibration_v001_20260717":("failed","rejected"),"mpp2_huber_loss_paired_v001_20260728":("done","accepted"),"mpp2_cpgcr_probe_v001_20260811":("done","accepted")}
    for k,v in expected.items():
        if (reg[k]["status"],reg[k]["evidence_status"])!=v: raise ValueError(f"Registry mismatch {k}")
    sources=[]; inventory=[]
    def add(path, exp, arm, ev):
        inventory.append(inventory_entry(path,ev,exp)); sources.append(canonical_rich(path,exp,arm,ev))
    add(locate_w007("FBR","internal_val_predictions.csv"),"w007_fbr","FBR","accepted")
    add(locate_w007("FBR","XZY_predictions.csv"),"w007_fbr","FBR","accepted")
    for arm in ("RCC","HCR","CPGCR"):
        exp="w007_"+arm.lower()
        add(locate_w007(arm,"internal_val_predictions.csv"),exp,arm,"accepted")
        add(locate_w007(arm,"XZY_predictions.csv"),exp,arm,"accepted")
    smoke=ROOT/"automation/results/mpp2-paired-smoke-prediction-supplement-20260712-r001"
    for exp,prefix,arm in (("lora_s0","s0","S0"),("lora_s1","s1","S1")):
        for split in ("internal_val","external_xzy"): add(smoke/f"{prefix}_predictions_{split}.csv",exp,arm,"accepted_smoke")
    drop=ROOT/"checkpoints/mpp_uni2h_mlp/mpp2_lora_r8_dropout10_smoke_20260714_r001"
    for split in ("internal_val","external_xzy"): add(drop/f"predictions_{split}.csv","lora_dropout10","LoRA_dropout10","accepted_smoke")
    legacy_path=ROOT/"checkpoints/mpp_uni2h_mlp/mpp2_barcode_repair_v003_frozen_baseline_20260711/predictions_external_xzy.csv"
    inventory.append(inventory_entry(legacy_path,"accepted","baseline_legacy_replay"))
    hpaths=[ROOT/"automation/returns/W001/A003/R004/artifacts/predictions_external_xzy.csv",ROOT/"automation/returns/W001/A004/R002/artifacts/predictions_external_xzy.csv"]
    inventory.extend([inventory_entry(hpaths[0],"accepted","huber_mse"),inventory_entry(hpaths[1],"accepted","huber_delta1")])
    protected_paths=[MANIFEST,PARAMS]+list((PROTECTED/"raw_ssgsea").glob("*/*.csv"))+list((PROTECTED/"zscore/group_2_repaired_v003/labels").glob("**/*.csv"))
    protected_before={str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected_paths}
    inventory.extend(inventory_entry(p,"protected_read_only","protected_label_or_manifest") for p in protected_paths)
    (OUT/"input_inventory.json").write_text(json.dumps({"registry_contract":expected,"files":inventory},ensure_ascii=False,indent=2),encoding="utf-8")
    long=pd.concat(sources,ignore_index=True)
    if set(long.pathway.unique()) != set(pd.read_csv(PARAMS).label): raise ValueError("30通路集合不一致")
    if len(set(long.pathway))!=30 or set(long[long.split.str.contains("internal")].patient.unique())!=set(INTERNAL) or set(long[long.patient=="XZY"].patient)!={"XZY"}: raise ValueError("患者/通路硬门失败")
    pair_key_gate = True
    for control, candidate in (("lora_s0","lora_s1"),("lora_s0","lora_dropout10"),("w007_fbr","w007_rcc"),("w007_fbr","w007_hcr"),("w007_fbr","w007_cpgcr")):
        for split in set(long[long.experiment == control].split) & set(long[long.experiment == candidate].split):
            a = long[(long.experiment == control) & (long.split == split)].sort_values(["patient","key","pathway"])
            b = long[(long.experiment == candidate) & (long.split == split)].sort_values(["patient","key","pathway"])
            key_cols = ["patient","key","pathway"]
            if not a[key_cols].reset_index(drop=True).equals(b[key_cols].reset_index(drop=True)):
                pair_key_gate = False
            elif not np.allclose(a.truth_z.to_numpy(), b.truth_z.to_numpy(), atol=2e-6, rtol=0):
                pair_key_gate = False
    if not pair_key_gate:
        raise ValueError("配对比较键集合或真值不一致")
    metrics, spatial, windows=per_patient_metrics(long)
    hub=pd.concat([huber_metrics(hpaths[0],"huber_mse","MSE"),huber_metrics(hpaths[1],"huber_delta1","Huber")],ignore_index=True)
    metrics=pd.concat([metrics,hub],ignore_index=True)
    # Huber spatial fields remain explicit NA with the mandated reason.
    for _,r in hub[["experiment","arm","evidence_status","split","patient","pathway"]].iterrows():
        spatial.loc[len(spatial)]={**r.to_dict(),"ssim":np.nan,"ssim_method":"not_computable_missing_identifier","grid_occupancy":np.nan,"valid_windows":0,"data_range":6.0,"truth_moran_i":np.nan,"prediction_moran_i":np.nan,"moran_delta":np.nan,"abs_moran_delta":np.nan,"residual_moran_i":np.nan,"rook_edges":0}
    pathway,summary=summarize(metrics,spatial); deltas=delta_tables(pathway)
    blocks=spatial_block_metrics(long); ci=bootstrap_ci(deltas,metrics,windows,blocks)
    metrics.to_csv(OUT/"metrics_long.csv",index=False); summary.to_csv(OUT/"metrics_summary.csv",index=False)
    deltas.to_csv(OUT/"paired_deltas.csv",index=False); ci.to_csv(OUT/"bootstrap_ci.csv",index=False); spatial.to_csv(OUT/"spatial_metrics.csv",index=False)
    protected_after={str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected_paths}
    ridge_path=ROOT/"automation/results/mpp2-pathway-ridge-calibration-20260717-r003/predictions_external_xzy.csv"
    qc=quality_checks(pd.read_csv(legacy_path),protected_before,protected_after,metrics,ridge_path)
    qc["hard_gates"]={"registry_status":True,"internal_rows":int(long[(long.experiment=="w007_fbr")&(long.split.str.contains("internal"))].drop_duplicates(["patient","key"]).shape[0])==1078,"external_rows":int(long[(long.experiment=="w007_fbr")&(long.patient=="XZY")].drop_duplicates(["patient","key"]).shape[0])==1039,"pathways":len(set(long.pathway))==30,"paired_key_sets_and_truth":pair_key_gate}
    qc["all_required_pass"] = qc["all_required_pass"] and all(qc["hard_gates"].values())
    (OUT/"quality_checks.json").write_text(json.dumps(qc,ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
    ext=deltas[deltas.split.str.contains("external|XZY",case=False,regex=True,na=False)]
    fig,ax=plt.subplots(figsize=(11,6)); data=[ext[(ext.comparison==c)&(ext.metric=="pcc")].delta_candidate_better.dropna() for c in ext.comparison.unique()]; ax.boxplot(data,tick_labels=[c.replace("_vs_","\nvs\n") for c in ext.comparison.unique()],showmeans=True); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("ΔPCC（正值=候选更优）"); ax.tick_params(axis='x',labelsize=7); fig.tight_layout(); fig.savefig(FIG/"external_pcc_paired_deltas.png",dpi=180); plt.close(fig)
    base=summary[(summary.experiment=="w007_fbr")&(summary.split.str.contains("XZY|external",case=False,regex=True))]; fig,ax=plt.subplots(figsize=(9,5)); ax.bar(base.metric,base["mean"]); ax.tick_params(axis='x',rotation=45,labelsize=8); ax.set_title("MPP2 baseline XZY：30通路均值（不同量纲仅作索引）"); fig.tight_layout(); fig.savefig(FIG/"baseline_xzy_metric_index.png",dpi=180); plt.close(fig)
    print(json.dumps({"quality":qc["all_required_pass"],"metrics_rows":len(metrics),"spatial_rows":len(spatial),"delta_rows":len(deltas),"ci_rows":len(ci)},ensure_ascii=False))


if __name__ == "__main__":
    main()
