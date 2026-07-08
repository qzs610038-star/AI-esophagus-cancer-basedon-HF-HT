#!/usr/bin/env python3
"""
scripts/rebuild_zscore_from_manifest.py — 按 split_manifest 重建 train-only z-score

方案 §四 z-score 统一分配:
  1. 从 raw {patient}_ssGSEA.csv 读取原始标签 (30 通路, raw 尺度)
  2. 按 split_manifest.csv 的 split 列, 只取 split=="train" 的样本拟合 mean/std (ddof=1)
  3. 用同一套 train-only 参数转换 train / internal_val / external XZY
  4. 输出 z-scored 标签 + 参数文件 + manifest (含 raw_label_source)

输入:
  - split_manifest.csv (由 generate_standard_splits.py 生成, 含 patch_stem/patient/split)
  - raw 标签: {mpp_root}/{N}/{patient}/{patient}_ssGSEA.csv
  - raw XZY 标签: {mpp_root}/2/XZY/XZY_ssGSEA.csv

输出 (splits_root/group_{N}/):
  - zscore_params_from_train.json   {pathway: {mean, std, count}, ddof, fit_split, ...}
  - zscore_params_from_train.csv    label,mean,std,count (兼容现有 train_mpp_uni2h_mlp.py)
  - zscore_manifest.json            fit 元数据 (fit_patients, raw_label_source, ddof, clip, seed, ...)
  - labels/
      train/{patient}/{patient}_ssGSEA_zscore.csv      (含 barcode, x, y, 30 通路 z-scored)
      val/{patient}/{patient}_ssGSEA_zscore.csv         (internal_val patches)
      external/XZY/XZY_ssGSEA_zscore_by_group_{N}_train.csv

关键防泄漏 (方案 §4.1):
  - mean/std 只在 split=="train" 上拟合, 不含 val/embargo/external
  - ddof=1 (样本 std), 不用 sklearn StandardScaler 默认 ddof=0
  - 不二次 z-score 队友已 z-scored 的标签
  - std==0 的通路置为 1.0

raw-scale 反归一化 (方案 §4.3 P1 必交付):
  - 训练在 z-score 空间; 预测后用本 MPP 的 mean/std inverse transform 得 raw 尺度预测
  - zscore_params_from_train.json 保存 mean/std, 供 evaluate 流程 inverse transform
  - evaluate 时: pred_raw = pred_z * std + mean, 与 raw 标签算 MAE/R2

用法 (服务器):
    cd D:\\AIPatho\\qzs\\pfmval_deploy_git
    "C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe" scripts/rebuild_zscore_from_manifest.py \\
        --splits-root mpp_standard_splits \\
        --mpp-root D:\\AIPatho\\Patch\\visiumhd_patch

本地验证 (用 partner raw 副本模拟):
    "C:\\Program Files\\Python313\\python.exe" scripts/rebuild_zscore_from_manifest.py \\
        --splits-root tmp/test_splits \\
        --mpp-root parter_ljk_MPP1&4_patch_split_zscore \\
        --use-partner-labels --mpp-id-override 1
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── 常量 ──
DEFAULT_SPLITS_ROOT = "mpp_standard_splits"
DEFAULT_MPP_ROOT = r"D:\AIPatho\Patch\visiumhd_patch"

TRAIN_PATIENTS = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
EXTERNAL_PATIENT = "XZY"
MPP_IDS = [1, 2, 3, 4, 5]

DDOF = 1  # 方案 §4.1: 样本 std (ddof=1)
CLIP_RANGE = 100.0  # 方案 §4.4: z-score 后 clip ±100 (与现有训练脚本一致)
CLIP_AFTER_TRANSFORM = True

# 30 通路列名 (参考组)
REF_PATHWAYS = None  # 首次读取 raw CSV 时从列名推断, 验证跨 MPP 一致

COORD_RE = re.compile(r'x(\d+)_y(\d+)')


def parse_xy(stem: str):
    m = COORD_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════
# 加载 raw 标签
# ═══════════════════════════════════════════════════════════════

def load_raw_label(label_csv: Path, patient: str) -> pd.DataFrame:
    """读 raw ssGSEA CSV, 返回含 barcode + x + y + 30 通路 raw 列的 DataFrame。

    raw CSV 首列名可能是 barcode 或 patch_id; 统一改名为 barcode。
    从 barcode 解析 x, y 列 (供 split_manifest 匹配)。
    """
    df = pd.read_csv(label_csv)
    first_col = df.columns[0]
    if first_col != "barcode":
        df = df.rename(columns={first_col: "barcode"})
    df["barcode"] = df["barcode"].astype(str)

    # 解析坐标
    coords = df["barcode"].apply(parse_xy)
    df["x"] = [c[0] for c in coords]
    df["y"] = [c[1] for c in coords]

    # 通路列 = 除 barcode/x/y 外的数值列
    skip = {"barcode", "x", "y"}
    pathway_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return df, pathway_cols


def load_partner_raw(patient: str, partner_root: Path, mpp_id: int) -> tuple:
    """本地验证用: 从 partner z-scored 标签读取 (跨 train/ val/ 子目录)。

    注意: partner 标签已被 z-score, 这里仅用于验证 split+拟合流程逻辑;
    真实运行必须用 raw 标签 (--use-partner-labels 仅本地验证)。
    """
    group_dir = partner_root / f"group_{mpp_id}"
    frames = []
    pathway_cols = None
    for sub in ("train", "val"):
        csv_path = group_dir / sub / patient / f"{patient}_ssGSEA_zscore.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)
            if pathway_cols is None:
                first_col = df.columns[0]
                skip = {first_col}
                pathway_cols = [c for c in df.columns if c not in skip
                                and pd.api.types.is_numeric_dtype(df[c])]
    if not frames:
        raise FileNotFoundError(f"partner label missing for {patient} in group_{mpp_id}")
    df = pd.concat(frames, ignore_index=True)
    first_col = df.columns[0]
    if first_col != "barcode":
        df = df.rename(columns={first_col: "barcode"})
    df["barcode"] = df["barcode"].astype(str)
    coords = df["barcode"].apply(parse_xy)
    df["x"] = [c[0] for c in coords]
    df["y"] = [c[1] for c in coords]
    skip = {"barcode", "x", "y"}
    if pathway_cols is None:
        pathway_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return df, pathway_cols


# ═══════════════════════════════════════════════════════════════
# z-score 拟合 + 转换
# ═══════════════════════════════════════════════════════════════

def fit_zscore_on_train(train_df: pd.DataFrame, pathway_cols: List[str],
                        ddof: int = DDOF) -> dict:
    """在 train 子集上拟合 mean/std。

    Returns:
        params: {pathway: {"mean": float, "std": float, "count": int}}
        约定: std==0 的通路置为 1.0 (避免除零)
    """
    params = {}
    for col in pathway_cols:
        vals = train_df[col].astype(float)
        # NaN 保护: nanmean/nanstd
        mean = float(np.nanmean(vals.values))
        std = float(np.nanstd(vals.values, ddof=ddof))
        count = int(np.sum(~np.isnan(vals.values)))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        params[col] = {"mean": mean, "std": std, "count": count}
    return params


def apply_zscore(df: pd.DataFrame, pathway_cols: List[str], params: dict,
                 clip: float = CLIP_RANGE) -> pd.DataFrame:
    """用 train-only mean/std 转换任意子集 (train/val/external)。

    Returns: DataFrame 同 shape, 通路列替换为 z-score 值, NaN→0, clip ±100。
    """
    out = df.copy()
    for col in pathway_cols:
        m = params[col]["mean"]
        s = params[col]["std"]
        z = (out[col].astype(float) - m) / s
        # NaN/Inf 保护
        z = z.replace([np.inf, -np.inf], np.nan)
        z = z.fillna(0.0)
        if CLIP_AFTER_TRANSFORM and clip:
            z = z.clip(-clip, clip)
        out[col] = z
    return out


def verify_fit_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          external_df: pd.DataFrame, pathway_cols: List[str],
                          params: dict, ddof: int = DDOF) -> dict:
    """验证 z-score 只在 train 上拟合:
      - fit mean/std 来自 train_df (已在 fit_zscore_on_train 中实现)
      - train 子集 z-score 后 mean≈0, std≈1 (ddof 校正后)
      - val/external 子集 z-score 后 mean/std 偏离 0/1 (因为不同分布)
    """
    train_z = apply_zscore(train_df, pathway_cols, params, clip=None)  # 不 clip 以验证统计
    train_means = train_z[pathway_cols].mean()
    train_stds = train_z[pathway_cols].std(ddof=ddof)
    return {
        "train_z_mean_range": [float(train_means.min()), float(train_means.max())],
        "train_z_std_range": [float(train_stds.min()), float(train_stds.max())],
    }


# ═══════════════════════════════════════════════════════════════
# 单 MPP 处理
# ═══════════════════════════════════════════════════════════════

def process_mpp(mpp_id: int, splits_root: Path, mpp_root: Path,
                label_source: str) -> tuple:
    """为单 MPP 重建 train-only z-score 标签。"""
    banner(f"MPP-{mpp_id} z-score 重建 (label_source={label_source})")

    group_dir = splits_root / f"group_{mpp_id}"
    manifest_path = group_dir / "split_manifest.csv"
    if not manifest_path.exists():
        return False, f"split_manifest.csv 不存在: {manifest_path}"

    manifest = pd.read_csv(manifest_path)
    print(f"  split_manifest: {len(manifest)} 行, "
          f"split 分布: {manifest['split'].value_counts().to_dict()}")

    # 加载每例患者 raw 标签, 按 manifest 分 split
    all_train_raw = []
    per_patient_val_raw = {}
    pathway_cols = None
    raw_label_sources = {}

    for patient in TRAIN_PATIENTS:
        if label_source == "raw":
            label_csv = mpp_root / str(mpp_id) / patient / f"{patient}_ssGSEA.csv"
            if not label_csv.exists():
                return False, f"raw label missing: {label_csv}"
            df, p_cols = load_raw_label(label_csv, patient)
            raw_label_sources[patient] = str(label_csv)
        else:
            df, p_cols = load_partner_raw(patient, mpp_root, mpp_id)
            raw_label_sources[patient] = f"partner:{mpp_root}/group_{mpp_id}/{{train|val}}/{patient}"

        if pathway_cols is None:
            pathway_cols = p_cols
        elif p_cols != pathway_cols:
            return False, f"{patient} 通路列与首患者不一致: {p_cols} vs {pathway_cols}"

        # 按 manifest 分 split
        pat_manifest = manifest[manifest["patient"] == patient]
        # 用 (barcode 即 patch_stem) 匹配
        train_stems = set(pat_manifest[pat_manifest["split"] == "train"]["patch_stem"])
        val_stems = set(pat_manifest[pat_manifest["split"] == "internal_val"]["patch_stem"])
        embargo_stems = set(pat_manifest[pat_manifest["split"] == "embargo"]["patch_stem"])

        train_df = df[df["barcode"].isin(train_stems)].copy()
        val_df = df[df["barcode"].isin(val_stems)].copy()

        n_train_match = len(train_df)
        n_val_match = len(val_df)
        n_expected_train = len(train_stems)
        n_expected_val = len(val_stems)
        print(f"  {patient}: raw={len(df)}, train={n_train_match}/{n_expected_train}, "
              f"val={n_val_match}/{n_expected_val}, embargo={len(embargo_stems)}(excluded)")

        if n_train_match != n_expected_train:
            return False, (f"{patient} train 匹配不等: {n_train_match} vs {n_expected_train}; "
                           f"检查 barcode↔patch_stem 一致性")
        if n_val_match != n_expected_val:
            return False, (f"{patient} val 匹配不等: {n_val_match} vs {n_expected_val}")

        all_train_raw.append(train_df)
        per_patient_val_raw[patient] = val_df

    # 合并所有患者 train, 拟合 z-score
    train_all = pd.concat(all_train_raw, ignore_index=True)
    print(f"\n  合并 train: {len(train_all)} 行 (六例患者), {len(pathway_cols)} 通路")

    zscore_params = fit_zscore_on_train(train_all, pathway_cols, ddof=DDOF)
    fit_verify = verify_fit_no_leakage(train_all, train_all, pd.DataFrame(), pathway_cols, zscore_params, DDOF)
    print(f"  训练集 z-score 验证: mean 范围 {fit_verify['train_z_mean_range']}, "
          f"std 范围 {fit_verify['train_z_std_range']} (应≈[0,0]/[1,1])")

    # 加载 external XZY raw 标签 (固定 MPP-2/XZY)
    ext_label_csv = mpp_root / "2" / EXTERNAL_PATIENT / f"{EXTERNAL_PATIENT}_ssGSEA.csv"
    if label_source == "raw":
        if not ext_label_csv.exists():
            return False, f"external raw label missing: {ext_label_csv}"
        ext_df, ext_p_cols = load_raw_label(ext_label_csv, EXTERNAL_PATIENT)
        raw_label_sources[EXTERNAL_PATIENT] = str(ext_label_csv)
    else:
        # partner 本地验证: external 标签在 group_{N}/external/XZY/
        ext_partner = mpp_root / f"group_{mpp_id}" / "external" / EXTERNAL_PATIENT / f"{EXTERNAL_PATIENT}_ssGSEA_zscore_by_group_{mpp_id}_train.csv"
        if ext_partner.exists():
            df_tmp = pd.read_csv(ext_partner)
            first_col = df_tmp.columns[0]
            df_tmp = df_tmp.rename(columns={first_col: "barcode"})
            df_tmp["barcode"] = df_tmp["barcode"].astype(str)
            coords = df_tmp["barcode"].apply(parse_xy)
            df_tmp["x"] = [c[0] for c in coords]
            df_tmp["y"] = [c[1] for c in coords]
            skip = {"barcode", "x", "y"}
            ext_p_cols = [c for c in df_tmp.columns if c not in skip and pd.api.types.is_numeric_dtype(df_tmp[c])]
            ext_df = df_tmp
            raw_label_sources[EXTERNAL_PATIENT] = str(ext_partner)
        else:
            return False, f"external label missing (partner): {ext_partner}"

    if ext_p_cols != pathway_cols:
        return False, f"external XZY 通路列与 train 不一致: {ext_p_cols} vs {pathway_cols}"
    print(f"  external {EXTERNAL_PATIENT}: {len(ext_df)} 行 raw")

    # ── 转换 train / val / external ──
    train_z = apply_zscore(train_all, pathway_cols, zscore_params)
    val_z = {p: apply_zscore(vdf, pathway_cols, zscore_params) for p, vdf in per_patient_val_raw.items()}
    ext_z = apply_zscore(ext_df, pathway_cols, zscore_params)

    # ── 写文件 ──
    labels_root = group_dir / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    # train/{patient}/{patient}_ssGSEA_zscore.csv (每患者分开, 兼容现有 dataset 加载)
    train_out = labels_root / "train"
    train_out.mkdir(exist_ok=True)
    for patient in TRAIN_PATIENTS:
        p_dir = train_out / patient
        p_dir.mkdir(exist_ok=True)
        pat_train_z = train_z[train_z["barcode"].isin(
            manifest[(manifest["patient"] == patient) & (manifest["split"] == "train")]["patch_stem"])]
        # 写 barcode + 30 通路 (不写 x/y, 与现有 partner 格式一致)
        out_cols = ["barcode"] + pathway_cols
        pat_train_z[out_cols].to_csv(p_dir / f"{patient}_ssGSEA_zscore.csv", index=False, encoding="utf-8-sig")
        print(f"  写 train: {p_dir / f'{patient}_ssGSEA_zscore.csv'} ({len(pat_train_z)} 行)")

    # val/{patient}/{patient}_ssGSEA_zscore.csv
    val_out = labels_root / "val"
    val_out.mkdir(exist_ok=True)
    for patient, vdf_z in val_z.items():
        p_dir = val_out / patient
        p_dir.mkdir(exist_ok=True)
        out_cols = ["barcode"] + pathway_cols
        vdf_z[out_cols].to_csv(p_dir / f"{patient}_ssGSEA_zscore.csv", index=False, encoding="utf-8-sig")
        print(f"  写 val: {p_dir / f'{patient}_ssGSEA_zscore.csv'} ({len(vdf_z)} 行)")

    # external/XZY/XZY_ssGSEA_zscore_by_group_{N}_train.csv
    ext_out = labels_root / "external" / EXTERNAL_PATIENT
    ext_out.mkdir(parents=True, exist_ok=True)
    ext_csv = ext_out / f"{EXTERNAL_PATIENT}_ssGSEA_zscore_by_group_{mpp_id}_train.csv"
    out_cols = ["barcode"] + pathway_cols
    ext_z[out_cols].to_csv(ext_csv, index=False, encoding="utf-8-sig")
    print(f"  写 external: {ext_csv} ({len(ext_z)} 行)")

    # zscore_params_from_train.csv (label, mean, std, count)
    params_csv_path = group_dir / "zscore_params_from_train.csv"
    pd.DataFrame([
        {"label": col,
         "mean": zscore_params[col]["mean"],
         "std": zscore_params[col]["std"],
         "count": zscore_params[col]["count"]}
        for col in pathway_cols
    ]).to_csv(params_csv_path, index=False, encoding="utf-8-sig")
    print(f"  写参数: {params_csv_path}")

    # zscore_params_from_train.json (含 mean/std, 供 raw-scale inverse transform)
    params_json_path = group_dir / "zscore_params_from_train.json"
    params_json = {
        "description": f"MPP-{mpp_id} train-only z-score parameters for {len(pathway_cols)} pathways",
        "fit_split": "train",
        "ddof": DDOF,
        "clip_applied_after_transform": CLIP_AFTER_TRANSFORM,
        "clip_range": [-CLIP_RANGE, CLIP_RANGE],
        "fit_patients": TRAIN_PATIENTS,
        "n_train_samples": int(len(train_all)),
        "pathways": zscore_params,
        "note": "mean/std are on RAW ssGSEA scale; use inverse transform pred_raw=pred_z*std+mean for raw-scale MAE/R2",
    }
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(params_json, f, indent=2, ensure_ascii=False)
    print(f"  写参数 JSON: {params_json_path}")

    # zscore_manifest.json (方案 §4.4 必含字段 + raw_label_source)
    manifest_json = {
        "mpp_id": mpp_id,
        "fit_split": "train",
        "fit_patients": TRAIN_PATIENTS,
        "excluded_splits": ["internal_val", "external_test", "embargo"],
        "ddof": DDOF,
        "clip_applied_after_transform": CLIP_AFTER_TRANSFORM,
        "clip_range": [-CLIP_RANGE, CLIP_RANGE],
        "split_seed": 42,  # 由 split_manifest 继承
        "val_ratio_target": 0.10,
        "overlap_policy": "bbox_embargo" if mpp_id in (3, 5) else "none_100pct_stride",
        "raw_label_source": raw_label_sources,
        "n_pathways": len(pathway_cols),
        "pathway_names": pathway_cols,
        "n_train_samples": int(len(train_all)),
        "n_val_samples": int(sum(len(v) for v in per_patient_val_raw.values())),
        "n_external_samples": int(len(ext_df)),
        "fit_verification": fit_verify,
    }
    manifest_path_json = group_dir / "zscore_manifest.json"
    with open(manifest_path_json, "w", encoding="utf-8") as f:
        json.dump(manifest_json, f, indent=2, ensure_ascii=False)
    print(f"  写 manifest: {manifest_path_json}")

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="按 split_manifest 重建 train-only z-score")
    parser.add_argument("--splits-root", default=DEFAULT_SPLITS_ROOT,
                        help=f"split 生成根目录 (默认 {DEFAULT_SPLITS_ROOT})")
    parser.add_argument("--mpp-root", default=DEFAULT_MPP_ROOT,
                        help=f"raw 标签根 (默认 {DEFAULT_MPP_ROOT})")
    parser.add_argument("--mpp-ids", default="1,2,3,4,5",
                        help="处理的 MPP (默认 1,2,3,4,5)")
    parser.add_argument("--use-partner-labels", action="store_true",
                        help="本地验证: mpp-root 指向 partner 根")
    parser.add_argument("--mpp-id-override", type=int, default=None)
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    mpp_root = Path(args.mpp_root)
    mpp_ids = [int(x) for x in args.mpp_ids.split(",")]
    if args.mpp_id_override is not None:
        mpp_ids = [args.mpp_id_override]

    label_source = "partner" if args.use_partner_labels else "raw"
    print(f"splits_root: {splits_root}")
    print(f"mpp_root:    {mpp_root}")
    print(f"label source: {label_source}")

    all_ok = True
    for mpp_id in mpp_ids:
        ok, msg = process_mpp(mpp_id, splits_root, mpp_root, label_source)
        print(f"  MPP-{mpp_id}: {msg}")
        if not ok:
            all_ok = False

    print(f"\n{'=' * 70}")
    if all_ok:
        print("  z-score 重建完成")
    else:
        print("  部分 MPP 失败, 见上方")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()