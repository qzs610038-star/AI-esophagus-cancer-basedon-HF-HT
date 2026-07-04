"""
prepare_mpp_zscore.py — MPP 实验 z-score 标准化标签准备

只在训练集患者上拟合 mean/std（ddof=1），用同一套参数转换训练集 + 内部验证集 + 外部测试集。
clip ±100 只在 z-score 变换之后做——拟合前不做 clip，防止 LMZ 极端值失真。
输出 CSV 保留原始坐标/ID 列，以便 dataset 按坐标匹配 .pt 文件与标签。

用法:
    # V3 模式（无内部验证）
    python prepare_mpp_zscore.py --train_mpp_id 3 --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL,ZHZ --external_mpp_id 2 --external_patient XZY

    # V3bis 模式（内部验证）
    python prepare_mpp_zscore.py --train_mpp_id 3 --train_patients HYZ15040,JFX,LMZ12939,TGC,XSL --val_strategy internal --val_patient ZHZ --external_mpp_id 2 --external_patient XZY
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_labels(mpp_root: str, mpp_id: int, patient: str) -> pd.DataFrame:
    """加载 MPP 患者原始 ssGSEA CSV。"""
    csv_path = Path(mpp_root) / str(mpp_id) / patient / f"{patient}_ssGSEA.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def classify_columns(df: pd.DataFrame) -> tuple:
    """将原始 CSV 列分为通路列、坐标/ID列。

    Returns:
        (pathway_cols, coord_cols) — 两个列名列表
    """
    # 假设 id/坐标列名：全小写后匹配
    skip_names = {"x", "y", "spot", "id", "spot_id", "barcode", "index",
                  "array_row", "array_col"}
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    pathway_cols = [c for c in numeric_cols if c.lower() not in skip_names]
    coord_cols = [c for c in numeric_cols if c.lower() in skip_names and c.lower() in {"x", "y"}]
    # 也捕获非数值的 id 列
    all_cols = set(df.columns)
    id_cols = [c for c in all_cols - set(numeric_cols)
               if c.lower() in skip_names]
    # 非通路的数值列 = total_numeric - pathway
    other_num = [c for c in numeric_cols if c not in pathway_cols]
    # coord_cols 是其中的 subset；其他 id 列可能是非数值
    return pathway_cols, coord_cols, id_cols


def main():
    parser = argparse.ArgumentParser(description="MPP z-score 标准化标签准备")
    parser.add_argument("--mpp_root", default=r"D:\AIPatho\Patch\visiumhd_patch",
                        help="MPP 数据根目录")
    parser.add_argument("--train_mpp_id", type=int, required=True,
                        help="训练集 MPP 编号")
    parser.add_argument("--train_patients", required=True,
                        help="训练集患者列表，逗号分隔，如 HYZ15040,JFX,LMZ12939")
    parser.add_argument("--external_mpp_id", type=int, default=2,
                        help="外部测试集 MPP 编号（默认 2）")
    parser.add_argument("--external_patient", default="XZY",
                        help="外部测试患者（默认 XZY）")
    parser.add_argument("--val_strategy", default="none",
                        choices=["none", "internal"],
                        help="验证策略：none=V3 模式(无内部验证), internal=V3bis 模式")
    parser.add_argument("--val_patient", default=None,
                        help="内部验证患者（val_strategy=internal 时必需，如 ZHZ）")
    parser.add_argument("--output_root", default="mpp_uni2h_cache",
                        help="输出根目录")
    parser.add_argument("--clip", type=float, default=100.0,
                        help="z-score 变换后的 clip 阈值（默认 100）")
    args = parser.parse_args()

    # ── 参数校验 ──
    if args.val_strategy == "internal" and args.val_patient is None:
        print("[ERROR] --val_strategy internal 必须同时指定 --val_patient")
        sys.exit(1)
    if args.val_strategy == "none" and args.val_patient is not None:
        print("[WARN] --val_patient 已指定但 --val_strategy=none，将忽略 val_patient")
        args.val_patient = None

    output_dir = Path(args.output_root) / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    params_dir = Path(args.output_root)
    params_dir.mkdir(parents=True, exist_ok=True)

    train_patients = [p.strip() for p in args.train_patients.split(",")]
    print(f"训练集患者 ({len(train_patients)}): {train_patients}")
    print(f"外部测试: MPP-{args.external_mpp_id}/{args.external_patient}")
    if args.val_patient:
        print(f"内部验证: MPP-{args.train_mpp_id}/{args.val_patient} (val_strategy={args.val_strategy}, 不参与 z-score 拟合)")
    # 防泄漏检查：val_patient 不应在 train_patients 中
    if args.val_patient and args.val_patient in train_patients:
        print(f"[ERROR] val_patient ({args.val_patient}) 出现在 train_patients 中，违反数据隔离规则")
        sys.exit(1)
    if args.external_patient in train_patients:
        print(f"[ERROR] external_patient ({args.external_patient}) 出现在 train_patients 中，"
              f"外部测试患者不应参与 z-score 拟合")
        sys.exit(1)
    if args.external_patient == args.val_patient:
        print(f"[ERROR] external_patient ({args.external_patient}) 与 val_patient 相同，违反数据隔离规则")
        sys.exit(1)
    print("=" * 60)

    # ── Step 1: 加载并分类列 ──
    print("\n[1/5] 加载训练集原始标签并分类列 ...")
    train_pathway_cols = None
    all_coord_cols = None
    all_id_cols = None
    train_raw_dfs = {}  # patient → full DataFrame

    for patient in train_patients:
        df = load_raw_labels(args.mpp_root, args.train_mpp_id, patient)
        pw_cols, cc, ic = classify_columns(df)
        if train_pathway_cols is None:
            train_pathway_cols = pw_cols
            all_coord_cols = cc
            all_id_cols = ic
        else:
            train_pathway_cols = [c for c in train_pathway_cols if c in pw_cols]
        train_raw_dfs[patient] = df
        print(f"  {patient}: {len(df)} 样本, {len(train_pathway_cols)} 通路, "
              f"coord={cc}, id_cols={ic}")

    # 合并通路值用于拟合
    train_pw_values = np.concatenate(
        [df[train_pathway_cols].values for df in train_raw_dfs.values()], axis=0)
    print(f"  训练集合并: {train_pw_values.shape}")
    pathway_cols = train_pathway_cols  # alias for later use

    # ── Step 2: 清理（拟合前不做 clip！） ──
    print("\n[2/5] 清理 NaN/Inf (拟合前不做 clip) ...")
    cleaned = np.nan_to_num(train_pw_values, nan=0.0)
    cleaned = np.where(np.isinf(cleaned), np.sign(cleaned) * 1e6, cleaned)
    nan_before = np.isnan(train_pw_values).sum()
    inf_before = np.isinf(train_pw_values).sum()
    print(f"  替换 NaN: {nan_before}, Inf: {inf_before}")

    # ── Step 3: 拟合 z-score 参数（手动计算，ddof=1） ──
    print("\n[3/5] 拟合 z-score 参数 (ddof=1, 手动 np.nanstd) ...")
    mean = np.nanmean(cleaned, axis=0)
    std = np.nanstd(cleaned, axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)  # 防除零

    # 打印极端通路
    for i, col in enumerate(pathway_cols):
        if abs(mean[i]) > 1 or std[i] > 3:
            print(f"  {col}: mean={mean[i]:.4f}, std={std[i]:.4f}")

    print(f"  完成: n_train_samples={cleaned.shape[0]}, n_pathways={len(pathway_cols)}, ddof=1")

    # ── Step 4: 保存参数 ──
    # 构建通路→参数映射（用于变换时的精确查找，不依赖顺序）
    pw_to_mean = {col: float(mean[i]) for i, col in enumerate(pathway_cols)}
    pw_to_std = {col: float(std[i]) for i, col in enumerate(pathway_cols)}

    params = {
        "pathways": pathway_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ddof": 1,
        "n_train_samples": int(cleaned.shape[0]),
        "fit_patients": train_patients,
        "train_mpp_id": args.train_mpp_id,
        "val_strategy": args.val_strategy,
        "val_patient": args.val_patient,
        "external_mpp_id": args.external_mpp_id,
        "external_patient": args.external_patient,
        "coord_cols": all_coord_cols,
        "id_cols": all_id_cols,
        "clip_applied_after_transform": True,
        "clip_range": [-args.clip, args.clip],
        "notes": "Manual np.nanmean/np.nanstd(ddof=1), NOT sklearn StandardScaler (which uses ddof=0). "
                 "nan_to_num+Inf sentinel pass BEFORE fitting; clip ±100 applied only AFTER (x-mean)/std transform. "
                 + (f"val_patient={args.val_patient} excluded from fit; used for internal val only." if args.val_patient else ""),
    }

    params_path = params_dir / f"zscore_params_mpp{args.train_mpp_id}.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n  参数已保存: {params_path}")

    # ── Step 5: 转换并保存（训练集 + 外部） ──
    def transform_and_save(patient: str, mpp_id: int, prefix: str):
        """加载原始标签 → 保留 coord/id 列 → z-score 变换通路列 → clip → 保存"""
        df_raw = load_raw_labels(args.mpp_root, mpp_id, patient)

        # 对齐通路列（用 dict 查找确保即使通路顺序不同也能正确匹配）
        avail_pw = [c for c in pathway_cols if c in df_raw.columns]
        if len(avail_pw) < len(pathway_cols):
            missing = set(pathway_cols) - set(avail_pw)
            print(f"  [WARN] {patient} 缺少 {len(missing)} 条通路: {missing}")

        # 数值清理
        values = np.nan_to_num(df_raw[avail_pw].values, nan=0.0)

        # 用训练集参数变换（dict 查找，不依赖位置）
        z = np.zeros_like(values)
        for i, col in enumerate(avail_pw):
            z[:, i] = (values[:, i] - pw_to_mean[col]) / pw_to_std[col]
        z = np.clip(z, -args.clip, args.clip)  # clip ONLY after transform

        # 构建输出 DataFrame: [坐标/ID列] + [z-scored 通路列]
        out_df = pd.DataFrame()
        # 保留所有坐标列（若存在）
        for c in all_coord_cols:
            if c in df_raw.columns:
                out_df[c] = df_raw[c].values
        # 保留所有 ID 列（若存在）
        for c in all_id_cols:
            if c in df_raw.columns:
                out_df[c] = df_raw[c].values
        # z-scored 通路列
        for i, col in enumerate(avail_pw):
            out_df[col] = z[:, i]

        out_path = output_dir / f"mpp{mpp_id}_{patient}_zscored.csv"
        out_df.to_csv(out_path, index=False)

        print(f"  {prefix} {patient} (MPP-{mpp_id}): "
              f"rows={len(out_df)}, cols={list(out_df.columns)[:4]}..., "
              f"z_min={z.min():.4f}, z_max={z.max():.4f}, z_mean={z.mean():.4f}, "
              f"NaN={np.isnan(z).sum()}, Inf={np.isinf(z).sum()}")
        return out_path

    print("\n[4/5] 转换训练集 ...")
    for patient in train_patients:
        transform_and_save(patient, args.train_mpp_id, "[TRAIN]")

    # ── 内部验证集（val_strategy=internal） ──
    if args.val_patient:
        print(f"\n[4a/5] 转换内部验证集 MPP-{args.train_mpp_id}/{args.val_patient} (不参与 z-score 拟合) ...")
        transform_and_save(args.val_patient, args.train_mpp_id, "[INTERNAL_VAL]")

    print("\n[5/5] 转换外部测试集 ...")
    transform_and_save(args.external_patient, args.external_mpp_id, "[EXTERNAL]")

    print(f"\n{'='*60}")
    print("z-score 标准化完成!")
    print(f"参数文件: {params_path}")
    print(f"标签输出: {output_dir}/")
    print(f"训练集拟合患者: {train_patients}")
    if args.val_patient:
        print(f"内部验证患者 (不参与拟合): {args.val_patient}")
    print(f"XZY 参与变换: 是  |  参与拟合: 否")


if __name__ == "__main__":
    main()
