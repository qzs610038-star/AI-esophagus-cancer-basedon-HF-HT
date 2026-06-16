#!/usr/bin/env python3
"""
scripts/zscore_labels.py — Z-score existing split labels (no re-split)
==============================================================

For patients that already have correct train/val split but raw ssGSEA labels.
Reads raw CSV, z-scores 30 pathway columns (ddof=1), saves output + params.

Usage (server):
    set PYTHONIOENCODING=utf-8
    python scripts/zscore_labels.py --input D:\...\HYZ15040_ssGSEA.csv --output-dir D:\...\ssGSEA
    python scripts/zscore_labels.py --input D:\...\LMZ12939_ssGSEA.csv --output-dir D:\...\ssGSEA
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EXPECTED_NUM_PATHWAYS = 30


def main():
    parser = argparse.ArgumentParser(description="Z-score normalize existing patient labels")
    parser.add_argument("--input", required=True,
                        help="Path to raw ssGSEA CSV (e.g. HYZ15040_ssGSEA.csv)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output files (z-scored CSV + params)")
    parser.add_argument("--ddof", type=int, default=1,
                        help="Degrees of freedom for std (default: 1, sample std)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ddof = args.ddof

    if not input_path.is_file():
        sys.exit(f"ERROR: input not found: {input_path}")

    # Derive patient name from filename: HYZ15040_ssGSEA.csv -> HYZ15040
    stem = input_path.stem  # e.g. HYZ15040_ssGSEA
    patient = stem.replace("_ssGSEA", "")

    # --- load ---
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # --- rename first col if needed ---
    old_name = df.columns[0]
    if old_name != "patch_id":
        df.rename(columns={old_name: "patch_id"}, inplace=True)
        print(f"Renamed: \"{old_name}\" -> \"patch_id\"")

    # --- target columns ---
    target_cols = list(df.columns[1:])
    if len(target_cols) != _EXPECTED_NUM_PATHWAYS:
        print(f"WARNING: expected {_EXPECTED_NUM_PATHWAYS} pathways, got {len(target_cols)}")
    print(f"Target columns: {len(target_cols)} pathways")

    # --- numeric safety ---
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nan_count = df[target_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values, filling with 0")
        df[target_cols] = df[target_cols].fillna(0.0)

    # --- z-score ---
    means = df[target_cols].mean()
    stds = df[target_cols].std(ddof=ddof)

    zero_std = stds[stds == 0].index.tolist()
    if zero_std:
        print(f"WARNING: {len(zero_std)} columns have zero std")

    df_z = df.copy()
    for col in target_cols:
        if stds[col] != 0:
            df_z[col] = (df_z[col] - means[col]) / stds[col]

    # --- NaN/Inf guard ---
    df_z[target_cols] = df_z[target_cols].replace([np.inf, -np.inf], np.nan)
    inf_nan = df_z[target_cols].isna().sum().sum()
    if inf_nan > 0:
        print(f"WARNING: {inf_nan} Inf->NaN, filling with 0")
        df_z[target_cols] = df_z[target_cols].fillna(0.0)

    # --- verify ---
    print(f"\nPost-zscore (ddof={ddof}):")
    for col in target_cols[:3]:
        print(f"  {col:30s} mean={df_z[col].mean():+.6f}  std={df_z[col].std(ddof=ddof):.6f}")
    extreme_5 = (df_z[target_cols].abs() > 5).sum().sum()
    max_abs = df_z[target_cols].abs().max().max()
    if extreme_5:
        print(f"  |z|>5 = {extreme_5}, max|z| = {max_abs:.2f}")
    print(f"  ... ({len(target_cols)-3} more cols)")

    # --- save z-scored CSV ---
    out_csv = output_dir / f"{patient}_ssGSEA_zscore.csv"
    df_z.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    # --- save params ---
    params_df = pd.DataFrame({
        "pathway": target_cols,
        "mean": [means[c] for c in target_cols],
        "std": [stds[c] for c in target_cols],
        "count": [int(df[target_cols].count()[c]) for c in target_cols],
    })

    csv_params = output_dir / f"{patient}_zscore_params.csv"
    json_params = output_dir / f"{patient}_zscore_params.json"

    params_df.to_csv(csv_params, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_params}")

    params_json = {
        "description": f"{patient} z-score parameters for {len(target_cols)} pathways (ddof={ddof})",
        "ddof": ddof,
        "num_samples": int(df_z.shape[0]),
        "pathways": {
            row["pathway"]: {"mean": float(row["mean"]), "std": float(row["std"]), "count": int(row["count"])}
            for _, row in params_df.iterrows()
        },
    }
    with open(json_params, "w", encoding="utf-8") as f:
        json.dump(params_json, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_params}")

    print("\nDone.")


if __name__ == "__main__":
    main()
