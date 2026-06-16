#!/usr/bin/env python3
"""
scripts/prepare_jfx_fixed_data.py — JFX0729 fixed data preparation
====================================================================

Parameterized pipeline to process JFX0729 corrected spatial transcriptomics data:
  1. Split patches -> train/val with configurable seed+fraction, copy (not move).
  2. Z-score normalize pathway columns (ddof=1, sample std).
  3. Output split_manifest.csv + zscore_params.csv + zscore_params.json.
  4. Run 7 integrity checks before exiting.

Why not split.py / zscore.py?
  - split.py  — hardcoded HYZ path, shutil.move, distance-based filtering.
  - zscore.py — hardcoded NUM_TARGET_COLS=8, hardcoded CSV_PATH.

Required Python environment:
  - Python 3.10+ with: numpy, pandas
  - Project root on sys.path (for dataset_online import in phase 3 check 7)
  - Actual execution: C:/Program Files/Python313/python.exe (torch 2.6.0+cu118)
  - NOT the pfmval_py310 conda env (lacks sklearn/torch)
  - Script sets sys.path automatically; just run from project root.

Usage:
    cd d:/AI空间转录病理研究/PFMval_new
    "C:/Program Files/Python313/python.exe" scripts/prepare_jfx_fixed_data.py

    # Or with explicit overrides:
    "C:/Program Files/Python313/python.exe" scripts/prepare_jfx_fixed_data.py \
        --patch-source data_new_3ST/JFX_fixed_data/JFX0729 \
        --label-source data_new_3ST/JFX_fixed_data/JFX0729_ssGSEA.csv \
        --patch-output data_new_3ST/patch_noov_spilt/JFX0729_noov_split \
        --label-output data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (used as argparse defaults; runtime values come from args.*)
# ---------------------------------------------------------------------------
_DEFAULT_SEED = 42
_DEFAULT_VAL_FRACTION = 0.1
_DEFAULT_DDOF = 1  # sample standard deviation (matches zscore.py:13)
_EXPECTED_NUM_PATHWAYS = 30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_coordinates(filename: str):
    """Extract (x, y) from 'patch_x10192_y10192.png' or bare stem.

    Returns (x: int, y: int) or (None, None).  Regex equivalent to split.py:12.
    """
    m = re.search(r'patch_x(\d+)_y(\d+)', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Phase 1: patch split
# ---------------------------------------------------------------------------

def phase1_split(patches_dir: Path, output_dir: Path,
                 seed: int, val_count: int) -> pd.DataFrame:
    """Split patches into train_patches/ and val_patches/ via copy.

    Args:
        patches_dir: source directory of .png files
        output_dir: where to create train_patches/ and val_patches/
        seed: random seed for deterministic shuffle
        val_count: exact number of validation patches (absolute count)

    Returns manifest DataFrame with columns:
        patch_id, split, x, y, source_path
    """
    banner("Phase 1 — Patch Split (copy, no distance filter)")

    # --- scan source ---
    patch_files = sorted(
        f for f in os.listdir(patches_dir) if f.lower().endswith(".png")
    )
    total = len(patch_files)
    print(f"   Found {total} .png patches")

    stems = [f[:-4] for f in patch_files]  # strip .png

    # --- split: use absolute val_count to avoid float rounding ---
    # Equivalent set split to sklearn train_test_split(..., test_size=val_count,
    # random_state=seed, shuffle=True), without requiring scikit-learn.
    if val_count <= 0 or val_count >= total:
        raise ValueError(f"val_count must be between 1 and total-1, got {val_count} for total={total}")
    shuffled_idx = np.random.RandomState(seed).permutation(total)
    val_idx = shuffled_idx[:val_count]
    train_idx = shuffled_idx[val_count:]
    train_stems = [stems[i] for i in train_idx]
    val_stems = [stems[i] for i in val_idx]
    print(f"   Train: {len(train_stems)}  |  Val: {len(val_stems)}  |  Total: {len(train_stems) + len(val_stems)}")

    # --- clean + create output dirs ---
    train_out = output_dir / "train_patches"
    val_out = output_dir / "val_patches"

    # Remove existing output to prevent stale-file contamination
    for d in [train_out, val_out]:
        if d.exists():
            shutil.rmtree(d)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    print(f"   Output dirs cleaned and re-created")

    # --- copy files + build manifest ---
    rows = []
    for stem_list, split_name, dst_dir in [
        (train_stems, "train", train_out),
        (val_stems, "val", val_out),
    ]:
        for stem in stem_list:
            fname = f"{stem}.png"
            src = patches_dir / fname
            dst = dst_dir / fname
            shutil.copy2(src, dst)
            x, y = parse_coordinates(stem)
            rows.append({
                "patch_id": stem,
                "split": split_name,
                "x": x,
                "y": y,
                "source_path": str(src),
            })

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"   Manifest saved -> {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# Phase 2: z-score
# ---------------------------------------------------------------------------

def phase2_zscore(csv_path: Path, output_path: Path, params_dir: Path,
                  seed: int, ddof: int):
    """Z-score normalize pathway columns.

    Rules:
      - Rename first column to 'patch_id'.
      - Target columns = df.columns[1:] (must match _EXPECTED_NUM_PATHWAYS).
      - ddof passed as arg (default 1 = sample std, matches zscore.py:13).
      - Save z-score params to both .csv and .json.
    """
    banner("Phase 2 — Z-Score Normalization")

    # --- load ---
    df = pd.read_csv(csv_path)
    print(f"   Loaded CSV: {df.shape[0]} rows x {df.shape[1]} cols")

    # --- rename first col ---
    old_name = df.columns[0]
    df.rename(columns={old_name: "patch_id"}, inplace=True)
    print(f"   Renamed first column: \"{old_name}\" -> \"patch_id\"")

    # --- validate target columns ---
    target_cols = list(df.columns[1:])
    if len(target_cols) != _EXPECTED_NUM_PATHWAYS:
        print(f"   WARNING: expected {_EXPECTED_NUM_PATHWAYS} pathway cols, got {len(target_cols)}")
        print(f"   Target cols: {target_cols}")
    else:
        print(f"   Target columns: {len(target_cols)} pathways")

    # --- convert to numeric (safety) ---
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nan_count = df[target_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"   WARNING: {nan_count} NaN values found in target columns, filling with 0")
        df[target_cols] = df[target_cols].fillna(0.0)

    # --- compute stats (using args.ddof, not hardcoded constant) ---
    means = df[target_cols].mean()
    stds = df[target_cols].std(ddof=ddof)
    counts = df[target_cols].count()

    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print(f"   WARNING: {len(zero_std_cols)} columns have zero std (skipping z-score for these):")
        for c in zero_std_cols:
            print(f"      - {c}")

    # --- z-score ---
    df_z = df.copy()
    valid_cols = [c for c in target_cols if stds[c] != 0]
    for col in valid_cols:
        df_z[col] = (df_z[col] - means[col]) / stds[col]

    # --- verify ---
    print(f"\n   Post-zscore verification (ddof={ddof}):")
    for col in target_cols[:3]:
        m = df_z[col].mean()
        s = df_z[col].std(ddof=ddof)
        print(f"      {col:30s}  mean={m:+.6f}  std={s:.6f}")
    if len(target_cols) > 3:
        print(f"      ... ({len(target_cols) - 3} more columns)")

    # --- NaN/Inf protection ---
    df_z[target_cols] = df_z[target_cols].replace([np.inf, -np.inf], np.nan)
    inf_nan_count = df_z[target_cols].isna().sum().sum()
    if inf_nan_count > 0:
        print(f"   WARNING: {inf_nan_count} Inf->NaN values after z-score, filling with 0")
        df_z[target_cols] = df_z[target_cols].fillna(0.0)

    # Report z-score extremes for training stability awareness
    extreme_5 = (df_z[target_cols].abs() > 5).sum().sum()
    extreme_6 = (df_z[target_cols].abs() > 6).sum().sum()
    max_abs = df_z[target_cols].abs().max().max()
    if extreme_5 > 0:
        print(f"\n   Z-score extreme values: |z|>5 = {extreme_5}, |z|>6 = {extreme_6}, max|z| = {max_abs:.2f}")
        print(f"   (LMZ12939-like extreme z-scores detected; monitor training loss/grad)")

    # --- save z-scored CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_z.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n   Z-scored CSV saved -> {output_path}")

    # --- save params ---
    params_df = pd.DataFrame({
        "pathway": target_cols,
        "mean": [means[c] for c in target_cols],
        "std": [stds[c] for c in target_cols],
        "count": [int(counts[c]) for c in target_cols],
    })

    csv_params_path = params_dir / "JFX0729_zscore_params.csv"
    json_params_path = params_dir / "JFX0729_zscore_params.json"
    params_dir.mkdir(parents=True, exist_ok=True)

    params_df.to_csv(csv_params_path, index=False, encoding="utf-8-sig")
    print(f"   Z-score params (CSV)  -> {csv_params_path}")

    params_json = {
        "description": "JFX0729 z-score parameters for 30 pathways",
        "seed": seed,
        "ddof": ddof,
        "num_samples": int(df_z.shape[0]),
        "pathways": {
            row["pathway"]: {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "count": int(row["count"]),
            }
            for _, row in params_df.iterrows()
        },
    }
    with open(json_params_path, "w", encoding="utf-8") as f:
        json.dump(params_json, f, indent=2, ensure_ascii=False)
    print(f"   Z-score params (JSON) -> {json_params_path}")

    return df_z


# ---------------------------------------------------------------------------
# Phase 3: integrity checks
# ---------------------------------------------------------------------------

def phase3_verify(manifest: pd.DataFrame, patch_output_dir: Path,
                  zscore_csv_path: Path, total: int, train_count: int,
                  val_count: int, ddof: int):
    """Run 7 acceptance checks.  Exit with code 1 on any failure."""
    banner("Phase 3 — Integrity Verification")
    all_ok = True

    # --- Check 0: disk file set == manifest set (no stale/orphan files) ---
    manifest_train = set(
        manifest[manifest["split"] == "train"]["patch_id"].values
    )
    manifest_val = set(
        manifest[manifest["split"] == "val"]["patch_id"].values
    )
    disk_train = set(
        f[:-4] for f in os.listdir(patch_output_dir / "train_patches")
        if f.endswith(".png")
    )
    disk_val = set(
        f[:-4] for f in os.listdir(patch_output_dir / "val_patches")
        if f.endswith(".png")
    )
    stale_train = disk_train - manifest_train
    missing_train = manifest_train - disk_train
    stale_val = disk_val - manifest_val
    missing_val = manifest_val - disk_val
    print(f"\n  Check 0 — Disk-vs-Manifest consistency:")
    ok0 = True
    if stale_train:
        print(f"  ❌ Stale files in train_patches/ (not in manifest): {len(stale_train)}")
        ok0 = False
    if missing_train:
        print(f"  ❌ Missing from train_patches/ (in manifest but not on disk): {len(missing_train)}")
        ok0 = False
    if stale_val:
        print(f"  ❌ Stale files in val_patches/ (not in manifest): {len(stale_val)}")
        ok0 = False
    if missing_val:
        print(f"  ❌ Missing from val_patches/ (in manifest but not on disk): {len(missing_val)}")
        ok0 = False
    if ok0:
        print(f"  ✓  PASS — disk files == manifest entries")
    else:
        all_ok = False

    # --- Check 1: patch counts ---
    train_n = len(manifest[manifest["split"] == "train"])
    val_n = len(manifest[manifest["split"] == "val"])
    total_n = train_n + val_n
    print(f"\n  Check 1 — Patch counts: train={train_n}, val={val_n}, total={total_n}")
    if total_n == total and train_n == train_count and val_n == val_count:
        print(f"  ✓  PASS")
    else:
        print(f"  ❌ FAIL — expected {total} total / {train_count} train / {val_count} val")
        all_ok = False

    # --- Check 2: CSV rows ---
    df_z = pd.read_csv(zscore_csv_path)
    csv_rows = df_z.shape[0]
    print(f"\n  Check 2 — CSV rows: {csv_rows}")
    if csv_rows == total:
        print(f"  ✓  PASS")
    else:
        print(f"  ❌ FAIL — expected {total}")
        all_ok = False

    # --- Check 3: patch_id <-> patch file intersection ---
    all_patches_on_disk = disk_train | disk_val
    csv_ids = set(str(i) for i in df_z["patch_id"])
    intersection = all_patches_on_disk & csv_ids
    print(f"\n  Check 3 — Patch<->CSV intersection: {len(intersection)} / {total}")
    if len(intersection) == total:
        print(f"  ✓  PASS")
    else:
        missing_in_csv = all_patches_on_disk - csv_ids
        missing_on_disk = csv_ids - all_patches_on_disk
        if missing_in_csv:
            print(f"  ❌ Patches on disk but NOT in CSV: {len(missing_in_csv)} — {list(missing_in_csv)[:5]}...")
        if missing_on_disk:
            print(f"  ❌ Patches in CSV but NOT on disk: {len(missing_on_disk)} — {list(missing_on_disk)[:5]}...")
        all_ok = False

    # --- Check 4: train/val no overlap ---
    overlap = disk_train & disk_val
    print(f"\n  Check 4 — Train/Val overlap: {len(overlap)}")
    if len(overlap) == 0:
        print(f"  ✓  PASS")
    else:
        print(f"  ❌ FAIL — {len(overlap)} overlapping patches: {list(overlap)[:5]}...")
        all_ok = False

    # --- Check 5: z-score mean~0, std~1 ---
    target_cols = list(df_z.columns[1:])
    means = df_z[target_cols].mean()
    stds = df_z[target_cols].std(ddof=ddof)
    mean_ok = means.abs().max() < 0.01
    std_ok = (stds - 1.0).abs().max() < 0.01
    print(f"\n  Check 5 — Z-score quality (ddof={ddof}):")
    print(f"         Mean range: [{means.min():+.6f}, {means.max():+.6f}]  (expect ~0)")
    print(f"         Std  range: [{stds.min():.6f}, {stds.max():.6f}]  (expect ~1)")
    if mean_ok and std_ok:
        print(f"  ✓  PASS")
    else:
        if not mean_ok:
            bad = means[means.abs() >= 0.01]
            print(f"  ❌ Pathway means far from 0: {dict(bad)}")
        if not std_ok:
            bad = stds[(stds - 1.0).abs() >= 0.01]
            print(f"  ❌ Pathway stds far from 1: {dict(bad)}")
        all_ok = False

    # --- Check 6: NaN/Inf in output CSV ---
    nan_inf = df_z[target_cols].replace([np.inf, -np.inf], np.nan).isna().sum().sum()
    print(f"\n  Check 6 — NaN/Inf in z-scored CSV: {nan_inf}")
    if nan_inf == 0:
        print(f"  ✓  PASS")
    else:
        print(f"  ❌ FAIL — {nan_inf} NaN/Inf values")
        all_ok = False

    # --- Check 7: dataset_online.py loading simulation ---
    print(f"\n  Check 7 — Simulate dataset_online.py loading:")
    try:
        from dataset_online import OnlinePatchDataset
        for split_name, expected_count in [("train", train_count), ("val", val_count)]:
            ds = OnlinePatchDataset(
                patches_dir=str(patch_output_dir / f"{split_name}_patches"),
                labels_csv=str(zscore_csv_path),
                transform=None,
            )
            actual = len(ds.samples)
            status = "✓" if actual == expected_count else "❌"
            print(f"         {split_name}: {actual} samples  {status} (expected {expected_count})")
            if actual != expected_count:
                all_ok = False
    except Exception as e:
        print(f"  ❌ FAIL — dataset_online.py loading crashed: {e}")
        print(f"         (Requires project Python env: C:/Program Files/Python313/python.exe)")
        all_ok = False

    # --- final verdict ---
    print(f"\n{'='*70}")
    if all_ok:
        print("  ✅  ALL CHECKS PASSED")
        print(f"{'='*70}\n")
    else:
        print("  ❌  SOME CHECKS FAILED — review output above")
        print(f"{'='*70}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JFX0729 fixed data: patch split + z-score + verification"
    )
    parser.add_argument(
        "--patch-source",
        default="data_new_3ST/JFX_fixed_data/JFX0729",
        help="Directory containing raw .png patches (default: %(default)s)",
    )
    parser.add_argument(
        "--label-source",
        default="data_new_3ST/JFX_fixed_data/JFX0729_ssGSEA.csv",
        help="Raw ssGSEA CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-output",
        default="data_new_3ST/patch_noov_spilt/JFX0729_noov_split",
        help="Output dir for train_patches/ + val_patches/ (default: %(default)s)",
    )
    parser.add_argument(
        "--label-output",
        default="data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv",
        help="Output path for z-scored CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--params-dir",
        default="data_new_3ST/ssGSEA_zscore",
        help="Directory for zscore_params.csv/json (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help=f"Random seed for split (default: {_DEFAULT_SEED})",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=_DEFAULT_VAL_FRACTION,
        help=f"Validation fraction (default: {_DEFAULT_VAL_FRACTION})",
    )
    parser.add_argument(
        "--ddof", type=int, default=_DEFAULT_DDOF,
        help=f"Degrees of freedom for std (default: {_DEFAULT_DDOF})",
    )
    args = parser.parse_args()

    # Resolve relative -> absolute (relative to project root)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))  # for dataset_online import in phase3
    print(f"Project root: {project_root}")

    patch_source = Path(args.patch_source)
    label_source = Path(args.label_source)
    patch_output = Path(args.patch_output)
    label_output = Path(args.label_output)
    params_dir = Path(args.params_dir)

    if not patch_source.is_dir():
        sys.exit(f"FATAL: Patch source not found: {patch_source}")
    if not label_source.is_file():
        sys.exit(f"FATAL: Label CSV not found: {label_source}")

    # Compute expected counts from actual source (not hardcoded)
    total_patches = len([f for f in os.listdir(patch_source) if f.lower().endswith(".png")])
    val_count = int(total_patches * args.val_fraction)
    train_count = total_patches - val_count

    print(f"Runtime config: seed={args.seed}, val_fraction={args.val_fraction}, "
          f"ddof={args.ddof}")
    print(f"Expected split: total={total_patches}, train={train_count}, val={val_count}")

    # --- run phases (all using args.* values) ---
    manifest = phase1_split(patch_source, patch_output, args.seed, val_count)
    df_z = phase2_zscore(label_source, label_output, params_dir, args.seed, args.ddof)
    phase3_verify(manifest, patch_output, label_output,
                  total_patches, train_count, val_count, args.ddof)


if __name__ == "__main__":
    main()
