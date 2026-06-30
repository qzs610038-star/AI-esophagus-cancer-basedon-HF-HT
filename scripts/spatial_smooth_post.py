"""Zero-parameter spatial smoothing for predictions.csv.

Example:
    python scripts/spatial_smooth_post.py \
        --predictions checkpoints/.../predictions.csv \
        --out_csv checkpoints/.../spatial_smooth_alpha_sweep.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def _pathways(df: pd.DataFrame) -> List[str]:
    true_cols = [c for c in df.columns if c.startswith("true_")]
    pathways = []
    for true_col in true_cols:
        pathway = true_col[5:]
        if f"pred_{pathway}" in df.columns:
            pathways.append(pathway)
    if not pathways:
        raise ValueError("No true_*/pred_* pathway column pairs found.")
    return pathways


def _mean_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        vals.append(float(np.corrcoef(yt, yp)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def _neighbor_means(coords: np.ndarray, preds: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n <= 1:
        return preds.copy()

    actual_k = min(k, n - 1)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    nn_idx = np.argpartition(dist_sq, kth=actual_k - 1, axis=1)[:, :actual_k]
    return preds[nn_idx].mean(axis=1)


def spatial_smooth(
    df: pd.DataFrame,
    pathways: Iterable[str],
    k: int,
    alpha: float,
    coord_cols: Tuple[str, str],
    group_col: str,
) -> np.ndarray:
    pathways = list(pathways)
    pred_cols = [f"pred_{p}" for p in pathways]
    smoothed = df[pred_cols].to_numpy(dtype=np.float64).copy()

    groups = df[group_col].fillna("").astype(str) if group_col in df.columns else pd.Series("", index=df.index)
    for _, idx in groups.groupby(groups).groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        coords = df.iloc[idx_arr][list(coord_cols)].to_numpy(dtype=np.float64)
        preds = smoothed[idx_arr]
        neigh = _neighbor_means(coords, preds, k=k)
        smoothed[idx_arr] = (1.0 - alpha) * preds + alpha * neigh

    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha sweep for zero-parameter spatial smoothing.")
    parser.add_argument("--predictions", required=True, help="Input predictions.csv with true_*/pred_* and x/y columns.")
    parser.add_argument("--out_csv", default=None, help="Output alpha sweep CSV. Defaults next to predictions.csv.")
    parser.add_argument("--k", type=int, default=6, help="Number of spatial neighbors.")
    parser.add_argument(
        "--alphas",
        default="0,0.03,0.05,0.1,0.15,0.2,0.3",
        help="Comma-separated alpha values.",
    )
    parser.add_argument("--coord_cols", default="x,y", help="Coordinate columns, e.g. x,y or pos_x,pos_y.")
    parser.add_argument("--group_col", default="patient", help="Group column to avoid smoothing across patients.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    df = pd.read_csv(pred_path)
    coord_cols = tuple(c.strip() for c in args.coord_cols.split(",", 1))
    if len(coord_cols) != 2 or any(c not in df.columns for c in coord_cols):
        raise ValueError(f"predictions.csv must contain coordinate columns {coord_cols}.")

    pathways = _pathways(df)
    y_true = df[[f"true_{p}" for p in pathways]].to_numpy(dtype=np.float64)
    y_pred = df[[f"pred_{p}" for p in pathways]].to_numpy(dtype=np.float64)
    baseline = _mean_pcc(y_true, y_pred)

    rows = []
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    for alpha in alphas:
        smoothed = spatial_smooth(
            df,
            pathways=pathways,
            k=args.k,
            alpha=alpha,
            coord_cols=coord_cols,
            group_col=args.group_col,
        )
        rows.append({
            "alpha": alpha,
            "k": args.k,
            "coord_cols": ",".join(coord_cols),
            "group_col": args.group_col,
            "mean_pcc": _mean_pcc(y_true, smoothed),
            "delta_vs_alpha0": _mean_pcc(y_true, smoothed) - baseline,
        })

    out_csv = Path(args.out_csv) if args.out_csv else pred_path.with_name("spatial_smooth_alpha_sweep.csv")
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    best = out.sort_values("mean_pcc", ascending=False).iloc[0]
    print(f"[OK] wrote {out_csv}")
    print(f"baseline mean PCC: {baseline:.6f}")
    print(f"best alpha={best['alpha']} mean PCC={best['mean_pcc']:.6f} delta={best['delta_vs_alpha0']:.6f}")


if __name__ == "__main__":
    main()
