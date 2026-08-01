import os
import sys
import json
import math
import hashlib
import gzip
import argparse
import platform
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def resolve_output_root(config, mode):
    """Keep smoke artifacts below the v001 root so they cannot replace Final."""
    base = Path(config['output_root'])
    return base / 'smoke' if mode == 'smoke' else base

def parse_barcode_coords(barcode):
    # e.g., patch_x336_y672 -> x=336, y=672
    parts = barcode.strip().split('_')
    x_val = None
    y_val = None
    for p in parts:
        if p.startswith('x'):
            x_val = int(p[1:])
        elif p.startswith('y'):
            y_val = int(p[1:])
    if x_val is None or y_val is None:
        raise ValueError(f"Cannot parse coordinates from barcode: {barcode}")
    return x_val, y_val

def build_spatial_weights(coords, graph_type='rook', step_size=None):
    """
    coords: N x 2 numpy array (x, y)
    returns: N x N sparse or dense numpy weight matrix (row standardized)
    """
    N = len(coords)
    if N == 0:
        return np.zeros((0, 0))
    
    if step_size is None:
        # infer step size as min non-zero dx/dy
        dx = np.abs(coords[:, 0][:, None] - coords[:, 0][None, :])
        dy = np.abs(coords[:, 1][:, None] - coords[:, 1][None, :])
        min_dx = np.min(dx[dx > 0]) if np.any(dx > 0) else 1
        min_dy = np.min(dy[dy > 0]) if np.any(dy > 0) else 1
        step_size = min(min_dx, min_dy)

    dist_matrix = cdist(coords, coords)
    adj = np.zeros((N, N), dtype=float)

    if graph_type == 'rook':
        # dx == step and dy == 0 OR dx == 0 and dy == step
        # using distance with tolerance 1.01 * step
        tol = 1.05 * step_size
        mask = (dist_matrix > 0) & (dist_matrix <= tol)
        # also verify orthogonal
        dx = np.abs(coords[:, 0][:, None] - coords[:, 0][None, :])
        dy = np.abs(coords[:, 1][:, None] - coords[:, 1][None, :])
        is_ortho = (dx < 0.1 * step_size) | (dy < 0.1 * step_size)
        adj[mask & is_ortho] = 1.0

    elif graph_type == 'queen':
        # dist <= sqrt(2) * 1.05 * step
        tol = math.sqrt(2) * 1.05 * step_size
        mask = (dist_matrix > 0) & (dist_matrix <= tol)
        adj[mask] = 1.0

    elif graph_type.startswith('knn'):
        k = int(graph_type[3:])
        for i in range(N):
            idx = np.argsort(dist_matrix[i])[1:k+1] # exclude self
            adj[i, idx] = 1.0
            # make symmetric for spatial graph
            for j in idx:
                adj[j, i] = 1.0

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Row-standardization
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w_std = adj / row_sums
    return w_std, adj, step_size

def compute_moran_i(x, w):
    N = len(x)
    std_x = np.std(x)
    if std_x == 0 or N <= 1:
        return np.nan
    x_diff = x - np.mean(x)
    S0 = w.sum()
    if S0 == 0:
        return np.nan
    numerator = np.dot(x_diff, w.dot(x_diff))
    denominator = np.sum(x_diff ** 2)
    return (N / S0) * (numerator / denominator)

def compute_geary_c(x, w, adj_unstd):
    N = len(x)
    std_x = np.std(x)
    if std_x == 0 or N <= 1:
        return np.nan
    x_diff = x - np.mean(x)
    denom = np.sum(x_diff ** 2)
    if denom == 0:
        return np.nan
    S0 = adj_unstd.sum()
    if S0 == 0:
        return np.nan
    deg = adj_unstd.sum(axis=1)
    num = 2.0 * (np.sum(deg * (x ** 2)) - np.dot(x, adj_unstd.dot(x)))
    return ((N - 1) / (2 * S0)) * (num / denom)

def compute_edge_ratio(x, adj_unstd, coords, seed=42):
    # true neighbors diff
    i_idx, j_idx = np.where(np.triu(adj_unstd) > 0)
    if len(i_idx) == 0:
        return np.nan
    true_diffs = np.abs(x[i_idx] - x[j_idx])
    median_true = np.median(true_diffs)
    
    # Global random-pair control.  This is deliberately not called
    # "distance matched": the former implementation calculated edge distances
    # but never used them when sampling controls.
    rng = np.random.default_rng(seed)
    
    # sample random pairs of same size
    N = len(x)
    rand_i = rng.integers(0, N, size=len(i_idx) * 3)
    rand_j = rng.integers(0, N, size=len(i_idx) * 3)
    valid = rand_i != rand_j
    rand_i, rand_j = rand_i[valid], rand_j[valid]
    
    rand_diffs = np.abs(x[rand_i] - x[rand_j])
    median_control = np.median(rand_diffs)
    
    if median_control == 0:
        return np.nan
    return median_true / median_control

def compute_edge_contrasts(x, adj_unstd, coords, step_size, seed=42):
    """Return two descriptive local-smoothness contrasts.

    ``global_random_ratio`` reproduces the original v001 statistic: median
    one-step edge difference divided by a global random-pair difference.
    ``near_far_ratio`` uses non-edge pairs 2--3 grid steps apart.  Neither is
    presented as a distance-matched causal contrast.
    """
    global_ratio = compute_edge_ratio(x, adj_unstd, coords, seed=seed)
    i_edge, j_edge = np.where(np.triu(adj_unstd, 1) > 0)
    if len(i_edge) == 0:
        return {"global_random_ratio": np.nan, "near_far_ratio": np.nan}
    edge_median = np.median(np.abs(x[i_edge] - x[j_edge]))
    distances = cdist(coords, coords)
    non_edge = np.triu(adj_unstd == 0, 1)
    annulus = non_edge & (distances >= 1.5 * step_size) & (distances <= 3.5 * step_size)
    i_far, j_far = np.where(annulus)
    if len(i_far) == 0:
        near_far = np.nan
    else:
        far_median = np.median(np.abs(x[i_far] - x[j_far]))
        near_far = np.nan if far_median == 0 else edge_median / far_median
    return {"global_random_ratio": global_ratio, "near_far_ratio": near_far}

def compute_variogram_fast(x, step_pairs_list, std_x):
    if std_x == 0:
        return [np.nan] * len(step_pairs_list)
    
    gamma = []
    for i_idx, j_idx in step_pairs_list:
        if len(i_idx) == 0:
            gamma.append(np.nan)
        else:
            diffs = ((x[i_idx] - x[j_idx]) / std_x) ** 2
            gamma.append(0.5 * np.mean(diffs))
    return gamma

def precompute_variogram_pairs(coords, max_steps=5, step_size=336):
    dist_matrix = cdist(coords, coords)
    step_pairs = []
    for step in range(1, max_steps + 1):
        target_dist = step * step_size
        tol = 0.2 * step_size
        mask = np.abs(dist_matrix - target_dist) <= tol
        i_idx, j_idx = np.where(np.triu(mask))
        step_pairs.append((i_idx, j_idx))
    return step_pairs

def compute_residual_moran(x, coords, w):
    N = len(x)
    if N < 5 or np.std(x) == 0:
        return np.nan
    # Fit 2D low-order polynomial trend (X, Y, X^2, Y^2, XY)
    X = coords[:, 0]
    Y = coords[:, 1]
    A = np.column_stack([np.ones(N), X, Y, X**2, Y**2, X*Y])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
        pred = A @ coeff
        residuals = x - pred
        return compute_moran_i(residuals, w)
    except Exception:
        return np.nan

def compute_within_patient_percentile(scores):
    N = len(scores)
    if N == 0:
        return np.array([])
    ranks = stats.rankdata(scores, method='average')
    percentiles = (ranks - 0.5) / N
    return percentiles

def build_percentile_frame(patient, df, coords, pathway_names):
    """Build a row-aligned percentile table for one patient."""
    out = pd.DataFrame({
        "patient": np.repeat(patient, len(df)),
        "barcode": df["barcode"].to_numpy(),
        "x": coords[:, 0],
        "y": coords[:, 1],
    })
    for pathway in pathway_names:
        out[pathway] = compute_within_patient_percentile(df[pathway].to_numpy())
    return out

def benjamini_hochberg(p_values):
    """BH-FDR correction with a stable fallback for older SciPy versions."""
    p_values = np.asarray(p_values, dtype=float)
    if hasattr(stats, "false_discovery_control"):
        return np.asarray(stats.false_discovery_control(p_values, method="bh"))
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def combine_stouffer_p(p_values):
    """Equal-weight one-sided Stouffer combination with a stable upper tail."""
    p_values = np.asarray(p_values, dtype=float)
    z_scores = stats.norm.isf(np.clip(p_values, np.finfo(float).tiny, 1.0))
    z_scores = np.clip(z_scores, -8.0, 8.0)
    stouffer_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return float(stats.norm.sf(stouffer_z))

def compare_raw_and_percentile_spatial_metrics(raw, w, adj_unstd, coords, step_size=1, seed=42):
    """Quantify what an ordinal transform preserves and what it changes."""
    percentile = compute_within_patient_percentile(raw)
    raw_edge = compute_edge_contrasts(raw, adj_unstd, coords, step_size, seed)
    pct_edge = compute_edge_contrasts(percentile, adj_unstd, coords, step_size, seed)
    n = len(raw)
    k = max(1, int(np.ceil(0.10 * n)))
    raw_order = np.argsort(raw)
    pct_order = np.argsort(percentile)
    top_overlap = len(set(raw_order[-k:]) & set(pct_order[-k:])) / k
    bottom_overlap = len(set(raw_order[:k]) & set(pct_order[:k])) / k
    tie_rate = 1.0 - (len(np.unique(raw)) / n)
    return {
        "raw_moran_i": compute_moran_i(raw, w),
        "percentile_moran_i": compute_moran_i(percentile, w),
        "raw_geary_c": compute_geary_c(raw, w, adj_unstd),
        "percentile_geary_c": compute_geary_c(percentile, w, adj_unstd),
        "raw_global_random_edge_ratio": raw_edge["global_random_ratio"],
        "percentile_global_random_edge_ratio": pct_edge["global_random_ratio"],
        "raw_near_far_ratio": raw_edge["near_far_ratio"],
        "percentile_near_far_ratio": pct_edge["near_far_ratio"],
        "moran_delta_percentile_minus_raw": compute_moran_i(percentile, w) - compute_moran_i(raw, w),
        "spearman_rho": stats.spearmanr(raw, percentile).statistic,
        "top10_overlap": top_overlap,
        "bottom10_overlap": bottom_overlap,
        "tie_rate": tie_rate,
    }

def compute_cliffs_delta(a, b):
    """Cliff's delta, positive when values in ``a`` tend to exceed ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = np.searchsorted(b, a, side="left").sum()
    less = (len(b) - np.searchsorted(b, a, side="right")).sum()
    return float((greater - less) / (len(a) * len(b)))

def compute_epsilon_squared(groups):
    """Descriptive Kruskal-Wallis epsilon-squared across patient groups."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    total_n = sum(len(g) for g in groups)
    if len(groups) < 2 or total_n <= len(groups):
        return np.nan
    h = stats.kruskal(*groups).statistic
    return float(max(0.0, (h - len(groups) + 1) / (total_n - len(groups))))

def compute_block_subsample_rank_stability(scores, coords, step_size, n_iterations=200, seed=42):
    """Estimate numeric percentile drift after retaining 80% of spatial blocks.

    Blocks are deterministic 4x4-step coordinate bins.  The statistic is not a
    performance estimate; it only measures how much empirical percentile
    values move when whole local regions are omitted.
    """
    scores = np.asarray(scores, dtype=float)
    full_percentile = compute_within_patient_percentile(scores)
    width = max(float(step_size) * 4.0, 1.0)
    block_ids = np.column_stack([
        np.floor((coords[:, 0] - coords[:, 0].min()) / width).astype(int),
        np.floor((coords[:, 1] - coords[:, 1].min()) / width).astype(int),
    ])
    _, inverse = np.unique(block_ids, axis=0, return_inverse=True)
    unique_blocks = np.unique(inverse)
    retain_n = max(1, int(np.ceil(0.8 * len(unique_blocks))))
    rng = np.random.default_rng(seed)
    shifts = []
    for _ in range(n_iterations):
        kept = rng.choice(unique_blocks, size=retain_n, replace=False)
        mask = np.isin(inverse, kept)
        subset_percentile = compute_within_patient_percentile(scores[mask])
        shifts.extend(np.abs(subset_percentile - full_percentile[mask]).tolist())
    shifts = np.asarray(shifts, dtype=float)
    return {
        "n_blocks": int(len(unique_blocks)),
        "n_iterations": int(n_iterations),
        "mean_abs_percentile_shift": float(np.mean(shifts)) if len(shifts) else np.nan,
        "p95_abs_percentile_shift": float(np.percentile(shifts, 95)) if len(shifts) else np.nan,
        "max_abs_percentile_shift": float(np.max(shifts)) if len(shifts) else np.nan,
    }

def permutation_test(x, w, adj_unstd, n_perms=499, seed=42):
    rng = np.random.default_rng(seed)
    obs_moran = compute_moran_i(x, w)
    if np.isnan(obs_moran):
        return np.nan, 1.0, np.array([])
    
    N = len(x)
    x_diff = x - np.mean(x)
    denom = np.sum(x_diff ** 2)
    S0 = w.sum()
    if S0 == 0 or denom == 0:
        return np.nan, 1.0, np.array([])
    factor = N / (S0 * denom)

    # Convert to CSR sparse matrix for ultra-fast multiplication
    w_sparse = csr_matrix(w)

    perms_mat = np.zeros((n_perms, N))
    for p in range(n_perms):
        perms_mat[p] = rng.permutation(x_diff)
    
    # perms_mat: (n_perms x N), w_sparse.T: (N x N)
    w_perms = w_sparse.dot(perms_mat.T).T # (n_perms x N)
    perm_morans = np.sum(perms_mat * w_perms, axis=1) * factor
    
    p_val = (np.sum(perm_morans >= obs_moran) + 1.0) / (n_perms + 1.0)
    return obs_moran, p_val, perm_morans

def run_full_pipeline(config_path, mode='final', replace_existing=False):
    run_started = datetime.now(timezone.utc)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    data_root = Path(config['data_root'])
    output_root = resolve_output_root(config, mode)
    
    if output_root.exists() and any(output_root.iterdir()) and not replace_existing:
        raise FileExistsError(
            f"Output directory {output_root} is non-empty. Preserve/archive it, then rerun with --replace-existing."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    
    n_perms = config['n_permutations_smoke'] if mode == 'smoke' else config['n_permutations_final']
    patients = config['patients']
    if mode == 'smoke':
        patients = config['patients'][:2]

    print(f"=== Starting MPP2 Raw Spatial Rank Analysis (Mode: {mode}, Permutations: {n_perms}) ===")

    # D0: Inputs QC & Assets check
    qc_results = {}
    patient_data = {}
    pathway_names = None

    for pt in config['patients']:
        raw_csv = data_root / f"source_snapshot/raw_ssgsea/{pt}/{pt}_ssGSEA.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw CSV for patient {pt}: {raw_csv}")
        
        df = pd.read_csv(raw_csv)
        # Parse barcode coords
        coords = []
        barcodes = df['barcode'].tolist()
        for bc in barcodes:
            cx, cy = parse_barcode_coords(bc)
            coords.append([cx, cy])
        coords = np.array(coords)
        
        pw_cols = [c for c in df.columns if c != 'barcode']
        if pathway_names is None:
            pathway_names = pw_cols
        
        # Build graphs
        w_rook, adj_rook, step_size = build_spatial_weights(coords, 'rook')
        w_queen, adj_queen, _ = build_spatial_weights(coords, 'queen', step_size)
        w_knn4, adj_knn4, _ = build_spatial_weights(coords, 'knn4', step_size)
        w_knn8, adj_knn8, _ = build_spatial_weights(coords, 'knn8', step_size)

        # Precompute variogram pairs for speed
        vario_pairs = precompute_variogram_pairs(coords, max_steps=5, step_size=step_size)

        patient_data[pt] = {
            'df': df,
            'coords': coords,
            'barcodes': barcodes,
            'step_size': step_size,
            'w_rook': w_rook, 'adj_rook': adj_rook,
            'w_queen': w_queen, 'adj_queen': adj_queen,
            'w_knn4': w_knn4, 'adj_knn4': adj_knn4,
            'w_knn8': w_knn8, 'adj_knn8': adj_knn8,
            'vario_pairs': vario_pairs,
            'n_spots': len(df)
        }

        qc_results[pt] = {
            'n_spots': len(df),
            'step_size': int(step_size),
            'rook_edges': int(adj_rook.sum() / 2),
            'queen_edges': int(adj_queen.sum() / 2),
            'status': 'PASS'
        }

    with open(output_root / "qc_summary.json", 'w', encoding='utf-8') as f:
        json.dump(qc_results, f, indent=2)

    # D1: Spatial Metrics & Permutation Tests
    metrics_rows = []
    seed = config.get('seed', 20260801)

    print("--> Computing spatial metrics for 210 patient x pathway units...")
    for pt in patients:
        pdata = patient_data[pt]
        df = pdata['df']
        coords = pdata['coords']
        w_rook = pdata['w_rook']
        adj_rook = pdata['adj_rook']
        step_size = pdata['step_size']

        for pw in pathway_names:
            x = df[pw].values
            
            # Primary graph: Rook
            moran_val, p_val, perms = permutation_test(x, w_rook, adj_rook, n_perms=n_perms, seed=seed)
            geary_val = compute_geary_c(x, w_rook, adj_rook)
            edge_contrasts = compute_edge_contrasts(x, adj_rook, coords, step_size, seed=seed)
            res_moran = compute_residual_moran(x, coords, w_rook)
            vario = compute_variogram_fast(x, pdata['vario_pairs'], np.std(x))

            # Sensitivity graphs
            moran_queen = compute_moran_i(x, pdata['w_queen'])
            moran_knn4 = compute_moran_i(x, pdata['w_knn4'])
            moran_knn8 = compute_moran_i(x, pdata['w_knn8'])

            metrics_rows.append({
                'patient': pt,
                'pathway': pw,
                'n_spots': len(x),
                'moran_i': moran_val,
                'p_value': p_val,
                'geary_c': geary_val,
                # Backward-compatible column; see explicit name below.
                'edge_ratio': edge_contrasts['global_random_ratio'],
                'global_random_edge_ratio': edge_contrasts['global_random_ratio'],
                'near_far_edge_ratio': edge_contrasts['near_far_ratio'],
                'residual_moran': res_moran,
                'variogram_h1': vario[0],
                'variogram_h2': vario[1],
                'variogram_h3': vario[2],
                'variogram_h4': vario[3],
                'variogram_h5': vario[4],
                'moran_queen': moran_queen,
                'moran_knn4': moran_knn4,
                'moran_knn8': moran_knn8
            })

    metrics_df = pd.DataFrame(metrics_rows)
    
    # Within-patient BH-FDR correction
    for pt, group in metrics_df.groupby('patient'):
        q_vals = benjamini_hochberg(group['p_value'].values)
        metrics_df.loc[group.index, 'q_value'] = q_vals

    metrics_df.to_csv(output_root / "spatial_metrics.csv", index=False)

    # D2: Patient Percentiles & Dual View
    percentile_dfs = []
    for pt in config['patients']:
        pdata = patient_data[pt]
        percentile_dfs.append(build_percentile_frame(
            pt, pdata['df'], pdata['coords'], pathway_names
        ))

    all_perc_df = pd.concat(percentile_dfs, ignore_index=True)
    
    # Save compressed CSV
    gz_path = output_root / "within_patient_percentiles.csv.gz"
    with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
        all_perc_df.to_csv(f, index=False)

    # Quantify the exact invariances and non-invariances of the ordinal view.
    percentile_metric_rows = []
    for pt in config['patients']:
        pdata = patient_data[pt]
        for pw in pathway_names:
            comparison = compare_raw_and_percentile_spatial_metrics(
                pdata['df'][pw].to_numpy(), pdata['w_rook'], pdata['adj_rook'],
                pdata['coords'], pdata['step_size'], seed
            )
            percentile_metric_rows.append({'patient': pt, 'pathway': pw, **comparison})
    percentile_metrics_df = pd.DataFrame(percentile_metric_rows)
    percentile_metrics_df.to_csv(output_root / "percentile_spatial_metrics.csv", index=False)

    bootstrap_iterations = (
        config.get('n_block_subsamples_smoke', 20) if mode == 'smoke'
        else config.get('n_block_subsamples_final', 200)
    )
    stability_rows = []
    for pt in config['patients']:
        pdata = patient_data[pt]
        for pw_idx, pw in enumerate(pathway_names):
            stability = compute_block_subsample_rank_stability(
                pdata['df'][pw].to_numpy(), pdata['coords'], pdata['step_size'],
                bootstrap_iterations, seed + pw_idx
            )
            stability_rows.append({'patient': pt, 'pathway': pw, **stability})
    pd.DataFrame(stability_rows).to_csv(output_root / "block_rank_stability_summary.csv", index=False)

    # D3: Cross-Patient Absolute Scale & Robust Summaries
    abs_summary_rows = []
    for pt in config['patients']:
        df = patient_data[pt]['df']
        for pw in pathway_names:
            vals = df[pw].values
            abs_summary_rows.append({
                'patient': pt,
                'pathway': pw,
                'n': len(vals),
                'mean': np.mean(vals),
                'std': np.std(vals),
                'median': np.median(vals),
                'iqr': stats.iqr(vals),
                'q05': np.percentile(vals, 5),
                'q10': np.percentile(vals, 10),
                'q90': np.percentile(vals, 90),
                'q95': np.percentile(vals, 95),
                'min': np.min(vals),
                'max': np.max(vals)
            })

    abs_df = pd.DataFrame(abs_summary_rows)
    abs_df.to_csv(output_root / "patient_pathway_absolute_summary.csv", index=False)

    # Pairwise absolute-scale effects. These are descriptive and do not identify
    # batch effects or disease severity without covariate/technical metadata.
    pairwise_rows = []
    pathway_effect_rows = []
    for pw in pathway_names:
        groups = [patient_data[pt]['df'][pw].to_numpy() for pt in config['patients']]
        pathway_effect_rows.append({
            'pathway': pw,
            'kruskal_epsilon_squared_patient': compute_epsilon_squared(groups),
        })
        for pt_a, pt_b in combinations(config['patients'], 2):
            a = patient_data[pt_a]['df'][pw].to_numpy()
            b = patient_data[pt_b]['df'][pw].to_numpy()
            pairwise_rows.append({
                'pathway': pw, 'patient_a': pt_a, 'patient_b': pt_b,
                'cliffs_delta_a_minus_b': compute_cliffs_delta(a, b),
                'wasserstein_distance': stats.wasserstein_distance(a, b),
                'median_difference_a_minus_b': float(np.median(a) - np.median(b)),
            })
    pd.DataFrame(pairwise_rows).to_csv(output_root / "cross_patient_pairwise_effects.csv", index=False)
    pd.DataFrame(pathway_effect_rows).to_csv(output_root / "cross_patient_pathway_effects.csv", index=False)

    # D4: Pathway Replication Summary & Decision Rules
    rep_summary = []
    supported_pathways = []
    patient_specific_pathways = []

    for pw in pathway_names:
        pw_metrics = metrics_df[metrics_df['pathway'] == pw]
        internal_metrics = pw_metrics[pw_metrics['patient'].isin(config['internal_patients'])]
        external_metrics = pw_metrics[pw_metrics['patient'].isin(config['external_patients'])]
        n_patients = len(internal_metrics)
        n_pos = np.sum(internal_metrics['moran_i'] > 0)
        med_moran = np.median(internal_metrics['moran_i'])
        med_edge = np.median(internal_metrics['global_random_edge_ratio'].dropna())
        
        # Stouffer combined p-value
        p_vals = internal_metrics['p_value'].values
        stouffer_p = combine_stouffer_p(p_vals)
        
        rep_summary.append({
            'pathway': pw,
            'internal_pos_patients_count': int(n_pos),
            'internal_n_patients': int(n_patients),
            'internal_median_moran_i': float(med_moran),
            'internal_median_global_random_edge_ratio': float(med_edge),
            'stouffer_p': float(stouffer_p),
            'external_moran_i': float(external_metrics['moran_i'].iloc[0]) if len(external_metrics) else np.nan,
            'external_direction_confirmed': bool(len(external_metrics) and external_metrics['moran_i'].iloc[0] > 0),
        })

    rep_df = pd.DataFrame(rep_summary)
    rep_df['stouffer_q_bh_30_pathways'] = benjamini_hochberg(rep_df['stouffer_p'].to_numpy())
    rep_df['internal_supported'] = (
        (rep_df['internal_pos_patients_count'] >= 5)
        & (rep_df['stouffer_q_bh_30_pathways'] < 0.05)
        & (rep_df['internal_median_moran_i'] >= 0.10)
        & (rep_df['internal_median_global_random_edge_ratio'] <= 0.85)
    )
    rep_df['evidence_tier'] = np.where(
        rep_df['internal_supported'], 'Supported',
        np.where((rep_df['internal_pos_patients_count'] >= 3) & (rep_df['internal_median_moran_i'] > 0.05),
                 'Weak/Pathway-Specific', 'Unsupported')
    )
    # Compatibility aliases for existing downstream readers.
    rep_df['pos_patients_count'] = rep_df['internal_pos_patients_count']
    rep_df['median_moran_i'] = rep_df['internal_median_moran_i']
    rep_df['median_edge_ratio'] = rep_df['internal_median_global_random_edge_ratio']
    supported_pathways = rep_df.loc[rep_df['internal_supported'], 'pathway'].tolist()
    patient_specific_pathways = rep_df.loc[
        (~rep_df['internal_supported']) & (rep_df['internal_pos_patients_count'] >= 3), 'pathway'
    ].tolist()
    rep_df.to_csv(output_root / "pathway_replication_summary.csv", index=False)

    # Decision Summary JSON
    evidence_class = "diagnostic_only"
    if len(supported_pathways) >= 15:
        spatial_supp = "broad_descriptive"
    elif len(supported_pathways) > 0:
        spatial_supp = "pathway_specific_descriptive"
    else:
        spatial_supp = "weak"

    decision_summary = {
        "evidence_class": evidence_class,
        "spatial_support": spatial_supp,
        "supported_pathways_count": len(supported_pathways),
        "supported_pathways": supported_pathways,
        "patient_specific_pathways": patient_specific_pathways,
        "primary_replication_cohort": config['internal_patients'],
        "external_confirmation_cohort": config['external_patients'],
        "combined_fdr_scope": "BH across 30 internal-six Stouffer p-values",
        "support_threshold_status": "exploratory diagnostic rule; not a preregistered training or GO gate",
        "edge_ratio_definition": "one-step edge median absolute difference / global random-pair median absolute difference; not distance matched",
        "rank_view_recommendation": "retain_as_candidate_dual_view" if len(supported_pathways) > 0 else "deprioritize",
        "a1s_recommendation": "eligible_for_preregistration_discussion" if spatial_supp in ["broad_descriptive", "pathway_specific_descriptive"] else "deprioritize",
        "does_not_prove": [
            "MPP2 prediction smoothness",
            "Phase 3 outcome benefit",
            "absolute gene expression differences"
            ,"batch effects or disease severity from cross-patient score shifts",
            "equality of Moran's I after percentile transformation"
        ]
    }

    with open(output_root / "decision_summary.json", 'w', encoding='utf-8') as f:
        json.dump(decision_summary, f, indent=2)

    # 10.2 Visualization Figures Generation
    print("--> Generating Human Visualizations (F01-F07)...")
    generate_visualizations(output_root, config, patient_data, metrics_df, all_perc_df, abs_df, rep_df)

    # Generate Human Report Markdown
    generate_human_report(output_root, decision_summary, rep_df, metrics_df, percentile_metrics_df, abs_df)

    # Write Manifest & Checksums
    input_hashes = {}
    for pt in config['patients']:
        raw_path = data_root / f"source_snapshot/raw_ssgsea/{pt}/{pt}_ssGSEA.csv"
        input_hashes[str(raw_path)] = compute_sha256(raw_path)
    git_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    ).stdout.strip()
    run_manifest = {
        "mode": mode,
        "n_permutations": n_perms,
        "seed": seed,
        "patients": config['patients'],
        "config": config,
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "code_sha256": compute_sha256(Path(__file__)),
        "config_sha256": compute_sha256(config_path),
        "input_sha256": input_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": getattr(__import__('scipy'), '__version__', 'unknown'),
            "matplotlib": matplotlib.__version__, "seaborn": sns.__version__,
        }
    }
    with open(output_root / "run_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(run_manifest, f, indent=2)

    # Checksums
    checksums = {}
    for p in output_root.glob("*"):
        if p.is_file() and p.name != "checksums.sha256":
            checksums[p.name] = compute_sha256(p)
            
    with open(output_root / "checksums.sha256", 'w', encoding='utf-8') as f:
        for fname, sha in checksums.items():
            f.write(f"{sha}  {fname}\n")

    print(f"=== Analysis Completed Successfully! Output in: {output_root} ===")

def generate_visualizations(output_root, config, patient_data, metrics_df, all_perc_df, abs_df, rep_df):
    # F01: Data Geometry
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, pt in enumerate(config['patients']):
        ax = axes[idx]
        coords = patient_data[pt]['coords']
        ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.7, c='navy')
        ax.set_title(f"{pt} (N={len(coords)}, step={patient_data[pt]['step_size']})")
        ax.set_aspect('equal')
        ax.axis('off')
    if len(config['patients']) < 8:
        axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(output_root / "F01_data_geometry.png", dpi=300)
    plt.close()

    # F02: Moran Heatmap
    pivot_moran = metrics_df.pivot(index='pathway', columns='patient', values='moran_i')
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_moran, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': "Moran's I"})
    plt.title("F02: Moran's I Heatmap Across 7 Patients x 30 Pathways")
    plt.tight_layout()
    plt.savefig(output_root / "F02_moran_heatmap.png", dpi=300)
    plt.close()

    # F03: Edge Ratio Heatmap
    pivot_edge = metrics_df.pivot(index='pathway', columns='patient', values='global_random_edge_ratio')
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_edge, annot=True, fmt=".2f", cmap="YlOrRd_r", vmin=0.5, vmax=1.0,
                cbar_kws={'label': "1-step / global-random median |difference|"})
    plt.title("F03: Global Random-Pair Edge Contrast (Not Distance-Matched)")
    plt.tight_layout()
    plt.savefig(output_root / "F03_edge_ratio_heatmap.png", dpi=300)
    plt.close()

    # F04: Variogram Panels
    plt.figure(figsize=(12, 6))
    vario_cols = ['variogram_h1', 'variogram_h2', 'variogram_h3', 'variogram_h4', 'variogram_h5']
    mean_vario = metrics_df.groupby('pathway')[vario_cols].mean()
    top_vario = rep_df.sort_values('internal_median_moran_i', ascending=False).head(10)['pathway']
    for pw in top_vario:
        plt.plot(range(1, 6), mean_vario.loc[pw], marker='o', label=pw[:15])
    plt.xlabel("Spatial Step Distance (h)")
    plt.ylabel("Mean Standardized Semi-variance γ(h)")
    plt.title("F04: Variogram Curves (Top 10 by Internal-Six Median Moran's I)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_root / "F04_variogram_panels.png", dpi=300)
    plt.close()

    # F05: Raw vs Percentile Maps (Representative Pathway)
    top_pw = rep_df.sort_values('median_moran_i', ascending=False).iloc[0]['pathway']
    pt_sample = config['patients'][0]
    pdata = patient_data[pt_sample]
    df_raw = pdata['df']
    df_perc_pt = all_perc_df[all_perc_df['patient'] == pt_sample]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = ax1.scatter(pdata['coords'][:, 0], pdata['coords'][:, 1], c=df_raw[top_pw].values, cmap='viridis', s=10)
    plt.colorbar(sc1, ax=ax1, label="Raw ssGSEA Score")
    ax1.set_title(f"Raw Score Map: {top_pw} ({pt_sample})")
    ax1.set_aspect('equal')
    ax1.axis('off')

    sc2 = ax2.scatter(pdata['coords'][:, 0], pdata['coords'][:, 1], c=df_perc_pt[top_pw].values, cmap='viridis', s=10)
    plt.colorbar(sc2, ax=ax2, label="Within-Patient Percentile (0-1)")
    ax2.set_title(f"Percentile Map: {top_pw} ({pt_sample})")
    ax2.set_aspect('equal')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_root / "F05_raw_vs_percentile_maps.png", dpi=300)
    plt.close()

    # F06: Cross-patient spot-level score distributions (descriptive only)
    plt.figure(figsize=(14, 8))
    top_5_pws = rep_df.sort_values('median_moran_i', ascending=False).head(5)['pathway'].tolist()
    distribution_parts = []
    for pt in config['patients']:
        raw = patient_data[pt]['df'][top_5_pws].copy()
        raw['patient'] = pt
        distribution_parts.append(raw.melt(id_vars='patient', var_name='pathway', value_name='score'))
    distribution_df = pd.concat(distribution_parts, ignore_index=True)
    sns.boxenplot(data=distribution_df, x='pathway', y='score', hue='patient', showfliers=False)
    plt.xticks(rotation=30, ha='right')
    plt.title("F06: Spot-Level ssGSEA Distributions (Descriptive; Not Batch Attribution)")
    plt.tight_layout()
    plt.savefig(output_root / "F06_cross_patient_distributions.png", dpi=300)
    plt.close()

    # F07: Patient effects plus internal-six median (not a confidence-interval forest plot)
    plt.figure(figsize=(10, 10))
    rep_sorted = rep_df.sort_values('median_moran_i', ascending=True)
    y_lookup = {pw: idx for idx, pw in enumerate(rep_sorted['pathway'])}
    for _, row in metrics_df.iterrows():
        marker = 'x' if row['patient'] in config['external_patients'] else 'o'
        color = 'crimson' if marker == 'x' else 'steelblue'
        plt.scatter(row['moran_i'], y_lookup[row['pathway']], marker=marker, color=color, s=18, alpha=0.75)
    plt.scatter(rep_sorted['internal_median_moran_i'], range(len(rep_sorted)),
                marker='|', color='black', s=130, label='Internal-six median')
    plt.axvline(0.10, color='black', linestyle='--', label='Threshold (Moran >= 0.10)')
    plt.yticks(range(len(rep_sorted)), rep_sorted['pathway'])
    plt.xlabel("Moran's I")
    plt.title("F07: Patient-Level Moran Effects (X = External XZY)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "F07_pathway_evidence_forest.png", dpi=300)
    plt.close()

def generate_human_report(output_root, decision_summary, rep_df, metrics_df, percentile_metrics_df, abs_df):
    pct_delta = percentile_metrics_df['moran_delta_percentile_minus_raw']
    exact_moran = int(np.isclose(pct_delta, 0.0, atol=1e-12).sum())
    monotone_variograms = int((metrics_df[[f'variogram_h{i}' for i in range(1, 6)]].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).sum())
    external_confirmed = int(rep_df['external_direction_confirmed'].sum())
    report_content = f"""# MPP2 原始基因集分数空间连续性与患者内排名分析报告（Codex 增量修复）

> **诊断模式**：本地 `diagnostic_only`
> **数据范围**：7 位患者（HYZ15040, JFX, LMZ12939, TGC, XSL, ZHZ, XZY）、11,589 个 Spot、30 维 ssGSEA 通路  
> **主复现队列**：internal 6；XZY 仅作外部方向确认
> **证据边界**：不改写训练划分，不构成训练授权或模型性能证据

---

## 核心结论

1. internal 6 的主分析中，**{decision_summary['supported_pathways_count']}/30** 通路达到当前诊断门槛；XZY 对 **{external_confirmed}/30** 通路给出同方向的正 Moran 结果。Stouffer 合并 p 值已在 30 通路范围做 BH-FDR。
2. 原始分数具有广泛的**全局空间自相关**；这支持把坐标利用列为后续可预注册候选，但不等于已经证明任意空间平滑正则会提升 MPP2。
3. 患者内百分位变换严格保持排序关系：Spearman 中位数为 **{percentile_metrics_df['spearman_rho'].median():.3f}**，top/bottom 10% 集合重合中位数均为 **{percentile_metrics_df['top10_overlap'].median():.3f}**。但是它不保持 Moran 数值：210 个患者×通路单元中精确不变仅 **{exact_moran}/210**，Moran 变化中位数为 **{pct_delta.median():+.4f}**，范围 **[{pct_delta.min():+.4f}, {pct_delta.max():+.4f}]**。
4. 跨患者绝对分布确有差异，但现有数据不能把差异归因为“批次效应”，也不能解释成“病情严重度”；因此 `Absolute + Ordinal` 只能表述为**值得后续验证的双视图候选**。

## 对双线方案的追加修正

- **患者内相对线 S1/B1**：原始证据支持“患者内排序可稳定构造并保持热点次序”，不支持“零损耗保持所有空间统计量”。新增 `percentile_spatial_metrics.csv` 和 `block_rank_stability_summary.csv` 用于量化该边界。
- **绝对线空间候选 S2-S/A1-S**：可进入预注册讨论；正则形式、通路范围和权重仍需独立消融，当前分析不能直接授权训练。
- **跨患者绝对视图**：新增 Cliff's delta、Wasserstein 距离和患者分组 epsilon-squared，仅作分布差异描述；缺少技术批次与临床协变量时不做原因归因。

## 图表与统计口径

- `F01`：患者内坐标几何；只在患者内构图。
- `F02`：每个患者×通路的原始 Moran's I。
- `F03`：一步邻边差异与全局随机对照差异之比。该指标**不是距离匹配对照**，旧报告中的对应称呼已撤回。
- `F04`：按 internal-six Moran 中位数选出的前 10 通路；210 个单元中有 **{monotone_variograms}/210** 的 h1-h5 半方差非递减，不能概括为全部单调。
- `F05`：原始分与患者内百分位空间图。冷热次序相同不等于 Moran 数值相同。
- `F06`：spot 级绝对分布，不再用“每患者一个中位数”的伪箱线图，也不作批次归因。
- `F07`：展示各患者 Moran 点和 internal-six 中位数；它是效应分布图，不再误称带置信区间的森林图。

## 明确不证明

1. 本分析**不能**证明 MPP2 模型预测结果已经具备空间连续性（模型预测连续性需等训练后评估）。
2. 本分析**不能**证明 Phase 3 结局预测会因此直接获益。
3. 本分析是对 **ssGSEA 基因集分数** 的分析，不能替代对原始单基因表达量（Counts）的跨患者比较。
4. 本分析不能从患者间分布位移单独识别批次效应、疾病严重度或生物学原因。

---
*报告生成时间：2026-08-01 | 自动生成自修复后的 `analyze.py`*
"""
    with open(output_root / "human_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)

    appendix = """# Gemini v001 结论追加修正（Codex，2026-08-01）

本文件不删除 Gemini 原始产出；原始文件已在本目录历史快照中保留。以下为追加性校正：

| 原结论 | 核查结果 | 修正表述 |
|---|---|---|
| 百分位完全保留 Moran's I | 不成立 | 排序与 top/bottom 热点集合保持，但 Moran 数值通常改变；见 `percentile_spatial_metrics.csv`。 |
| edge_ratio 是距离匹配随机对照 | 不成立 | 原实现未使用已计算的距离；现正名为全局随机对照比，并新增 2--3 步非邻边对照。 |
| 7 人共同决定可复现等级 | 不符合原方案队列边界 | internal 6 为主复现队列，XZY 仅作外部方向确认。 |
| Stouffer q < 0.05 | 原实现只有 p、未做 30 通路 BH | 已新增 `stouffer_q_bh_30_pathways` 并据此分级。 |
| 跨患者位移证明批次效应/保留病情严重度 | 证据不足 | 只能描述分布差异；新增效应量，但不做原因归因。 |
| 所有变异函数都单调上升 | 表述过度 | 报告改为给出实际非递减单元数。 |
| F07 是森林图 | 图形类型不符 | 改为患者级 Moran 效应点图，并单列 internal-six 中位数。 |

证据级别始终为 `diagnostic_only`；不构成训练授权、性能提升证明或 accepted 结果。
"""
    with open(output_root / "codex_correction_appendix.md", 'w', encoding='utf-8') as f:
        f.write(appendix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPP2 Raw Spatial Rank Analysis")
    parser.add_argument("--config", type=str, default="scripts/explorations/mpp2_raw_spatial_rank_v001/config.json")
    parser.add_argument("--mode", type=str, choices=['smoke', 'final'], default='final')
    parser.add_argument("--replace-existing", action="store_true",
                        help="Explicitly replace files in an already archived output directory")
    args = parser.parse_args()
    
    run_full_pipeline(args.config, mode=args.mode, replace_existing=args.replace_existing)
