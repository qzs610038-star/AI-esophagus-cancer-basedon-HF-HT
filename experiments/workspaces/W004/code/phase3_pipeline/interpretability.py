"""Blind-review packet helpers; image generation remains an explicit caller step."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def build_blinded_review_index(
    heatmap_manifest: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a reviewer index and a separate private key table."""
    required = {"case_id", "slide_id", "model", "heatmap_path"}
    if missing := required - set(heatmap_manifest.columns):
        raise ValueError(f"heatmap manifest missing columns: {sorted(missing)}")
    if heatmap_manifest.empty or heatmap_manifest["heatmap_path"].duplicated().any():
        raise ValueError("heatmap manifest must be non-empty with unique paths")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(heatmap_manifest))
    source = heatmap_manifest.iloc[order].reset_index(drop=True).copy()
    source["review_id"] = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:12] for i in range(len(source))]
    reviewer = source[["review_id", "heatmap_path"]].copy()
    key = source[["review_id", "case_id", "slide_id", "model"]].copy()
    return reviewer, key


def summarize_blinded_reviews(reviews: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    required = {"review_id", "reviewer_id", "score"}
    if missing := required - set(reviews.columns):
        raise ValueError(f"reviews missing columns: {sorted(missing)}")
    if not np.isfinite(reviews["score"].to_numpy(dtype=float)).all():
        raise ValueError("review scores must be finite")
    merged = reviews.merge(key, on="review_id", how="left", validate="many_to_one")
    if merged["model"].isna().any():
        raise ValueError("review contains unknown blinded IDs")
    return merged.groupby("model", as_index=False).agg(mean_score=("score", "mean"), n_reviews=("score", "size"), n_reviewers=("reviewer_id", "nunique"))
