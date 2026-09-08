from __future__ import annotations

import pandas as pd

from .protocol import Sampler, SamplingSpec

TRAIN_ONLY = ("train",)


class SpatialStrideSampler(Sampler):
    sampling_id = "spatial_stride"

    def __init__(self, stride_x: int = 2, stride_y: int = 2, phase_x: int = 0, phase_y: int = 0):
        if stride_x < 1 or stride_y < 1:
            raise ValueError("stride_x and stride_y must be >= 1")
        self.stride_x = int(stride_x)
        self.stride_y = int(stride_y)
        self.phase_x = int(phase_x) % self.stride_x
        self.phase_y = int(phase_y) % self.stride_y

    def spec(self) -> SamplingSpec:
        return SamplingSpec(
            self.sampling_id,
            apply_to=TRAIN_ONLY,
            details={
                "stride_x": self.stride_x,
                "stride_y": self.stride_y,
                "phase_x": self.phase_x,
                "phase_y": self.phase_y,
                "val_and_external": "remain_dense",
            },
        )

    def filter_manifest(self, manifest_df: pd.DataFrame, split: str) -> pd.DataFrame:
        if split != "train":
            return manifest_df.copy()
        if "x" not in manifest_df.columns or "y" not in manifest_df.columns:
            raise ValueError("spatial_stride requires x and y columns in split_manifest")
        kept = []
        for _, group in manifest_df.groupby("patient", sort=False):
            train = group[group["split"] == "train"].copy()
            other = group[group["split"] != "train"]
            if train.empty:
                kept.append(group)
                continue
            x_rank = {value: index for index, value in enumerate(sorted(train["x"].unique()))}
            y_rank = {value: index for index, value in enumerate(sorted(train["y"].unique()))}
            mask = train["x"].map(x_rank).mod(self.stride_x).eq(self.phase_x) & train["y"].map(y_rank).mod(self.stride_y).eq(self.phase_y)
            kept.append(pd.concat([train.loc[mask], other], axis=0))
        if not kept:
            return manifest_df.iloc[0:0].copy()
        return pd.concat(kept, axis=0).sort_index()
