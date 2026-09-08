from __future__ import annotations

import pandas as pd

from .protocol import Sampler, SamplingSpec


class DenseSampler(Sampler):
    sampling_id = "dense"

    def spec(self) -> SamplingSpec:
        return SamplingSpec(self.sampling_id, apply_to=(), details={"kept": "all_splits"})

    def filter_manifest(self, manifest_df: pd.DataFrame, split: str) -> pd.DataFrame:
        return manifest_df.copy()
