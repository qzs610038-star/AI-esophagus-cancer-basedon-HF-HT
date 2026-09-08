from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SamplingSpec:
    sampling_id: str
    apply_to: tuple[str, ...]
    details: dict

    def to_dict(self) -> dict:
        return {
            "sampling_id": self.sampling_id,
            "apply_to": list(self.apply_to),
            "details": self.details,
        }


class Sampler:
    sampling_id = ""

    def spec(self) -> SamplingSpec:
        raise NotImplementedError

    def filter_manifest(self, manifest_df: pd.DataFrame, split: str) -> pd.DataFrame:
        raise NotImplementedError
