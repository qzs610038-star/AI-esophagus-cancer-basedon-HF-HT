from __future__ import annotations

import json
from pathlib import Path

from .protocol import TargetAdapter, TargetSpec


class PathwaySsGseaTarget(TargetAdapter):
    target_id = "pathway_ssgsea"

    def __init__(self, names: list[str], source: str):
        if not names:
            raise ValueError("pathway target requires a non-empty name list")
        self._spec = TargetSpec(
            target_id=self.target_id,
            names=list(names),
            n_outputs=len(names),
            label_kind="ssgsea_zscore",
            source=source,
        )

    def spec(self) -> TargetSpec:
        return self._spec

    def label_csv(self, labels_root: Path, split: str, patient: str, train_mpp_id: int) -> Path:
        if split == "train":
            return labels_root / "train" / patient / f"{patient}_ssGSEA_zscore.csv"
        if split == "internal_val":
            return labels_root / "val" / patient / f"{patient}_ssGSEA_zscore.csv"
        if split == "external":
            return (
                labels_root / "external" / patient
                / f"{patient}_ssGSEA_zscore_by_group_{train_mpp_id}_train.csv"
            )
        raise ValueError(f"unknown split for pathway labels: {split}")


def load_pathway_names(zscore_manifest: Path) -> list[str]:
    payload = json.loads(zscore_manifest.read_text(encoding="utf-8-sig"))
    names = payload.get("pathway_names")
    if not names:
        raise ValueError(f"pathway_names missing in {zscore_manifest}")
    return [str(name) for name in names]
