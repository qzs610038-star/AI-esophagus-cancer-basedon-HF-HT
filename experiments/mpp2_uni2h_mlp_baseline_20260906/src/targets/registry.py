from __future__ import annotations

from pathlib import Path

from .gene_expression import GeneExpressionTarget
from .pathway_ssgsea import PathwaySsGseaTarget, load_pathway_names
from .protocol import TargetAdapter


def load_target(target_id: str, zscore_manifest: Path) -> TargetAdapter:
    if target_id == "pathway_ssgsea":
        names = load_pathway_names(zscore_manifest)
        return PathwaySsGseaTarget(names, source=str(zscore_manifest))
    if target_id == "gene_expression":
        return GeneExpressionTarget()
    raise ValueError(
        f"unknown target_id={target_id!r}. Implement src/targets/<id>.py and register it here."
    )
