"""Reserved Task-2 target: swap training labels from 30 pathway scores to genes.

Task 2 changes the training input table and the model output together.
Image features stay in the existing .pt cache. Do not concatenate gene
values onto those features. The teammate fills inputs.gene_labels_root
with the server path of the pathway-member gene matrices.
"""

from __future__ import annotations

from pathlib import Path

from .protocol import MissingTargetContract, TargetAdapter, TargetSpec

MISSING = (
    "spot_or_bin_gene_expression",
    "gene_identifier_mapping",
    "gene_set_membership",
    "pathway_scoring_software_and_parameters",
    "background_gene_universe",
    "train_only_gene_standardization_params",
)


class GeneExpressionTarget(TargetAdapter):
    target_id = "gene_expression"

    def spec(self) -> TargetSpec:
        raise MissingTargetContract(
            "target_id=gene_expression is reserved but not ready. Missing: "
            + ", ".join(MISSING)
        )

    def label_csv(self, labels_root: Path, split: str, patient: str, train_mpp_id: int) -> Path:
        raise MissingTargetContract(
            "gene expression labels are not configured. Missing: " + ", ".join(MISSING)
        )


def reserved_spec() -> TargetSpec:
    return TargetSpec(
        target_id="gene_expression",
        names=(),
        n_outputs=0,
        label_kind="gene_expression",
        source="reserved",
        missing=MISSING,
    )
