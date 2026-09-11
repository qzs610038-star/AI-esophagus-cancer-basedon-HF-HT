"""Self-contained model, graph, and metric primitives for this experiment."""

from .metrics import (
    SelectionCandidate,
    compute_metrics,
    compute_regression_metrics,
    flattened_pooled_pcc,
    patient_macro_pathway_pcc,
    patient_macro_pathway_pcc_details,
    pooled_pcc,
    select_best_candidate,
    selection_key,
    z_mse,
)
from .model import (
    RegressionHead,
    SharedProjection,
    SourceLoadReport,
    SpatialWarmstartModel,
    WarmstartOutput,
    load_stage1_hc,
)
from .spatial import SpatialGraph, SpatialPoint, build_spatial_graph

__all__ = [
    "RegressionHead",
    "SelectionCandidate",
    "SharedProjection",
    "SourceLoadReport",
    "SpatialGraph",
    "SpatialPoint",
    "SpatialWarmstartModel",
    "WarmstartOutput",
    "build_spatial_graph",
    "compute_metrics",
    "compute_regression_metrics",
    "flattened_pooled_pcc",
    "load_stage1_hc",
    "patient_macro_pathway_pcc",
    "patient_macro_pathway_pcc_details",
    "pooled_pcc",
    "select_best_candidate",
    "selection_key",
    "z_mse",
]
