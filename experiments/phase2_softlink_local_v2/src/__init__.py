"""Batch-1 public interface. Training / schedulers live in later batches."""

from config import load_config, validate_config
from data import (
    Normalization,
    PointIdentity,
    PointTable,
    attach_features,
    attach_labels,
    build_unlabeled_point_table_from_feature_directory,
    inverse_transform,
    iter_index_batches,
    load_normalization,
    load_normalization_from_config,
    load_split_point_table,
    make_point_table,
    require_verified_slides,
)
from errors import (
    ConfigError,
    IdentityMismatchError,
    NonFiniteDataError,
    SlideMappingMissingError,
    UnsupportedAlgorithmError,
)
from graph import NeighborBatch, SpatialGraph, build_split_graph, gather_neighbors
from model import SoftlinkModel, apply_center_dropout_mask, build_model, predict
from precheck import run_precheck
from relations import (
    BatchSupport,
    ThresholdResult,
    build_batch_support,
    compute_distance_thresholds,
    relation_loss,
    sample_threshold_pairs,
)

__all__ = [
    "BatchSupport",
    "ConfigError",
    "IdentityMismatchError",
    "NeighborBatch",
    "NonFiniteDataError",
    "Normalization",
    "PointIdentity",
    "PointTable",
    "SlideMappingMissingError",
    "SoftlinkModel",
    "SpatialGraph",
    "ThresholdResult",
    "UnsupportedAlgorithmError",
    "apply_center_dropout_mask",
    "attach_features",
    "attach_labels",
    "build_batch_support",
    "build_model",
    "build_split_graph",
    "build_unlabeled_point_table_from_feature_directory",
    "compute_distance_thresholds",
    "gather_neighbors",
    "inverse_transform",
    "iter_index_batches",
    "load_config",
    "load_normalization",
    "load_normalization_from_config",
    "load_split_point_table",
    "make_point_table",
    "predict",
    "relation_loss",
    "require_verified_slides",
    "run_precheck",
    "sample_threshold_pairs",
    "validate_config",
]
