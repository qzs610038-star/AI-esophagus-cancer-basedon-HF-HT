"""Fixed pathway graph and interpretation utilities for optional Phase 5 use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_gene_overlap_graph(
    pathway_names: Sequence[str],
    gene_sets: Mapping[str, Sequence[str]],
    *,
    minimum_jaccard: float = 0.0,
) -> np.ndarray:
    """Build a fixed, symmetric Jaccard graph in the supplied pathway order."""

    if not 0 <= minimum_jaccard <= 1:
        raise ValueError("minimum_jaccard must be in [0, 1]")
    names = [str(name) for name in pathway_names]
    if len(names) != len(set(names)) or any(name not in gene_sets for name in names):
        raise ValueError("pathway names must be unique and present in gene_sets")
    sets = [{str(gene) for gene in gene_sets[name]} for name in names]
    if any(not genes for genes in sets):
        raise ValueError("gene sets must be non-empty")
    graph = np.eye(len(names), dtype=np.float64)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            score = len(sets[left] & sets[right]) / len(sets[left] | sets[right])
            if score >= minimum_jaccard:
                graph[left, right] = graph[right, left] = score
    return graph


def normalized_graph(graph: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(graph, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph must be square")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any() or not np.allclose(adjacency, adjacency.T):
        raise ValueError("graph must be finite, non-negative, and symmetric")
    degree = adjacency.sum(axis=1)
    if (degree <= 0).any():
        raise ValueError("graph contains an isolated node")
    inverse = np.diag(1.0 / np.sqrt(degree))
    return inverse @ adjacency @ inverse


def fixed_graph_diffusion(pathway_values: np.ndarray, graph: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply a non-learned pathway-graph sensitivity transform."""

    values = np.asarray(pathway_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != np.asarray(graph).shape[0]:
        raise ValueError("pathway values and graph dimensions differ")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return (1 - alpha) * values + alpha * values @ normalized_graph(graph)


def intervention_delta(
    predict_probability,
    pathway_values: np.ndarray,
    *,
    pathway_index: int,
    replacement: float = 0.0,
) -> np.ndarray:
    """Compute probability change after a frozen single-pathway intervention."""

    values = np.asarray(pathway_values, dtype=float)
    if values.ndim != 2 or not 0 <= pathway_index < values.shape[1]:
        raise ValueError("invalid pathway intervention input")
    baseline = np.asarray(predict_probability(values), dtype=float)
    changed = values.copy()
    changed[:, pathway_index] = replacement
    intervened = np.asarray(predict_probability(changed), dtype=float)
    if baseline.shape != intervened.shape:
        raise ValueError("predict_probability returned inconsistent shapes")
    return intervened - baseline
