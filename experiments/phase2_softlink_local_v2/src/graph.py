"""Same-slide, same-split one-hop directed graph. Labels never enter this module's inputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from data import PointIdentity, PointTable, geometry_by_slide, load_slide_geometry, require_verified_slides
from errors import IdentityMismatchError, SlideMappingMissingError


@dataclass
class GraphEdge:
    neighbor: PointIdentity
    neighbor_index: int
    distance: float
    rho: float
    cosine: float
    b: float
    a: float


@dataclass
class GraphNode:
    identity: PointIdentity
    index: int
    a_self: float
    degree: int
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class SpatialGraph:
    split: str
    nodes: dict[tuple[str, str, str], GraphNode]
    index_of: dict[tuple[str, str, str], int]
    identities: list[PointIdentity]
    use_image_similarity: bool
    radius_in_native_steps: float
    max_neighbors: int
    self_raw_weight: float

    def node(self, ident: PointIdentity) -> GraphNode:
        return self.nodes[ident.key()]


def _l2_normalize(vectors: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _edge_weight(
    rho: float,
    cosine: float,
    *,
    sigma: float,
    use_image_similarity: bool,
    image_temperature: float,
) -> float:
    distance_term = float(np.exp(-(rho ** 2) / (2.0 * (sigma ** 2))))
    if use_image_similarity:
        morph = float(np.exp((cosine - 1.0) / float(image_temperature)))
        return distance_term * morph
    return distance_term


def build_split_graph(
    table: PointTable,
    split: str,
    config: dict,
    *,
    geometry_table=None,
    require_features: bool = True,
) -> SpatialGraph:
    require_verified_slides(table, context=f"build_split_graph(split={split})")
    graph_cfg = config["graph"]
    if int(graph_cfg["hops"]) != 1:
        raise IdentityMismatchError("只实现 hops=1")
    if graph_cfg.get("directed") is not True:
        raise IdentityMismatchError("只实现有向图，不对称化")
    if table.features is None and require_features:
        raise IdentityMismatchError("建图需要冻结图像特征；不得用标签构图")

    if geometry_table is None:
        geometry_table = load_slide_geometry(config["data"]["slide_geometry_file"])
    by_slide = geometry_by_slide(geometry_table)

    mask = np.where(table.split == split)[0]
    if mask.size == 0:
        raise IdentityMismatchError(f"split={split} 没有点")

    groups: dict[str, list[int]] = defaultdict(list)
    for index in mask.tolist():
        ident = table.identities[int(index)]
        if table.split[int(index)] != split:
            raise IdentityMismatchError("建图时出现跨 split 点")
        groups[ident.slide_id].append(int(index))

    radius_steps = float(graph_cfg["radius_in_native_steps"])
    max_neighbors = int(graph_cfg["max_neighbors"])
    sigma = float(graph_cfg["distance_sigma"])
    use_image = bool(graph_cfg["use_image_similarity"])
    image_temperature = float(graph_cfg["image_temperature"])
    self_raw = float(graph_cfg["self_raw_weight"])
    eps = float(config["model"]["normalization_epsilon"])

    features = None if table.features is None else _l2_normalize(np.asarray(table.features, dtype=np.float64), eps)
    nodes: dict[tuple[str, str, str], GraphNode] = {}

    for slide_id, members in groups.items():
        if slide_id not in by_slide:
            raise SlideMappingMissingError(
                f"SLIDE_MAPPING_UNVERIFIED: slide_id={slide_id} 在 slide_geometry.csv 中没有绑定原生步长 s。"
                " 不得用患者步长顶替未经核实的切片几何。"
            )
        step = float(by_slide[slide_id]["s"])
        if not np.isfinite(step) or step <= 0:
            raise IdentityMismatchError(f"slide_id={slide_id} 的原生步长非法: {step}")
        radius = radius_steps * step
        coords = np.stack([table.x[members], table.y[members]], axis=1)
        delta = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        for local_i, global_i in enumerate(members):
            ident_i = table.identities[global_i]
            ranked: list[tuple[float, tuple[str, str, str], int, int]] = []
            for local_j, global_j in enumerate(members):
                if local_j == local_i:
                    continue
                d_ij = float(dist[local_i, local_j])
                if d_ij > radius:
                    continue
                ident_j = table.identities[global_j]
                if ident_j.slide_id != ident_i.slide_id:
                    raise IdentityMismatchError("图边跨 slide")
                if table.split[global_j] != split or table.split[global_i] != split:
                    raise IdentityMismatchError("图边跨 split")
                ranked.append((d_ij, ident_j.key(), local_j, global_j))
            ranked.sort()
            chosen = ranked[:max_neighbors]
            weights_b = []
            edges = []
            for d_ij, _, local_j, global_j in chosen:
                ident_j = table.identities[global_j]
                rho = d_ij / step
                if features is None:
                    cosine = float("nan")
                else:
                    cosine = float(np.dot(features[global_i], features[global_j]))
                b_ij = _edge_weight(
                    rho,
                    cosine,
                    sigma=sigma,
                    use_image_similarity=use_image,
                    image_temperature=image_temperature,
                )
                weights_b.append(b_ij)
                edges.append(
                    GraphEdge(
                        neighbor=ident_j,
                        neighbor_index=global_j,
                        distance=d_ij,
                        rho=float(rho),
                        cosine=cosine,
                        b=float(b_ij),
                        a=0.0,
                    )
                )
            sum_b = float(np.sum(weights_b)) if weights_b else 0.0
            denom = self_raw + sum_b
            a_self = self_raw / denom if denom != 0 else 1.0
            for edge, b_ij in zip(edges, weights_b):
                edge.a = float(b_ij / denom) if denom != 0 else 0.0
            nodes[ident_i.key()] = GraphNode(
                identity=ident_i,
                index=global_i,
                a_self=float(a_self),
                degree=len(edges),
                edges=edges,
            )

    identities = [table.identities[int(i)] for i in mask.tolist()]
    return SpatialGraph(
        split=split,
        nodes=nodes,
        index_of={table.identities[int(i)].key(): int(i) for i in mask.tolist()},
        identities=identities,
        use_image_similarity=use_image,
        radius_in_native_steps=radius_steps,
        max_neighbors=max_neighbors,
        self_raw_weight=self_raw,
    )


@dataclass
class NeighborBatch:
    center_identities: list[PointIdentity]
    center_index: np.ndarray
    neighbor_index: np.ndarray
    neighbor_mask: np.ndarray
    neighbor_weights: np.ndarray
    neighbor_rho: np.ndarray
    neighbor_cosine: np.ndarray
    neighbor_b: np.ndarray
    a_self: np.ndarray
    k_max: int

    def neighbor_features(self, table: PointTable) -> np.ndarray | None:
        if table.features is None:
            return None
        batch, k_max = self.neighbor_index.shape
        out = np.zeros((batch, k_max, table.features.shape[1]), dtype=np.float32)
        for i in range(batch):
            for k in range(k_max):
                if self.neighbor_mask[i, k]:
                    out[i, k] = table.features[int(self.neighbor_index[i, k])]
        return out


def gather_neighbors(graph: SpatialGraph, center_identities: list[PointIdentity]) -> NeighborBatch:
    k_max = max((graph.node(ident).degree for ident in center_identities), default=0)
    batch = len(center_identities)
    neighbor_index = np.full((batch, k_max), -1, dtype=np.int64)
    mask = np.zeros((batch, k_max), dtype=bool)
    weights = np.zeros((batch, k_max), dtype=np.float64)
    rho = np.zeros((batch, k_max), dtype=np.float64)
    cosine = np.zeros((batch, k_max), dtype=np.float64)
    b_values = np.zeros((batch, k_max), dtype=np.float64)
    a_self = np.zeros(batch, dtype=np.float64)
    centers = np.zeros(batch, dtype=np.int64)
    for i, ident in enumerate(center_identities):
        node = graph.node(ident)
        centers[i] = node.index
        a_self[i] = node.a_self
        for k, edge in enumerate(node.edges):
            neighbor_index[i, k] = edge.neighbor_index
            mask[i, k] = True
            weights[i, k] = edge.a
            rho[i, k] = edge.rho
            cosine[i, k] = edge.cosine
            b_values[i, k] = edge.b
    return NeighborBatch(
        center_identities=list(center_identities),
        center_index=centers,
        neighbor_index=neighbor_index,
        neighbor_mask=mask,
        neighbor_weights=weights,
        neighbor_rho=rho,
        neighbor_cosine=cosine,
        neighbor_b=b_values,
        a_self=a_self,
        k_max=k_max,
    )


def graph_stats(graph: SpatialGraph, table: PointTable) -> dict:
    degrees = np.array([graph.nodes[ident.key()].degree for ident in graph.identities], dtype=np.int64)
    a_self = np.array([graph.nodes[ident.key()].a_self for ident in graph.identities], dtype=np.float64)
    b_all = []
    cosine_all = []
    for ident in graph.identities:
        for edge in graph.nodes[ident.key()].edges:
            b_all.append(edge.b)
            cosine_all.append(edge.cosine)
    b_all = np.asarray(b_all, dtype=np.float64) if b_all else np.asarray([], dtype=np.float64)
    cosine_all = np.asarray(cosine_all, dtype=np.float64) if cosine_all else np.asarray([], dtype=np.float64)
    per_patient: dict[str, dict] = {}
    for ident in graph.identities:
        rec = per_patient.setdefault(ident.patient_id, {"n": 0, "isolated": 0, "degree_sum": 0})
        rec["n"] += 1
        deg = graph.nodes[ident.key()].degree
        rec["degree_sum"] += deg
        if deg == 0:
            rec["isolated"] += 1
    for rec in per_patient.values():
        rec["isolated_rate"] = rec["isolated"] / rec["n"] if rec["n"] else 0.0
        rec["mean_degree"] = rec["degree_sum"] / rec["n"] if rec["n"] else 0.0
    return {
        "split": graph.split,
        "n_nodes": len(graph.identities),
        "n_edges": int(np.sum(degrees)),
        "mean_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "isolated_count": int(np.sum(degrees == 0)),
        "isolated_rate": float(np.mean(degrees == 0)) if degrees.size else 0.0,
        "a_self_mean": float(np.mean(a_self)) if a_self.size else float("nan"),
        "a_self_median": float(np.median(a_self)) if a_self.size else float("nan"),
        "b_mean": float(np.mean(b_all)) if b_all.size else float("nan"),
        "cosine_mean": float(np.mean(cosine_all)) if cosine_all.size else float("nan"),
        "use_image_similarity": graph.use_image_similarity,
        "per_patient": per_patient,
        "degree_histogram": {int(k): int(np.sum(degrees == k)) for k in range(0, int(graph.max_neighbors) + 1)},
        "n_points_in_table_split": int((table.split == graph.split).sum()),
    }


def edges_as_records(graph: SpatialGraph) -> list[dict]:
    rows = []
    for ident in graph.identities:
        node = graph.nodes[ident.key()]
        if not node.edges:
            rows.append(
                {
                    "patient_id": ident.patient_id,
                    "slide_id": ident.slide_id,
                    "spot_id": ident.spot_id,
                    "neighbor_patient_id": "",
                    "neighbor_slide_id": "",
                    "neighbor_spot_id": "",
                    "distance": None,
                    "rho": None,
                    "cosine": None,
                    "b": None,
                    "a": None,
                    "a_self": node.a_self,
                    "degree": 0,
                    "isolated": True,
                }
            )
            continue
        for edge in node.edges:
            rows.append(
                {
                    "patient_id": ident.patient_id,
                    "slide_id": ident.slide_id,
                    "spot_id": ident.spot_id,
                    "neighbor_patient_id": edge.neighbor.patient_id,
                    "neighbor_slide_id": edge.neighbor.slide_id,
                    "neighbor_spot_id": edge.neighbor.spot_id,
                    "distance": edge.distance,
                    "rho": edge.rho,
                    "cosine": edge.cosine,
                    "b": edge.b,
                    "a": edge.a,
                    "a_self": node.a_self,
                    "degree": node.degree,
                    "isolated": False,
                }
            )
    return rows
