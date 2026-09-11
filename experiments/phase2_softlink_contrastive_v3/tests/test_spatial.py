from pathlib import Path
import sys

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from src.data import OnlineRecord
from src.spatial import build_spatial_graph


def _record(patient, stem, x, y):
    return OnlineRecord(2, patient, stem, x, y, "train", "b", Path("unused"), np.zeros(30, np.float32))


def test_fixed_graph_is_patient_local_and_uses_neighbor_difference():
    records = [
        _record("A", "a0", 0, 0), _record("A", "a1", 1, 0),
        _record("B", "b0", 0, 0), _record("B", "b1", 1, 0),
    ]
    features = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    graph = build_spatial_graph(records, features, native_steps={"A": 1, "B": 1}, max_neighbors=8)
    for center, neighbors in enumerate(graph.neighbor_index):
        for neighbor in neighbors[neighbors >= 0]:
            assert records[center].patient == records[int(neighbor)].patient
    hidden = torch.tensor([[0.0], [2.0], [10.0], [14.0]])
    delta = graph.spatial_delta(hidden)
    assert delta[0] > 0 and delta[1] < 0 and delta[2] > 0 and delta[3] < 0


def test_radius_and_neighbor_cap_are_enforced():
    records = [_record("A", str(i), i, 0) for i in range(5)]
    graph = build_spatial_graph(
        records, np.ones((5, 2)), native_steps={"A": 1},
        radius_in_native_steps=1.5, max_neighbors=1,
    )
    assert np.all(graph.degree <= 1)
