from __future__ import annotations

import numpy as np
import torch

from relations import build_batch_support, compute_distance_thresholds, relation_loss, sample_threshold_pairs


def fill_rows(values):
    out = np.zeros((len(values), 30), dtype=np.float64)
    for i, value in enumerate(values):
        out[i] = float(value)
    return out


def test_self_and_ignore_leave_denominator():
    labels = fill_rows([0.0, 0.1, 1.0, 5.0])
    # D(0,1)=0.01, D(0,2)=1.0, D(0,3)=25
    support = build_batch_support(
        labels,
        patient_ids=["p0", "p0", "p1", "p2"],
        fixed_indices=[0, 1, 2, 3],
        q10=0.05,
        q50=2.0,
        max_near=8,
        max_far=32,
        far_patient_scope="all",
        run_seed=7,
        epoch=1,
        batch_index=0,
    )
    assert support.distances[0, 0] == 0.0
    assert 0 not in support.near_indices[0]
    assert 0 not in support.far_indices[0]
    assert support.near_indices[0] == [1]
    assert 2 not in support.near_indices[0] and 2 not in support.far_indices[0]
    assert support.far_indices[0] == [3]
    assert bool(support.valid[0]) is True

    z = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.2, 0.98], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        dim=-1,
    )
    z.requires_grad_(True)
    y = torch.as_tensor(labels, dtype=torch.float32)
    loss, stats = relation_loss(z, y, support, tau_y=1.0, tau_z=0.1)
    assert stats["denominator"] == int(support.n_valid)

    near = support.near_indices[0]
    far = support.far_indices[0]
    s_idx = near + far
    d_near = torch.mean((y[0] - y[near]) ** 2, dim=-1)
    q = torch.softmax(-d_near / 1.0, dim=0)
    assert torch.allclose(q.sum(), torch.tensor(1.0), atol=1e-6)
    logits = (z.detach()[s_idx] * z.detach()[0]).sum(dim=-1) / 0.1
    p = torch.softmax(logits, dim=0)
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-6)
    ignore_logit = (z.detach()[2] * z.detach()[0]).sum() / 0.1
    p_if_ignore = torch.softmax(torch.cat([logits, ignore_logit.unsqueeze(0)]), dim=0)
    assert not torch.allclose(p, p_if_ignore[: len(s_idx)])
    assert float(loss.detach()) >= 0.0


def test_valid_center_requires_near_and_far():
    labels = fill_rows([0.0, 0.01, 0.02])
    support = build_batch_support(
        labels,
        patient_ids=["a", "a", "a"],
        fixed_indices=[0, 1, 2],
        q10=1.0,
        q50=10.0,
        max_near=8,
        max_far=32,
        far_patient_scope="all",
        run_seed=1,
        epoch=1,
        batch_index=0,
    )
    assert support.n_valid == 0
    assert all(len(n) > 0 for n in support.near_indices)
    assert all(len(f) == 0 for f in support.far_indices)
    z = torch.randn(3, 4, requires_grad=True)
    loss, stats = relation_loss(z, torch.as_tensor(labels, dtype=torch.float32), support, tau_y=1.0, tau_z=0.1)
    assert stats["denominator"] == 0
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.count_nonzero(z.grad) == 0


def test_ninth_near_is_ignored_and_far_scope():
    values = [0.0] + [0.01 * (i + 1) for i in range(9)] + [9.0, 9.5]
    labels = fill_rows(values)
    patients = ["p0"] * 11 + ["p1"]
    support = build_batch_support(
        labels,
        patient_ids=patients,
        fixed_indices=list(range(12)),
        q10=1.0,
        q50=2.0,
        max_near=8,
        max_far=32,
        far_patient_scope="all",
        run_seed=3,
        epoch=1,
        batch_index=0,
    )
    assert len(support.near_indices[0]) == 8
    assert 9 not in support.near_indices[0]
    support_same = build_batch_support(
        labels,
        patient_ids=patients,
        fixed_indices=list(range(12)),
        q10=1.0,
        q50=2.0,
        max_near=8,
        max_far=32,
        far_patient_scope="same_patient",
        run_seed=3,
        epoch=1,
        batch_index=0,
    )
    assert support_same.far_indices[0]
    assert all(patients[j] == "p0" for j in support_same.far_indices[0])


def test_relation_backprop_to_h_not_labels():
    labels = fill_rows([0.0, 0.05, 6.0])
    support = build_batch_support(
        labels,
        patient_ids=["a", "b", "c"],
        fixed_indices=[0, 1, 2],
        q10=0.1,
        q50=1.0,
        max_near=8,
        max_far=32,
        far_patient_scope="all",
        run_seed=11,
        epoch=1,
        batch_index=0,
    )
    assert bool(support.valid[0])
    y = torch.as_tensor(labels, dtype=torch.float32)
    y.requires_grad_(True)
    h = torch.nn.Parameter(torch.randn(3, 6))
    proj = torch.nn.Linear(6, 4, bias=False)
    z = torch.nn.functional.normalize(proj(h), dim=-1)
    loss, stats = relation_loss(z, y, support, tau_y=1.0, tau_z=0.1)
    assert stats["denominator"] >= 1
    loss.backward()
    assert h.grad is not None
    assert float(h.grad.abs().sum()) > 0
    assert y.grad is None or float(y.grad.abs().sum()) == 0


def test_threshold_pairs_exclude_self_and_are_reproducible():
    pairs_a = sample_threshold_pairs(10, 20, 20260907)
    pairs_b = sample_threshold_pairs(10, 20, 20260907)
    np.testing.assert_array_equal(pairs_a, pairs_b)
    assert np.all(pairs_a[:, 0] != pairs_a[:, 1])
    labels = np.random.default_rng(0).normal(size=(20, 30))
    thr = compute_distance_thresholds(labels, n_pairs=50, seed=20260907)
    assert thr.q10 <= thr.q50
    assert thr.pair_indices.shape == (50, 2)
