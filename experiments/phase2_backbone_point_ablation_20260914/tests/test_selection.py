from __future__ import annotations

import numpy as np
import pytest

from selection import (
    CheckpointChoice,
    EarlyStopState,
    patient_macro_metrics,
    update_checkpoint_choice,
    update_early_stop,
)


def test_constant_prediction_penalty_is_separate_from_measured_pcc():
    target = np.array([[0.0], [1.0], [0.0], [2.0]])
    prediction = np.ones_like(target)
    metrics = patient_macro_metrics(
        prediction,
        target,
        ["a", "a", "b", "b"],
        pathway_names=["p"],
        require_patient_count=2,
    )

    assert metrics.patient_macro_pathway_pcc == -1.0
    assert np.isnan(metrics.patient_macro_pathway_pcc_measured)
    assert metrics.n_constant_prediction_penalized == 2


def test_patient_macro_is_patient_equal_not_point_pooled():
    # Patient a has four observations and perfect PCC; patient b has two and
    # perfect negative PCC. Equal patient weighting therefore gives zero.
    target = np.array([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]])
    prediction = np.array([[0.0], [1.0], [2.0], [3.0], [1.0], [0.0]])
    metrics = patient_macro_metrics(
        prediction,
        target,
        ["a", "a", "a", "a", "b", "b"],
        require_patient_count=2,
    )
    assert metrics.patient_macro_pathway_pcc == pytest.approx(0.0, abs=1e-12)


def test_checkpoint_tie_prefers_lower_mse_then_earlier_epoch():
    current = CheckpointChoice(epoch=6, score=0.2, mse=1.0)
    better_mse = update_checkpoint_choice(
        current, epoch=7, score=0.2000005, mse=0.9, tolerance=1e-6
    )
    assert better_mse.epoch == 7
    same = update_checkpoint_choice(
        better_mse, epoch=8, score=better_mse.score, mse=better_mse.mse
    )
    assert same.epoch == 7


def test_formal_window_and_early_stop_count_from_epoch_16():
    state = EarlyStopState()
    for epoch in range(1, 6):
        state = update_early_stop(state, epoch=epoch, score=0.9, patience=2)
        assert state.reference is None
    for epoch in range(6, 16):
        state = update_early_stop(state, epoch=epoch, score=0.2 + epoch / 1000, patience=2)
    assert state.reference == pytest.approx(0.215)
    state = update_early_stop(state, epoch=16, score=0.21505, patience=2)
    assert state.count == 1 and not state.stopped
    state = update_early_stop(state, epoch=17, score=0.21505, patience=2)
    assert state.stopped
