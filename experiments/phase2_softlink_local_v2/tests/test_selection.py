import pytest

from selection import CheckpointChoice, EarlyStopState, relation_lambda, update_checkpoint_choice, update_early_stop


def test_lambda_and_checkpoint_tie_rules():
    assert relation_lambda(5) == 0.0
    assert relation_lambda(6) == 0.005
    assert relation_lambda(15) == 0.05
    current = CheckpointChoice(epoch=6, score=0.2, mse=1.0, kind="formal")
    lower_mse = update_checkpoint_choice(current, epoch=7, score=0.2000005, mse=0.9, lambda_value=0.01)
    assert lower_mse.epoch == 7
    earlier = update_checkpoint_choice(lower_mse, epoch=8, score=lower_mse.score, mse=lower_mse.mse, lambda_value=0.02)
    assert earlier.epoch == 7


def test_early_stop_reference_window_and_count_start():
    state = EarlyStopState()
    for epoch in range(6, 16):
        state = update_early_stop(state, epoch=epoch, score=0.2 + epoch / 1000, patience=2)
    assert state.reference == pytest.approx(0.215)
    state = update_early_stop(state, epoch=16, score=0.21505, patience=2)
    assert state.count == 1 and not state.stopped
    state = update_early_stop(state, epoch=17, score=0.21505, patience=2)
    assert state.stopped
