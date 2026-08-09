import numpy as np
import pytest

from pfmval_core.metrics import compute_metrics


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2])),
        (np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([1.0]), np.array([1.5])),
    ],
)
def test_neutral_metrics_returns_expected_keys_and_finite_values(y_true, y_pred):
    actual = compute_metrics(y_true, y_pred)
    assert actual.keys() == {"mse", "mae", "r2", "pcc"}
    assert all(np.isfinite(actual[key]) for key in ("mse", "mae", "pcc"))
    assert np.isfinite(actual["r2"]) or np.isnan(actual["r2"])


def test_neutral_metrics_rejects_nan():
    values = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        compute_metrics(values, values)


def test_active_mpp_trainers_use_neutral_metrics_module():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    for name in ("train_mpp_uni2h_mlp.py", "train_mpp_uni2h_lora.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "from pfmval_core.metrics import compute_metrics" in text
        assert "from histogene.utils import compute_metrics" not in text
