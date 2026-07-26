import numpy as np
import pytest

from histogene.utils import compute_metrics as legacy_compute_metrics
from pfmval_core.metrics import compute_metrics


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2])),
        (np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([1.0]), np.array([1.5])),
    ],
)
def test_neutral_metrics_matches_legacy_for_supported_inputs(y_true, y_pred):
    expected = legacy_compute_metrics(y_true, y_pred)
    actual = compute_metrics(y_true, y_pred)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], nan_ok=True)


def test_neutral_metrics_matches_legacy_nan_rejection():
    values = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        legacy_compute_metrics(values, values)
    with pytest.raises(ValueError):
        compute_metrics(values, values)


def test_active_mpp_trainers_use_neutral_metrics_module():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    for name in ("train_mpp_uni2h_mlp.py", "train_mpp_uni2h_lora.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "from pfmval_core.metrics import compute_metrics" in text
        assert "from histogene.utils import compute_metrics" not in text
