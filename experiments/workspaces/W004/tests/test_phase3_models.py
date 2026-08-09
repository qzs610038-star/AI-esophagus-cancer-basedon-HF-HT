import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).parents[1] / "code" / "phase3_pipeline" / "models.py"
_SPEC = importlib.util.spec_from_file_location("phase3_models", _PATH)
models = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(models)


def test_pooling_masks_padding_and_attention_sums_to_one():
    x = torch.tensor([[[1., 0.], [3., 0.], [99., 0.]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(models.mean_mil_pool(x, mask), torch.tensor([[2., 0.]]))
    pooled, weights = models.AttentionMILPooling(2)(x, mask, return_attention=True)
    assert pooled.shape == (1, 2)
    assert weights[0, 2] == 0 and torch.allclose(weights.sum(1), torch.ones(1))


@pytest.mark.parametrize("fusion", ["h_and_e", "pathway", "concat", "dual_linear", "adaptive_gated", "low_rank"])
def test_unified_bag_classifier_all_fusions(fusion):
    model = models.MILBagClassifier(5, hidden_dim=8, fusion=fusion, pooling="attention", num_outputs=2)
    image, pathways, mask = torch.randn(2, 4, 5), torch.randn(2, 4, 30), torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    logits, attention = model(image, pathways, mask, return_attention=True)
    assert logits.shape == (2, 2) and attention.shape == (2, 4)


def test_residual_adapters_start_as_exact_identity():
    values = torch.randn(2, 3, 30)
    generic = models.GenericEqualParameterAdapter()
    pathway = models.PathwayViewResidualAdapter()
    assert torch.equal(generic(values), values)
    assert torch.equal(pathway(values, values + 1, values - 1, values.abs()), values)


def test_patient_aggregation_monotonic_head_and_slice_weights():
    slides = torch.randn(2, 3, 7)
    pooled, attention = models.MultiSlideSetAttention(7)(slides, torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool), True)
    logits = models.MonotonicPCRMPRHead(7)(pooled)
    assert attention.shape == (2, 3)
    assert torch.all(torch.sigmoid(logits[:, 0]) <= torch.sigmoid(logits[:, 1]))
    assert torch.allclose(models.inverse_slide_count_weights(torch.tensor([1, 2, 4])), torch.tensor([1., .5, .25]))


def test_shape_errors_are_explicit():
    with pytest.raises(ValueError, match="rank 3"):
        models.mean_mil_pool(torch.ones(2, 3))
    with pytest.raises(ValueError, match="positive"):
        models.inverse_slide_count_weights(torch.tensor([1, 0]))


def test_pathology_residual_and_joint_loss_preserve_contracts():
    image = torch.randn(2, 3, 5)
    views = [torch.rand(2, 3, 30) for _ in range(4)]
    adapter = models.PathologyPathwayResidualAdapter(5)
    assert torch.equal(adapter(image, *views), image)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    optimizer.zero_grad()
    adapter(image, *views).sum().backward()
    assert adapter.epsilon.grad is not None and adapter.epsilon.grad.abs() > 0
    optimizer.step()
    assert not torch.equal(adapter(image, *views), image)
    logits = models.MonotonicPCRMPRHead(5)(torch.randn(2, 5))
    targets = torch.tensor([[1.0, 1.0], [float("nan"), 0.0]])
    loss = models.monotonic_joint_bce_loss(logits, targets, torch.tensor([0.5, 1.0]))
    assert loss.ndim == 0 and torch.isfinite(loss)
