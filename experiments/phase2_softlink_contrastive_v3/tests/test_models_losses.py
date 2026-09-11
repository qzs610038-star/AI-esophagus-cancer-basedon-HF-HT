"""Small, CPU-only contract tests for the Phase2 v4 model primitives."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.losses import (
    bidirectional_soft_contrastive_loss,
    build_teacher_targets,
    select_teacher_temperature,
)
from src.diagnostics import delta_w_ratios, gradient_norms, representation_statistics
from src.models import (
    Stage1Model,
    Stage2Head,
    assert_lora_contract,
    count_trainable_parameters,
    inject_independent_qv_lora,
    lora_disabled,
)


class TinyAttention(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.qkv(x)[..., : x.shape[-1]])


class TinyBlock(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.attn = TinyAttention(dim)
        self.mlp = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.mlp(self.attn(x)) + x)


class TinyBackbone(nn.Module):
    def __init__(self, dim=12, blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x[:, 0]


class ForwardFeaturesOnlyBackbone(TinyBackbone):
    def __init__(self):
        super().__init__()
        self.forward_features_calls = 0

    def forward_features(self, x):
        self.forward_features_calls += 1
        return torch.ones(x.shape[0], 2, 12, device=x.device)

    def forward(self, x):
        raise AssertionError("Stage1Model must prefer forward_features for cache-compatible CLS")


def test_zero_delta_keeps_backbone_forward_and_k_is_untouched():
    torch.manual_seed(1)
    backbone = TinyBackbone().eval()
    x = torch.randn(3, 2, 12)
    expected = backbone(x)
    inject_independent_qv_lora(backbone, [0, 1, 2, 3], rank=2, alpha=4, dropout=0.05)
    actual = backbone(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_lora_contract(backbone, [0, 1, 2, 3])
    for block in backbone.blocks:
        wrapper = block.attn.qkv
        assert wrapper.q_A.data_ptr() != wrapper.v_A.data_ptr()
        assert wrapper.delta_weight("k").abs().sum() == 0


def test_lora_gradients_reach_b_after_first_update_and_a_after_second():
    torch.manual_seed(2)
    backbone = TinyBackbone()
    inject_independent_qv_lora(backbone, [0, 1, 2, 3], rank=2, alpha=4)
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    x = torch.randn(4, 2, 12)
    target = torch.randn(4, 12)
    for _ in range(2):
        optimizer.zero_grad()
        (backbone(x) - target).square().mean().backward()
        optimizer.step()
    wrapper = backbone.blocks[0].attn.qkv
    assert wrapper.q_B.grad is not None and wrapper.q_B.grad.norm() > 0
    assert wrapper.q_A.grad is not None and wrapper.q_A.grad.norm() > 0
    assert delta_w_ratios(backbone)[0]["k"] == 0.0
    assert gradient_norms(backbone)["blocks.0.attn.qkv.q_A"] > 0
    stats = representation_statistics(backbone(x).detach())
    assert stats["variance"] >= 0 and stats["effective_rank"] > 0


def test_lora_can_be_disabled_temporarily_for_frozen_reference_without_model_copy():
    torch.manual_seed(7)
    backbone = TinyBackbone().eval()
    x = torch.randn(3, 2, 12)
    frozen_reference = backbone(x)
    inject_independent_qv_lora(backbone, [0, 1, 2, 3], rank=2, alpha=4)
    backbone.eval()
    for block in backbone.blocks:
        block.attn.qkv.q_B.data.fill_(0.1)
        block.attn.qkv.v_B.data.fill_(0.1)

    adapted = backbone(x)
    with lora_disabled(backbone):
        disabled = backbone(x)
    restored = backbone(x)

    assert not torch.allclose(adapted, frozen_reference)
    torch.testing.assert_close(disabled, frozen_reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(restored, adapted)


def test_lora_parameter_budget_is_v4_exact():
    class BudgetAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(1536, 4608)

    class BudgetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = BudgetAttention()

    class BudgetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([BudgetBlock() for _ in range(24)])

    for rank, expected in [(8, 196608), (2, 49152)]:
        backbone = BudgetBackbone()
        inject_independent_qv_lora(backbone, [20, 21, 22, 23], rank=rank, alpha=2 * rank)
        assert count_trainable_parameters(backbone) == expected


def test_stage1_and_stage2_contracts():
    model = Stage1Model(TinyBackbone(), input_dim=12, embedding_dim=8, output_dim=30)
    output = model(torch.randn(2, 2, 12), teacher_inputs=torch.randn(2, 30))
    assert set(output) == {"cls", "h", "pred", "teacher_embedding"}
    assert output["pred"].shape == (2, 30)
    shared = model.shared
    head = Stage2Head(shared, mode="spatial", input_dim=8, output_dim=30)
    assert torch.count_nonzero(head.B) == 0
    original = [p.detach().clone() for p in shared.parameters()]
    head.set_shared_mode("freeze")
    assert not any(p.requires_grad for p in shared.parameters())
    head.set_shared_mode("continue")
    assert all(p.requires_grad for p in shared.parameters())
    head.set_shared_mode("reset")
    assert all(p.requires_grad for p in shared.parameters())
    assert any(not torch.equal(before, after) for before, after in zip(original, shared.parameters()))


def test_stage1_prefers_forward_features_for_cls_extraction():
    backbone = ForwardFeaturesOnlyBackbone()
    model = Stage1Model(backbone, input_dim=12, embedding_dim=8, output_dim=30)
    result = model(torch.randn(2, 2, 12))
    assert backbone.forward_features_calls == 1
    torch.testing.assert_close(result["cls"], torch.ones(2, 12))


def test_teacher_rows_anchor_and_both_directions_are_used():
    labels = torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0]], requires_grad=True)
    q_i2y, q_y2i = build_teacher_targets(labels, ["a", "a", "b"], tau_y=0.3)
    torch.testing.assert_close(q_i2y.sum(1), torch.ones(3))
    torch.testing.assert_close(q_y2i.sum(1), torch.ones(3))
    torch.testing.assert_close(q_i2y.diag(), torch.full((3,), 0.25))
    torch.testing.assert_close(q_y2i.diag(), torch.full((3,), 0.25))
    assert not q_i2y.requires_grad and not q_y2i.requires_grad
    h = torch.randn(3, 5, requires_grad=True)
    t = torch.randn(3, 5, requires_grad=True)
    loss_a = bidirectional_soft_contrastive_loss(h, t, q_i2y, q_y2i)
    loss_b = bidirectional_soft_contrastive_loss(h, t, q_i2y, torch.eye(3))
    assert not torch.allclose(loss_a, loss_b)
    loss_a.backward()
    assert h.grad is not None and t.grad is not None


def test_patient_centered_teacher_requires_prefit_means_and_uses_fixed_fold_values():
    means = {"a": torch.tensor([100.0]), "b": torch.tensor([100.0]), "c": torch.tensor([105.0])}
    with pytest.raises(ValueError, match="pre-fitted patient_means"):
        build_teacher_targets(torch.tensor([[101.0], [99.0], [100.0]]), ["a", "a", "b"], mode="patient_centered")
    # a:101 always maps to +1 with the protocol/fold mean of 100.  Its
    # co-batch patient changes, but not this sample's centred representation.
    first_labels, first_ids = torch.tensor([[101.0], [99.0], [100.0]]), ["a", "a", "b"]
    second_labels, second_ids = torch.tensor([[101.0], [99.0], [105.0]]), ["a", "a", "c"]
    first, _ = build_teacher_targets(first_labels, first_ids, mode="patient_centered", tau_y=1.0, patient_means=means)
    second, _ = build_teacher_targets(second_labels, second_ids, mode="patient_centered", tau_y=1.0, patient_means=means)
    expected, _ = build_teacher_targets(torch.tensor([[1.0], [-1.0], [0.0]]), first_ids, mode="global", tau_y=1.0)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    batches = [(first_labels, first_ids) for _ in range(20)]
    with pytest.raises(ValueError, match="pre-fitted patient_means"):
        select_teacher_temperature(batches, mode="patient_centered")
    assert isinstance(select_teacher_temperature(batches, mode="patient_centered", patient_means=means), float)


def test_teacher_temperature_uses_larger_candidate_for_ties():
    labels = torch.tensor([[0.0], [1.0], [2.0]])
    batches = [(labels, ["a", "b", "c"]) for _ in range(20)]
    result = select_teacher_temperature(batches, candidates=(0.1, 0.3), target_entropy=1.0)
    assert result == 0.3
