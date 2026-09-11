"""Phase2 soft-link contrastive v4 model, loss, and diagnostic primitives."""

from .diagnostics import (
    covariance_effective_rank,
    delta_w_ratios,
    gradient_cosine,
    gradient_norms,
    off_diagonal_cosine_statistics,
    representation_statistics,
)
from .losses import (
    DEFAULT_TEMPERATURE_CANDIDATES,
    bidirectional_soft_contrastive_loss,
    build_teacher_targets,
    select_teacher_temperature,
)
from .models import (
    IndependentQVLoRA,
    PathwayTeacher,
    RegressionHead,
    SharedProjection,
    Stage1Model,
    Stage2Head,
    assert_lora_contract,
    count_trainable_parameters,
    inject_independent_qv_lora,
    lora_delta_report,
)

__all__ = [
    "DEFAULT_TEMPERATURE_CANDIDATES", "IndependentQVLoRA", "PathwayTeacher", "RegressionHead",
    "SharedProjection", "Stage1Model", "Stage2Head", "assert_lora_contract",
    "bidirectional_soft_contrastive_loss", "build_teacher_targets", "count_trainable_parameters",
    "covariance_effective_rank", "delta_w_ratios", "gradient_cosine", "gradient_norms",
    "inject_independent_qv_lora", "lora_delta_report", "off_diagonal_cosine_statistics",
    "representation_statistics", "select_teacher_temperature",
]
