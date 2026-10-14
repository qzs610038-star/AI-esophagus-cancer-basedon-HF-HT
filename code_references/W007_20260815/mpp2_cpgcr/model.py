"""冻结 MPP2 基线与容量匹配的残差模型。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch
from torch import nn


ARM_ORDER = ("FBR", "RCC", "HCR", "CPGCR")
TRAINABLE_ARMS = ARM_ORDER[1:]


@dataclass(frozen=True)
class ModelOutput:
    """一次前向的预测与训练期表示。"""

    prediction: torch.Tensor
    baseline_prediction: torch.Tensor
    image_embedding: torch.Tensor | None
    pathway_embedding: torch.Tensor | None


class FrozenBaseline(nn.Module):
    """强制永久 eval、无梯度前向的 accepted B0 包装器。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenBaseline":
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(features)


class ImageProjector(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PathwayEncoder(nn.Module):
    def __init__(self, pathway_dim: int, projection_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(pathway_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        return self.layers(targets)


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def build_shared_initial_states(
    *,
    seed: int,
    input_dim: int = 1536,
    pathway_dim: int = 30,
    projection_dim: int = 256,
) -> dict[str, dict[str, torch.Tensor]]:
    """建立三臂共享图像初始化及 HCR/CPGCR 共享通路初始化。"""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        image_projector = ImageProjector(input_dim, projection_dim)
        residual_head = nn.Linear(projection_dim, pathway_dim)
        _zero_linear(residual_head)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        pathway_encoder = PathwayEncoder(pathway_dim, projection_dim)
    return {
        "image_projector": copy.deepcopy(image_projector.state_dict()),
        "residual_head": copy.deepcopy(residual_head.state_dict()),
        "pathway_encoder": copy.deepcopy(pathway_encoder.state_dict()),
    }


class ResidualContrastiveModel(nn.Module):
    """RCC/HCR/CPGCR 共用的 image-only 推理结构。"""

    def __init__(
        self,
        baseline: nn.Module,
        *,
        arm: str,
        input_dim: int = 1536,
        pathway_dim: int = 30,
        projection_dim: int = 256,
        initial_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        if arm not in TRAINABLE_ARMS:
            raise ValueError(f"arm must be one of {TRAINABLE_ARMS}, got {arm}")
        self.arm = arm
        self.baseline = FrozenBaseline(baseline)
        self.image_projector = ImageProjector(input_dim, projection_dim)
        self.residual_head = nn.Linear(projection_dim, pathway_dim)
        _zero_linear(self.residual_head)
        self.pathway_encoder = (
            PathwayEncoder(pathway_dim, projection_dim)
            if arm in {"HCR", "CPGCR"}
            else None
        )
        if initial_states is not None:
            self.image_projector.load_state_dict(initial_states["image_projector"])
            self.residual_head.load_state_dict(initial_states["residual_head"])
            if self.pathway_encoder is not None:
                self.pathway_encoder.load_state_dict(initial_states["pathway_encoder"])
        self.baseline.eval()

    def train(self, mode: bool = True) -> "ResidualContrastiveModel":
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> ModelOutput:
        baseline_prediction = self.baseline(features)
        image_embedding = self.image_projector(features)
        prediction = baseline_prediction + self.residual_head(image_embedding)
        pathway_embedding = None
        if self.pathway_encoder is not None:
            if targets is None:
                raise ValueError(f"{self.arm} training forward requires pathway targets")
            pathway_embedding = self.pathway_encoder(targets)
        return ModelOutput(
            prediction=prediction,
            baseline_prediction=baseline_prediction,
            image_embedding=image_embedding,
            pathway_embedding=pathway_embedding,
        )

    def inference(self, features: torch.Tensor) -> torch.Tensor:
        """部署端只使用图像分支，不调用训练期通路编码器。"""

        baseline_prediction = self.baseline(features)
        image_embedding = self.image_projector(features)
        return baseline_prediction + self.residual_head(image_embedding)

    def optimizer_parameters(self) -> Iterator[nn.Parameter]:
        modules: list[nn.Module] = [self.image_projector, self.residual_head]
        if self.pathway_encoder is not None:
            modules.append(self.pathway_encoder)
        for module in modules:
            yield from module.parameters()


def load_accepted_baseline(
    checkpoint_path: str | Path,
    *,
    input_dim: int = 1536,
    hidden_dim: int = 1024,
    pathway_dim: int = 30,
    dropout: float = 0.3,
) -> nn.Module:
    """按现有 MPPMLPHead 键合同加载 accepted B0。"""

    from train_mpp_uni2h_mlp import MPPMLPHead

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model = MPPMLPHead(
        in_dim=input_dim,
        hidden=hidden_dim,
        out_dim=pathway_dim,
        dropout=dropout,
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
