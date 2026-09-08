from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Predictor:
    model_id: str
    model: nn.Module
    in_dim: int
    out_dim: int

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)
