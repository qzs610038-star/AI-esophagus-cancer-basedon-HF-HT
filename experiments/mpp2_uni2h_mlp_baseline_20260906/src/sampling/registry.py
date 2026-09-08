from __future__ import annotations

from .dense import DenseSampler
from .protocol import Sampler
from .spatial_stride import SpatialStrideSampler


def load_sampler(sampling_id: str, parameters: dict) -> Sampler:
    if sampling_id == "dense":
        return DenseSampler()
    if sampling_id == "spatial_stride":
        return SpatialStrideSampler(
            stride_x=int(parameters.get("stride_x", 2)),
            stride_y=int(parameters.get("stride_y", 2)),
            phase_x=int(parameters.get("phase_x", 0)),
            phase_y=int(parameters.get("phase_y", 0)),
        )
    raise ValueError(
        f"unknown sampling_id={sampling_id!r}. Implement src/sampling/<id>.py and register it here."
    )
