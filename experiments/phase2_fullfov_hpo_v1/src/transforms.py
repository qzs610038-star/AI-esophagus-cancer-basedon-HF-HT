"""冻结的图像预处理协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from PIL import Image

from errors import ConfigError


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LEGACY_CROP_PROTOCOL = "legacy_crop_0875_v1"
FULL_FOV_PROTOCOL = "full_fov_224_bicubic_v1"
SUPPORTED_PROTOCOLS = (LEGACY_CROP_PROTOCOL, FULL_FOV_PROTOCOL)


def _bicubic() -> int:
    return Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def describe_transform(protocol: str = FULL_FOV_PROTOCOL) -> dict:
    """Return the complete, JSON-safe contract used in cache metadata."""
    if protocol == LEGACY_CROP_PROTOCOL:
        steps = ["rgb", "resize_shorter_side_256_bicubic", "center_crop_224", "to_float_tensor_0_1", "imagenet_normalize"]
        geometry = {"resize_shorter_side": 256, "crop_size": 224, "nominal_square_fov_fraction": 0.875}
    elif protocol == FULL_FOV_PROTOCOL:
        steps = ["rgb", "require_square_input", "resize_exact_224x224_bicubic", "to_float_tensor_0_1", "imagenet_normalize"]
        geometry = {"input_must_be_square": True, "resize_size": [224, 224], "crop_size": None}
    else:
        raise ConfigError(f"未知 preprocessing protocol: {protocol!r}")
    return {"protocol": protocol, "steps": steps, **geometry, "interpolation": "bicubic", "antialias": True, "mean": list(MEAN), "std": list(STD), "random_augmentation": False}


def validate_protocols(configured: Mapping[str, object] | None) -> tuple[str, ...]:
    if configured is None:
        return SUPPORTED_PROTOCOLS
    names = tuple(str(key) for key in configured)
    if set(names) != set(SUPPORTED_PROTOCOLS):
        raise ConfigError(f"preprocessing.protocols 必须包含且只包含 {list(SUPPORTED_PROTOCOLS)}")
    return names


@dataclass(frozen=True)
class FrozenTransform:
    protocol: str

    def __post_init__(self) -> None:
        describe_transform(self.protocol)

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Apply geometry only; exposed for field-of-view verification tests."""
        if not isinstance(image, Image.Image):
            raise TypeError("预处理输入必须是 PIL.Image")
        rgb = image.convert("RGB")
        if self.protocol == FULL_FOV_PROTOCOL:
            if rgb.width != rgb.height:
                raise ConfigError(f"{FULL_FOV_PROTOCOL} 只接受方形 patch，实际尺寸={rgb.size}；拒绝静默裁切或拉伸")
            return rgb.resize((224, 224), resample=_bicubic())
        scale = 256.0 / min(rgb.width, rgb.height)
        resized = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), resample=_bicubic())
        left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
        if left < 0 or top < 0:
            raise ConfigError("历史裁剪路径得到小于 224 的图像")
        return resized.crop((left, top, left + 224, top + 224))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(self.prepare_image(image), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


def build_transform(protocol: str = FULL_FOV_PROTOCOL) -> FrozenTransform:
    return FrozenTransform(protocol)


def build_shared_transform(protocol: str = FULL_FOV_PROTOCOL) -> FrozenTransform:
    """Migration alias for copied package callers."""
    return build_transform(protocol)
