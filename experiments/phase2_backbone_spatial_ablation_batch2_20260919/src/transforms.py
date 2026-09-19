"""Split full-FOV geometry from encoder-native RGB normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from PIL import Image

from errors import ConfigError


GEOMETRY_PROTOCOL = "full_fov_224_bicubic_v1"
FULL_FOV_PROTOCOL = GEOMETRY_PROTOCOL
IMAGENET_RGB_V1 = "imagenet_rgb_v1"
HOPTIMUS_RGB_V1 = "hoptimus_rgb_v1"
NORMALIZATION_PROFILES = {
    IMAGENET_RGB_V1: {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    HOPTIMUS_RGB_V1: {"mean": (0.707223, 0.578729, 0.703617), "std": (0.211883, 0.230117, 0.177517)},
}


def _bicubic() -> int:
    return Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def describe_geometry(protocol: str = GEOMETRY_PROTOCOL) -> dict:
    if protocol != GEOMETRY_PROTOCOL:
        raise ConfigError(f"本批只实现 {GEOMETRY_PROTOCOL}，实际={protocol!r}")
    return {
        "geometry_protocol": protocol,
        "steps": ["rgb", "require_square_input", "resize_exact_224x224_bicubic", "to_float_tensor_0_1"],
        "input_must_be_square": True,
        "resize_size": [224, 224],
        "crop_size": None,
        "interpolation": "bicubic",
        "antialias": True,
        "random_augmentation": False,
        "center_crop": False,
        "uses_auto_image_processor": False,
    }


def describe_normalization(profile: str) -> dict:
    if profile not in NORMALIZATION_PROFILES:
        raise ConfigError(f"未知图像归一化配置: {profile!r}")
    values = NORMALIZATION_PROFILES[profile]
    return {
        "normalization_profile": profile,
        "object": "image_input_before_encoder",
        "mean": list(values["mean"]),
        "std": list(values["std"]),
        "formula": "(pixel_channel - channel_mean) / channel_std",
        "pixel_range_before_normalize": [0.0, 1.0],
    }


def describe_transform(protocol: str = GEOMETRY_PROTOCOL, normalization_profile: str = IMAGENET_RGB_V1) -> dict:
    return {**describe_geometry(protocol), **describe_normalization(normalization_profile)}


@dataclass(frozen=True)
class FrozenTransform:
    protocol: str = GEOMETRY_PROTOCOL
    normalization_profile: str = IMAGENET_RGB_V1

    def __post_init__(self) -> None:
        describe_transform(self.protocol, self.normalization_profile)

    def prepare_image(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("预处理输入必须是 PIL.Image")
        rgb = image.convert("RGB")
        if rgb.width != rgb.height:
            raise ConfigError(f"{self.protocol} 只接受方形 patch，实际尺寸={rgb.size}；拒绝静默裁切或拉伸")
        return rgb.resize((224, 224), resample=_bicubic())

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(self.prepare_image(image), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        profile = describe_normalization(self.normalization_profile)
        mean = torch.tensor(profile["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(profile["std"], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


def build_transform(protocol: str = GEOMETRY_PROTOCOL, normalization_profile: str = IMAGENET_RGB_V1) -> FrozenTransform:
    return FrozenTransform(protocol, normalization_profile)


def build_shared_transform(protocol: str = GEOMETRY_PROTOCOL, normalization_profile: str = IMAGENET_RGB_V1) -> FrozenTransform:
    return build_transform(protocol, normalization_profile)
