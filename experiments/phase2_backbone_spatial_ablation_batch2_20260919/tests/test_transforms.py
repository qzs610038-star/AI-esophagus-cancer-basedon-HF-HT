from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from errors import ConfigError
from transforms import (
    GEOMETRY_PROTOCOL,
    HOPTIMUS_RGB_V1,
    IMAGENET_RGB_V1,
    FrozenTransform,
    build_transform,
    describe_transform,
)


def _edge_image(size: int = 256) -> Image.Image:
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[:9, :, 0] = 255
    pixels[-9:, :, 1] = 255
    pixels[:, :9, 2] = 255
    pixels[:, -9:, :2] = 255
    return Image.fromarray(pixels, "RGB")


def test_full_fov_keeps_all_four_edges():
    image = _edge_image(256)
    full = np.asarray(build_transform(GEOMETRY_PROTOCOL, IMAGENET_RGB_V1).prepare_image(image))
    probes = ((3, 112), (220, 112), (112, 3), (112, 220))
    assert all(full[y, x].max() > 180 for y, x in probes)


def test_non_square_input_fails():
    with pytest.raises(ConfigError, match="方形"):
        build_transform()(Image.new("RGB", (224, 223)))


def test_hoptimus_and_imagenet_normalization_are_distinct():
    image = Image.new("RGB", (224, 224), (80, 40, 120))
    imagenet = build_transform(GEOMETRY_PROTOCOL, IMAGENET_RGB_V1)(image)
    hopt = build_transform(GEOMETRY_PROTOCOL, HOPTIMUS_RGB_V1)(image)
    assert imagenet.shape == hopt.shape == (3, 224, 224)
    assert not torch.allclose(imagenet, hopt)
    meta = describe_transform(GEOMETRY_PROTOCOL, HOPTIMUS_RGB_V1)
    assert meta["geometry_protocol"] == GEOMETRY_PROTOCOL
    assert meta["normalization_profile"] == HOPTIMUS_RGB_V1
    assert meta["object"] == "image_input_before_encoder"
    assert meta["uses_auto_image_processor"] is False


def test_transform_module_does_not_use_auto_image_processor():
    source = Path(__file__).resolve().parents[1] / "src" / "transforms.py"
    text = source.read_text(encoding="utf-8")
    assert "AutoImageProcessor" not in text
    assert "rescale" not in text.lower() or "to_float_tensor_0_1" in describe_transform()["steps"]


def test_geometry_and_normalization_are_separate_fields():
    transform = FrozenTransform(GEOMETRY_PROTOCOL, HOPTIMUS_RGB_V1)
    assert transform.protocol == GEOMETRY_PROTOCOL
    assert transform.normalization_profile == HOPTIMUS_RGB_V1
