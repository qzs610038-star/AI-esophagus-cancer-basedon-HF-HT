from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from transforms import build_shared_transform, describe_transform


def test_shared_transform_is_deterministic_rgb_resize_crop_and_normalize():
    transform = build_shared_transform()
    # Non-square RGBA image exercises RGB conversion and short-side resizing.
    image = Image.fromarray(np.full((100, 200, 4), [255, 0, 0, 127], dtype=np.uint8), "RGBA")
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    expected_red = torch.tensor(
        [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (0.0 - 0.406) / 0.225]
    )
    torch.testing.assert_close(first[:, 0, 0], expected_red)


def test_transform_description_is_explicit_and_has_no_random_augmentation():
    description = describe_transform()
    assert description["resize_shorter_side"] == 256
    assert description["crop_size"] == 224
    assert description["antialias"] is True
    assert description["random_augmentation"] is False
    assert description["steps"][0] == "rgb"
