"""One deterministic preprocessing chain shared by every new encoder."""

from __future__ import annotations

from PIL import Image


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def describe_transform() -> dict:
    return {
        "steps": [
            "rgb",
            "resize_shorter_side_256_bicubic_antialias",
            "center_crop_224",
            "to_float_tensor_0_1",
            "imagenet_normalize",
        ],
        "resize_shorter_side": 256,
        "crop_size": 224,
        "interpolation": "bicubic",
        "antialias": True,
        "mean": list(MEAN),
        "std": list(STD),
        "random_augmentation": False,
    }


class SharedTransform:
    def __init__(self) -> None:
        try:
            from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
            from torchvision.transforms import InterpolationMode
        except ImportError as exc:  # pragma: no cover - checked by environment command
            raise RuntimeError("缺少 torchvision，无法执行冻结预处理") from exc
        self._transform = Compose(
            [
                Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=MEAN, std=STD),
            ]
        )

    def __call__(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError("预处理输入必须是 PIL.Image")
        return self._transform(image.convert("RGB"))


def build_shared_transform() -> SharedTransform:
    return SharedTransform()
