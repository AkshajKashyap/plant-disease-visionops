"""Deterministic image corruptions for evaluation-time robustness checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from plant_disease_visionops.data.transforms import build_eval_transform

CorruptionName = Literal[
    "brightness_decrease",
    "brightness_increase",
    "gaussian_blur",
    "gaussian_noise",
    "contrast_decrease",
    "rotation",
    "zoom_in",
]
CORRUPTION_NAMES: tuple[CorruptionName, ...] = (
    "brightness_decrease",
    "brightness_increase",
    "gaussian_blur",
    "gaussian_noise",
    "contrast_decrease",
    "rotation",
    "zoom_in",
)
SEVERITY_LEVELS = (1, 2, 3)

_BRIGHTNESS_DECREASE = (0.8, 0.6, 0.4)
_BRIGHTNESS_INCREASE = (1.2, 1.5, 1.8)
_BLUR_SIGMA = (0.8, 1.5, 2.5)
_NOISE_STANDARD_DEVIATION = (0.03, 0.07, 0.12)
_CONTRAST_DECREASE = (0.8, 0.6, 0.4)
_ROTATION_DEGREES = (5.0, 15.0, 30.0)
_ZOOM_RETAINED_FRACTION = (0.9, 0.75, 0.6)


def _validate_corruption(name: str, severity: int) -> None:
    if name not in CORRUPTION_NAMES:
        raise ValueError(f"Unknown corruption {name!r}; expected one of {CORRUPTION_NAMES}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}; got {severity}")


def _noise_seed(image: Image.Image, seed: int, severity: int) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode())
    digest.update(str(severity).encode())
    digest.update(image.mode.encode())
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())
    return int.from_bytes(digest.digest()[:8], byteorder="big")


def _gaussian_noise(image: Image.Image, severity: int, seed: int) -> Image.Image:
    tensor = transform_functional.pil_to_tensor(image).to(dtype=torch.float32) / 255.0
    generator = torch.Generator().manual_seed(_noise_seed(image, seed, severity))
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    corrupted = (tensor + noise * _NOISE_STANDARD_DEVIATION[severity - 1]).clamp(0.0, 1.0)
    return transform_functional.to_pil_image(corrupted)


def _zoom_in(image: Image.Image, severity: int) -> Image.Image:
    width, height = image.size
    retained_fraction = _ZOOM_RETAINED_FRACTION[severity - 1]
    crop_width = max(1, round(width * retained_fraction))
    crop_height = max(1, round(height * retained_fraction))
    cropped = transform_functional.center_crop(image, [crop_height, crop_width])
    return transform_functional.resize(
        cropped,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


@dataclass(frozen=True, slots=True)
class CorruptionTransform:
    """Callable PIL transform for one named corruption and severity."""

    name: CorruptionName
    severity: int
    seed: int = 42

    def __post_init__(self) -> None:
        _validate_corruption(self.name, self.severity)

    def __call__(self, image: Image.Image) -> Image.Image:
        level = self.severity - 1
        if self.name == "brightness_decrease":
            return transform_functional.adjust_brightness(image, _BRIGHTNESS_DECREASE[level])
        if self.name == "brightness_increase":
            return transform_functional.adjust_brightness(image, _BRIGHTNESS_INCREASE[level])
        if self.name == "gaussian_blur":
            kernel_size = 3 + 2 * level
            return transform_functional.gaussian_blur(
                image,
                kernel_size=[kernel_size, kernel_size],
                sigma=[_BLUR_SIGMA[level], _BLUR_SIGMA[level]],
            )
        if self.name == "gaussian_noise":
            return _gaussian_noise(image, self.severity, self.seed)
        if self.name == "contrast_decrease":
            return transform_functional.adjust_contrast(image, _CONTRAST_DECREASE[level])
        if self.name == "rotation":
            return transform_functional.rotate(
                image,
                angle=_ROTATION_DEGREES[level],
                interpolation=InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            )
        if self.name == "zoom_in":
            return _zoom_in(image, self.severity)
        raise AssertionError(f"Unhandled corruption: {self.name}")


def create_corruption(
    name: CorruptionName | str,
    severity: int,
    seed: int = 42,
) -> CorruptionTransform:
    """Create a validated corruption transform from registry values."""
    _validate_corruption(name, severity)
    return CorruptionTransform(
        name=cast(CorruptionName, name),
        severity=severity,
        seed=seed,
    )


def build_corrupted_eval_transform(
    image_size: int,
    corruption_name: CorruptionName | str,
    severity: int,
    seed: int = 42,
) -> transforms.Compose:
    """Apply one corruption before the existing deterministic evaluation transform."""
    corruption = create_corruption(corruption_name, severity, seed)
    evaluation = build_eval_transform(image_size)
    return transforms.Compose([corruption, *evaluation.transforms])
