"""Image transforms shared by training and evaluation data pipelines."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_image_size(image_size: int) -> None:
    if image_size <= 0:
        raise ValueError(f"image_size must be greater than zero; got {image_size}")


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build augmentation and ImageNet normalization for training images."""
    _validate_image_size(image_size)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Build deterministic resizing, center cropping, and normalization."""
    _validate_image_size(image_size)
    resize_size = round(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
