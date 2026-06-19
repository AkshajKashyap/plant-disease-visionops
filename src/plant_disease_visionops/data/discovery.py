"""Discover and validate class-organized image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class DatasetDiscoveryError(ValueError):
    """Base error for datasets that cannot be discovered."""


class DatasetNotFoundError(DatasetDiscoveryError):
    """Raised when the requested dataset directory does not exist."""


class EmptyDatasetError(DatasetDiscoveryError):
    """Raised when no supported images exist in class directories."""


@dataclass(frozen=True, slots=True)
class DiscoveredImage:
    """An image candidate found beneath one class directory."""

    path: Path
    class_name: str


@dataclass(frozen=True, slots=True)
class ValidImage:
    """Metadata extracted from an image that Pillow could decode."""

    path: Path
    class_name: str
    width: int
    height: int
    image_format: str | None


@dataclass(frozen=True, slots=True)
class InvalidImage:
    """An image candidate that Pillow could not decode."""

    path: Path
    class_name: str
    error: str


@dataclass(frozen=True, slots=True)
class DatasetScan:
    """Complete result of discovering and validating a dataset."""

    data_dir: Path
    discovered_images: tuple[DiscoveredImage, ...]
    valid_images: tuple[ValidImage, ...]
    invalid_images: tuple[InvalidImage, ...]


def discover_images(data_dir: Path | str) -> tuple[DiscoveredImage, ...]:
    """Return supported images stored directly under class directories.

    Args:
        data_dir: Root containing one directory per class.

    Raises:
        DatasetNotFoundError: If ``data_dir`` does not exist or is not a directory.
        EmptyDatasetError: If no JPG, JPEG, or PNG files are found under class directories.
    """
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise DatasetNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset path is not a directory: {root}")

    images = tuple(
        DiscoveredImage(path=image_path, class_name=class_dir.name)
        for class_dir in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for image_path in sorted(
            class_dir.iterdir(),
            key=lambda path: (path.name.casefold(), path.name),
        )
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise EmptyDatasetError(
            f"No supported images found in class directories under {root}. "
            f"Expected extensions: {extensions}"
        )
    return images


def inspect_image(image: DiscoveredImage) -> ValidImage | InvalidImage:
    """Validate one discovered image and return its metadata or decode error."""
    try:
        with Image.open(image.path) as opened_image:
            opened_image.verify()
        with Image.open(image.path) as opened_image:
            opened_image.load()
            width, height = opened_image.size
            image_format = opened_image.format
    except Exception as exc:  # Pillow decoders can raise several format-specific errors.
        message = str(exc).strip() or exc.__class__.__name__
        return InvalidImage(
            path=image.path,
            class_name=image.class_name,
            error=message,
        )

    return ValidImage(
        path=image.path,
        class_name=image.class_name,
        width=width,
        height=height,
        image_format=image_format,
    )


def scan_dataset(data_dir: Path | str) -> DatasetScan:
    """Discover all image candidates and validate each with Pillow."""
    root = Path(data_dir).expanduser()
    discovered = discover_images(root)
    inspected = tuple(inspect_image(image) for image in discovered)
    valid = tuple(image for image in inspected if isinstance(image, ValidImage))
    invalid = tuple(image for image in inspected if isinstance(image, InvalidImage))
    return DatasetScan(
        data_dir=root,
        discovered_images=discovered,
        valid_images=valid,
        invalid_images=invalid,
    )
