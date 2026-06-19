from pathlib import Path

import pytest
from PIL import Image

from plant_disease_visionops.data.discovery import (
    DatasetNotFoundError,
    EmptyDatasetError,
    discover_images,
    scan_dataset,
)


def _create_image(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 120, 50)).save(path)


def test_discover_images_uses_class_directories_and_supported_extensions(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_image(data_dir / "healthy" / "one.JPG")
    _create_image(data_dir / "rust" / "two.png")
    _create_image(data_dir / "rust" / "nested" / "ignored.jpeg")
    (data_dir / "healthy" / "notes.txt").write_text("not an image", encoding="utf-8")

    images = discover_images(data_dir)

    assert [(image.class_name, image.path.name) for image in images] == [
        ("healthy", "one.JPG"),
        ("rust", "two.png"),
    ]


def test_scan_dataset_detects_corrupt_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_image(data_dir / "healthy" / "valid.png", size=(20, 10))
    corrupt_path = data_dir / "rust" / "corrupt.jpg"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"this is not a jpeg")

    scan = scan_dataset(data_dir)

    assert len(scan.discovered_images) == 2
    assert [(image.width, image.height) for image in scan.valid_images] == [(20, 10)]
    assert [image.path for image in scan.invalid_images] == [corrupt_path]
    assert scan.invalid_images[0].error


def test_missing_dataset_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(DatasetNotFoundError, match="does not exist"):
        discover_images(missing)


def test_empty_dataset_has_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "healthy").mkdir(parents=True)

    with pytest.raises(EmptyDatasetError, match="No supported images"):
        discover_images(data_dir)
