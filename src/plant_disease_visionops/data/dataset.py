"""PyTorch dataset backed by Milestone 2 split metadata."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional

REQUIRED_CSV_COLUMNS = frozenset({"filepath", "label", "class_index", "width", "height"})
CLASS_MAPPING_FILENAME = "class_to_index.json"


class DatasetLoadingError(Exception):
    """Base error for invalid or unavailable dataset inputs."""


class SplitCsvNotFoundError(FileNotFoundError, DatasetLoadingError):
    """Raised when a requested split CSV does not exist."""


class ClassMappingNotFoundError(FileNotFoundError, DatasetLoadingError):
    """Raised when class_to_index.json does not exist."""


class MissingImageFileError(FileNotFoundError, DatasetLoadingError):
    """Raised when a CSV references an image that no longer exists."""


class InvalidSplitMetadataError(ValueError, DatasetLoadingError):
    """Raised when a split CSV has malformed or inconsistent metadata."""


class InvalidClassMappingError(ValueError, DatasetLoadingError):
    """Raised when class_to_index.json is malformed."""


class InvalidLabelError(ValueError, DatasetLoadingError):
    """Raised when a row label or class index disagrees with the class mapping."""


class EmptySplitError(ValueError, DatasetLoadingError):
    """Raised when a split CSV has no image rows."""


class ImageLoadError(RuntimeError, DatasetLoadingError):
    """Raised when an image cannot be decoded at access time."""


class SampleMetadata(TypedDict):
    """Metadata returned alongside each image tensor and target."""

    filepath: str
    label: str
    class_index: int
    width: int
    height: int


ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class ImageSample:
    """Validated metadata for one split row."""

    filepath: str
    image_path: Path
    label: str
    class_index: int
    width: int
    height: int


def load_class_mapping(mapping_path: Path | str) -> dict[str, int]:
    """Load and validate a label-to-index JSON mapping."""
    path = Path(mapping_path).expanduser()
    if not path.is_file():
        raise ClassMappingNotFoundError(f"Class mapping file does not exist: {path}")
    try:
        raw_mapping = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidClassMappingError(f"Could not read class mapping {path}: {exc}") from exc

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise InvalidClassMappingError(f"Class mapping must be a non-empty JSON object: {path}")

    mapping: dict[str, int] = {}
    for label, class_index in raw_mapping.items():
        if not isinstance(label, str) or not label.strip():
            raise InvalidClassMappingError(f"Class mapping contains an invalid label: {label!r}")
        if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
            raise InvalidClassMappingError(
                f"Class index for '{label}' must be a non-negative integer; got {class_index!r}"
            )
        mapping[label] = class_index

    if len(set(mapping.values())) != len(mapping):
        raise InvalidClassMappingError("Class mapping indices must be unique")
    return mapping


def _parse_positive_integer(value: str | None, field: str, row_number: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError as exc:
        raise InvalidSplitMetadataError(f"Row {row_number} has invalid {field}: {value!r}") from exc
    if parsed <= 0:
        raise InvalidSplitMetadataError(f"Row {row_number} has non-positive {field}: {parsed}")
    return parsed


def _resolve_image_path(raw_data_dir: Path, filepath: str, row_number: int) -> Path:
    relative_path = Path(filepath)
    if relative_path.is_absolute():
        raise InvalidSplitMetadataError(
            f"Row {row_number} filepath must be relative to the raw data directory: {filepath}"
        )
    image_path = (raw_data_dir / relative_path).resolve()
    if not image_path.is_relative_to(raw_data_dir):
        raise InvalidSplitMetadataError(
            f"Row {row_number} filepath escapes the raw data directory: {filepath}"
        )
    return image_path


def _read_samples(
    csv_path: Path,
    raw_data_dir: Path,
    class_to_index: dict[str, int],
) -> tuple[ImageSample, ...]:
    try:
        csv_file = csv_path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise InvalidSplitMetadataError(f"Could not read split CSV {csv_path}: {exc}") from exc

    with csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_CSV_COLUMNS - columns)
        if missing_columns:
            raise InvalidSplitMetadataError(
                f"Split CSV {csv_path} is missing required columns: {', '.join(missing_columns)}"
            )

        samples: list[ImageSample] = []
        seen_filepaths: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            filepath = (row.get("filepath") or "").strip()
            label = (row.get("label") or "").strip()
            if not filepath:
                raise InvalidSplitMetadataError(f"Row {row_number} has an empty filepath")
            if filepath in seen_filepaths:
                raise InvalidSplitMetadataError(
                    f"Split CSV contains duplicate filepath at row {row_number}: {filepath}"
                )
            seen_filepaths.add(filepath)

            if label not in class_to_index:
                raise InvalidLabelError(
                    f"Row {row_number} label '{label}' is not present in the class mapping"
                )
            try:
                class_index = int(row.get("class_index") or "")
            except ValueError as exc:
                raise InvalidLabelError(
                    f"Row {row_number} has invalid class_index: {row.get('class_index')!r}"
                ) from exc
            expected_index = class_to_index[label]
            if class_index != expected_index:
                raise InvalidLabelError(
                    f"Row {row_number} label '{label}' expects class_index {expected_index}, "
                    f"got {class_index}"
                )

            samples.append(
                ImageSample(
                    filepath=filepath,
                    image_path=_resolve_image_path(raw_data_dir, filepath, row_number),
                    label=label,
                    class_index=class_index,
                    width=_parse_positive_integer(row.get("width"), "width", row_number),
                    height=_parse_positive_integer(row.get("height"), "height", row_number),
                )
            )

    if not samples:
        raise EmptySplitError(f"Split CSV contains no image rows: {csv_path}")
    return tuple(samples)


class ImageClassificationDataset(Dataset[tuple[Tensor, int, SampleMetadata]]):
    """Load RGB images described by one Milestone 2 split CSV."""

    def __init__(
        self,
        csv_path: Path | str,
        raw_data_dir: Path | str,
        transform: ImageTransform | None = None,
        class_mapping_path: Path | str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser()
        self.raw_data_dir = Path(raw_data_dir).expanduser().resolve()
        if not self.csv_path.is_file():
            raise SplitCsvNotFoundError(f"Split CSV does not exist: {self.csv_path}")
        if not self.raw_data_dir.is_dir():
            raise DatasetLoadingError(
                f"Raw data directory does not exist or is not a directory: {self.raw_data_dir}"
            )

        mapping_path = (
            Path(class_mapping_path).expanduser()
            if class_mapping_path is not None
            else self.csv_path.parent / CLASS_MAPPING_FILENAME
        )
        self.class_to_index = load_class_mapping(mapping_path)
        self.samples = _read_samples(self.csv_path, self.raw_data_dir, self.class_to_index)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, SampleMetadata]:
        sample = self.samples[index]
        if not sample.image_path.is_file():
            raise MissingImageFileError(
                f"Image file referenced by {self.csv_path} does not exist: {sample.image_path}"
            )

        try:
            with Image.open(sample.image_path) as image:
                rgb_image = image.convert("RGB")
                image_tensor = (
                    self.transform(rgb_image)
                    if self.transform is not None
                    else transform_functional.to_tensor(rgb_image)
                )
        except OSError as exc:
            raise ImageLoadError(f"Could not load image {sample.image_path}: {exc}") from exc

        if not isinstance(image_tensor, torch.Tensor):
            raise ImageLoadError(
                f"Transform must return a torch.Tensor for {sample.image_path}; "
                f"got {type(image_tensor).__name__}"
            )

        metadata: SampleMetadata = {
            "filepath": sample.filepath,
            "label": sample.label,
            "class_index": sample.class_index,
            "width": sample.width,
            "height": sample.height,
        }
        return image_tensor, sample.class_index, metadata
