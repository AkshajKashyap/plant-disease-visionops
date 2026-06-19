"""Create deterministic, stratified metadata splits from validated images."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from plant_disease_visionops.data.discovery import DatasetScan, ValidImage

SPLIT_NAMES = ("train", "val", "test")


class SplitGenerationError(ValueError):
    """Base error for split configurations that cannot be generated."""


class InvalidSplitRatiosError(SplitGenerationError):
    """Raised when train, validation, and test ratios are invalid."""


class InsufficientClassImagesError(SplitGenerationError):
    """Raised when a class cannot contribute to every requested split."""


class DataLeakageError(SplitGenerationError):
    """Raised if an image filepath occurs in more than one generated split."""


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Requested proportions for train, validation, and test metadata."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        """Validate that every split is requested and ratios total one."""
        values = self.as_dict()
        for name, value in values.items():
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise InvalidSplitRatiosError(
                    f"{name} ratio must be greater than 0 and less than 1; got {value}"
                )

        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise InvalidSplitRatiosError(f"Split ratios must sum to 1.0; got {total:.12g}")

    def as_dict(self) -> dict[str, float]:
        """Return ratios keyed by their output split names."""
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Portable image metadata consumed by later training pipelines."""

    filepath: str
    label: str
    class_index: int
    width: int
    height: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a row suitable for CSV serialization."""
        return {
            "filepath": self.filepath,
            "label": self.label,
            "class_index": self.class_index,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Generated split metadata and the class mapping used to create it."""

    data_dir: Path
    ratios: SplitRatios
    seed: int
    class_to_index: dict[str, int]
    records: dict[str, tuple[SplitRecord, ...]]
    total_discovered_images: int
    invalid_image_paths: tuple[str, ...]


def allocate_split_counts(image_count: int, ratios: SplitRatios) -> dict[str, int]:
    """Allocate one class across splits while staying near requested ratios."""
    ratios.validate()
    minimum_images = len(SPLIT_NAMES)
    if image_count < minimum_images:
        raise InsufficientClassImagesError(
            f"At least {minimum_images} valid images are required per class for "
            f"train/val/test splits; got {image_count}"
        )

    raw_ratio_values = ratios.as_dict()
    ratio_total = sum(raw_ratio_values.values())
    ratio_values = {name: raw_ratio_values[name] / ratio_total for name in SPLIT_NAMES}
    ideal_counts = {name: image_count * ratio_values[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(ideal_counts[name]) for name in SPLIT_NAMES}
    remaining = image_count - sum(counts.values())
    by_remainder = sorted(
        SPLIT_NAMES,
        key=lambda name: (
            -(ideal_counts[name] - counts[name]),
            SPLIT_NAMES.index(name),
        ),
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1

    for empty_split in (name for name in SPLIT_NAMES if counts[name] == 0):
        donors = [name for name in SPLIT_NAMES if counts[name] > 1]
        donor = max(
            donors,
            key=lambda name: (
                counts[name] - ideal_counts[name],
                counts[name],
                -SPLIT_NAMES.index(name),
            ),
        )
        counts[donor] -= 1
        counts[empty_split] = 1

    return counts


def _class_seed(seed: int, class_name: str) -> int:
    encoded = f"{seed}\0{class_name}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], byteorder="big")


def _portable_path(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def _record(image: ValidImage, data_dir: Path, class_index: int) -> SplitRecord:
    return SplitRecord(
        filepath=_portable_path(image.path, data_dir),
        label=image.class_name,
        class_index=class_index,
        width=image.width,
        height=image.height,
    )


def _ensure_no_overlap(records: dict[str, tuple[SplitRecord, ...]]) -> None:
    seen: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for record in records[split_name]:
            previous_split = seen.get(record.filepath)
            if previous_split is not None:
                raise DataLeakageError(
                    f"Image filepath occurs in both {previous_split} and {split_name}: "
                    f"{record.filepath}"
                )
            seen[record.filepath] = split_name


def create_stratified_splits(
    scan: DatasetScan,
    ratios: SplitRatios,
    seed: int = 42,
) -> DatasetSplits:
    """Create deterministic train/val/test records from a validated dataset scan."""
    ratios.validate()
    valid_by_class: defaultdict[str, list[ValidImage]] = defaultdict(list)
    for image in scan.valid_images:
        valid_by_class[image.class_name].append(image)

    class_names = sorted(
        {image.class_name for image in scan.discovered_images},
        key=lambda name: (name.casefold(), name),
    )
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    split_records: dict[str, list[SplitRecord]] = {name: [] for name in SPLIT_NAMES}

    for class_name in class_names:
        class_images = valid_by_class[class_name]
        try:
            counts = allocate_split_counts(len(class_images), ratios)
        except InsufficientClassImagesError as exc:
            raise InsufficientClassImagesError(
                f"Class '{class_name}' has {len(class_images)} valid images. {exc}"
            ) from exc

        shuffled_images = class_images.copy()
        random.Random(_class_seed(seed, class_name)).shuffle(shuffled_images)
        start = 0
        for split_name in SPLIT_NAMES:
            end = start + counts[split_name]
            split_records[split_name].extend(
                _record(image, scan.data_dir, class_to_index[class_name])
                for image in shuffled_images[start:end]
            )
            start = end

    immutable_records = {name: tuple(split_records[name]) for name in SPLIT_NAMES}
    _ensure_no_overlap(immutable_records)
    invalid_paths = tuple(
        _portable_path(image.path, scan.data_dir) for image in scan.invalid_images
    )
    return DatasetSplits(
        data_dir=scan.data_dir,
        ratios=ratios,
        seed=seed,
        class_to_index=class_to_index,
        records=immutable_records,
        total_discovered_images=len(scan.discovered_images),
        invalid_image_paths=invalid_paths,
    )
