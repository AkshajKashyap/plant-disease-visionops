"""Analyze and prepare real image datasets for the project's raw layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from plant_disease_visionops.data.discovery import SUPPORTED_IMAGE_EXTENSIONS

EXPECTED_LAYOUT = "<data-dir>/<class_name>/*.{jpg,jpeg,png}"
PreparationMode = Literal["copy", "symlink"]


class RawLayoutError(ValueError):
    """Base error for raw layout analysis and preparation."""


class InputDirectoryNotFoundError(RawLayoutError):
    """Raised when a requested input directory does not exist."""


class NoUsableClassFoldersError(RawLayoutError):
    """Raised when no direct class image folders can be used."""


class NoSourceImagesError(RawLayoutError):
    """Raised when preparation cannot discover supported source images."""


class OutputDirectoryNotEmptyError(RawLayoutError):
    """Raised before preparation would replace existing output data."""


@dataclass(frozen=True, slots=True)
class ClassFolderSummary:
    """One directly usable class folder and its candidate count."""

    class_name: str
    relative_path: str
    candidate_images: int


@dataclass(frozen=True, slots=True)
class RawLayoutAnalysis:
    """Analysis of files accepted or ignored by the current raw scanner."""

    data_dir: Path
    class_folders: tuple[ClassFolderSummary, ...]
    unsupported_files: tuple[str, ...]
    nested_image_folders: tuple[str, ...]
    root_image_files: tuple[str, ...]

    @property
    def total_candidate_images(self) -> int:
        return sum(summary.candidate_images for summary in self.class_folders)

    @property
    def is_usable(self) -> bool:
        return bool(self.class_folders)


@dataclass(frozen=True, slots=True)
class SourceImage:
    """One supported source image assigned to its immediate parent class."""

    path: Path
    relative_path: str
    class_name: str


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Source and destination metadata for one prepared image."""

    source: str
    destination: str
    class_name: str


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Completed raw layout preparation and its manifest data."""

    input_dir: Path
    output_dir: Path
    manifest_path: Path
    mode: PreparationMode
    files: tuple[PreparedImage, ...]


def _sorted_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )


def analyze_raw_layout(data_dir: Path | str) -> RawLayoutAnalysis:
    """Analyze direct candidates and files ignored by the existing scanner."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise InputDirectoryNotFoundError(f"Raw data directory does not exist: {root}")
    if not root.is_dir():
        raise InputDirectoryNotFoundError(f"Raw data path is not a directory: {root}")

    class_counts: Counter[str] = Counter()
    unsupported_files: list[str] = []
    nested_folders: set[str] = set()
    root_images: list[str] = []
    for path in _sorted_files(root):
        relative_path = path.relative_to(root)
        relative_text = relative_path.as_posix()
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            unsupported_files.append(relative_text)
        elif len(relative_path.parts) == 2:
            class_counts[relative_path.parts[0]] += 1
        elif len(relative_path.parts) == 1:
            root_images.append(relative_text)
        else:
            nested_folders.add(relative_path.parent.as_posix())

    class_folders = tuple(
        ClassFolderSummary(
            class_name=class_name,
            relative_path=class_name,
            candidate_images=class_counts[class_name],
        )
        for class_name in sorted(class_counts, key=lambda name: (name.casefold(), name))
    )
    return RawLayoutAnalysis(
        data_dir=root,
        class_folders=class_folders,
        unsupported_files=tuple(unsupported_files),
        nested_image_folders=tuple(
            sorted(nested_folders, key=lambda path: (path.casefold(), path))
        ),
        root_image_files=tuple(root_images),
    )


def require_usable_layout(analysis: RawLayoutAnalysis) -> None:
    """Raise a clear error unless the existing scanner can use the layout."""
    if not analysis.is_usable:
        raise NoUsableClassFoldersError(
            f"No usable class folders found under {analysis.data_dir}. "
            f"Expected layout: {EXPECTED_LAYOUT}"
        )


def discover_source_images(input_dir: Path | str) -> tuple[SourceImage, ...]:
    """Discover supported images recursively and use each parent folder as its class."""
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise InputDirectoryNotFoundError(f"Input dataset directory does not exist: {root}")
    if not root.is_dir():
        raise InputDirectoryNotFoundError(f"Input dataset path is not a directory: {root}")

    images = tuple(
        SourceImage(
            path=path,
            relative_path=path.relative_to(root).as_posix(),
            class_name=path.parent.name,
        )
        for path in _sorted_files(root)
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise NoSourceImagesError(
            f"No supported images found recursively under {root}. Expected extensions: {extensions}"
        )
    return images


def _validate_distinct_directories(input_dir: Path, output_dir: Path) -> None:
    if (
        input_dir == output_dir
        or output_dir.is_relative_to(input_dir)
        or input_dir.is_relative_to(output_dir)
    ):
        raise RawLayoutError(
            "Input and output directories must be separate and must not contain each other: "
            f"input={input_dir}, output={output_dir}"
        )


def _destination_name(source: SourceImage) -> str:
    source_hash = hashlib.sha256(source.relative_path.encode()).hexdigest()[:12]
    return f"{source.path.stem}__{source_hash}{source.path.suffix.lower()}"


def _prepare_in_staging_directory(
    source_images: tuple[SourceImage, ...],
    staging_dir: Path,
    mode: PreparationMode,
) -> tuple[PreparedImage, ...]:
    prepared: list[PreparedImage] = []
    destinations: set[str] = set()
    for source in source_images:
        relative_destination = Path(source.class_name) / _destination_name(source)
        destination_text = relative_destination.as_posix()
        if destination_text in destinations:
            raise RawLayoutError(
                f"Generated destination collision for source image: {source.relative_path}"
            )
        destinations.add(destination_text)

        destination = staging_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(source.path, destination)
        else:
            destination.symlink_to(source.path.resolve())
        prepared.append(
            PreparedImage(
                source=source.relative_path,
                destination=destination_text,
                class_name=source.class_name,
            )
        )
    return tuple(prepared)


def build_raw_layout_manifest(result: PreparationResult, overwrite: bool) -> dict[str, object]:
    """Build a serializable record of a completed preparation operation."""
    class_counts = Counter(prepared.class_name for prepared in result.files)
    return {
        "schema_version": 1,
        "input_directory": str(result.input_dir),
        "output_directory": str(result.output_dir),
        "mode": result.mode,
        "overwrite": overwrite,
        "supported_extensions": sorted(SUPPORTED_IMAGE_EXTENSIONS),
        "total_images": len(result.files),
        "class_counts": dict(
            sorted(class_counts.items(), key=lambda item: (item[0].casefold(), item[0]))
        ),
        "files": [
            {
                "source": prepared.source,
                "destination": prepared.destination,
                "class_name": prepared.class_name,
            }
            for prepared in result.files
        ],
    }


def prepare_raw_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    mode: PreparationMode = "copy",
    manifest_path: Path | str = Path("reports/raw_layout_manifest.json"),
    overwrite: bool = False,
) -> PreparationResult:
    """Copy or symlink a recursively organized dataset into the expected raw layout."""
    if mode not in {"copy", "symlink"}:
        raise RawLayoutError(f"mode must be 'copy' or 'symlink'; got {mode!r}")
    source_root = Path(input_dir).expanduser().resolve()
    destination_path = Path(output_dir).expanduser().absolute()
    if destination_path.is_symlink():
        raise OutputDirectoryNotEmptyError(
            f"Output directory cannot be a symlink: {destination_path}"
        )
    destination_root = destination_path.resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    source_images = discover_source_images(source_root)
    _validate_distinct_directories(source_root, destination_root)

    if destination_root.exists() and not destination_root.is_dir():
        raise OutputDirectoryNotEmptyError(
            f"Output path exists and is not a directory: {destination_root}"
        )
    output_has_data = destination_root.exists() and any(destination_root.iterdir())
    if output_has_data and not overwrite:
        raise OutputDirectoryNotEmptyError(
            f"Output directory is not empty: {destination_root}. "
            "Pass --overwrite to replace it explicitly."
        )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.prepare-",
            dir=destination_root.parent,
        )
    )
    try:
        prepared = _prepare_in_staging_directory(source_images, staging_path, mode)
        if destination_root.exists():
            shutil.rmtree(destination_root)
        staging_path.replace(destination_root)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    result = PreparationResult(
        input_dir=source_root,
        output_dir=destination_root,
        manifest_path=manifest,
        mode=mode,
        files=prepared,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(build_raw_layout_manifest(result, overwrite), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
