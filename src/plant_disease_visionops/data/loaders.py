"""Factories for split datasets and PyTorch DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from plant_disease_visionops.data.dataset import ImageClassificationDataset, ImageTransform
from plant_disease_visionops.data.transforms import build_eval_transform, build_train_transform

SplitName = Literal["train", "val", "test"]
VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True, slots=True)
class DatasetCollection:
    """Train, validation, and test datasets."""

    train: ImageClassificationDataset
    val: ImageClassificationDataset
    test: ImageClassificationDataset


@dataclass(frozen=True, slots=True)
class DataLoaderCollection:
    """Train, validation, and test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def create_split_dataset(
    split: SplitName,
    raw_data_dir: Path | str,
    processed_dir: Path | str,
    image_size: int = 224,
    transform: ImageTransform | None = None,
) -> ImageClassificationDataset:
    """Create one CSV-backed split dataset with its appropriate default transform."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; got {split!r}")
    processed_path = Path(processed_dir).expanduser()
    selected_transform = transform
    if selected_transform is None:
        selected_transform = (
            build_train_transform(image_size)
            if split == "train"
            else build_eval_transform(image_size)
        )
    return ImageClassificationDataset(
        csv_path=processed_path / f"{split}.csv",
        raw_data_dir=raw_data_dir,
        transform=selected_transform,
        class_mapping_path=processed_path / "class_to_index.json",
    )


def create_datasets(
    raw_data_dir: Path | str,
    processed_dir: Path | str,
    image_size: int = 224,
    train_transform: ImageTransform | None = None,
    eval_transform: ImageTransform | None = None,
) -> DatasetCollection:
    """Create all three datasets used by later training and evaluation code."""
    return DatasetCollection(
        train=create_split_dataset(
            "train", raw_data_dir, processed_dir, image_size, train_transform
        ),
        val=create_split_dataset("val", raw_data_dir, processed_dir, image_size, eval_transform),
        test=create_split_dataset("test", raw_data_dir, processed_dir, image_size, eval_transform),
    )


def create_dataloader(
    dataset: ImageClassificationDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Create one validated DataLoader."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than zero; got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative; got {num_workers}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_dataloaders(
    raw_data_dir: Path | str,
    processed_dir: Path | str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    test_shuffle: bool = False,
) -> DataLoaderCollection:
    """Create train/val/test DataLoaders with conventional shuffle defaults."""
    datasets = create_datasets(
        raw_data_dir=raw_data_dir,
        processed_dir=processed_dir,
        image_size=image_size,
    )
    return DataLoaderCollection(
        train=create_dataloader(datasets.train, batch_size, num_workers, train_shuffle),
        val=create_dataloader(datasets.val, batch_size, num_workers, val_shuffle),
        test=create_dataloader(datasets.test, batch_size, num_workers, test_shuffle),
    )
