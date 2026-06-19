"""Inspect one transformed image batch and save a visual grid."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from torch import Tensor
from torchvision.utils import make_grid, save_image

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.data.loaders import (
    VALID_SPLITS,
    SplitName,
    create_dataloader,
    create_split_dataset,
)
from plant_disease_visionops.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize_batch(images: Tensor) -> Tensor:
    """Undo ImageNet normalization and clamp images for visualization."""
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def inspect_batch(
    raw_data_dir: Path | str,
    processed_dir: Path | str,
    split: SplitName,
    batch_size: int,
    image_size: int,
    out_dir: Path | str,
    num_workers: int = 0,
) -> Path:
    """Load one batch, print its summary, and save a denormalized image grid."""
    dataset = create_split_dataset(
        split=split,
        raw_data_dir=raw_data_dir,
        processed_dir=processed_dir,
        image_size=image_size,
    )
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == "train",
    )
    images, class_indices, metadata = next(iter(loader))
    labels = [str(label) for label in metadata["label"]]
    class_counts = dict(sorted(Counter(labels).items()))

    print(f"Batch tensor shape: {tuple(images.shape)}")
    print(f"Batch class indices: {class_indices.tolist()}")
    print(f"Batch class counts: {json.dumps(class_counts, sort_keys=True)}")

    output_directory = Path(out_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"sample_batch_{split}.png"
    grid = make_grid(
        denormalize_batch(images),
        nrow=min(4, len(images)),
        padding=2,
    )
    save_image(grid, output_path)
    print(f"Saved batch grid to {output_path}")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load one transformed image batch and save a visual inspection grid."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing class image folders (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing split CSVs and class mapping (default: data/processed).",
    )
    parser.add_argument(
        "--split",
        choices=sorted(VALID_SPLITS),
        default="train",
        help="Split to inspect (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of images in the inspected batch (default: 8).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square transformed image size in pixels (default: 224).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/figures"),
        help="Directory for the saved sample grid (default: artifacts/figures).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch inspection command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        inspect_batch(
            raw_data_dir=args.raw_data_dir,
            processed_dir=args.processed_dir,
            split=args.split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
        )
    except (DatasetLoadingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
