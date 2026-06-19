"""Generalized CLI for baseline and transfer-learning experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.models.factory import SUPPORTED_MODELS
from plant_disease_visionops.training.experiment import ExperimentConfig, run_experiment


def parse_boolean(value: str) -> bool:
    """Parse explicit true/false CLI values without Python truthiness surprises."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false; got {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a named image-classification experiment."
    )
    parser.add_argument(
        "--model-name",
        choices=SUPPORTED_MODELS,
        default="resnet18",
        help="Model architecture (default: resnet18).",
    )
    parser.add_argument(
        "--experiment-name",
        help="Output filename prefix; defaults to the --out-dir folder name.",
    )
    parser.add_argument(
        "--pretrained",
        type=parse_boolean,
        default=True,
        metavar="{true,false}",
        help="Load torchvision pretrained weights (default: true).",
    )
    parser.add_argument(
        "--freeze-backbone",
        type=parse_boolean,
        default=False,
        metavar="{true,false}",
        help="Train only the replacement classifier (default: false).",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/models/resnet18_transfer"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured baseline or ResNet18 experiment."""
    args = _build_parser().parse_args(argv)
    experiment_name = args.experiment_name or args.out_dir.name
    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        raw_data_dir=args.raw_data_dir,
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        dropout=args.dropout,
        device=args.device,
    )
    try:
        run_experiment(config)
    except (DatasetLoadingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
