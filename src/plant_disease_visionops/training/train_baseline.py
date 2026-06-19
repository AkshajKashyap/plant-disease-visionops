"""Backward-compatible CLI wrapper for the baseline CNN experiment."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.training.experiment import ExperimentConfig, run_experiment


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    """Original Milestone 4 configuration retained for Python callers."""

    raw_data_dir: Path
    processed_dir: Path
    out_dir: Path
    reports_dir: Path
    figures_dir: Path
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 0.001
    num_workers: int = 2
    seed: int = 42
    dropout: float = 0.3
    device: str = "auto"

    def as_experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name="baseline_cnn",
            model_name="baseline_cnn",
            pretrained=False,
            freeze_backbone=False,
            raw_data_dir=self.raw_data_dir,
            processed_dir=self.processed_dir,
            out_dir=self.out_dir,
            reports_dir=self.reports_dir,
            figures_dir=self.figures_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            num_workers=self.num_workers,
            seed=self.seed,
            dropout=self.dropout,
            device=self.device,
        )

    def validate(self) -> None:
        self.as_experiment_config().validate()


def run_baseline_training(config: BaselineTrainingConfig) -> dict[str, Any]:
    """Run the baseline through the generalized experiment engine."""
    return run_experiment(config.as_experiment_config())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline CNN.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/models/baseline_cnn"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the original baseline command through the shared engine."""
    args = _build_parser().parse_args(argv)
    config = BaselineTrainingConfig(
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
        run_baseline_training(config)
    except (DatasetLoadingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
