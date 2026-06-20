"""CLI for clean and corrupted evaluation of a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.data.loaders import VALID_SPLITS
from plant_disease_visionops.evaluation.corruptions import CORRUPTION_NAMES, SEVERITY_LEVELS
from plant_disease_visionops.evaluation.model_loading import ModelLoadingError
from plant_disease_visionops.evaluation.robustness import (
    RobustnessConfig,
    run_robustness_evaluation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on clean and corrupted image splits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTION_NAMES, default=None)
    parser.add_argument("--severities", nargs="+", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run robustness evaluation from command-line arguments."""
    args = _build_parser().parse_args(argv)
    config = RobustnessConfig(
        checkpoint=args.checkpoint,
        experiment_name=args.experiment_name,
        raw_data_dir=args.raw_data_dir,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        corruptions=tuple(args.corruptions or CORRUPTION_NAMES),
        severities=tuple(args.severities or SEVERITY_LEVELS),
    )
    try:
        results = run_robustness_evaluation(config)
    except (DatasetLoadingError, ModelLoadingError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Robustness report written for {results['experiment_name']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
