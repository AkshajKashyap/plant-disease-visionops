"""Command-line prediction for one local image."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.evaluation.model_loading import ModelLoadingError
from plant_disease_visionops.inference.predictor import (
    ImagePreprocessingError,
    load_model_for_inference,
    predict_image,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one leaf image with a saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a checkpoint and print one honest prediction as JSON."""
    args = _build_parser().parse_args(argv)
    try:
        inference_model = load_model_for_inference(
            checkpoint_path=args.checkpoint,
            processed_dir=args.processed_dir,
            device=args.device,
        )
        result = predict_image(
            inference_model,
            args.image_path,
            image_size=args.image_size,
            top_k=args.top_k,
        )
    except (
        DatasetLoadingError,
        ImagePreprocessingError,
        ModelLoadingError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
