"""Train and evaluate the compact baseline CNN."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from plant_disease_visionops.data.dataset import DatasetLoadingError, load_class_mapping
from plant_disease_visionops.data.loaders import create_dataloaders
from plant_disease_visionops.models.baseline_cnn import create_baseline_cnn, summarize_model
from plant_disease_visionops.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_json_history,
)
from plant_disease_visionops.training.engine import evaluate, train_one_epoch
from plant_disease_visionops.training.reporting import (
    save_confusion_matrix_figure,
    save_training_curves,
    write_baseline_results,
)
from plant_disease_visionops.training.reproducibility import get_device, set_seed


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    """Filesystem and hyperparameter configuration for one baseline run."""

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

    def validate(self) -> None:
        if self.image_size < 8:
            raise ValueError(f"image_size must be at least 8; got {self.image_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than zero; got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be greater than zero; got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be greater than zero; got {self.learning_rate}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative; got {self.num_workers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")


def _hyperparameters(config: BaselineTrainingConfig) -> dict[str, Any]:
    return {
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "dropout": config.dropout,
    }


def _ordered_class_names(class_to_index: dict[str, int]) -> list[str]:
    expected_indices = list(range(len(class_to_index)))
    actual_indices = sorted(class_to_index.values())
    if actual_indices != expected_indices:
        raise ValueError(
            "class_to_index.json indices must be contiguous from 0 to "
            f"{len(class_to_index) - 1}; got {actual_indices}"
        )
    return sorted(class_to_index, key=class_to_index.__getitem__)


def run_baseline_training(config: BaselineTrainingConfig) -> dict[str, Any]:
    """Execute baseline training, best-model selection, evaluation, and reporting."""
    config.validate()
    set_seed(config.seed)
    device = get_device(config.device)
    class_to_index = load_class_mapping(config.processed_dir / "class_to_index.json")
    class_names = _ordered_class_names(class_to_index)
    num_classes = len(class_names)
    loaders = create_dataloaders(
        raw_data_dir=config.raw_data_dir,
        processed_dir=config.processed_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    model = create_baseline_cnn(num_classes=num_classes, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    hyperparameters = _hyperparameters(config)
    checkpoint_metadata = {
        "model_name": "baseline_cnn",
        "model_config": {"num_classes": num_classes, "dropout": config.dropout},
        "class_to_index": class_to_index,
        "hyperparameters": hyperparameters,
    }

    config.out_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = config.out_dir / "best_model.pt"
    last_checkpoint_path = config.out_dir / "last_model.pt"
    history_path = config.out_dir / "history.json"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, loaders.train, criterion, optimizer, device)
        validation_metrics = evaluate(
            model,
            loaders.val,
            criterion,
            device,
            num_classes,
            class_names,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": {
                    "loss": validation_metrics["loss"],
                    "accuracy": validation_metrics["accuracy"],
                    "macro_f1": validation_metrics["macro_f1"],
                },
            }
        )
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            validation_metrics,
            checkpoint_metadata,
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
                checkpoint_metadata,
            )
        save_json_history(
            {
                "schema_version": 1,
                "model_name": "baseline_cnn",
                "hyperparameters": hyperparameters,
                "epochs": history,
                "best_validation_epoch": best_epoch,
            },
            history_path,
        )
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train loss={train_metrics['loss']:.4f}, "
            f"train accuracy={train_metrics['accuracy']:.4f}, "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"val accuracy={validation_metrics['accuracy']:.4f}, "
            f"val macro F1={validation_metrics['macro_f1']:.4f}"
        )

    load_checkpoint(best_checkpoint_path, model, device=device)
    best_validation_metrics = evaluate(
        model,
        loaders.val,
        criterion,
        device,
        num_classes,
        class_names,
    )
    test_metrics = evaluate(
        model,
        loaders.test,
        criterion,
        device,
        num_classes,
        class_names,
    )

    confusion_matrix_path = config.figures_dir / "baseline_cnn_confusion_matrix.png"
    training_curves_path = config.figures_dir / "baseline_cnn_training_curves.png"
    save_confusion_matrix_figure(
        test_metrics["confusion_matrix"],
        class_names,
        confusion_matrix_path,
    )
    save_training_curves(history, training_curves_path)

    results = {
        "schema_version": 1,
        "model": {"name": "baseline_cnn", **summarize_model(model)},
        "dataset": {
            "num_classes": num_classes,
            "class_to_index": class_to_index,
            "split_sizes": {
                "train": len(loaders.train.dataset),
                "val": len(loaders.val.dataset),
                "test": len(loaders.test.dataset),
            },
        },
        "device": str(device),
        "hyperparameters": hyperparameters,
        "best_validation_epoch": best_epoch,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "checkpoints": {
            "best": str(best_checkpoint_path.resolve()),
            "last": str(last_checkpoint_path.resolve()),
            "history": str(history_path.resolve()),
        },
        "figures": {
            "confusion_matrix": str(confusion_matrix_path.resolve()),
            "training_curves": str(training_curves_path.resolve()),
        },
        "note": "This is a baseline CNN for comparison, not the final model.",
    }
    write_baseline_results(
        results,
        config.reports_dir / "baseline_cnn_results.json",
        config.reports_dir / "baseline_cnn_results.md",
    )
    print(f"Best validation epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline CNN.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw class-folder image root (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing split CSVs and class mapping (default: data/processed).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/models/baseline_cnn"),
        help="Checkpoint and history directory (default: artifacts/models/baseline_cnn).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="JSON and Markdown report directory (default: reports).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures"),
        help="Training curve and confusion matrix directory (default: artifacts/figures).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square transformed image size (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per optimizer step (default: 16).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of complete training epochs (default: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Python, NumPy, and PyTorch random seed (default: 42).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Classifier dropout probability (default: 0.3).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device selection (default: auto).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline CNN training from command-line arguments."""
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
