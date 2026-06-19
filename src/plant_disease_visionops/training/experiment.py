"""Model-agnostic image-classification experiment runner."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from plant_disease_visionops.data.dataset import load_class_mapping
from plant_disease_visionops.data.loaders import create_dataloaders
from plant_disease_visionops.models.baseline_cnn import summarize_model
from plant_disease_visionops.models.factory import SUPPORTED_MODELS, create_model
from plant_disease_visionops.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_json_history,
)
from plant_disease_visionops.training.engine import evaluate, train_one_epoch
from plant_disease_visionops.training.reporting import (
    save_confusion_matrix_figure,
    save_training_curves,
    write_experiment_results,
)
from plant_disease_visionops.training.reproducibility import get_device, set_seed

EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for a baseline or transfer-learning run."""

    experiment_name: str
    model_name: str
    pretrained: bool
    freeze_backbone: bool
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
        if not EXPERIMENT_NAME_PATTERN.fullmatch(self.experiment_name):
            raise ValueError(
                "experiment_name must contain only letters, numbers, dots, underscores, and "
                f"hyphens; got {self.experiment_name!r}"
            )
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {SUPPORTED_MODELS}; got {self.model_name!r}"
            )
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


def _ordered_class_names(class_to_index: dict[str, int]) -> list[str]:
    expected_indices = list(range(len(class_to_index)))
    actual_indices = sorted(class_to_index.values())
    if actual_indices != expected_indices:
        raise ValueError(
            "class_to_index.json indices must be contiguous from 0 to "
            f"{len(class_to_index) - 1}; got {actual_indices}"
        )
    return sorted(class_to_index, key=class_to_index.__getitem__)


def _hyperparameters(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "dropout": config.dropout,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train, select, evaluate, and report one configured model experiment."""
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
    model = create_model(
        model_name=config.model_name,
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Selected model configuration has no trainable parameters")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    hyperparameters = _hyperparameters(config)
    checkpoint_metadata = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "model_config": {
            "num_classes": num_classes,
            "pretrained": config.pretrained,
            "freeze_backbone": config.freeze_backbone,
            "dropout": config.dropout,
        },
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
                "schema_version": 2,
                "experiment_name": config.experiment_name,
                "model_name": config.model_name,
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
    confusion_matrix_path = config.figures_dir / f"{config.experiment_name}_confusion_matrix.png"
    training_curves_path = config.figures_dir / f"{config.experiment_name}_training_curves.png"
    save_confusion_matrix_figure(
        test_metrics["confusion_matrix"],
        class_names,
        confusion_matrix_path,
        title=f"{config.experiment_name} Test Confusion Matrix",
    )
    save_training_curves(history, training_curves_path)

    split_sizes = {
        "train": len(loaders.train.dataset),
        "val": len(loaders.val.dataset),
        "test": len(loaders.test.dataset),
    }
    weakest_test_classes = sorted(
        test_metrics["per_class"],
        key=lambda item: (item["f1"], item["class_name"]),
    )[:10]
    results = {
        "schema_version": 2,
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "device": str(device),
        "num_classes": num_classes,
        "split_sizes": split_sizes,
        "best_validation_epoch": best_epoch,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "weakest_test_classes": weakest_test_classes,
        "checkpoint_path": str(best_checkpoint_path.resolve()),
        "model": {"name": config.model_name, **summarize_model(model)},
        "dataset": {
            "num_classes": num_classes,
            "class_to_index": class_to_index,
            "split_sizes": split_sizes,
        },
        "hyperparameters": hyperparameters,
        "checkpoints": {
            "best": str(best_checkpoint_path.resolve()),
            "last": str(last_checkpoint_path.resolve()),
            "history": str(history_path.resolve()),
        },
        "figures": {
            "confusion_matrix": str(confusion_matrix_path.resolve()),
            "training_curves": str(training_curves_path.resolve()),
        },
        "note": "Metrics are specific to this experiment's saved best checkpoint.",
    }
    result_stem = f"{config.experiment_name}_results"
    write_experiment_results(
        results,
        config.reports_dir / f"{result_stem}.json",
        config.reports_dir / f"{result_stem}.md",
    )
    print(f"Experiment: {config.experiment_name}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    return results
