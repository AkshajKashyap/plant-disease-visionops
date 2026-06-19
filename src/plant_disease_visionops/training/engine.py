"""Framework-level loops for training and evaluating classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from plant_disease_visionops.evaluation.metrics import compute_classification_metrics


def _batch_tensors(batch: object) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Each DataLoader batch must contain images and class indices")
    images, targets = batch[0], batch[1]
    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        raise ValueError("DataLoader images and class indices must be torch tensors")
    return images, targets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    """Train for one complete epoch and return sample-weighted loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in dataloader:
        images, targets = _batch_tensors(batch)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a classifier and compute aggregate, per-class, and confusion metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    for batch in dataloader:
        images, targets = _batch_tensors(batch)
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(int(value) for value in targets.cpu().tolist())
        all_predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader produced no samples")
    return compute_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
        num_classes=num_classes,
        loss=total_loss / total_samples,
        class_names=class_names,
    )
