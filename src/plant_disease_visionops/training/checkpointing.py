"""Checkpoint and JSON history persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save model state and enough metadata to resume or evaluate."""
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "metadata": metadata or {},
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model and optional optimizer state and return the checkpoint payload."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def save_json_history(history: dict[str, Any], path: Path | str) -> Path:
    """Atomically save JSON-compatible training history."""
    history_path = Path(path).expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = history_path.with_suffix(history_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(history_path)
    return history_path
