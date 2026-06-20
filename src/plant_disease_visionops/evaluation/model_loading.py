"""Offline reconstruction of trained models from project checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from plant_disease_visionops.models.factory import create_model


class ModelLoadingError(RuntimeError):
    """Raised when checkpoint metadata cannot reconstruct a trained model."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A reconstructed model and the class/model metadata needed for evaluation."""

    model: nn.Module
    model_name: str
    class_to_index: dict[str, int]
    checkpoint_metadata: dict[str, Any]


def _read_report_metadata(report_path: Path | None) -> dict[str, Any]:
    if report_path is None or not report_path.is_file():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadingError(f"Could not read model report {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ModelLoadingError(f"Model report must contain a JSON object: {report_path}")
    return report


def _class_mapping(metadata: dict[str, Any], report: dict[str, Any]) -> dict[str, int]:
    raw_mapping = metadata.get("class_to_index")
    if raw_mapping is None:
        dataset = report.get("dataset", {})
        raw_mapping = dataset.get("class_to_index") if isinstance(dataset, dict) else None
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ModelLoadingError("Checkpoint/report metadata is missing a non-empty class_to_index")
    mapping: dict[str, int] = {}
    for class_name, class_index in raw_mapping.items():
        if (
            not isinstance(class_name, str)
            or isinstance(class_index, bool)
            or not isinstance(class_index, int)
        ):
            raise ModelLoadingError("Checkpoint class_to_index contains invalid entries")
        mapping[class_name] = class_index
    expected = list(range(len(mapping)))
    if sorted(mapping.values()) != expected:
        raise ModelLoadingError(f"Checkpoint class indices must be contiguous; got {mapping}")
    return mapping


def load_trained_model(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
    report_path: Path | str | None = None,
) -> LoadedModel:
    """Reconstruct baseline CNN or ResNet18 without downloading pretrained weights."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ModelLoadingError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ModelLoadingError(f"Invalid checkpoint payload: {path}")

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ModelLoadingError(f"Checkpoint metadata must be a dictionary: {path}")
    report = _read_report_metadata(
        Path(report_path).expanduser() if report_path is not None else None
    )
    report_model = report.get("model", {})
    report_model_name = report.get("model_name")
    if report_model_name is None and isinstance(report_model, dict):
        report_model_name = report_model.get("name")
    model_name = metadata.get("model_name", report_model_name)
    if not isinstance(model_name, str):
        raise ModelLoadingError("Checkpoint/report metadata is missing model_name")

    mapping = _class_mapping(metadata, report)
    model_config = metadata.get("model_config", {})
    if not isinstance(model_config, dict):
        raise ModelLoadingError("Checkpoint model_config must be a dictionary")
    num_classes = model_config.get("num_classes", len(mapping))
    if not isinstance(num_classes, int) or num_classes != len(mapping):
        raise ModelLoadingError(
            f"Checkpoint num_classes does not match class mapping: {num_classes} vs {len(mapping)}"
        )
    dropout = model_config.get("dropout", 0.3)
    freeze_backbone = model_config.get("freeze_backbone", False)
    if not isinstance(dropout, (int, float)) or not isinstance(freeze_backbone, bool):
        raise ModelLoadingError("Checkpoint model_config has invalid dropout/freeze_backbone")

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=freeze_backbone,
        dropout=float(dropout),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise ModelLoadingError(
            f"Checkpoint state does not match reconstructed model: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        model_name=model_name,
        class_to_index=mapping,
        checkpoint_metadata=metadata,
    )
