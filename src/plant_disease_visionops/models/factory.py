"""Model registry for baseline and transfer-learning experiments."""

from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from plant_disease_visionops.models.baseline_cnn import create_baseline_cnn

ModelName = Literal["baseline_cnn", "resnet18"]
SUPPORTED_MODELS = ("baseline_cnn", "resnet18")


class ModelConfigurationError(ValueError):
    """Raised when model options are invalid or incompatible."""


class PretrainedWeightsError(RuntimeError):
    """Raised when requested pretrained weights cannot be loaded."""


def create_resnet18(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create ResNet18 with a task-specific classifier and optional frozen backbone."""
    if num_classes <= 0:
        raise ModelConfigurationError(f"num_classes must be greater than zero; got {num_classes}")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            raise PretrainedWeightsError(
                "Could not load pretrained ResNet18 weights. Check network/cache access or rerun "
                "with --pretrained false."
            ) from exc
        raise

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


def create_model(
    model_name: ModelName | str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a supported model from experiment-level configuration."""
    if model_name == "baseline_cnn":
        if pretrained:
            raise ModelConfigurationError("baseline_cnn does not support pretrained weights")
        if freeze_backbone:
            raise ModelConfigurationError("baseline_cnn does not have a separable backbone")
        return create_baseline_cnn(num_classes=num_classes, dropout=dropout)
    if model_name == "resnet18":
        return create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ModelConfigurationError(
        f"Unsupported model_name {model_name!r}; expected one of {SUPPORTED_MODELS}"
    )
