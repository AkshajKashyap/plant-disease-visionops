"""A compact convolutional baseline for plant disease classification."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


def _convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class BaselineCNN(nn.Module):
    """Three convolution blocks followed by global pooling and one classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be greater than zero; got {num_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.num_classes = num_classes
        self.dropout = dropout
        self.features = nn.Sequential(
            _convolution_block(3, 32),
            _convolution_block(32, 64),
            _convolution_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features))


def create_baseline_cnn(num_classes: int, dropout: float = 0.3) -> BaselineCNN:
    """Create the baseline CNN without hardcoding its output dimension."""
    return BaselineCNN(num_classes=num_classes, dropout=dropout)


def summarize_model(model: nn.Module) -> dict[str, Any]:
    """Return parameter counts suitable for JSON reports."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "architecture": model.__class__.__name__,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
