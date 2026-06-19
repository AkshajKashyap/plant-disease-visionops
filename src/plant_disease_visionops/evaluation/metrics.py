"""Classification metrics with explicit per-class results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
    loss: float,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate and per-class metrics using a stable class order."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be greater than zero; got {num_classes}")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must contain the same number of items")
    if len(targets) == 0:
        raise ValueError("targets and predictions cannot be empty")
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(f"class_names must contain {num_classes} entries; got {len(class_names)}")

    valid_labels = set(range(num_classes))
    observed_labels = set(targets) | set(predictions)
    if not observed_labels <= valid_labels:
        raise ValueError(
            f"targets and predictions must be in [0, {num_classes - 1}]; "
            f"got {sorted(observed_labels - valid_labels)}"
        )

    labels = list(range(num_classes))
    names = list(class_names) if class_names is not None else [str(index) for index in labels]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    per_class = [
        {
            "class_index": class_index,
            "class_name": names[class_index],
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(per_class_f1[class_index]),
            "support": int(support[class_index]),
        }
        for class_index in labels
    ]
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "num_samples": len(targets),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "per_class_support": [int(value) for value in support],
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }
