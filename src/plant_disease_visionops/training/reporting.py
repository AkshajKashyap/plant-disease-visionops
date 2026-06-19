"""JSON, Markdown, and figure outputs for baseline training runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "plant-disease-visionops-matplotlib"
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_lines(title: str, metrics: dict[str, Any]) -> list[str]:
    return [
        f"## {title}",
        "",
        f"- Loss: {metrics['loss']:.6f}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Samples: {metrics['num_samples']}",
    ]


def render_baseline_results_markdown(results: dict[str, Any]) -> str:
    """Render a concise, human-readable baseline evaluation report."""
    dataset = results["dataset"]
    hyperparameters = results["hyperparameters"]
    validation_metrics = results["validation_metrics"]
    test_metrics = results["test_metrics"]
    weakest_classes = sorted(
        test_metrics["per_class"],
        key=lambda item: (item["f1"], item["class_name"]),
    )[:10]
    lines = [
        "# Baseline CNN Results",
        "",
        "> This is a small baseline CNN for pipeline validation and comparison. It is not the "
        "final model.",
        "",
        "## Run Summary",
        "",
        f"- Number of classes: {dataset['num_classes']}",
        f"- Device: `{results['device']}`",
        f"- Best validation epoch: {results['best_validation_epoch']}",
        f"- Best checkpoint: `{results['checkpoints']['best']}`",
        "",
        "## Dataset Splits",
        "",
        "| Split | Images |",
        "|---|---:|",
        f"| Train | {dataset['split_sizes']['train']} |",
        f"| Validation | {dataset['split_sizes']['val']} |",
        f"| Test | {dataset['split_sizes']['test']} |",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    for name, value in hyperparameters.items():
        lines.append(f"| {_escape_markdown(name)} | {_escape_markdown(value)} |")

    lines.extend(["", *_metric_lines("Best Validation Metrics", validation_metrics)])
    lines.extend(["", *_metric_lines("Test Metrics", test_metrics)])
    lines.extend(
        [
            "",
            "## Weakest Test Classes",
            "",
            "The ten classes with the lowest test F1 are shown, or all classes when fewer than ten "
            "exist.",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_metrics in weakest_classes:
        lines.append(
            f"| {_escape_markdown(class_metrics['class_name'])} | "
            f"{class_metrics['support']} | {class_metrics['precision']:.6f} | "
            f"{class_metrics['recall']:.6f} | {class_metrics['f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def write_baseline_results(
    results: dict[str, Any],
    json_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Write matching machine-readable and Markdown result reports."""
    output_json = Path(json_path).expanduser()
    output_markdown = Path(markdown_path).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_baseline_results_markdown(results), encoding="utf-8")
    return output_json, output_markdown


def render_experiment_results_markdown(results: dict[str, Any]) -> str:
    """Render a method-specific experiment report with transfer-learning metadata."""
    weakest_classes = (
        results.get("weakest_test_classes")
        or sorted(
            results["test_metrics"]["per_class"],
            key=lambda item: (item["f1"], item["class_name"]),
        )[:10]
    )
    lines = [
        f"# Experiment Results: {_escape_markdown(results['experiment_name'])}",
        "",
        "## Experiment Metadata",
        "",
        f"- Model: `{results['model_name']}`",
        f"- Pretrained: {results['pretrained']}",
        f"- Freeze backbone: {results['freeze_backbone']}",
        f"- Device: `{results['device']}`",
        f"- Number of classes: {results['num_classes']}",
        f"- Best validation epoch: {results['best_validation_epoch']}",
        f"- Best checkpoint: `{results['checkpoint_path']}`",
        "",
        "## Dataset Splits",
        "",
        "| Split | Images |",
        "|---|---:|",
        f"| Train | {results['split_sizes']['train']} |",
        f"| Validation | {results['split_sizes']['val']} |",
        f"| Test | {results['split_sizes']['test']} |",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| image_size | {results['image_size']} |",
        f"| batch_size | {results['batch_size']} |",
        f"| epochs | {results['epochs']} |",
        f"| learning_rate | {results['learning_rate']} |",
        f"| num_workers | {results['hyperparameters']['num_workers']} |",
        f"| seed | {results['hyperparameters']['seed']} |",
        f"| dropout | {results['hyperparameters']['dropout']} |",
        "",
        *_metric_lines("Best Validation Metrics", results["validation_metrics"]),
        "",
        *_metric_lines("Test Metrics", results["test_metrics"]),
        "",
        "## Weakest Test Classes",
        "",
        "The ten classes with the lowest test F1 are shown, or all classes when fewer than ten "
        "exist.",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_metrics in weakest_classes:
        lines.append(
            f"| {_escape_markdown(class_metrics['class_name'])} | "
            f"{class_metrics['support']} | {class_metrics['precision']:.6f} | "
            f"{class_metrics['recall']:.6f} | {class_metrics['f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This report records one experiment configuration. Metrics are measured from its "
            "saved best checkpoint and are not cross-experiment estimates.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_experiment_results(
    results: dict[str, Any],
    json_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Write JSON and Markdown outputs for one named experiment."""
    output_json = Path(json_path).expanduser()
    output_markdown = Path(markdown_path).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_experiment_results_markdown(results), encoding="utf-8")
    return output_json, output_markdown


def save_training_curves(history: list[dict[str, Any]], output_path: Path | str) -> Path:
    """Save train/validation loss and score curves without requiring a display server."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not history:
        raise ValueError("Training history cannot be empty")
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]

    figure = Figure(figsize=(11, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    loss_axis, score_axis = figure.subplots(1, 2)
    loss_axis.plot(epochs, [entry["train"]["loss"] for entry in history], label="Train")
    loss_axis.plot(epochs, [entry["val"]["loss"] for entry in history], label="Validation")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.25)

    score_axis.plot(
        epochs,
        [entry["train"]["accuracy"] for entry in history],
        label="Train accuracy",
    )
    score_axis.plot(
        epochs,
        [entry["val"]["accuracy"] for entry in history],
        label="Validation accuracy",
    )
    score_axis.plot(
        epochs,
        [entry["val"]["macro_f1"] for entry in history],
        label="Validation macro F1",
    )
    score_axis.set(title="Scores", xlabel="Epoch", ylabel="Score", ylim=(0.0, 1.0))
    score_axis.legend()
    score_axis.grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    figure.clear()
    return path


def save_confusion_matrix_figure(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: Path | str,
    title: str = "Baseline CNN Test Confusion Matrix",
) -> Path:
    """Save a raw-count confusion matrix with stable class ordering."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(matrix) != len(class_names) or any(len(row) != len(class_names) for row in matrix):
        raise ValueError("Confusion matrix dimensions must match class_names")
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_size = max(7.0, min(20.0, len(class_names) * 0.45))
    figure = Figure(figsize=(figure_size, figure_size), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    if len(class_names) <= 15:
        maximum = max((max(row) for row in matrix), default=0)
        threshold = maximum / 2
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
    figure.savefig(path, dpi=150)
    figure.clear()
    return path
