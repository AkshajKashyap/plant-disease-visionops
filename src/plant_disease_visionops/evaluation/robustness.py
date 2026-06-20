"""Robustness evaluation orchestration, reports, and figures."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import torch
from torch import nn

from plant_disease_visionops.data.dataset import load_class_mapping
from plant_disease_visionops.data.loaders import (
    VALID_SPLITS,
    SplitName,
    create_dataloader,
    create_split_dataset,
)
from plant_disease_visionops.data.transforms import build_eval_transform
from plant_disease_visionops.evaluation.corruptions import (
    CORRUPTION_NAMES,
    SEVERITY_LEVELS,
    build_corrupted_eval_transform,
)
from plant_disease_visionops.evaluation.model_loading import load_trained_model
from plant_disease_visionops.training.engine import evaluate
from plant_disease_visionops.training.reproducibility import get_device, set_seed

_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "plant-disease-visionops-matplotlib"
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Configuration for clean and corrupted evaluation of one saved experiment."""

    checkpoint: Path
    experiment_name: str
    raw_data_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    split: str = "test"
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"
    corruptions: tuple[str, ...] = CORRUPTION_NAMES
    severities: tuple[int, ...] = SEVERITY_LEVELS

    def validate(self) -> None:
        if not _EXPERIMENT_NAME_PATTERN.fullmatch(self.experiment_name):
            raise ValueError(f"Invalid experiment_name: {self.experiment_name!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; got {self.split!r}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be greater than zero; got {self.image_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than zero; got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative; got {self.num_workers}")
        if not self.corruptions or any(name not in CORRUPTION_NAMES for name in self.corruptions):
            raise ValueError(
                f"corruptions must come from {CORRUPTION_NAMES}; got {self.corruptions}"
            )
        if not self.severities or any(level not in SEVERITY_LEVELS for level in self.severities):
            raise ValueError(f"severities must come from {SEVERITY_LEVELS}; got {self.severities}")


def _ordered_class_names(class_to_index: dict[str, int]) -> list[str]:
    return sorted(class_to_index, key=class_to_index.__getitem__)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "num_samples": metrics["num_samples"],
        "per_class_f1": [
            {
                "class_index": item["class_index"],
                "class_name": item["class_name"],
                "f1": item["f1"],
                "support": item["support"],
            }
            for item in metrics["per_class"]
        ],
    }


def _evaluate_transform(
    model: nn.Module,
    config: RobustnessConfig,
    transform: Any,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    split = cast(SplitName, config.split)
    dataset = create_split_dataset(
        split=split,
        raw_data_dir=config.raw_data_dir,
        processed_dir=config.processed_dir,
        image_size=config.image_size,
        transform=transform,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return evaluate(
        model=model,
        dataloader=dataloader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
    )


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_robustness_markdown(results: dict[str, Any]) -> str:
    """Render clean and corruption results with explicit performance drops."""
    clean = results["clean_metrics"]
    lines = [
        f"# Robustness Evaluation: {_escape_markdown(results['experiment_name'])}",
        "",
        f"- Checkpoint: `{results['checkpoint_path']}`",
        f"- Evaluated split: `{results['split']}`",
        f"- Model: `{results['model_name']}`",
        f"- Clean accuracy: {clean['accuracy']:.6f}",
        f"- Clean macro F1: {clean['macro_f1']:.6f}",
        "",
        "## Corruption Results",
        "",
        "| Corruption | Severity | Accuracy | Accuracy drop | Macro F1 | Macro F1 drop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in results["corruption_results"]:
        lines.append(
            f"| {_escape_markdown(condition['corruption'])} | {condition['severity']} | "
            f"{condition['accuracy']:.6f} | {condition['accuracy_drop']:.6f} | "
            f"{condition['macro_f1']:.6f} | {condition['macro_f1_drop']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Worst Corruption Settings",
            "",
            "| Corruption | Severity | Macro F1 | Macro F1 drop |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in results["worst_5_by_macro_f1_drop"]:
        lines.append(
            f"| {_escape_markdown(condition['corruption'])} | {condition['severity']} | "
            f"{condition['macro_f1']:.6f} | {condition['macro_f1_drop']:.6f} |"
        )
    lines.extend(
        [
            "",
            "> High clean accuracy on curated leaf datasets may not guarantee robustness to "
            "real-world lighting, focus, framing, and camera changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_metric_figure(
    results: dict[str, Any],
    metric: str,
    output_path: Path,
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for corruption in results["corruptions"]:
        conditions = [
            item for item in results["corruption_results"] if item["corruption"] == corruption
        ]
        axis.plot(
            [item["severity"] for item in conditions],
            [item[metric] for item in conditions],
            marker="o",
            label=corruption,
        )
    clean_value = results["clean_metrics"][metric]
    axis.axhline(clean_value, color="black", linestyle="--", linewidth=1.2, label="clean")
    axis.set(
        title=f"{results['experiment_name']} Robustness: {metric.replace('_', ' ').title()}",
        xlabel="Severity",
        ylabel=metric.replace("_", " ").title(),
        xticks=list(results["severities"]),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output_path, dpi=150)
    figure.clear()
    return output_path


def write_robustness_outputs(results: dict[str, Any], config: RobustnessConfig) -> None:
    """Write method-specific JSON, Markdown, and accuracy/F1 figures."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    result_stem = f"{config.experiment_name}_robustness"
    json_path = config.reports_dir / f"{result_stem}.json"
    markdown_path = config.reports_dir / f"{result_stem}.md"
    json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_robustness_markdown(results), encoding="utf-8")
    _save_metric_figure(
        results,
        "accuracy",
        config.figures_dir / f"{result_stem}_accuracy.png",
    )
    _save_metric_figure(
        results,
        "macro_f1",
        config.figures_dir / f"{result_stem}_macro_f1.png",
    )


def run_robustness_evaluation(config: RobustnessConfig) -> dict[str, Any]:
    """Evaluate clean data and every configured corruption/severity condition."""
    config.validate()
    set_seed(config.seed)
    device = get_device(config.device)
    report_path = config.reports_dir / f"{config.experiment_name}_results.json"
    loaded = load_trained_model(config.checkpoint, device=device, report_path=report_path)
    processed_mapping = load_class_mapping(config.processed_dir / "class_to_index.json")
    if loaded.class_to_index != processed_mapping:
        raise ValueError("Checkpoint class mapping does not match processed class_to_index.json")
    class_names = _ordered_class_names(loaded.class_to_index)

    clean_full = _evaluate_transform(
        loaded.model,
        config,
        build_eval_transform(config.image_size),
        device,
        class_names,
    )
    clean = _compact_metrics(clean_full)
    print(f"clean: accuracy={clean['accuracy']:.4f}, macro F1={clean['macro_f1']:.4f}")
    corruption_results: list[dict[str, Any]] = []
    for corruption in config.corruptions:
        for severity in config.severities:
            metrics = _compact_metrics(
                _evaluate_transform(
                    loaded.model,
                    config,
                    build_corrupted_eval_transform(
                        config.image_size,
                        corruption,
                        severity,
                        config.seed,
                    ),
                    device,
                    class_names,
                )
            )
            condition = {
                "corruption": corruption,
                "severity": severity,
                **metrics,
                "accuracy_drop": clean["accuracy"] - metrics["accuracy"],
                "macro_f1_drop": clean["macro_f1"] - metrics["macro_f1"],
            }
            corruption_results.append(condition)
            print(
                f"{corruption} severity {severity}: accuracy={metrics['accuracy']:.4f}, "
                f"macro F1={metrics['macro_f1']:.4f}"
            )

    worst_5 = sorted(
        corruption_results,
        key=lambda item: (-item["macro_f1_drop"], item["corruption"], item["severity"]),
    )[:5]
    results = {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "model_name": loaded.model_name,
        "checkpoint_path": str(config.checkpoint.resolve()),
        "split": config.split,
        "device": str(device),
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "num_classes": len(class_names),
        "corruptions": list(config.corruptions),
        "severities": list(config.severities),
        "clean_metrics": clean,
        "corruption_results": corruption_results,
        "worst_5_by_macro_f1_drop": worst_5,
        "average_corrupted_macro_f1": fmean(item["macro_f1"] for item in corruption_results),
        "note": (
            "High clean accuracy on curated leaf datasets may not guarantee robustness to "
            "real-world imaging changes."
        ),
    }
    write_robustness_outputs(results, config)
    return results
