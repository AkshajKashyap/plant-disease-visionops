"""Prediction-level failure analysis, reports, and visual galleries."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from PIL import Image

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
    create_corruption,
)
from plant_disease_visionops.evaluation.model_loading import load_trained_model
from plant_disease_visionops.training.reproducibility import get_device, set_seed

_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TOP_K = 10
_MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "plant-disease-visionops-matplotlib"
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))


class PredictionRecord(TypedDict):
    """Serializable prediction details for one evaluated image."""

    filepath: str
    true_label: str
    true_index: int
    predicted_label: str
    predicted_index: int
    confidence: float
    true_class_probability: float
    correct: bool
    corruption: str | None
    severity: int | None


@dataclass(frozen=True, slots=True)
class FailureAnalysisConfig:
    """Configuration for one clean or corrupted prediction analysis."""

    checkpoint: Path
    experiment_name: str
    raw_data_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    split: str = "test"
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    max_examples: int = 80
    device: str = "auto"
    corruption: str | None = None
    severity: int | None = None

    def validate(self) -> None:
        """Reject ambiguous or unsafe analysis settings."""
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
        if self.max_examples <= 0:
            raise ValueError(f"max_examples must be greater than zero; got {self.max_examples}")
        if self.corruption is None and self.severity is not None:
            raise ValueError("--severity requires --corruption")
        if self.corruption is not None:
            if self.corruption not in CORRUPTION_NAMES:
                raise ValueError(
                    f"corruption must be one of {CORRUPTION_NAMES}; got {self.corruption!r}"
                )
            if self.severity not in SEVERITY_LEVELS:
                raise ValueError(
                    f"severity must be one of {SEVERITY_LEVELS} when corruption is set; "
                    f"got {self.severity}"
                )

    @property
    def condition_name(self) -> str:
        """Return the filename-safe name for this evaluation condition."""
        if self.corruption is None:
            return "clean"
        return f"{self.corruption}_s{self.severity}"

    @property
    def output_stem(self) -> str:
        """Return the shared stem for all condition-specific artifacts."""
        return f"{self.experiment_name}_failures_{self.condition_name}"


def _count_rows(counter: Counter[str], key: str) -> list[dict[str, Any]]:
    return [{key: name, "mistake_count": count} for name, count in counter.most_common(_TOP_K)]


def _record_sort_key(record: PredictionRecord) -> tuple[float, str]:
    return (-record["confidence"], record["filepath"])


def summarize_predictions(records: list[PredictionRecord]) -> dict[str, Any]:
    """Compute mistake, confusion, confidence, and per-class summaries."""
    if not records:
        raise ValueError("Cannot summarize an empty prediction collection")

    mistakes = [record for record in records if not record["correct"]]
    correct = [record for record in records if record["correct"]]
    true_mistakes = Counter(record["true_label"] for record in mistakes)
    predicted_mistakes = Counter(record["predicted_label"] for record in mistakes)
    confusion_counts = Counter(
        (record["true_label"], record["predicted_label"]) for record in mistakes
    )

    class_totals: Counter[tuple[int, str]] = Counter()
    class_mistakes: Counter[tuple[int, str]] = Counter()
    for record in records:
        key = (record["true_index"], record["true_label"])
        class_totals[key] += 1
        if not record["correct"]:
            class_mistakes[key] += 1

    per_class = []
    for (class_index, class_name), total in sorted(class_totals.items()):
        mistake_count = class_mistakes[(class_index, class_name)]
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "total": total,
                "mistakes": mistake_count,
                "error_rate": mistake_count / total,
            }
        )

    common_confusions = [
        {"true_label": pair[0], "predicted_label": pair[1], "mistake_count": count}
        for pair, count in confusion_counts.most_common(_TOP_K)
    ]
    return {
        "total_images": len(records),
        "total_mistakes": len(mistakes),
        "error_rate": len(mistakes) / len(records),
        "top_true_labels_by_mistake_count": _count_rows(true_mistakes, "true_label"),
        "top_predicted_labels_in_mistakes": _count_rows(
            predicted_mistakes, "predicted_label"
        ),
        "most_common_confusions": common_confusions,
        "lowest_confidence_correct_predictions": sorted(
            correct, key=lambda record: (record["confidence"], record["filepath"])
        )[:_TOP_K],
        "highest_confidence_wrong_predictions": sorted(mistakes, key=_record_sort_key)[
            :_TOP_K
        ],
        "per_class_error_rates": sorted(
            per_class,
            key=lambda item: (-item["error_rate"], -item["mistakes"], item["class_index"]),
        ),
    }


def _ordered_class_names(class_to_index: dict[str, int]) -> list[str]:
    return sorted(class_to_index, key=class_to_index.__getitem__)


def _collect_predictions(
    config: FailureAnalysisConfig,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
) -> list[PredictionRecord]:
    transform = (
        build_eval_transform(config.image_size)
        if config.corruption is None
        else build_corrupted_eval_transform(
            config.image_size,
            config.corruption,
            cast(int, config.severity),
            config.seed,
        )
    )
    dataset = create_split_dataset(
        split=cast(SplitName, config.split),
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

    model.eval()
    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for images, targets, metadata in dataloader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted_indices = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values
            target_indices = targets.tolist()
            filepaths = cast(list[str], metadata["filepath"])
            for row, (filepath, true_index, predicted_index) in enumerate(
                zip(filepaths, target_indices, predicted_indices.tolist(), strict=True)
            ):
                records.append(
                    {
                        "filepath": filepath,
                        "true_label": class_names[true_index],
                        "true_index": true_index,
                        "predicted_label": class_names[predicted_index],
                        "predicted_index": predicted_index,
                        "confidence": float(confidences[row].item()),
                        "true_class_probability": float(probabilities[row, true_index].item()),
                        "correct": predicted_index == true_index,
                        "corruption": config.corruption,
                        "severity": config.severity,
                    }
                )
    return records


def _compact_label(label: str, max_length: int = 24) -> str:
    readable = label.replace("___", " / ").replace("_", " ")
    return readable if len(readable) <= max_length else f"{readable[: max_length - 3]}..."


def save_failure_gallery(
    records: list[PredictionRecord],
    config: FailureAnalysisConfig,
    output_path: Path,
) -> Path:
    """Save a readable gallery of selected misclassified source images."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(4, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    figure = Figure(figsize=(3.4 * columns, 3.2 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)

    if not records:
        axis = axes[0, 0]
        axis.text(0.5, 0.5, "No misclassifications found", ha="center", va="center")
        axis.set_axis_off()
    else:
        corruption = (
            None
            if config.corruption is None
            else create_corruption(
                config.corruption,
                cast(int, config.severity),
                config.seed,
            )
        )
        for axis, record in zip(axes.flat, records, strict=False):
            image_path = config.raw_data_dir / record["filepath"]
            with Image.open(image_path) as image:
                display_image = image.convert("RGB")
                if corruption is not None:
                    display_image = corruption(display_image)
                axis.imshow(display_image)
            axis.set_title(
                f"True: {_compact_label(record['true_label'])}\n"
                f"Pred: {_compact_label(record['predicted_label'])}\n"
                f"Confidence: {record['confidence']:.3f}",
                fontsize=8,
            )
            axis.set_axis_off()

    for axis in axes.flat[len(records) :]:
        axis.set_axis_off()
    figure.suptitle(
        f"{config.experiment_name}: {config.condition_name} misclassifications",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=140)
    figure.clear()
    return output_path


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _append_count_table(
    lines: list[str],
    heading: str,
    rows: list[dict[str, Any]],
    label_key: str,
) -> None:
    lines.extend(["", f"## {heading}", "", "| Label | Mistakes |", "|---|---:|"])
    if not rows:
        lines.append("| None | 0 |")
        return
    for row in rows:
        lines.append(
            f"| {_escape_markdown(row[label_key])} | {row['mistake_count']} |"
        )


def render_failure_markdown(results: dict[str, Any]) -> str:
    """Render a concise, condition-specific failure analysis report."""
    summary = results["summary"]
    corruption = results["corruption"] or "none (clean evaluation)"
    severity = results["severity"] if results["severity"] is not None else "n/a"
    lines = [
        f"# Failure Analysis: {_escape_markdown(results['experiment_name'])}",
        "",
        f"- Checkpoint: `{results['checkpoint_path']}`",
        f"- Model: `{results['model_name']}`",
        f"- Evaluated split: `{results['split']}`",
        f"- Corruption: `{corruption}`",
        f"- Severity: `{severity}`",
        f"- Total images: {summary['total_images']}",
        f"- Total mistakes: {summary['total_mistakes']}",
        f"- Error rate: {summary['error_rate']:.6f}",
        f"- Misclassification grid: `{results['gallery_path']}`",
    ]
    _append_count_table(
        lines,
        "Top True Labels by Mistake Count",
        summary["top_true_labels_by_mistake_count"],
        "true_label",
    )
    _append_count_table(
        lines,
        "Top Predicted Labels in Mistakes",
        summary["top_predicted_labels_in_mistakes"],
        "predicted_label",
    )
    lines.extend(
        [
            "",
            "## Most Common Confusions",
            "",
            "| True label | Predicted label | Mistakes |",
            "|---|---|---:|",
        ]
    )
    if not summary["most_common_confusions"]:
        lines.append("| None | None | 0 |")
    for row in summary["most_common_confusions"]:
        lines.append(
            f"| {_escape_markdown(row['true_label'])} | "
            f"{_escape_markdown(row['predicted_label'])} | {row['mistake_count']} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Confidence Wrong Predictions",
            "",
            "| Filepath | True label | Predicted label | Confidence | True probability |",
            "|---|---|---|---:|---:|",
        ]
    )
    if not summary["highest_confidence_wrong_predictions"]:
        lines.append("| None | None | None | n/a | n/a |")
    for record in summary["highest_confidence_wrong_predictions"]:
        lines.append(
            f"| {_escape_markdown(record['filepath'])} | "
            f"{_escape_markdown(record['true_label'])} | "
            f"{_escape_markdown(record['predicted_label'])} | "
            f"{record['confidence']:.6f} | {record['true_class_probability']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest-Confidence Correct Predictions",
            "",
            "| Filepath | Label | Confidence |",
            "|---|---|---:|",
        ]
    )
    if not summary["lowest_confidence_correct_predictions"]:
        lines.append("| None | None | n/a |")
    for record in summary["lowest_confidence_correct_predictions"]:
        lines.append(
            f"| {_escape_markdown(record['filepath'])} | "
            f"{_escape_markdown(record['true_label'])} | {record['confidence']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Highest Per-Class Error Rates",
            "",
            "| Class | Images | Mistakes | Error rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["per_class_error_rates"][:_TOP_K]:
        lines.append(
            f"| {_escape_markdown(row['class_name'])} | {row['total']} | "
            f"{row['mistakes']} | {row['error_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gallery is a targeted diagnostic for this checkpoint and evaluation condition. "
            "Repeated confusion pairs can identify classes that need closer data or model review, "
            "but the images may also expose labeling ambiguity, duplicated acquisition settings, "
            "or clean-background dataset artifacts. These results should not be interpreted as "
            "field performance without evaluation on independently collected field images.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_failure_outputs(
    results: dict[str, Any],
    config: FailureAnalysisConfig,
    gallery_records: list[PredictionRecord],
) -> tuple[Path, Path, Path]:
    """Write condition-specific JSON, Markdown, and image gallery outputs."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_dir / f"{config.output_stem}.json"
    markdown_path = config.reports_dir / f"{config.output_stem}.md"
    gallery_path = config.figures_dir / f"{config.output_stem}.png"
    save_failure_gallery(gallery_records, config, gallery_path)
    results["gallery_path"] = str(gallery_path.resolve())
    json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_failure_markdown(results), encoding="utf-8")
    return json_path, markdown_path, gallery_path


def run_failure_analysis(config: FailureAnalysisConfig) -> dict[str, Any]:
    """Run prediction-level analysis and persist reports for one condition."""
    config.validate()
    set_seed(config.seed)
    device = get_device(config.device)
    report_path = config.reports_dir / f"{config.experiment_name}_results.json"
    loaded = load_trained_model(config.checkpoint, device=device, report_path=report_path)
    processed_mapping = load_class_mapping(config.processed_dir / "class_to_index.json")
    if loaded.class_to_index != processed_mapping:
        raise ValueError("Checkpoint class mapping does not match processed class_to_index.json")
    class_names = _ordered_class_names(loaded.class_to_index)

    records = _collect_predictions(config, loaded.model, device, class_names)
    summary = summarize_predictions(records)
    mistakes = sorted(
        (record for record in records if not record["correct"]),
        key=_record_sort_key,
    )
    selected_mistakes = mistakes[: config.max_examples]
    results: dict[str, Any] = {
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
        "condition": config.condition_name,
        "corruption": config.corruption,
        "severity": config.severity,
        "summary": summary,
        "misclassified_examples": selected_mistakes,
        "stored_example_count": len(selected_mistakes),
        "max_examples": config.max_examples,
        "gallery_path": "",
        "note": (
            "This is a checkpoint-specific diagnostic. Curated backgrounds, acquisition patterns, "
            "and label ambiguity can influence the observed failures."
        ),
    }
    write_failure_outputs(results, config, selected_mistakes)
    return results
