import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from plant_disease_visionops.evaluation.analyze_failures import main as failure_main
from plant_disease_visionops.evaluation.failure_analysis import (
    FailureAnalysisConfig,
    PredictionRecord,
    run_failure_analysis,
    summarize_predictions,
)
from plant_disease_visionops.models.factory import create_model
from plant_disease_visionops.training.checkpointing import save_checkpoint

CSV_COLUMNS = ("filepath", "label", "class_index", "width", "height")


def _prediction(
    filepath: str,
    true_label: str,
    true_index: int,
    predicted_label: str,
    predicted_index: int,
    confidence: float,
) -> PredictionRecord:
    return {
        "filepath": filepath,
        "true_label": true_label,
        "true_index": true_index,
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "true_class_probability": confidence if true_index == predicted_index else 1 - confidence,
        "correct": true_index == predicted_index,
        "corruption": None,
        "severity": None,
    }


def test_confusion_summary_uses_only_wrong_predictions() -> None:
    records = [
        _prediction("a.png", "healthy", 0, "healthy", 0, 0.7),
        _prediction("b.png", "rust", 1, "healthy", 0, 0.9),
        _prediction("c.png", "rust", 1, "healthy", 0, 0.8),
        _prediction("d.png", "healthy", 0, "rust", 1, 0.6),
    ]

    summary = summarize_predictions(records)

    assert summary["total_images"] == 4
    assert summary["total_mistakes"] == 3
    assert summary["error_rate"] == pytest.approx(0.75)
    assert summary["top_true_labels_by_mistake_count"][0] == {
        "true_label": "rust",
        "mistake_count": 2,
    }
    assert summary["most_common_confusions"][0] == {
        "true_label": "rust",
        "predicted_label": "healthy",
        "mistake_count": 2,
    }
    assert summary["highest_confidence_wrong_predictions"][0]["filepath"] == "b.png"
    healthy_rates = [
        row for row in summary["per_class_error_rates"] if row["class_name"] == "healthy"
    ]
    assert healthy_rates[0]["error_rate"] == pytest.approx(0.5)


def _create_toy_checkpoint_and_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    mapping = {"healthy": 0, "rust": 1}
    (processed_dir / "class_to_index.json").write_text(json.dumps(mapping), encoding="utf-8")
    rows = []
    colors = {"healthy": (30, 150, 60), "rust": (150, 60, 30)}
    for label in mapping:
        for image_number in range(2):
            relative_path = f"{label}/test_{image_number}.png"
            image_path = raw_dir / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), color=colors[label]).save(image_path)
            rows.append(
                {
                    "filepath": relative_path,
                    "label": label,
                    "class_index": mapping[label],
                    "width": 24,
                    "height": 24,
                }
            )
    with (processed_dir / "test.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    model = create_model("baseline_cnn", num_classes=2, pretrained=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias[0] = 2.0
    checkpoint_path = tmp_path / "models" / "tiny_baseline" / "best_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        epoch=1,
        metrics={"macro_f1": 0.0},
        metadata={
            "experiment_name": "tiny_baseline",
            "model_name": "baseline_cnn",
            "model_config": {"num_classes": 2, "dropout": 0.3},
            "class_to_index": mapping,
        },
    )
    return raw_dir, processed_dir, checkpoint_path


def test_clean_failure_cli_writes_json_markdown_and_grid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir, processed_dir, checkpoint_path = _create_toy_checkpoint_and_data(tmp_path)
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"

    exit_code = failure_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--experiment-name",
            "tiny_baseline",
            "--raw-data-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--figures-dir",
            str(figures_dir),
            "--image-size",
            "16",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--max-examples",
            "1",
            "--device",
            "cpu",
        ]
    )

    json_path = reports_dir / "tiny_baseline_failures_clean.json"
    markdown_path = reports_dir / "tiny_baseline_failures_clean.md"
    grid_path = figures_dir / "tiny_baseline_failures_clean.png"
    assert exit_code == 0
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert grid_path.is_file() and grid_path.stat().st_size > 0
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_images"] == 4
    assert saved["summary"]["total_mistakes"] == 2
    assert saved["stored_example_count"] == 1
    example = saved["misclassified_examples"][0]
    assert {
        "filepath",
        "true_label",
        "true_index",
        "predicted_label",
        "predicted_index",
        "confidence",
        "true_class_probability",
        "corruption",
        "severity",
    } <= example.keys()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Most Common Confusions" in markdown
    assert "## Highest-Confidence Wrong Predictions" in markdown
    assert "dataset artifacts" in markdown
    assert "Analyzed 4 images" in capsys.readouterr().out


def test_corrupted_failure_analysis_uses_condition_specific_outputs(tmp_path: Path) -> None:
    raw_dir, processed_dir, checkpoint_path = _create_toy_checkpoint_and_data(tmp_path)
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    config = FailureAnalysisConfig(
        checkpoint=checkpoint_path,
        experiment_name="tiny_baseline",
        raw_data_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        image_size=16,
        batch_size=2,
        num_workers=0,
        max_examples=2,
        device="cpu",
        corruption="brightness_decrease",
        severity=3,
    )

    results = run_failure_analysis(config)

    stem = "tiny_baseline_failures_brightness_decrease_s3"
    assert results["condition"] == "brightness_decrease_s3"
    assert results["corruption"] == "brightness_decrease"
    assert results["severity"] == 3
    assert all(
        record["corruption"] == "brightness_decrease"
        and record["severity"] == 3
        for record in results["misclassified_examples"]
    )
    assert (reports_dir / f"{stem}.json").is_file()
    assert (reports_dir / f"{stem}.md").is_file()
    assert (figures_dir / f"{stem}.png").is_file()


def test_failure_cli_fails_clearly_for_missing_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = failure_main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--experiment-name",
            "missing_model",
            "--raw-data-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
    assert "Checkpoint does not exist" in capsys.readouterr().err
