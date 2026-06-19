import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plant_disease_visionops.models.baseline_cnn import (
    create_baseline_cnn,
    summarize_model,
)
from plant_disease_visionops.training.checkpointing import load_checkpoint, save_checkpoint
from plant_disease_visionops.training.engine import evaluate, train_one_epoch
from plant_disease_visionops.training.reporting import write_baseline_results

CSV_COLUMNS = ("filepath", "label", "class_index", "width", "height")


def _tiny_loader() -> DataLoader:
    images = torch.rand(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)


def _tiny_linear_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))


def test_baseline_model_forward_shape_and_summary() -> None:
    model = create_baseline_cnn(num_classes=5, dropout=0.2)

    logits = model(torch.rand(2, 3, 32, 32))
    summary = summarize_model(model)

    assert logits.shape == (2, 5)
    assert summary["architecture"] == "BaselineCNN"
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] == summary["total_parameters"]


def test_train_one_epoch_runs_on_tiny_cpu_data() -> None:
    model = _tiny_linear_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_one_epoch(
        model=model,
        dataloader=_tiny_loader(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics["num_samples"] == 4
    assert metrics["loss"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_returns_aggregate_per_class_and_confusion_metrics() -> None:
    metrics = evaluate(
        model=_tiny_linear_model(),
        dataloader=_tiny_loader(),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        num_classes=2,
        class_names=["healthy", "rust"],
    )

    required_keys = {
        "loss",
        "accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class",
        "confusion_matrix",
    }
    assert required_keys <= metrics.keys()
    assert len(metrics["per_class"]) == 2
    assert len(metrics["confusion_matrix"]) == 2
    assert metrics["num_samples"] == 4


def test_checkpoint_save_and_load_restores_model(tmp_path: Path) -> None:
    model = _tiny_linear_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = tmp_path / "model.pt"
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        metrics={"macro_f1": 0.5},
        metadata={"model_name": "tiny"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)

    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, device="cpu")

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["macro_f1"] == 0.5
    for restored, original in zip(model.parameters(), original_parameters, strict=True):
        assert torch.equal(restored, original)


def _report_metrics() -> dict[str, object]:
    return {
        "loss": 0.6,
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "num_samples": 4,
        "per_class": [
            {
                "class_index": 0,
                "class_name": "healthy",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.74,
                "support": 2,
            },
            {
                "class_index": 1,
                "class_name": "rust",
                "precision": 0.7,
                "recall": 0.8,
                "f1": 0.74,
                "support": 2,
            },
        ],
        "confusion_matrix": [[1, 1], [0, 2]],
    }


def test_report_writing_creates_json_and_markdown(tmp_path: Path) -> None:
    results = {
        "schema_version": 1,
        "dataset": {
            "num_classes": 2,
            "split_sizes": {"train": 8, "val": 4, "test": 4},
        },
        "device": "cpu",
        "hyperparameters": {"epochs": 1, "batch_size": 2},
        "best_validation_epoch": 1,
        "validation_metrics": _report_metrics(),
        "test_metrics": _report_metrics(),
        "checkpoints": {"best": "/tmp/best_model.pt"},
    }

    json_path, markdown_path = write_baseline_results(
        results,
        tmp_path / "results.json",
        tmp_path / "results.md",
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == results
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Baseline CNN Results" in markdown
    assert "not the final model" in markdown
    assert "## Weakest Test Classes" in markdown


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_split(
    processed_dir: Path,
    split: str,
    rows: list[dict[str, str | int]],
) -> None:
    with (processed_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_training_cli_smoke_creates_all_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    processed_dir.mkdir()
    mapping = {"healthy": 0, "rust": 1}
    (processed_dir / "class_to_index.json").write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )
    split_counts = {"train": 4, "val": 2, "test": 2}
    colors = {"healthy": (30, 150, 60), "rust": (150, 60, 30)}
    for split, count in split_counts.items():
        rows = []
        for index in range(count):
            label = "healthy" if index % 2 == 0 else "rust"
            filepath = f"{label}/{split}_{index}.png"
            _create_image(raw_dir / filepath, colors[label])
            rows.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "class_index": mapping[label],
                    "width": 16,
                    "height": 16,
                }
            )
        _write_split(processed_dir, split, rows)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(Path(__file__).parents[1] / "src"),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plant_disease_visionops.training.train_baseline",
            "--raw-data-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--out-dir",
            str(models_dir),
            "--reports-dir",
            str(reports_dir),
            "--figures-dir",
            str(figures_dir),
            "--image-size",
            "16",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--learning-rate",
            "0.001",
            "--num-workers",
            "0",
            "--seed",
            "42",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )

    expected_files = [
        models_dir / "best_model.pt",
        models_dir / "last_model.pt",
        models_dir / "history.json",
        reports_dir / "baseline_cnn_results.json",
        reports_dir / "baseline_cnn_results.md",
        figures_dir / "baseline_cnn_confusion_matrix.png",
        figures_dir / "baseline_cnn_training_curves.png",
    ]
    assert result.returncode == 0, result.stderr
    assert "Epoch 1/1" in result.stdout
    assert all(path.is_file() for path in expected_files)
    saved_results = json.loads(expected_files[3].read_text(encoding="utf-8"))
    assert saved_results["dataset"]["split_sizes"] == split_counts
    assert saved_results["best_validation_epoch"] == 1
    assert saved_results["device"] == "cpu"
