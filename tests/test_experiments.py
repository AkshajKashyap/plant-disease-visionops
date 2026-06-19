import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from plant_disease_visionops.evaluation.compare_experiments import main as compare_main
from plant_disease_visionops.models.factory import create_model, create_resnet18

CSV_COLUMNS = ("filepath", "label", "class_index", "width", "height")


def test_resnet18_forward_pass_shape_without_pretrained_weights() -> None:
    model = create_resnet18(num_classes=4, pretrained=False, freeze_backbone=False)
    model.eval()

    with torch.inference_mode():
        logits = model(torch.rand(2, 3, 64, 64))

    assert logits.shape == (2, 4)
    assert model.fc.out_features == 4


def test_freeze_backbone_only_leaves_resnet_classifier_trainable() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )

    parameters = dict(model.named_parameters())
    assert parameters["fc.weight"].requires_grad
    assert parameters["fc.bias"].requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in parameters.items()
        if not name.startswith("fc.")
    )


def _create_toy_splits(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True)
    mapping = {"healthy": 0, "rust": 1}
    (processed_dir / "class_to_index.json").write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )
    colors = {"healthy": (30, 150, 60), "rust": (150, 60, 30)}
    for split in ("train", "val", "test"):
        rows = []
        for index, label in enumerate(("healthy", "rust")):
            filepath = f"{label}/{split}_{index}.png"
            image_path = raw_dir / filepath
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=colors[label]).save(image_path)
            rows.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "class_index": mapping[label],
                    "width": 32,
                    "height": 32,
                }
            )
        with (processed_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return raw_dir, processed_dir


@pytest.mark.parametrize(
    ("model_name", "experiment_name", "freeze_backbone"),
    [
        ("baseline_cnn", "tiny_baseline", "false"),
        ("resnet18", "tiny_resnet18", "true"),
    ],
)
def test_train_experiment_cli_smoke_without_network(
    tmp_path: Path,
    model_name: str,
    experiment_name: str,
    freeze_backbone: str,
) -> None:
    raw_dir, processed_dir = _create_toy_splits(tmp_path)
    models_dir = tmp_path / "models" / experiment_name
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
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
            "plant_disease_visionops.training.train_experiment",
            "--model-name",
            model_name,
            "--experiment-name",
            experiment_name,
            "--pretrained",
            "false",
            "--freeze-backbone",
            freeze_backbone,
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
            "32",
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
        timeout=120,
    )

    expected_files = [
        models_dir / "best_model.pt",
        models_dir / "last_model.pt",
        models_dir / "history.json",
        reports_dir / f"{experiment_name}_results.json",
        reports_dir / f"{experiment_name}_results.md",
        figures_dir / f"{experiment_name}_confusion_matrix.png",
        figures_dir / f"{experiment_name}_training_curves.png",
    ]
    assert result.returncode == 0, result.stderr
    assert all(path.is_file() for path in expected_files)
    report = json.loads(expected_files[3].read_text(encoding="utf-8"))
    assert report["experiment_name"] == experiment_name
    assert report["model_name"] == model_name
    assert report["pretrained"] is False
    assert report["freeze_backbone"] is (freeze_backbone == "true")
    assert report["best_validation_epoch"] == 1
    assert len(report["weakest_test_classes"]) == 2


def _fake_result(
    experiment_name: str,
    model_name: str,
    test_accuracy: float,
    test_macro_f1: float,
    val_macro_f1: float,
) -> dict[str, object]:
    return {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "pretrained": model_name == "resnet18",
        "freeze_backbone": False,
        "epochs": 3,
        "image_size": 128,
        "batch_size": 16,
        "validation_metrics": {"macro_f1": val_macro_f1},
        "test_metrics": {"accuracy": test_accuracy, "macro_f1": test_macro_f1},
    }


def test_comparison_cli_reads_result_jsons_and_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    baseline = {
        "model": {"name": "baseline_cnn"},
        "hyperparameters": {"epochs": 3, "image_size": 128, "batch_size": 16},
        "validation_metrics": {"macro_f1": 0.80},
        "test_metrics": {"accuracy": 0.81, "macro_f1": 0.79},
    }
    (reports_dir / "baseline_cnn_results.json").write_text(json.dumps(baseline), encoding="utf-8")
    (reports_dir / "resnet_transfer_results.json").write_text(
        json.dumps(_fake_result("resnet_transfer", "resnet18", 0.91, 0.90, 0.89)),
        encoding="utf-8",
    )
    markdown_path = reports_dir / "experiment_comparison.md"
    json_path = reports_dir / "experiment_comparison.json"

    exit_code = compare_main(
        [
            "--reports-dir",
            str(reports_dir),
            "--out-md",
            str(markdown_path),
            "--out-json",
            str(json_path),
        ]
    )

    assert exit_code == 0
    comparison = json.loads(json_path.read_text(encoding="utf-8"))
    assert [item["experiment_name"] for item in comparison["experiments"]] == [
        "baseline_cnn",
        "resnet_transfer",
    ]
    assert comparison["experiments"][0]["pretrained"] is False
    assert comparison["experiments"][1]["test_macro_f1"] == 0.90
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Experiment Comparison" in markdown
    assert "resnet18" in markdown
    assert "Compared 2 experiments." in capsys.readouterr().out
