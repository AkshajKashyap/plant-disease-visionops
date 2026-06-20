import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from plant_disease_visionops.evaluation.compare_robustness import main as compare_main
from plant_disease_visionops.evaluation.corruptions import (
    CORRUPTION_NAMES,
    build_corrupted_eval_transform,
    create_corruption,
)
from plant_disease_visionops.evaluation.evaluate_robustness import main as robustness_main
from plant_disease_visionops.evaluation.robustness import (
    RobustnessConfig,
    run_robustness_evaluation,
)
from plant_disease_visionops.models.factory import create_model
from plant_disease_visionops.training.checkpointing import save_checkpoint

CSV_COLUMNS = ("filepath", "label", "class_index", "width", "height")


def _gradient_image(size: tuple[int, int] = (32, 24)) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)
    return image


def test_corruptions_preserve_image_and_eval_tensor_shapes() -> None:
    image = _gradient_image()

    for corruption_name in CORRUPTION_NAMES:
        corrupted = create_corruption(corruption_name, severity=2, seed=42)(image)
        tensor = build_corrupted_eval_transform(
            image_size=16,
            corruption_name=corruption_name,
            severity=2,
            seed=42,
        )(image)
        assert corrupted.size == image.size
        assert tensor.shape == (3, 16, 16)


def test_corruption_severity_changes_output() -> None:
    image = _gradient_image()

    mild = create_corruption("brightness_decrease", severity=1)(image)
    severe = create_corruption("brightness_decrease", severity=3)(image)

    assert mild.tobytes() != severe.tobytes()


def test_gaussian_noise_is_deterministic_for_fixed_seed() -> None:
    image = _gradient_image()

    first = create_corruption("gaussian_noise", severity=2, seed=17)(image)
    second = create_corruption("gaussian_noise", severity=2, seed=17)(image)
    different_seed = create_corruption("gaussian_noise", severity=2, seed=18)(image)

    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != different_seed.tobytes()


def _create_toy_checkpoint_and_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    mapping = {"healthy": 0, "rust": 1}
    (processed_dir / "class_to_index.json").write_text(json.dumps(mapping), encoding="utf-8")
    rows = []
    for index, label in enumerate(mapping):
        relative_path = f"{label}/test_{index}.png"
        image_path = raw_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        color = (30, 150, 60) if label == "healthy" else (150, 60, 30)
        Image.new("RGB", (32, 32), color=color).save(image_path)
        rows.append(
            {
                "filepath": relative_path,
                "label": label,
                "class_index": mapping[label],
                "width": 32,
                "height": 32,
            }
        )
    with (processed_dir / "test.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    model = create_model("baseline_cnn", num_classes=2, pretrained=False)
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


def test_robustness_evaluator_writes_honest_reports_and_figures(tmp_path: Path) -> None:
    raw_dir, processed_dir, checkpoint_path = _create_toy_checkpoint_and_data(tmp_path)
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    config = RobustnessConfig(
        checkpoint=checkpoint_path,
        experiment_name="tiny_baseline",
        raw_data_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        split="test",
        image_size=16,
        batch_size=2,
        num_workers=0,
        seed=42,
        device="cpu",
        corruptions=("brightness_decrease", "gaussian_noise"),
        severities=(1,),
    )

    results = run_robustness_evaluation(config)

    expected_files = [
        reports_dir / "tiny_baseline_robustness.json",
        reports_dir / "tiny_baseline_robustness.md",
        figures_dir / "tiny_baseline_robustness_accuracy.png",
        figures_dir / "tiny_baseline_robustness_macro_f1.png",
    ]
    assert all(path.is_file() for path in expected_files)
    assert results["experiment_name"] == "tiny_baseline"
    assert results["clean_metrics"]["num_samples"] == 2
    assert len(results["corruption_results"]) == 2
    assert all("macro_f1_drop" in item for item in results["corruption_results"])
    saved = json.loads(expected_files[0].read_text(encoding="utf-8"))
    assert saved == results
    markdown = expected_files[1].read_text(encoding="utf-8")
    assert "## Worst Corruption Settings" in markdown
    assert "may not guarantee robustness" in markdown


def _fake_robustness(experiment_name: str, clean_f1: float, corrupted_f1: float) -> dict:
    return {
        "experiment_name": experiment_name,
        "clean_metrics": {"accuracy": clean_f1 + 0.01, "macro_f1": clean_f1},
        "corruption_results": [
            {
                "corruption": "gaussian_noise",
                "severity": 3,
                "accuracy": corrupted_f1 + 0.01,
                "macro_f1": corrupted_f1,
                "macro_f1_drop": clean_f1 - corrupted_f1,
            }
        ],
    }


def test_robustness_comparison_reads_fake_reports_and_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "baseline_robustness.json").write_text(
        json.dumps(_fake_robustness("baseline", 0.90, 0.60)),
        encoding="utf-8",
    )
    (reports_dir / "resnet_robustness.json").write_text(
        json.dumps(_fake_robustness("resnet", 0.98, 0.80)),
        encoding="utf-8",
    )
    markdown_path = reports_dir / "robustness_comparison.md"
    json_path = reports_dir / "robustness_comparison.json"

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
        "baseline",
        "resnet",
    ]
    assert comparison["experiments"][0]["largest_macro_f1_drop"] == pytest.approx(0.30)
    assert "# Robustness Comparison" in markdown_path.read_text(encoding="utf-8")
    assert "Compared robustness for 2 experiments." in capsys.readouterr().out


def test_robustness_cli_fails_clearly_for_missing_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = robustness_main(
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
