import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from plant_disease_visionops.data.dataset import (
    EmptySplitError,
    ImageClassificationDataset,
    InvalidLabelError,
    MissingImageFileError,
    SplitCsvNotFoundError,
)
from plant_disease_visionops.data.loaders import create_dataloaders
from plant_disease_visionops.data.transforms import build_eval_transform

CSV_COLUMNS = ("filepath", "label", "class_index", "width", "height")


def _create_image(
    path: Path,
    mode: str = "RGB",
    size: tuple[int, int] = (24, 18),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color: int | tuple[int, int, int] = 128 if mode == "L" else (40, 130, 70)
    Image.new(mode, size, color=color).save(path)


def _write_mapping(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "class_to_index.json").write_text(
        json.dumps({"healthy": 0, "rust": 1}),
        encoding="utf-8",
    )


def _write_split(processed_dir: Path, split: str, rows: list[dict[str, str | int]]) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / f"{split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _row(filepath: str, label: str, class_index: int) -> dict[str, str | int]:
    return {
        "filepath": filepath,
        "label": label,
        "class_index": class_index,
        "width": 24,
        "height": 18,
    }


def test_dataset_length_rgb_conversion_label_and_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _create_image(raw_dir / "healthy" / "gray.png", mode="L")
    _write_mapping(processed_dir)
    csv_path = _write_split(processed_dir, "train", [_row("healthy/gray.png", "healthy", 0)])

    dataset = ImageClassificationDataset(csv_path=csv_path, raw_data_dir=raw_dir)
    image, class_index, metadata = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 18, 24)
    assert torch.equal(image[0], image[1])
    assert torch.equal(image[1], image[2])
    assert class_index == 0
    assert metadata == {
        "filepath": "healthy/gray.png",
        "label": "healthy",
        "class_index": 0,
        "width": 24,
        "height": 18,
    }


def test_eval_transform_returns_configured_tensor_shape(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _create_image(raw_dir / "rust" / "leaf.png", size=(40, 20))
    _write_mapping(processed_dir)
    csv_path = _write_split(processed_dir, "val", [_row("rust/leaf.png", "rust", 1)])
    dataset = ImageClassificationDataset(
        csv_path=csv_path,
        raw_data_dir=raw_dir,
        transform=build_eval_transform(image_size=32),
    )

    image, class_index, _ = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert class_index == 1


def test_dataset_reports_missing_csv_and_image_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_mapping(processed_dir)

    with pytest.raises(SplitCsvNotFoundError, match="Split CSV does not exist"):
        ImageClassificationDataset(processed_dir / "missing.csv", raw_dir)

    csv_path = _write_split(
        processed_dir,
        "train",
        [_row("healthy/missing.png", "healthy", 0)],
    )
    dataset = ImageClassificationDataset(csv_path, raw_dir)
    with pytest.raises(MissingImageFileError, match="does not exist"):
        dataset[0]


def test_dataset_rejects_invalid_labels_and_empty_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_mapping(processed_dir)
    invalid_csv = _write_split(
        processed_dir,
        "train",
        [_row("blight/leaf.png", "blight", 0)],
    )

    with pytest.raises(InvalidLabelError, match="not present in the class mapping"):
        ImageClassificationDataset(invalid_csv, raw_dir)

    empty_csv = _write_split(processed_dir, "val", [])
    with pytest.raises(EmptySplitError, match="contains no image rows"):
        ImageClassificationDataset(empty_csv, raw_dir)


def test_dataloader_factories_create_expected_batch_shapes_and_samplers(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_mapping(processed_dir)
    rows_by_split = {
        "train": [
            _row("healthy/train_1.png", "healthy", 0),
            _row("rust/train_2.png", "rust", 1),
        ],
        "val": [
            _row("healthy/val_1.png", "healthy", 0),
            _row("rust/val_2.png", "rust", 1),
        ],
        "test": [
            _row("healthy/test_1.png", "healthy", 0),
            _row("rust/test_2.png", "rust", 1),
        ],
    }
    for split, rows in rows_by_split.items():
        for row in rows:
            _create_image(raw_dir / str(row["filepath"]))
        _write_split(processed_dir, split, rows)

    loaders = create_dataloaders(
        raw_data_dir=raw_dir,
        processed_dir=processed_dir,
        batch_size=2,
        num_workers=0,
        image_size=32,
    )
    images, class_indices, metadata = next(iter(loaders.train))

    assert images.shape == (2, 3, 32, 32)
    assert class_indices.shape == (2,)
    assert sorted(class_indices.tolist()) == [0, 1]
    assert len(metadata["filepath"]) == 2
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.val.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_inspect_batch_cli_prints_summary_and_saves_grid(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    figures_dir = tmp_path / "figures"
    _write_mapping(processed_dir)
    rows = [
        _row("healthy/one.png", "healthy", 0),
        _row("rust/two.png", "rust", 1),
    ]
    for row in rows:
        _create_image(raw_dir / str(row["filepath"]))
    _write_split(processed_dir, "train", rows)
    environment = os.environ.copy()
    source_dir = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_dir), environment.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plant_disease_visionops.data.inspect_batch",
            "--raw-data-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--split",
            "train",
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--out-dir",
            str(figures_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    output_path = figures_dir / "sample_batch_train.png"
    assert result.returncode == 0, result.stderr
    assert "Batch tensor shape: (2, 3, 32, 32)" in result.stdout
    assert 'Batch class counts: {"healthy": 1, "rust": 1}' in result.stdout
    assert output_path.is_file()
    with Image.open(output_path) as grid_image:
        assert grid_image.format == "PNG"
        assert grid_image.mode == "RGB"
