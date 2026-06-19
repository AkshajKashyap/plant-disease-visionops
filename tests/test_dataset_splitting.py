import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from plant_disease_visionops.data.discovery import scan_dataset
from plant_disease_visionops.data.make_splits import generate_split_files
from plant_disease_visionops.data.splitting import (
    InsufficientClassImagesError,
    InvalidSplitRatiosError,
    SplitRatios,
    create_stratified_splits,
)


def _create_image(path: Path, size: tuple[int, int] = (18, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(60, 140, 70)).save(path)


def _create_class(data_dir: Path, class_name: str, image_count: int) -> None:
    for index in range(image_count):
        _create_image(
            data_dir / class_name / f"image_{index:03d}.png",
            size=(18 + index, 12 + index),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_stratified_splits_exclude_corrupt_images_and_have_no_overlap(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_class(data_dir, "healthy", 20)
    _create_class(data_dir, "rust", 20)
    corrupt_path = data_dir / "rust" / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image")

    splits = create_stratified_splits(
        scan_dataset(data_dir),
        ratios=SplitRatios(train=0.7, val=0.15, test=0.15),
        seed=42,
    )

    expected_per_class = {"train": 14, "val": 3, "test": 3}
    all_paths: set[str] = set()
    for split_name, expected_count in expected_per_class.items():
        records = splits.records[split_name]
        assert Counter(record.label for record in records) == {
            "healthy": expected_count,
            "rust": expected_count,
        }
        split_paths = {record.filepath for record in records}
        assert split_paths.isdisjoint(all_paths)
        all_paths.update(split_paths)

    assert len(all_paths) == 40
    assert "rust/corrupt.jpg" not in all_paths
    assert splits.invalid_image_paths == ("rust/corrupt.jpg",)
    assert splits.class_to_index == {"healthy": 0, "rust": 1}


def test_split_files_have_expected_schema_and_summaries(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    _create_class(data_dir, "healthy", 8)
    _create_class(data_dir, "rust", 8)

    summary = generate_split_files(
        data_dir=data_dir,
        out_dir=out_dir,
        reports_dir=reports_dir,
        ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
        seed=7,
    )

    for split_name, expected_total in {"train": 8, "val": 4, "test": 4}.items():
        rows = _read_csv(out_dir / f"{split_name}.csv")
        assert len(rows) == expected_total
        assert list(rows[0]) == ["filepath", "label", "class_index", "width", "height"]
        assert all(not Path(row["filepath"]).is_absolute() for row in rows)
        assert {row["class_index"] for row in rows} == {"0", "1"}

    mapping = json.loads((out_dir / "class_to_index.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((out_dir / "split_summary.json").read_text(encoding="utf-8"))
    markdown = (reports_dir / "split_summary.md").read_text(encoding="utf-8")
    assert mapping == {"healthy": 0, "rust": 1}
    assert saved_summary == summary
    assert saved_summary["leakage_check"]["passed"] is True
    assert "# Dataset Split Summary" in markdown


def test_same_seed_produces_identical_csv_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_class(data_dir, "healthy", 20)
    _create_class(data_dir, "rust", 20)
    ratios = SplitRatios(train=0.7, val=0.15, test=0.15)

    for run_name in ("first", "second"):
        generate_split_files(
            data_dir=data_dir,
            out_dir=tmp_path / run_name / "processed",
            reports_dir=tmp_path / run_name / "reports",
            ratios=ratios,
            seed=42,
        )

    processed_files = (
        "train.csv",
        "val.csv",
        "test.csv",
        "class_to_index.json",
        "split_summary.json",
    )
    for filename in processed_files:
        first = (tmp_path / "first" / "processed" / filename).read_bytes()
        second = (tmp_path / "second" / "processed" / filename).read_bytes()
        assert first == second

    first_markdown = (tmp_path / "first" / "reports" / "split_summary.md").read_bytes()
    second_markdown = (tmp_path / "second" / "reports" / "split_summary.md").read_bytes()
    assert first_markdown == second_markdown


def test_different_seed_changes_split_membership(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_class(data_dir, "healthy", 20)
    _create_class(data_dir, "rust", 20)
    scan = scan_dataset(data_dir)
    ratios = SplitRatios(train=0.7, val=0.15, test=0.15)

    first = create_stratified_splits(scan, ratios=ratios, seed=42)
    second = create_stratified_splits(scan, ratios=ratios, seed=99)

    first_train = {record.filepath for record in first.records["train"]}
    second_train = {record.filepath for record in second.records["train"]}
    assert first_train != second_train


def test_too_small_class_has_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _create_class(data_dir, "healthy", 5)
    _create_class(data_dir, "rare_disease", 2)

    with pytest.raises(
        InsufficientClassImagesError,
        match="Class 'rare_disease' has 2 valid images",
    ):
        create_stratified_splits(
            scan_dataset(data_dir),
            ratios=SplitRatios(),
            seed=42,
        )


@pytest.mark.parametrize(
    "ratios, message",
    [
        (SplitRatios(train=0.7, val=0.2, test=0.2), "sum to 1.0"),
        (SplitRatios(train=0.8, val=-0.1, test=0.3), "val ratio"),
        (SplitRatios(train=1.0, val=0.0, test=0.0), "train ratio"),
    ],
)
def test_invalid_ratios_have_clear_errors(ratios: SplitRatios, message: str) -> None:
    with pytest.raises(InvalidSplitRatiosError, match=message):
        ratios.validate()


def test_make_splits_cli_runs_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    _create_class(data_dir, "healthy", 4)
    _create_class(data_dir, "rust", 4)
    environment = os.environ.copy()
    source_dir = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_dir), environment.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plant_disease_visionops.data.make_splits",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--reports-dir",
            str(reports_dir),
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Created splits from 8 valid images: train=4, val=2, test=2." in result.stdout
    assert (out_dir / "train.csv").is_file()
    assert (reports_dir / "split_summary.md").is_file()
