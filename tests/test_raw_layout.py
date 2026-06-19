import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from plant_disease_visionops.data.prepare_raw_layout import main as prepare_main
from plant_disease_visionops.data.raw_layout import (
    OutputDirectoryNotEmptyError,
    analyze_raw_layout,
    prepare_raw_dataset,
)
from plant_disease_visionops.data.validate_layout import main as validate_main


def _create_image(
    path: Path,
    color: tuple[int, int, int] = (40, 140, 70),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def test_valid_flat_layout_lists_classes_counts_and_unsupported_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    _create_image(raw_dir / "healthy" / "one.jpg")
    _create_image(raw_dir / "healthy" / "two.PNG")
    _create_image(raw_dir / "rust" / "three.jpeg")
    (raw_dir / "healthy" / "notes.txt").write_text("metadata", encoding="utf-8")

    analysis = analyze_raw_layout(raw_dir)

    assert [(item.class_name, item.candidate_images) for item in analysis.class_folders] == [
        ("healthy", 2),
        ("rust", 1),
    ]
    assert analysis.total_candidate_images == 3
    assert analysis.unsupported_files == ("healthy/notes.txt",)
    assert analysis.nested_image_folders == ()
    assert validate_main(["--data-dir", str(raw_dir)]) == 0
    output = capsys.readouterr().out
    assert "healthy: 2 candidate images" in output
    assert "WARNING: 1 unsupported files will be ignored" in output
    assert "Layout is usable: 2 classes and 3 candidate images." in output


def test_nested_layout_warns_and_exits_with_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    _create_image(raw_dir / "train" / "healthy" / "one.jpg")
    _create_image(raw_dir / "valid" / "rust" / "two.png")

    analysis = analyze_raw_layout(raw_dir)

    assert analysis.class_folders == ()
    assert analysis.nested_image_folders == ("train/healthy", "valid/rust")
    assert validate_main(["--data-dir", str(raw_dir)]) == 2
    captured = capsys.readouterr()
    assert "nested folders will not be scanned" in captured.out
    assert "No usable class folders found" in captured.err
    assert "Expected layout" in captured.err


def test_prepare_copy_mode_handles_collisions_and_writes_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "external" / "plant_village"
    output_dir = tmp_path / "raw"
    manifest_path = tmp_path / "reports" / "raw_layout_manifest.json"
    _create_image(input_dir / "train" / "healthy" / "leaf.jpg", color=(10, 20, 30))
    _create_image(input_dir / "test" / "healthy" / "leaf.jpg", color=(40, 50, 60))
    _create_image(input_dir / "valid" / "rust" / "spot.png")

    result = prepare_raw_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        mode="copy",
        manifest_path=manifest_path,
    )

    healthy_files = sorted((output_dir / "healthy").iterdir())
    assert len(result.files) == 3
    assert len(healthy_files) == 2
    assert healthy_files[0].name != healthy_files[1].name
    assert all(path.name.startswith("leaf__") for path in healthy_files)
    assert all(path.is_file() and not path.is_symlink() for path in healthy_files)
    assert (output_dir / "rust").is_dir()
    prepared_analysis = analyze_raw_layout(output_dir)
    assert prepared_analysis.is_usable
    assert prepared_analysis.total_candidate_images == 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "copy"
    assert manifest["total_images"] == 3
    assert manifest["class_counts"] == {"healthy": 2, "rust": 1}
    assert len({item["destination"] for item in manifest["files"]}) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="symlink behavior requires Linux semantics")
def test_prepare_symlink_mode_creates_links_to_source_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "external" / "dataset"
    output_dir = tmp_path / "raw"
    source_image = input_dir / "color" / "healthy" / "leaf.jpg"
    _create_image(source_image)

    result = prepare_raw_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        mode="symlink",
        manifest_path=tmp_path / "manifest.json",
    )

    destination = output_dir / result.files[0].destination
    assert destination.is_symlink()
    assert destination.resolve() == source_image.resolve()


def test_prepare_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "external" / "dataset"
    output_dir = tmp_path / "raw"
    _create_image(input_dir / "train" / "healthy" / "leaf.jpg")
    existing_file = output_dir / "existing" / "keep.jpg"
    _create_image(existing_file)

    with pytest.raises(OutputDirectoryNotEmptyError, match="Pass --overwrite"):
        prepare_raw_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            manifest_path=tmp_path / "first_manifest.json",
        )

    assert existing_file.is_file()
    result = prepare_raw_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=tmp_path / "second_manifest.json",
        overwrite=True,
    )
    assert not existing_file.exists()
    assert (output_dir / result.files[0].destination).is_file()


def test_prepare_cli_executes_with_custom_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "external" / "dataset"
    output_dir = tmp_path / "raw"
    manifest_path = tmp_path / "reports" / "manifest.json"
    _create_image(input_dir / "PlantVillage" / "healthy" / "leaf.png")

    exit_code = prepare_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "copy",
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    assert manifest_path.is_file()
    assert "Prepared 1 images across 1 classes using copy mode." in capsys.readouterr().out
