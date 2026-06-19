import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from plant_disease_visionops.data.audit_dataset import audit_dataset


def _create_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 160, 80)).save(path)


def test_audit_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    out_dir = tmp_path / "reports"
    _create_image(data_dir / "healthy" / "one.jpg", (10, 20))
    _create_image(data_dir / "healthy" / "two.png", (30, 40))
    _create_image(data_dir / "rust" / "three.jpeg", (20, 30))
    corrupt_path = data_dir / "rust" / "broken.png"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"broken")

    audit = audit_dataset(data_dir, out_dir)

    assert audit["total_images"] == 4
    assert audit["valid_images"] == 3
    assert audit["invalid_images"] == 1
    assert audit["class_counts"] == {"healthy": 2, "rust": 2}
    assert audit["valid_class_counts"] == {"healthy": 2, "rust": 1}
    assert audit["image_size_statistics"]["width_pixels"] == {
        "min": 10,
        "max": 30,
        "mean": 20.0,
        "median": 20,
    }
    assert audit["class_imbalance"]["max_to_min_ratio"] == 2.0
    assert audit["invalid_image_details"][0]["path"] == "rust/broken.png"

    json_report = json.loads((out_dir / "data_audit.json").read_text(encoding="utf-8"))
    markdown_report = (out_dir / "data_audit.md").read_text(encoding="utf-8")
    assert json_report == audit
    assert "# Dataset Audit" in markdown_report
    assert "rust/broken.png" in markdown_report


def test_cli_runs_end_to_end_on_toy_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    out_dir = tmp_path / "reports"
    _create_image(data_dir / "healthy" / "leaf.png", (16, 12))
    environment = os.environ.copy()
    source_dir = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_dir), environment.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plant_disease_visionops.data.audit_dataset",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Audited 1 images across 1 classes (0 invalid)." in result.stdout
    assert (out_dir / "data_audit.json").is_file()
    assert (out_dir / "data_audit.md").is_file()


def test_cli_returns_nonzero_for_empty_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    environment = os.environ.copy()
    source_dir = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = str(source_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plant_disease_visionops.data.audit_dataset",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "reports"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "No supported images found" in result.stderr
    assert not (tmp_path / "reports").exists()
