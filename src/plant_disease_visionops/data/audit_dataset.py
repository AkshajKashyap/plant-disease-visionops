"""Audit a class-organized image dataset and write JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plant_disease_visionops.data.discovery import DatasetDiscoveryError, DatasetScan, scan_dataset

JSON_REPORT_NAME = "data_audit.json"
MARKDOWN_REPORT_NAME = "data_audit.md"


def _numeric_statistics(values: Sequence[int | float]) -> dict[str, int | float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def build_audit(scan: DatasetScan) -> dict[str, Any]:
    """Build a JSON-serializable audit summary from a dataset scan."""
    total_counts = Counter(image.class_name for image in scan.discovered_images)
    valid_counts = Counter(image.class_name for image in scan.valid_images)
    invalid_counts = Counter(image.class_name for image in scan.invalid_images)
    class_names = sorted(total_counts, key=str.casefold)

    valid_total = len(scan.valid_images)
    per_class = {
        class_name: {
            "total_images": total_counts[class_name],
            "valid_images": valid_counts[class_name],
            "invalid_images": invalid_counts[class_name],
            "percentage_of_valid_images": (
                round(valid_counts[class_name] / valid_total * 100, 2) if valid_total else 0.0
            ),
        }
        for class_name in class_names
    }

    counts = [valid_counts[class_name] for class_name in class_names]
    majority_count = max(counts)
    minority_count = min(counts)
    imbalance = {
        "basis": "valid_images",
        "number_of_classes": len(class_names),
        "majority_classes": [
            class_name for class_name in class_names if valid_counts[class_name] == majority_count
        ],
        "majority_count": majority_count,
        "minority_classes": [
            class_name for class_name in class_names if valid_counts[class_name] == minority_count
        ],
        "minority_count": minority_count,
        "max_to_min_ratio": (
            round(majority_count / minority_count, 2) if minority_count > 0 else None
        ),
        "mean_images_per_class": round(statistics.fmean(counts), 2),
        "median_images_per_class": round(statistics.median(counts), 2),
    }

    widths = [image.width for image in scan.valid_images]
    heights = [image.height for image in scan.valid_images]
    pixel_counts = [image.width * image.height for image in scan.valid_images]
    aspect_ratios = [image.width / image.height for image in scan.valid_images]

    return {
        "schema_version": 1,
        "data_directory": str(scan.data_dir.resolve()),
        "total_images": len(scan.discovered_images),
        "valid_images": valid_total,
        "invalid_images": len(scan.invalid_images),
        "class_counts": dict(total_counts),
        "valid_class_counts": {name: valid_counts[name] for name in class_names},
        "classes": per_class,
        "image_size_statistics": {
            "basis": "valid_images",
            "width_pixels": _numeric_statistics(widths),
            "height_pixels": _numeric_statistics(heights),
            "pixel_count": _numeric_statistics(pixel_counts),
            "aspect_ratio": _numeric_statistics(aspect_ratios),
        },
        "class_imbalance": imbalance,
        "invalid_image_details": [
            {
                "path": image.path.relative_to(scan.data_dir).as_posix(),
                "class_name": image.class_name,
                "error": image.error,
            }
            for image in scan.invalid_images
        ],
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(audit: dict[str, Any]) -> str:
    """Render a human-readable Markdown version of an audit summary."""
    imbalance = audit["class_imbalance"]
    size_stats = audit["image_size_statistics"]
    lines = [
        "# Dataset Audit",
        "",
        f"- Data directory: `{audit['data_directory']}`",
        f"- Total image files: {audit['total_images']}",
        f"- Valid images: {audit['valid_images']}",
        f"- Invalid/corrupt images: {audit['invalid_images']}",
        f"- Classes: {imbalance['number_of_classes']}",
        "",
        "## Class Distribution",
        "",
        "Counts include every supported image file; percentages use valid images only.",
        "",
        "| Class | Total | Valid | Invalid | Valid share |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, class_summary in audit["classes"].items():
        lines.append(
            f"| {_markdown_cell(class_name)} | {class_summary['total_images']} | "
            f"{class_summary['valid_images']} | {class_summary['invalid_images']} | "
            f"{class_summary['percentage_of_valid_images']:.2f}% |"
        )

    ratio = imbalance["max_to_min_ratio"]
    ratio_text = (
        f"{ratio:.2f}" if ratio is not None else "not defined (a class has no valid images)"
    )
    lines.extend(
        [
            "",
            "## Class Imbalance",
            "",
            "Imbalance statistics use valid images only.",
            "",
            f"- Majority class(es): {', '.join(map(_markdown_cell, imbalance['majority_classes']))} "
            f"({imbalance['majority_count']} images)",
            f"- Minority class(es): {', '.join(map(_markdown_cell, imbalance['minority_classes']))} "
            f"({imbalance['minority_count']} images)",
            f"- Maximum-to-minimum ratio: {ratio_text}",
            f"- Mean images per class: {imbalance['mean_images_per_class']:.2f}",
            f"- Median images per class: {imbalance['median_images_per_class']:.2f}",
            "",
            "## Image Sizes",
            "",
        ]
    )
    if size_stats["width_pixels"] is None:
        lines.append("No valid images were available for image size statistics.")
    else:
        lines.extend(
            [
                "Statistics use valid images only.",
                "",
                "| Metric | Min | Max | Mean | Median |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "width_pixels": "Width (pixels)",
            "height_pixels": "Height (pixels)",
            "pixel_count": "Pixel count",
            "aspect_ratio": "Aspect ratio (width / height)",
        }
        for key, label in labels.items():
            stats = size_stats[key]
            lines.append(
                f"| {label} | {stats['min']} | {stats['max']} | "
                f"{stats['mean']} | {stats['median']} |"
            )

    lines.extend(["", "## Invalid Images", ""])
    if not audit["invalid_image_details"]:
        lines.append("No invalid or corrupt images were detected.")
    else:
        lines.extend(["| Path | Class | Error |", "|---|---|---|"])
        for invalid in audit["invalid_image_details"]:
            lines.append(
                f"| `{_markdown_cell(invalid['path'])}` | "
                f"{_markdown_cell(invalid['class_name'])} | "
                f"{_markdown_cell(invalid['error'])} |"
            )
    return "\n".join(lines) + "\n"


def write_audit_reports(audit: dict[str, Any], out_dir: Path | str) -> tuple[Path, Path]:
    """Write JSON and Markdown audit reports and return their paths."""
    output_directory = Path(out_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / JSON_REPORT_NAME
    markdown_path = output_directory / MARKDOWN_REPORT_NAME
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def audit_dataset(data_dir: Path | str, out_dir: Path | str) -> dict[str, Any]:
    """Scan a dataset, write both audit reports, and return the audit summary."""
    audit = build_audit(scan_dataset(data_dir))
    write_audit_reports(audit, out_dir)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a class-organized plant disease image dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Dataset root containing one directory per class (default: data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for data_audit.json and data_audit.md (default: reports).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset audit command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        audit = audit_dataset(args.data_dir, args.out_dir)
    except DatasetDiscoveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Audited {audit['total_images']} images across "
        f"{audit['class_imbalance']['number_of_classes']} classes "
        f"({audit['invalid_images']} invalid)."
    )
    print(f"Reports written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
