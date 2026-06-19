"""Generate train, validation, and test CSV metadata for a local dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plant_disease_visionops.data.discovery import DatasetDiscoveryError, scan_dataset
from plant_disease_visionops.data.splitting import (
    SPLIT_NAMES,
    DatasetSplits,
    SplitGenerationError,
    SplitRatios,
    create_stratified_splits,
)

CSV_FIELDNAMES = ("filepath", "label", "class_index", "width", "height")
CLASS_MAPPING_NAME = "class_to_index.json"
SUMMARY_JSON_NAME = "split_summary.json"
SUMMARY_MARKDOWN_NAME = "split_summary.md"


def build_split_summary(splits: DatasetSplits) -> dict[str, Any]:
    """Build a JSON-serializable description of generated metadata splits."""
    split_summaries = {}
    all_paths: set[str] = set()
    overlap_count = 0
    for split_name in SPLIT_NAMES:
        records = splits.records[split_name]
        paths = {record.filepath for record in records}
        overlap_count += len(paths & all_paths)
        all_paths.update(paths)
        class_counts = Counter(record.label for record in records)
        split_summaries[split_name] = {
            "total_images": len(records),
            "class_counts": {
                class_name: class_counts[class_name] for class_name in splits.class_to_index
            },
        }

    valid_images = sum(summary["total_images"] for summary in split_summaries.values())
    return {
        "schema_version": 1,
        "data_directory": str(splits.data_dir.resolve()),
        "seed": splits.seed,
        "ratios": splits.ratios.as_dict(),
        "number_of_classes": len(splits.class_to_index),
        "class_to_index": splits.class_to_index,
        "total_discovered_images": splits.total_discovered_images,
        "valid_images": valid_images,
        "excluded_invalid_images": len(splits.invalid_image_paths),
        "excluded_invalid_image_paths": list(splits.invalid_image_paths),
        "splits": split_summaries,
        "leakage_check": {
            "overlap_count": overlap_count,
            "unique_filepaths": len(all_paths),
            "passed": overlap_count == 0 and len(all_paths) == valid_images,
        },
    }


def render_split_summary_markdown(summary: dict[str, Any]) -> str:
    """Render split summary metadata as Markdown."""
    ratios = summary["ratios"]
    lines = [
        "# Dataset Split Summary",
        "",
        f"- Data directory: `{summary['data_directory']}`",
        f"- Random seed: {summary['seed']}",
        f"- Ratios: train={ratios['train']}, val={ratios['val']}, test={ratios['test']}",
        f"- Discovered image files: {summary['total_discovered_images']}",
        f"- Valid images included: {summary['valid_images']}",
        f"- Invalid/corrupt images excluded: {summary['excluded_invalid_images']}",
        "",
        "## Split Counts",
        "",
        "| Split | Images |",
        "|---|---:|",
    ]
    for split_name in SPLIT_NAMES:
        lines.append(f"| {split_name} | {summary['splits'][split_name]['total_images']} |")

    lines.extend(
        [
            "",
            "## Per-Class Counts",
            "",
            "| Class | Class index | Train | Validation | Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, class_index in summary["class_to_index"].items():
        escaped_name = class_name.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {escaped_name} | {class_index} | "
            f"{summary['splits']['train']['class_counts'][class_name]} | "
            f"{summary['splits']['val']['class_counts'][class_name]} | "
            f"{summary['splits']['test']['class_counts'][class_name]} |"
        )

    leakage = summary["leakage_check"]
    status = "passed" if leakage["passed"] else "failed"
    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            f"Leakage check **{status}** with {leakage['overlap_count']} overlapping filepaths.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_split_outputs(
    splits: DatasetSplits,
    out_dir: Path | str,
    reports_dir: Path | str,
) -> dict[str, Any]:
    """Write split CSVs, class mapping, and JSON/Markdown summaries."""
    output_directory = Path(out_dir).expanduser()
    report_directory = Path(reports_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        csv_path = output_directory / f"{split_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(record.as_dict() for record in splits.records[split_name])

    mapping_path = output_directory / CLASS_MAPPING_NAME
    mapping_path.write_text(
        json.dumps(splits.class_to_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = build_split_summary(splits)
    summary_path = output_directory / SUMMARY_JSON_NAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = report_directory / SUMMARY_MARKDOWN_NAME
    markdown_path.write_text(render_split_summary_markdown(summary), encoding="utf-8")
    return summary


def generate_split_files(
    data_dir: Path | str,
    out_dir: Path | str,
    reports_dir: Path | str,
    ratios: SplitRatios,
    seed: int = 42,
) -> dict[str, Any]:
    """Scan valid images, generate stratified splits, and write all metadata files."""
    ratios.validate()
    scan = scan_dataset(data_dir)
    splits = create_stratified_splits(scan, ratios=ratios, seed=seed)
    return write_split_outputs(splits, out_dir=out_dir, reports_dir=reports_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic stratified train/val/test image metadata."
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
        default=Path("data/processed"),
        help="Directory for CSV and JSON metadata (default: data/processed).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for split_summary.md (default: reports).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction requested for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction requested for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction requested for testing (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic class shuffling (default: 42).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split generation command-line interface."""
    args = _build_parser().parse_args(argv)
    ratios = SplitRatios(
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
    )
    try:
        summary = generate_split_files(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            reports_dir=args.reports_dir,
            ratios=ratios,
            seed=args.seed,
        )
    except (DatasetDiscoveryError, SplitGenerationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    split_counts = summary["splits"]
    print(
        "Created splits from "
        f"{summary['valid_images']} valid images: "
        f"train={split_counts['train']['total_images']}, "
        f"val={split_counts['val']['total_images']}, "
        f"test={split_counts['test']['total_images']}."
    )
    if summary["excluded_invalid_images"]:
        print(f"Excluded {summary['excluded_invalid_images']} invalid/corrupt images.")
    print(f"Split metadata written to {args.out_dir}")
    print(f"Markdown summary written to {args.reports_dir / SUMMARY_MARKDOWN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
