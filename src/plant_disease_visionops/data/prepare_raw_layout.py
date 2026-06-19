"""Command-line preparation of nested image datasets for the raw scanner."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from plant_disease_visionops.data.raw_layout import RawLayoutError, prepare_raw_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy or symlink recursively discovered images into class folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Downloaded dataset root to search recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Prepared raw dataset directory (default: data/raw).",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Materialize files by copying or creating absolute symlinks (default: copy).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("reports/raw_layout_manifest.json"),
        help="Preparation manifest path (default: reports/raw_layout_manifest.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace a non-empty output directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw layout preparation command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        result = prepare_raw_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            manifest_path=args.manifest_path,
            overwrite=args.overwrite,
        )
    except (RawLayoutError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    class_counts = Counter(prepared.class_name for prepared in result.files)
    print(
        f"Prepared {len(result.files)} images across {len(class_counts)} classes "
        f"using {result.mode} mode."
    )
    for class_name in sorted(class_counts, key=lambda name: (name.casefold(), name)):
        print(f"  - {class_name}: {class_counts[class_name]} images")
    print(f"Raw dataset written to {result.output_dir}")
    print(f"Manifest written to {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
