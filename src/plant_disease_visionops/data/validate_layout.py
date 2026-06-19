"""Command-line validation for the direct class-folder raw dataset layout."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from plant_disease_visionops.data.raw_layout import (
    EXPECTED_LAYOUT,
    RawLayoutAnalysis,
    RawLayoutError,
    analyze_raw_layout,
    require_usable_layout,
)

MAX_WARNING_PATHS = 10


def _append_paths(lines: list[str], paths: tuple[str, ...]) -> None:
    for path in paths[:MAX_WARNING_PATHS]:
        lines.append(f"  - {path}")
    omitted = len(paths) - MAX_WARNING_PATHS
    if omitted > 0:
        lines.append(f"  - ... and {omitted} more")


def render_layout_report(analysis: RawLayoutAnalysis) -> str:
    """Render usable classes and ignored-file warnings for terminal output."""
    lines = [
        f"Raw data directory: {analysis.data_dir}",
        f"Expected layout: {EXPECTED_LAYOUT}",
        "",
        "Usable class folders:",
    ]
    if analysis.class_folders:
        for class_folder in analysis.class_folders:
            lines.append(
                f"  - {class_folder.class_name}: {class_folder.candidate_images} candidate images"
            )
    else:
        lines.append("  - none")

    if analysis.unsupported_files:
        lines.extend(
            [
                "",
                f"WARNING: {len(analysis.unsupported_files)} unsupported files will be ignored:",
            ]
        )
        _append_paths(lines, analysis.unsupported_files)
    if analysis.nested_image_folders:
        lines.extend(
            [
                "",
                "WARNING: Supported images in these nested folders will not be scanned by "
                "audit or split commands:",
            ]
        )
        _append_paths(lines, analysis.nested_image_folders)
    if analysis.root_image_files:
        lines.extend(
            [
                "",
                "WARNING: Supported images directly under the data directory have no class "
                "folder and will be ignored:",
            ]
        )
        _append_paths(lines, analysis.root_image_files)

    if analysis.is_usable:
        lines.extend(
            [
                "",
                f"Layout is usable: {len(analysis.class_folders)} classes and "
                f"{analysis.total_candidate_images} candidate images.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Layout is not usable by the current audit and split workflows.",
                "Run prepare_raw_layout for nested datasets, then validate again.",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that raw images are direct children of class folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset directory to validate (default: data/raw).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw layout validation command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        analysis = analyze_raw_layout(args.data_dir)
    except RawLayoutError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Expected layout: {EXPECTED_LAYOUT}", file=sys.stderr)
        return 2

    print(render_layout_report(analysis), end="")
    try:
        require_usable_layout(analysis)
    except RawLayoutError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
