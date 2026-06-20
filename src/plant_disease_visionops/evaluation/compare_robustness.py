"""Compare robustness summaries across trained model experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean


class RobustnessComparisonError(ValueError):
    """Raised when robustness reports are absent or invalid."""


@dataclass(frozen=True, slots=True)
class RobustnessSummary:
    """Cross-model robustness fields derived from one actual evaluation report."""

    experiment_name: str
    clean_accuracy: float
    clean_macro_f1: float
    worst_corruption: str
    worst_severity: int
    worst_corrupted_accuracy: float
    worst_corrupted_macro_f1: float
    largest_macro_f1_drop: float
    average_corrupted_macro_f1: float
    source_report: str


def _load_summary(path: Path) -> RobustnessSummary:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessComparisonError(f"Could not read robustness report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise RobustnessComparisonError(f"Robustness report must be a JSON object: {path}")
    clean = report.get("clean_metrics")
    conditions = report.get("corruption_results")
    experiment_name = report.get("experiment_name")
    if not isinstance(clean, dict) or not isinstance(conditions, list) or not conditions:
        raise RobustnessComparisonError(f"Missing clean/corruption metrics in report: {path}")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise RobustnessComparisonError(f"Missing experiment_name in report: {path}")
    if any(not isinstance(item, dict) for item in conditions):
        raise RobustnessComparisonError(f"Invalid corruption result entries in report: {path}")
    try:
        worst = min(
            conditions,
            key=lambda item: (
                float(item["macro_f1"]),
                str(item["corruption"]),
                int(item["severity"]),
            ),
        )
        drops = [float(item["macro_f1_drop"]) for item in conditions]
        corrupted_f1 = [float(item["macro_f1"]) for item in conditions]
        return RobustnessSummary(
            experiment_name=experiment_name,
            clean_accuracy=float(clean["accuracy"]),
            clean_macro_f1=float(clean["macro_f1"]),
            worst_corruption=str(worst["corruption"]),
            worst_severity=int(worst["severity"]),
            worst_corrupted_accuracy=float(worst["accuracy"]),
            worst_corrupted_macro_f1=float(worst["macro_f1"]),
            largest_macro_f1_drop=max(drops),
            average_corrupted_macro_f1=fmean(corrupted_f1),
            source_report=str(path.resolve()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RobustnessComparisonError(
            f"Invalid metric value in robustness report {path}: {exc}"
        ) from exc


def discover_robustness_summaries(reports_dir: Path | str) -> list[RobustnessSummary]:
    """Read all method-specific robustness reports from one directory."""
    directory = Path(reports_dir).expanduser()
    if not directory.is_dir():
        raise RobustnessComparisonError(f"Reports directory does not exist: {directory}")
    report_paths = sorted(
        directory.glob("*_robustness.json"),
        key=lambda path: path.name.casefold(),
    )
    if not report_paths:
        raise RobustnessComparisonError(f"No *_robustness.json files found in {directory}")
    summaries = [_load_summary(path) for path in report_paths]
    names = [summary.experiment_name for summary in summaries]
    if len(names) != len(set(names)):
        raise RobustnessComparisonError(f"Duplicate robustness experiment names: {names}")
    return sorted(
        summaries,
        key=lambda item: (item.experiment_name.casefold(), item.experiment_name),
    )


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_robustness_comparison(summaries: Sequence[RobustnessSummary]) -> str:
    """Render a Markdown comparison of clean and worst corrupted performance."""
    lines = [
        "# Robustness Comparison",
        "",
        "| Experiment | Clean accuracy | Clean macro F1 | Worst corruption | Severity | "
        "Worst accuracy | Worst macro F1 | Largest F1 drop | Average corrupted F1 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {_escape_markdown(summary.experiment_name)} | {summary.clean_accuracy:.6f} | "
            f"{summary.clean_macro_f1:.6f} | {_escape_markdown(summary.worst_corruption)} | "
            f"{summary.worst_severity} | {summary.worst_corrupted_accuracy:.6f} | "
            f"{summary.worst_corrupted_macro_f1:.6f} | "
            f"{summary.largest_macro_f1_drop:.6f} | "
            f"{summary.average_corrupted_macro_f1:.6f} |"
        )
    lines.extend(
        [
            "",
            "All values are copied or derived from completed robustness evaluation reports; no "
            "missing conditions are estimated.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_robustness_comparison(
    reports_dir: Path | str,
    out_markdown: Path | str,
    out_json: Path | str,
) -> list[RobustnessSummary]:
    """Write JSON and Markdown comparisons from existing robustness reports."""
    summaries = discover_robustness_summaries(reports_dir)
    markdown_path = Path(out_markdown).expanduser()
    json_path = Path(out_json).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_robustness_comparison(summaries), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [asdict(summary) for summary in summaries],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare saved robustness evaluation reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/robustness_comparison.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/robustness_comparison.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run robustness comparison from command-line arguments."""
    args = _build_parser().parse_args(argv)
    try:
        summaries = write_robustness_comparison(
            args.reports_dir,
            args.out_md,
            args.out_json,
        )
    except RobustnessComparisonError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Compared robustness for {len(summaries)} experiments.")
    print(f"Markdown comparison written to {args.out_md}")
    print(f"JSON comparison written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
