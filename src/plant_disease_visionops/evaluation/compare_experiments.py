"""Compare metrics from baseline and transfer-learning result reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ExperimentComparisonError(ValueError):
    """Raised when experiment reports are missing or malformed."""


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Fields shown in the cross-experiment comparison table."""

    experiment_name: str
    model_name: str
    test_accuracy: float
    test_macro_f1: float
    best_val_macro_f1: float
    epochs: int
    image_size: int
    batch_size: int
    pretrained: bool
    freeze_backbone: bool
    source_report: str


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _required(mapping: dict[str, Any], key: str, report_path: Path) -> Any:
    if key not in mapping:
        raise ExperimentComparisonError(f"Missing '{key}' in result report: {report_path}")
    return mapping[key]


def _top_level_or_nested(
    raw: dict[str, Any],
    top_level_key: str,
    nested: dict[str, Any],
    nested_key: str,
    report_path: Path,
) -> Any:
    if top_level_key in raw:
        return raw[top_level_key]
    return _required(nested, nested_key, report_path)


def _boolean_value(value: Any, field: str, report_path: Path) -> bool:
    if not isinstance(value, bool):
        raise ExperimentComparisonError(
            f"'{field}' must be true or false in result report: {report_path}"
        )
    return value


def _summary_from_report(report_path: Path) -> ExperimentSummary:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentComparisonError(
            f"Could not read result report {report_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ExperimentComparisonError(f"Result report must contain a JSON object: {report_path}")

    hyperparameters = raw.get("hyperparameters", {})
    model = raw.get("model", {})
    if not isinstance(hyperparameters, dict) or not isinstance(model, dict):
        raise ExperimentComparisonError(f"Invalid nested metadata in result report: {report_path}")
    validation_metrics = _required(raw, "validation_metrics", report_path)
    test_metrics = _required(raw, "test_metrics", report_path)
    if not isinstance(validation_metrics, dict) or not isinstance(test_metrics, dict):
        raise ExperimentComparisonError(f"Invalid metrics objects in result report: {report_path}")

    fallback_name = report_path.name.removesuffix("_results.json")
    try:
        return ExperimentSummary(
            experiment_name=str(raw.get("experiment_name", fallback_name)),
            model_name=str(_top_level_or_nested(raw, "model_name", model, "name", report_path)),
            test_accuracy=float(_required(test_metrics, "accuracy", report_path)),
            test_macro_f1=float(_required(test_metrics, "macro_f1", report_path)),
            best_val_macro_f1=float(_required(validation_metrics, "macro_f1", report_path)),
            epochs=int(_top_level_or_nested(raw, "epochs", hyperparameters, "epochs", report_path)),
            image_size=int(
                _top_level_or_nested(raw, "image_size", hyperparameters, "image_size", report_path)
            ),
            batch_size=int(
                _top_level_or_nested(raw, "batch_size", hyperparameters, "batch_size", report_path)
            ),
            pretrained=_boolean_value(
                raw.get("pretrained", hyperparameters.get("pretrained", False)),
                "pretrained",
                report_path,
            ),
            freeze_backbone=_boolean_value(
                raw.get("freeze_backbone", hyperparameters.get("freeze_backbone", False)),
                "freeze_backbone",
                report_path,
            ),
            source_report=str(report_path.resolve()),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentComparisonError(
            f"Invalid comparison value in result report {report_path}: {exc}"
        ) from exc


def discover_experiment_summaries(reports_dir: Path | str) -> list[ExperimentSummary]:
    """Read every method-specific result JSON, including Milestone 4 legacy reports."""
    directory = Path(reports_dir).expanduser()
    if not directory.is_dir():
        raise ExperimentComparisonError(f"Reports directory does not exist: {directory}")
    report_paths = sorted(directory.glob("*_results.json"), key=lambda path: path.name.casefold())
    if not report_paths:
        raise ExperimentComparisonError(f"No *_results.json files found in {directory}")
    summaries = [_summary_from_report(path) for path in report_paths]
    names = [summary.experiment_name for summary in summaries]
    if len(names) != len(set(names)):
        raise ExperimentComparisonError(f"Duplicate experiment names found: {names}")
    return sorted(
        summaries, key=lambda item: (item.experiment_name.casefold(), item.experiment_name)
    )


def render_comparison_markdown(summaries: Sequence[ExperimentSummary]) -> str:
    """Render experiment summaries as a compact Markdown table."""
    lines = [
        "# Experiment Comparison",
        "",
        "| Experiment | Model | Test accuracy | Test macro F1 | Best val macro F1 | Epochs | "
        "Image size | Batch size | Pretrained | Freeze backbone |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {_escape_markdown(summary.experiment_name)} | "
            f"{_escape_markdown(summary.model_name)} | "
            f"{summary.test_accuracy:.6f} | {summary.test_macro_f1:.6f} | "
            f"{summary.best_val_macro_f1:.6f} | {summary.epochs} | {summary.image_size} | "
            f"{summary.batch_size} | {str(summary.pretrained).lower()} | "
            f"{str(summary.freeze_backbone).lower()} |"
        )
    lines.extend(
        [
            "",
            "Rows reproduce metrics from each experiment's saved result JSON. They do not rerun "
            "evaluation or infer missing scores.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_experiment_comparison(
    reports_dir: Path | str,
    out_markdown: Path | str,
    out_json: Path | str,
) -> list[ExperimentSummary]:
    """Discover result reports and write matching Markdown and JSON comparisons."""
    summaries = discover_experiment_summaries(reports_dir)
    markdown_path = Path(out_markdown).expanduser()
    json_path = Path(out_json).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_comparison_markdown(summaries), encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Compare saved model experiment reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/experiment_comparison.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/experiment_comparison.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment comparison command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        summaries = write_experiment_comparison(
            reports_dir=args.reports_dir,
            out_markdown=args.out_md,
            out_json=args.out_json,
        )
    except ExperimentComparisonError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Compared {len(summaries)} experiments.")
    print(f"Markdown comparison written to {args.out_md}")
    print(f"JSON comparison written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
