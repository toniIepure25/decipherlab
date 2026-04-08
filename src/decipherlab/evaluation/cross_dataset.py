from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.utils.io import ensure_directory, write_csv, write_json, write_text


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _row_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        if name in row:
            value = _to_float(row.get(name))
            if value is not None:
                return value
    return None


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _dataset_label(run_dir: Path, explicit_label: str | None = None) -> str:
    if explicit_label is not None:
        return explicit_label
    metadata_path = run_dir / "dataset_summary.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        if "dataset_name" in metadata:
            return str(metadata["dataset_name"])
    return run_dir.name


def _main_condition_rows(run_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(run_dir / "main_comparison_with_ci.csv")
    return {row["condition"]: row for row in rows}


def _pairwise_rows(run_dir: Path) -> list[dict[str, str]]:
    return _read_csv_rows(run_dir / "pairwise_effect_summary.csv")


def _failure_rows(run_dir: Path) -> list[dict[str, str]]:
    return _read_csv_rows(run_dir / "failure_case_summary.csv")


def _dataset_metadata(run_dir: Path) -> dict[str, Any]:
    metadata = _read_json(run_dir / "dataset_summary.json")
    experiment = _read_json(run_dir / "experiment_metadata.json")
    return {"dataset": metadata, "experiment": experiment}


def build_cross_dataset_outputs(
    datasets: list[dict[str, str | Path]],
    output_root: str | Path = "outputs",
) -> dict[str, str]:
    output_dir = ensure_directory(output_root)
    summary_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    condition_ci_rows: list[dict[str, Any]] = []

    for item in datasets:
        run_dir = Path(item["run_dir"])
        label = _dataset_label(run_dir, explicit_label=str(item.get("dataset_label")) if item.get("dataset_label") else None)
        conditions = _main_condition_rows(run_dir)
        pairwise = _pairwise_rows(run_dir)
        failures = _failure_rows(run_dir)
        metadata = _dataset_metadata(run_dir)
        dataset_summary = metadata["dataset"]

        row_a = conditions["A. Fixed Transcript + Heuristic Posterior"]
        row_b = conditions["B. Fixed Transcript + Calibrated Posterior"]
        row_c = conditions["C. Uncertainty-Aware + Heuristic Posterior"]
        row_d = conditions["D. Uncertainty-Aware + Calibrated Posterior"]

        for condition_name, condition_row in conditions.items():
            condition_ci_rows.append(
                {
                    "dataset": label,
                    "run_dir": str(run_dir),
                    "condition": condition_name,
                    "mean_symbol_top1_accuracy": _row_float(condition_row, "mean_symbol_top1_accuracy"),
                    "mean_symbol_topk_accuracy": _row_float(condition_row, "mean_symbol_topk_accuracy"),
                    "mean_symbol_topk_accuracy_ci_lower": _row_float(
                        condition_row, "mean_symbol_topk_accuracy_ci_lower"
                    ),
                    "mean_symbol_topk_accuracy_ci_upper": _row_float(
                        condition_row, "mean_symbol_topk_accuracy_ci_upper"
                    ),
                    "mean_symbol_nll": _row_float(
                        condition_row, "mean_symbol_nll", "mean_symbol_negative_log_likelihood"
                    ),
                    "mean_symbol_nll_ci_lower": _row_float(
                        condition_row,
                        "mean_symbol_nll_ci_lower",
                        "mean_symbol_negative_log_likelihood_ci_lower",
                    ),
                    "mean_symbol_nll_ci_upper": _row_float(
                        condition_row,
                        "mean_symbol_nll_ci_upper",
                        "mean_symbol_negative_log_likelihood_ci_upper",
                    ),
                    "mean_symbol_ece": _row_float(
                        condition_row, "mean_symbol_ece", "mean_symbol_expected_calibration_error"
                    ),
                    "mean_symbol_ece_ci_lower": _row_float(
                        condition_row,
                        "mean_symbol_ece_ci_lower",
                        "mean_symbol_expected_calibration_error_ci_lower",
                    ),
                    "mean_symbol_ece_ci_upper": _row_float(
                        condition_row,
                        "mean_symbol_ece_ci_upper",
                        "mean_symbol_expected_calibration_error_ci_upper",
                    ),
                    "mean_family_topk_accuracy": _row_float(condition_row, "mean_family_topk_accuracy"),
                    "mean_family_topk_accuracy_ci_lower": _row_float(
                        condition_row, "mean_family_topk_accuracy_ci_lower"
                    ),
                    "mean_family_topk_accuracy_ci_upper": _row_float(
                        condition_row, "mean_family_topk_accuracy_ci_upper"
                    ),
                }
            )

        summary_rows.append(
            {
                "dataset": label,
                "run_dir": str(run_dir),
                "dataset_name": dataset_summary["dataset_name"],
                "sequence_count": dataset_summary["sequence_count"],
                "record_count": dataset_summary["record_count"],
                "train_group_count": dataset_summary["split_group_counts"].get("train", 0),
                "train_symbol_class_count": len(dataset_summary.get("train_symbol_counts", {})),
                "a_fixed_heuristic_topk": _to_float(row_a["mean_symbol_topk_accuracy"]),
                "b_fixed_calibrated_topk": _to_float(row_b["mean_symbol_topk_accuracy"]),
                "c_uncertainty_heuristic_topk": _to_float(row_c["mean_symbol_topk_accuracy"]),
                "d_uncertainty_calibrated_topk": _to_float(row_d["mean_symbol_topk_accuracy"]),
                "heuristic_uncertainty_topk_delta_mean": _mean(
                    [_to_float(row["heuristic_uncertainty_topk_delta"]) for row in pairwise]
                ),
                "calibrated_uncertainty_topk_delta_mean": _mean(
                    [_to_float(row["calibrated_uncertainty_topk_delta"]) for row in pairwise]
                ),
                "fixed_calibration_topk_delta_mean": _mean(
                    [_to_float(row["fixed_calibration_topk_delta"]) for row in pairwise]
                ),
                "uncertainty_calibration_topk_delta_mean": _mean(
                    [_to_float(row["uncertainty_calibration_topk_delta"]) for row in pairwise]
                ),
                "combined_topk_delta_mean": _mean(
                    [_to_float(row["combined_topk_delta"]) for row in pairwise]
                ),
                "combined_nll_delta_mean": _mean(
                    [_to_float(row["combined_nll_delta"]) for row in pairwise]
                ),
                "combined_ece_delta_mean": _mean(
                    [_to_float(row["combined_ece_delta"]) for row in pairwise]
                ),
                "family_signal_available": float(
                    any(
                        _to_float(row_key.get("mean_family_topk_accuracy")) not in {None, 0.0}
                        for row_key in conditions.values()
                    )
                ),
            }
        )

        for pair_row in pairwise:
            ambiguity_rows.append(
                {
                    "dataset": label,
                    "ambiguity_level": _to_float(pair_row["ambiguity_level"]),
                    "heuristic_uncertainty_topk_delta": _to_float(pair_row["heuristic_uncertainty_topk_delta"]),
                    "calibrated_uncertainty_topk_delta": _to_float(pair_row["calibrated_uncertainty_topk_delta"]),
                    "fixed_calibration_topk_delta": _to_float(pair_row["fixed_calibration_topk_delta"]),
                    "uncertainty_calibration_topk_delta": _to_float(pair_row["uncertainty_calibration_topk_delta"]),
                    "combined_topk_delta": _to_float(pair_row["combined_topk_delta"]),
                    "combined_nll_delta": _to_float(pair_row["combined_nll_delta"]),
                    "combined_ece_delta": _to_float(pair_row["combined_ece_delta"]),
                }
            )

        for failure_row in failures:
            failure_rows.append(
                {
                    "dataset": label,
                    "case_type": failure_row["case_type"],
                    "ambiguity_level": _to_float(failure_row["ambiguity_level"]),
                    "condition_group": failure_row["condition_group"],
                    "count": int(failure_row["count"]),
                }
            )

    write_csv(output_dir / "cross_dataset_summary.csv", summary_rows)
    write_csv(output_dir / "cross_dataset_ambiguity_rows.csv", ambiguity_rows)
    write_csv(output_dir / "cross_dataset_failure_summary.csv", failure_rows)
    write_csv(output_dir / "cross_dataset_tables_with_ci.csv", condition_ci_rows)
    write_json(
        output_dir / "cross_dataset_summary.json",
        {
            "summary_rows": summary_rows,
            "ambiguity_rows": ambiguity_rows,
            "failure_rows": failure_rows,
            "condition_ci_rows": condition_ci_rows,
        },
    )

    figure_path = output_dir / "cross_dataset_effects_plot.png"
    _write_effects_plot(ambiguity_rows, figure_path)
    write_text(output_dir / "cross_dataset_summary.md", _build_cross_dataset_markdown(summary_rows, ambiguity_rows, failure_rows))

    return {
        "summary_csv": str(output_dir / "cross_dataset_summary.csv"),
        "summary_md": str(output_dir / "cross_dataset_summary.md"),
        "effects_plot": str(figure_path),
        "failure_csv": str(output_dir / "cross_dataset_failure_summary.csv"),
        "ci_table_csv": str(output_dir / "cross_dataset_tables_with_ci.csv"),
    }


def _write_effects_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for dataset in datasets:
        subset = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: float(row["ambiguity_level"]),
        )
        axes[0].plot(
            [row["ambiguity_level"] for row in subset],
            [row["heuristic_uncertainty_topk_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
        axes[1].plot(
            [row["ambiguity_level"] for row in subset],
            [row["calibrated_uncertainty_topk_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
    axes[0].set_title("Heuristic Uncertainty Top-k Delta")
    axes[1].set_title("Calibrated Uncertainty Top-k Delta")
    for axis in axes:
        axis.set_xlabel("Ambiguity Level")
        axis.set_ylabel("Top-k Delta")
        axis.grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_cross_dataset_markdown(
    summary_rows: list[dict[str, Any]],
    ambiguity_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Cross-Dataset Summary",
        "",
        "## Headline",
    ]
    for row in summary_rows:
        lines.append(
            "- {dataset}: heuristic uncertainty top-k delta mean {hu}; calibrated uncertainty top-k delta mean {cu}; fixed-calibration top-k delta mean {fc}.".format(
                dataset=row["dataset"],
                hu=_fmt(row["heuristic_uncertainty_topk_delta_mean"]),
                cu=_fmt(row["calibrated_uncertainty_topk_delta_mean"]),
                fc=_fmt(row["fixed_calibration_topk_delta_mean"]),
            )
        )

    lines.extend(["", "## Ambiguity Sweep"])
    for dataset in sorted({row["dataset"] for row in ambiguity_rows}):
        subset = sorted(
            [row for row in ambiguity_rows if row["dataset"] == dataset],
            key=lambda row: float(row["ambiguity_level"]),
        )
        if not subset:
            continue
        highest = subset[-1]
        lines.append(
            "- {dataset}: at ambiguity {ambiguity}, heuristic uncertainty top-k delta = {hu}, calibrated uncertainty top-k delta = {cu}.".format(
                dataset=dataset,
                ambiguity=_fmt(highest["ambiguity_level"]),
                hu=_fmt(highest["heuristic_uncertainty_topk_delta"]),
                cu=_fmt(highest["calibrated_uncertainty_topk_delta"]),
            )
        )

    lines.extend(["", "## Failure Modes"])
    grouped_failures: dict[tuple[str, str], int] = {}
    for row in failure_rows:
        key = (str(row["dataset"]), str(row["case_type"]))
        grouped_failures[key] = grouped_failures.get(key, 0) + int(row["count"])
    for (dataset, case_type), count in sorted(grouped_failures.items()):
        lines.append(f"- {dataset}: {case_type} = {count}")

    lines.extend(
        [
            "",
            "## Guardrail",
            "- These comparisons remain symbol-level and ambiguity-focused.",
            "- They do not by themselves support semantic decipherment or broad historical-generalization claims.",
        ]
    )
    return "\n".join(lines) + "\n"
