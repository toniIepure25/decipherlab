from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-sequence-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.utils.io import ensure_directory, write_csv, write_json, write_text


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_sequence_cross_dataset_outputs(
    datasets: list[dict[str, str | Path]],
    output_root: str | Path = "outputs",
) -> dict[str, str]:
    output_dir = ensure_directory(output_root)
    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []

    for dataset_item in datasets:
        dataset_label = str(dataset_item["dataset_label"])
        task_label = str(dataset_item["task_label"])
        run_map: dict[str, Path] = {
            "cluster_distance": Path(dataset_item["cluster_distance_run"]),
            "calibrated_classifier": Path(dataset_item["calibrated_classifier_run"]),
        }
        for strategy_label, run_dir in run_map.items():
            rows = _read_csv_rows(run_dir / "sequence_branch_summary.csv")
            pairwise = _read_csv_rows(run_dir / "sequence_pairwise_effects.csv")
            failures = _read_csv_rows(run_dir / "sequence_failure_summary.csv")
            dataset_summary = _read_json(run_dir / "dataset_summary.json")
            benchmark_summary = _read_json(run_dir / "benchmark_summary.json")

            for row in rows:
                summary_rows.append(
                    {
                        "dataset": dataset_label,
                        "task_name": task_label,
                        "posterior_strategy_requested": strategy_label,
                        "ambiguity_level": _to_float(row["ambiguity_level"]),
                        "method": row["method"],
                        "sequence_exact_match": _to_float(row["sequence_exact_match"]),
                        "sequence_token_accuracy": _to_float(row["sequence_token_accuracy"]),
                        "sequence_topk_recovery": _to_float(row["sequence_topk_recovery"]),
                        "sequence_cer": _to_float(row["sequence_cer"]),
                        "symbol_topk_accuracy": _to_float(row["symbol_topk_accuracy"]),
                        "prediction_set_coverage": _to_float(row.get("prediction_set_coverage")),
                        "prediction_set_avg_size": _to_float(row.get("prediction_set_avg_size")),
                        "family_identification_accuracy": _to_float(row.get("family_identification_accuracy")),
                        "source_dataset_name": benchmark_summary["metadata"]["source_dataset_name"],
                        "selected_symbol_count": len(benchmark_summary["alphabet"]),
                        "sequence_length": dataset_summary["sequence_length"],
                    }
                )
            for row in pairwise:
                effect_rows.append(
                    {
                        "dataset": dataset_label,
                        "task_name": task_label,
                        "posterior_strategy_requested": strategy_label,
                        **{
                            key: (_to_float(value) if key != "ambiguity_regime" and key != "posterior_strategy_requested" else value)
                            for key, value in row.items()
                        },
                    }
                )
            for row in failures:
                failure_rows.append(
                    {
                        "dataset": dataset_label,
                        "task_name": task_label,
                        "posterior_strategy_requested": strategy_label,
                        "case_type": row["case_type"],
                        "ambiguity_level": _to_float(row["ambiguity_level"]),
                        "count": int(float(row["count"])),
                    }
                )
            for row in _read_csv_rows(run_dir / "sequence_ambiguity_regime_table.csv"):
                regime_rows.append(
                    {
                        "dataset": dataset_label,
                        "task_name": task_label,
                        **{
                            key: (_to_float(value) if key not in {"ambiguity_regime", "posterior_strategy_requested"} else value)
                            for key, value in row.items()
                        },
                    }
                )

    write_csv(output_dir / "sequence_cross_dataset_summary.csv", summary_rows)
    write_csv(output_dir / "sequence_cross_dataset_failure_summary.csv", failure_rows)
    write_csv(output_dir / "sequence_ambiguity_regime_table.csv", regime_rows)
    write_csv(output_dir / "sequence_cross_dataset_effect_rows.csv", effect_rows)

    aggregate_rows = _aggregate_effect_rows(effect_rows)
    write_csv(output_dir / "sequence_cross_dataset_tables.csv", aggregate_rows)
    figure_path = output_dir / "sequence_cross_dataset_effects_plot.png"
    _write_effects_plot(effect_rows, figure_path)
    summary_md = _build_summary_markdown(aggregate_rows, failure_rows, regime_rows)
    write_text(output_dir / "sequence_cross_dataset_summary.md", summary_md)
    write_json(
        output_dir / "sequence_cross_dataset_summary.json",
        {
            "summary_rows": summary_rows,
            "aggregate_rows": aggregate_rows,
            "failure_rows": failure_rows,
            "regime_rows": regime_rows,
        },
    )
    return {
        "summary_csv": str(output_dir / "sequence_cross_dataset_summary.csv"),
        "summary_md": str(output_dir / "sequence_cross_dataset_summary.md"),
        "effects_plot": str(figure_path),
        "failure_csv": str(output_dir / "sequence_cross_dataset_failure_summary.csv"),
        "regime_csv": str(output_dir / "sequence_ambiguity_regime_table.csv"),
    }


def _aggregate_effect_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in effect_rows:
        key = (str(row["dataset"]), str(row["posterior_strategy_requested"]))
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, strategy), rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "dataset": dataset,
                "posterior_strategy_requested": strategy,
                "mean_uncertainty_sequence_exact_delta": _mean(
                    [row["uncertainty_sequence_exact_delta"] for row in rows]
                ),
                "mean_uncertainty_sequence_topk_delta": _mean(
                    [row["uncertainty_sequence_topk_delta"] for row in rows]
                ),
                "mean_uncertainty_symbol_topk_delta": _mean(
                    [row["uncertainty_symbol_topk_delta"] for row in rows]
                ),
                "mean_conformal_sequence_exact_delta": _mean(
                    [row["conformal_sequence_exact_delta"] for row in rows]
                ),
                "mean_conformal_sequence_topk_delta": _mean(
                    [row["conformal_sequence_topk_delta"] for row in rows]
                ),
                "mean_conformal_coverage_delta": _mean(
                    [row["conformal_coverage_delta"] for row in rows]
                ),
                "peak_uncertainty_exact_delta": max(
                    [row["uncertainty_sequence_exact_delta"] for row in rows if row["uncertainty_sequence_exact_delta"] is not None],
                    default=None,
                ),
                "peak_ambiguity_for_uncertainty_exact_delta": _peak_ambiguity(
                    rows,
                    metric_key="uncertainty_sequence_exact_delta",
                ),
            }
        )
    return aggregate_rows


def _peak_ambiguity(rows: list[dict[str, Any]], metric_key: str) -> float | None:
    candidates = [row for row in rows if row.get(metric_key) is not None]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: float(row[metric_key]))["ambiguity_level"])


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _write_effects_plot(effect_rows: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for dataset in sorted({str(row["dataset"]) for row in effect_rows}):
        subset = sorted(
            [row for row in effect_rows if row["dataset"] == dataset and row["posterior_strategy_requested"] == "calibrated_classifier"],
            key=lambda row: float(row["ambiguity_level"]),
        )
        if not subset:
            continue
        axes[0].plot(
            [row["ambiguity_level"] for row in subset],
            [row["uncertainty_sequence_exact_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
        axes[1].plot(
            [row["ambiguity_level"] for row in subset],
            [row["conformal_coverage_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Uncertainty Beam Sequence Exact Delta")
    axes[0].set_xlabel("Ambiguity")
    axes[0].set_ylabel("Delta vs Fixed Greedy")
    axes[1].set_title("Conformal Coverage Delta")
    axes[1].set_xlabel("Ambiguity")
    axes[1].set_ylabel("Delta vs Uncertainty Beam")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_summary_markdown(
    aggregate_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Sequence Cross-Dataset Summary",
        "",
        "These results are synthetic-from-real sequence findings and should not be read as semantic decipherment evidence.",
        "",
        "## Aggregate Effects",
    ]
    for row in aggregate_rows:
        lines.append(
            "- {dataset} ({strategy}): mean uncertainty sequence exact delta = {exact}, mean sequence top-k delta = {topk}, mean conformal coverage delta = {coverage}, peak exact delta at ambiguity {ambiguity}.".format(
                dataset=row["dataset"],
                strategy=row["posterior_strategy_requested"],
                exact=_fmt(row["mean_uncertainty_sequence_exact_delta"]),
                topk=_fmt(row["mean_uncertainty_sequence_topk_delta"]),
                coverage=_fmt(row["mean_conformal_coverage_delta"]),
                ambiguity=_fmt(row["peak_ambiguity_for_uncertainty_exact_delta"]),
            )
        )
    lines.extend(["", "## Ambiguity Regimes"])
    for row in regime_rows:
        lines.append(
            "- {dataset} {strategy} {regime}: mean uncertainty exact delta = {exact}, mean uncertainty top-k delta = {topk}, mean conformal coverage delta = {coverage}.".format(
                dataset=row["dataset"],
                strategy=row["posterior_strategy_requested"],
                regime=row["ambiguity_regime"],
                exact=_fmt(row["mean_uncertainty_sequence_exact_delta"]),
                topk=_fmt(row["mean_uncertainty_sequence_topk_delta"]),
                coverage=_fmt(row["mean_conformal_coverage_delta"]),
            )
        )
    if failure_rows:
        lines.extend(["", "## Recurring Failure Modes"])
        grouped: dict[tuple[str, str], int] = {}
        for row in failure_rows:
            key = (str(row["dataset"]), str(row["case_type"]))
            grouped[key] = grouped.get(key, 0) + int(row["count"])
        for (dataset, case_type), count in sorted(grouped.items()):
            lines.append(f"- {dataset}: `{case_type}` count = {count}.")
    return "\n".join(lines)
