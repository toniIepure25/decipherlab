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


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _peak_ambiguity(rows: list[dict[str, Any]], metric_key: str) -> float | None:
    candidates = [row for row in rows if row.get(metric_key) is not None]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: float(row[metric_key]))["ambiguity_level"])


def _collect_run_rows(
    datasets: list[dict[str, str | Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
                        "prediction_set_singleton_rate": _to_float(row.get("prediction_set_singleton_rate")),
                        "prediction_set_rescue_rate": _to_float(row.get("prediction_set_rescue_rate")),
                        "family_identification_accuracy": _to_float(row.get("family_identification_accuracy")),
                        "family_identification_topk_recovery": _to_float(row.get("family_identification_topk_recovery")),
                        "source_dataset_name": benchmark_summary["metadata"]["source_dataset_name"],
                        "selected_symbol_count": len(benchmark_summary["alphabet"]),
                        "sequence_length": dataset_summary["sequence_length"],
                        "synthetic_from_real": bool(benchmark_summary["metadata"]["synthetic_from_real"]),
                    }
                )
            for row in pairwise:
                effect_rows.append(
                    {
                        "dataset": dataset_label,
                        "task_name": task_label,
                        "posterior_strategy_requested": strategy_label,
                        **{
                            key: (_to_float(value) if key not in {"ambiguity_regime", "posterior_strategy_requested"} else value)
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

    return summary_rows, effect_rows, failure_rows, regime_rows


def _aggregate_effect_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in effect_rows:
        key = (str(row["dataset"]), str(row["task_name"]), str(row["posterior_strategy_requested"]))
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, task_name, strategy), rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "dataset": dataset,
                "task_name": task_name,
                "posterior_strategy_requested": strategy,
                "mean_uncertainty_sequence_exact_delta": _mean(
                    [row.get("uncertainty_sequence_exact_delta") for row in rows]
                ),
                "mean_uncertainty_sequence_topk_delta": _mean(
                    [row.get("uncertainty_sequence_topk_delta") for row in rows]
                ),
                "mean_uncertainty_symbol_topk_delta": _mean(
                    [row.get("uncertainty_symbol_topk_delta") for row in rows]
                ),
                "mean_uncertainty_family_delta": _mean(
                    [row.get("uncertainty_family_delta") for row in rows]
                ),
                "mean_conformal_sequence_exact_delta": _mean(
                    [row.get("conformal_sequence_exact_delta") for row in rows]
                ),
                "mean_conformal_sequence_topk_delta": _mean(
                    [row.get("conformal_sequence_topk_delta") for row in rows]
                ),
                "mean_conformal_coverage_delta": _mean(
                    [row.get("conformal_coverage_delta") for row in rows]
                ),
                "mean_conformal_set_size_delta": _mean(
                    [row.get("conformal_set_size_delta") for row in rows]
                ),
                "mean_conformal_family_delta": _mean(
                    [row.get("conformal_family_delta") for row in rows]
                ),
                "mean_trigram_sequence_exact_delta": _mean(
                    [row.get("trigram_sequence_exact_delta") for row in rows]
                ),
                "mean_trigram_sequence_topk_delta": _mean(
                    [row.get("trigram_sequence_topk_delta") for row in rows]
                ),
                "mean_trigram_family_delta": _mean(
                    [row.get("trigram_family_delta") for row in rows]
                ),
                "mean_crf_sequence_exact_delta": _mean(
                    [row.get("crf_sequence_exact_delta") for row in rows]
                ),
                "mean_crf_sequence_topk_delta": _mean(
                    [row.get("crf_sequence_topk_delta") for row in rows]
                ),
                "mean_crf_family_delta": _mean(
                    [row.get("crf_family_delta") for row in rows]
                ),
                "mean_conformal_trigram_sequence_exact_delta": _mean(
                    [row.get("conformal_trigram_sequence_exact_delta") for row in rows]
                ),
                "mean_conformal_trigram_sequence_topk_delta": _mean(
                    [row.get("conformal_trigram_sequence_topk_delta") for row in rows]
                ),
                "mean_conformal_trigram_coverage_delta": _mean(
                    [row.get("conformal_trigram_coverage_delta") for row in rows]
                ),
                "mean_conformal_trigram_set_size_delta": _mean(
                    [row.get("conformal_trigram_set_size_delta") for row in rows]
                ),
                "mean_conformal_trigram_family_delta": _mean(
                    [row.get("conformal_trigram_family_delta") for row in rows]
                ),
                "mean_conformal_crf_sequence_exact_delta": _mean(
                    [row.get("conformal_crf_sequence_exact_delta") for row in rows]
                ),
                "mean_conformal_crf_sequence_topk_delta": _mean(
                    [row.get("conformal_crf_sequence_topk_delta") for row in rows]
                ),
                "mean_conformal_crf_coverage_delta": _mean(
                    [row.get("conformal_crf_coverage_delta") for row in rows]
                ),
                "mean_conformal_crf_set_size_delta": _mean(
                    [row.get("conformal_crf_set_size_delta") for row in rows]
                ),
                "mean_conformal_crf_family_delta": _mean(
                    [row.get("conformal_crf_family_delta") for row in rows]
                ),
                "peak_uncertainty_exact_delta": max(
                    [row["uncertainty_sequence_exact_delta"] for row in rows if row.get("uncertainty_sequence_exact_delta") is not None],
                    default=None,
                ),
                "peak_ambiguity_for_uncertainty_exact_delta": _peak_ambiguity(
                    rows,
                    metric_key="uncertainty_sequence_exact_delta",
                ),
                "peak_ambiguity_for_trigram_sequence_exact_delta": _peak_ambiguity(
                    rows,
                    metric_key="trigram_sequence_exact_delta",
                ),
                "peak_ambiguity_for_crf_sequence_exact_delta": _peak_ambiguity(
                    rows,
                    metric_key="crf_sequence_exact_delta",
                ),
                "peak_ambiguity_for_family_delta": _peak_ambiguity(
                    rows,
                    metric_key="uncertainty_family_delta",
                ),
            }
        )
    return aggregate_rows


def _family_signal_present(summary_rows: list[dict[str, Any]]) -> bool:
    return any(row.get("family_identification_accuracy") is not None for row in summary_rows)


def _write_effects_plot(
    effect_rows: list[dict[str, Any]],
    output_path: Path,
    family_focus: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for dataset in sorted({str(row["dataset"]) for row in effect_rows}):
        subset = sorted(
            [row for row in effect_rows if row["dataset"] == dataset and row["posterior_strategy_requested"] == "calibrated_classifier"],
            key=lambda row: float(row["ambiguity_level"]),
        )
        if not subset:
            continue
        axes[0, 0].plot(
            [row["ambiguity_level"] for row in subset],
            [row["uncertainty_sequence_exact_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
        axes[0, 1].plot(
            [row["ambiguity_level"] for row in subset],
            [row["uncertainty_sequence_topk_delta"] for row in subset],
            marker="o",
            label=dataset,
        )
        if family_focus:
            axes[1, 0].plot(
                [row["ambiguity_level"] for row in subset],
                [row["trigram_family_delta"] for row in subset],
                marker="o",
                label=dataset,
            )
            axes[1, 1].plot(
                [row["ambiguity_level"] for row in subset],
                [row["conformal_trigram_family_delta"] for row in subset],
                marker="o",
                label=dataset,
            )
        else:
            axes[1, 0].plot(
                [row["ambiguity_level"] for row in subset],
                [row["conformal_coverage_delta"] for row in subset],
                marker="o",
                label=dataset,
            )
            axes[1, 1].plot(
                [row["ambiguity_level"] for row in subset],
                [row["conformal_set_size_delta"] for row in subset],
                marker="o",
                label=dataset,
            )
    for axis in axes.flatten():
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Ambiguity")
    axes[0, 0].set_title("Uncertainty Beam Sequence Exact Delta")
    axes[0, 0].set_ylabel("Delta")
    axes[0, 1].set_title("Uncertainty Beam Sequence Top-k Delta")
    axes[0, 1].set_ylabel("Delta")
    if family_focus:
        axes[1, 0].set_title("Trigram Family Delta")
        axes[1, 0].set_ylabel("Delta vs Uncertainty Beam")
        axes[1, 1].set_title("Conformal Trigram Family Delta")
        axes[1, 1].set_ylabel("Delta vs Trigram Beam")
    else:
        axes[1, 0].set_title("Conformal Coverage Delta")
        axes[1, 0].set_ylabel("Delta vs Uncertainty Beam")
        axes[1, 1].set_title("Conformal Set-Size Delta")
        axes[1, 1].set_ylabel("Delta vs Uncertainty Beam")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_summary_markdown(
    *,
    title: str,
    aggregate_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    family_focus: bool,
    scope_note: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        scope_note,
        "",
        "## Aggregate Effects",
    ]
    for row in aggregate_rows:
        detail = (
            ", mean uncertainty family delta = {unc_family}, mean trigram family delta = {trigram_family}, mean CRF family delta = {crf_family}, mean conformal trigram family delta = {conformal_trigram_family}, mean conformal CRF family delta = {conformal_crf_family}".format(
                unc_family=_fmt(row["mean_uncertainty_family_delta"]),
                trigram_family=_fmt(row["mean_trigram_family_delta"]),
                crf_family=_fmt(row["mean_crf_family_delta"]),
                conformal_trigram_family=_fmt(row["mean_conformal_trigram_family_delta"]),
                conformal_crf_family=_fmt(row["mean_conformal_crf_family_delta"]),
            )
            if family_focus
            else ", mean conformal coverage delta = {coverage}, mean CRF exact delta = {crf_exact}, peak exact delta at ambiguity {ambiguity}".format(
                coverage=_fmt(row["mean_conformal_coverage_delta"]),
                crf_exact=_fmt(row["mean_crf_sequence_exact_delta"]),
                ambiguity=_fmt(row["peak_ambiguity_for_uncertainty_exact_delta"]),
            )
        )
        lines.append(
            "- {dataset} / {task} ({strategy}): mean uncertainty exact delta = {exact}, mean uncertainty top-k delta = {topk}{detail}.".format(
                dataset=row["dataset"],
                task=row["task_name"],
                strategy=row["posterior_strategy_requested"],
                exact=_fmt(row["mean_uncertainty_sequence_exact_delta"]),
                topk=_fmt(row["mean_uncertainty_sequence_topk_delta"]),
                detail=detail,
            )
        )
    lines.extend(["", "## Ambiguity Regimes"])
    for row in regime_rows:
        regime_detail = (
            ", mean uncertainty family delta = {unc_family}, mean trigram family delta = {trigram_family}, mean CRF family delta = {crf_family}, mean conformal trigram family delta = {conformal_trigram_family}, mean conformal CRF family delta = {conformal_crf_family}".format(
                unc_family=_fmt(row["mean_uncertainty_family_delta"]),
                trigram_family=_fmt(row["mean_trigram_family_delta"]),
                crf_family=_fmt(row["mean_crf_family_delta"]),
                conformal_trigram_family=_fmt(row["mean_conformal_trigram_family_delta"]),
                conformal_crf_family=_fmt(row["mean_conformal_crf_family_delta"]),
            )
            if family_focus
            else ", mean conformal coverage delta = {coverage}, mean CRF exact delta = {crf_exact}".format(
                coverage=_fmt(row["mean_conformal_coverage_delta"]),
                crf_exact=_fmt(row["mean_crf_sequence_exact_delta"]),
            )
        )
        lines.append(
            "- {dataset} {strategy} {regime}: mean uncertainty exact delta = {exact}, mean uncertainty top-k delta = {topk}{detail}.".format(
                dataset=row["dataset"],
                strategy=row["posterior_strategy_requested"],
                regime=row["ambiguity_regime"],
                exact=_fmt(row["mean_uncertainty_sequence_exact_delta"]),
                topk=_fmt(row["mean_uncertainty_sequence_topk_delta"]),
                detail=regime_detail,
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


def build_sequence_cross_dataset_outputs(
    datasets: list[dict[str, str | Path]],
    output_root: str | Path = "outputs",
    *,
    output_prefix: str = "sequence_cross_dataset",
    markdown_title: str = "Sequence Cross-Dataset Summary",
    alias_stem: str | None = None,
) -> dict[str, str]:
    output_dir = ensure_directory(output_root)
    summary_rows, effect_rows, failure_rows, regime_rows = _collect_run_rows(datasets)
    family_focus = _family_signal_present(summary_rows)
    synthetic_flags = {
        bool(row.get("synthetic_from_real"))
        for row in summary_rows
        if row.get("synthetic_from_real") is not None
    }
    if synthetic_flags == {True}:
        scope_note = (
            "These results are synthetic-from-real sequence findings and should not be read as semantic decipherment evidence."
        )
    elif synthetic_flags == {False}:
        scope_note = (
            "These results come from a real grouped manifest-backed sequence dataset and should be read as preliminary grouped evidence, not semantic decipherment evidence."
        )
    else:
        scope_note = (
            "This summary mixes synthetic-from-real and real grouped sequence findings; any real grouped claims are preliminary and should not be read as semantic decipherment evidence."
        )

    summary_csv_path = output_dir / f"{output_prefix}_summary.csv"
    failure_csv_path = output_dir / f"{output_prefix}_failure_summary.csv"
    regime_csv_path = output_dir / f"{output_prefix}_ambiguity_regime_table.csv"
    effect_rows_path = output_dir / f"{output_prefix}_effect_rows.csv"
    aggregate_rows = _aggregate_effect_rows(effect_rows)
    tables_path = output_dir / f"{output_prefix}_tables.csv"
    figure_path = output_dir / f"{output_prefix}_effects_plot.png"
    summary_md_path = output_dir / f"{output_prefix}_summary.md"
    summary_json_path = output_dir / f"{output_prefix}_summary.json"

    write_csv(summary_csv_path, summary_rows)
    write_csv(failure_csv_path, failure_rows)
    write_csv(regime_csv_path, regime_rows)
    write_csv(effect_rows_path, effect_rows)
    write_csv(tables_path, aggregate_rows)
    _write_effects_plot(effect_rows, figure_path, family_focus=family_focus)
    summary_md = _build_summary_markdown(
        title=markdown_title,
        aggregate_rows=aggregate_rows,
        failure_rows=failure_rows,
        regime_rows=regime_rows,
        family_focus=family_focus,
        scope_note=scope_note,
    )
    write_text(summary_md_path, summary_md)
    write_json(
        summary_json_path,
        {
            "summary_rows": summary_rows,
            "aggregate_rows": aggregate_rows,
            "failure_rows": failure_rows,
            "regime_rows": regime_rows,
        },
    )

    if alias_stem is not None:
        write_csv(output_dir / f"{alias_stem}.csv", aggregate_rows)
        write_text(output_dir / f"{alias_stem}.md", summary_md)

    return {
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "effects_plot": str(figure_path),
        "failure_csv": str(failure_csv_path),
        "regime_csv": str(regime_csv_path),
        "tables_csv": str(tables_path),
        "effect_rows_csv": str(effect_rows_path),
    }


def build_sequence_decoder_comparison_outputs(
    task_inputs: list[dict[str, str | Path]],
    output_root: str | Path = "outputs",
) -> dict[str, str]:
    output_dir = ensure_directory(output_root)
    effect_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for task_input in task_inputs:
        task_label = str(task_input["task_label"])
        for row in _read_csv_rows(Path(task_input["effect_rows_csv"])):
            effect_rows.append(
                {
                    "task_name": task_label,
                    **{
                        key: (_to_float(value) if key not in {"dataset", "task_name", "ambiguity_regime", "posterior_strategy_requested"} else value)
                        for key, value in row.items()
                    },
                }
            )
        for row in _read_csv_rows(Path(task_input["failure_csv"])):
            failure_rows.append(
                {
                    "task_name": task_label,
                    "dataset": row["dataset"],
                    "posterior_strategy_requested": row["posterior_strategy_requested"],
                    "case_type": row["case_type"],
                    "ambiguity_level": _to_float(row["ambiguity_level"]),
                    "count": int(float(row["count"])),
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in effect_rows:
        key = (str(row["dataset"]), str(row["task_name"]), str(row["posterior_strategy_requested"]))
        grouped.setdefault(key, []).append(row)

    comparison_rows: list[dict[str, Any]] = []
    for (dataset, task_name, strategy), rows in sorted(grouped.items()):
        comparison_rows.append(
            {
                "dataset": dataset,
                "task_name": task_name,
                "posterior_strategy_requested": strategy,
                "mean_uncertainty_sequence_exact_delta": _mean(
                    [row.get("uncertainty_sequence_exact_delta") for row in rows]
                ),
                "mean_uncertainty_sequence_topk_delta": _mean(
                    [row.get("uncertainty_sequence_topk_delta") for row in rows]
                ),
                "mean_conformal_sequence_exact_delta": _mean(
                    [row.get("conformal_sequence_exact_delta") for row in rows]
                ),
                "mean_crf_sequence_exact_delta": _mean(
                    [row.get("crf_sequence_exact_delta") for row in rows]
                ),
                "mean_crf_sequence_topk_delta": _mean(
                    [row.get("crf_sequence_topk_delta") for row in rows]
                ),
                "mean_trigram_sequence_exact_delta": _mean(
                    [row.get("trigram_sequence_exact_delta") for row in rows]
                ),
                "mean_trigram_sequence_topk_delta": _mean(
                    [row.get("trigram_sequence_topk_delta") for row in rows]
                ),
                "mean_crf_family_delta": _mean(
                    [row.get("crf_family_delta") for row in rows]
                ),
                "mean_conformal_trigram_sequence_exact_delta": _mean(
                    [row.get("conformal_trigram_sequence_exact_delta") for row in rows]
                ),
                "mean_conformal_crf_sequence_exact_delta": _mean(
                    [row.get("conformal_crf_sequence_exact_delta") for row in rows]
                ),
                "mean_conformal_coverage_delta": _mean(
                    [row.get("conformal_coverage_delta") for row in rows]
                ),
                "mean_conformal_trigram_coverage_delta": _mean(
                    [row.get("conformal_trigram_coverage_delta") for row in rows]
                ),
                "mean_conformal_set_size_delta": _mean(
                    [row.get("conformal_set_size_delta") for row in rows]
                ),
                "mean_conformal_crf_coverage_delta": _mean(
                    [row.get("conformal_crf_coverage_delta") for row in rows]
                ),
                "mean_conformal_trigram_set_size_delta": _mean(
                    [row.get("conformal_trigram_set_size_delta") for row in rows]
                ),
                "mean_conformal_crf_set_size_delta": _mean(
                    [row.get("conformal_crf_set_size_delta") for row in rows]
                ),
                "mean_conformal_family_delta": _mean(
                    [row.get("conformal_family_delta") for row in rows]
                ),
                "mean_trigram_family_delta": _mean(
                    [row.get("trigram_family_delta") for row in rows]
                ),
                "mean_conformal_crf_family_delta": _mean(
                    [row.get("conformal_crf_family_delta") for row in rows]
                ),
                "mean_conformal_trigram_family_delta": _mean(
                    [row.get("conformal_trigram_family_delta") for row in rows]
                ),
                "best_ambiguity_for_crf_sequence_exact_delta": _peak_ambiguity(
                    rows,
                    "crf_sequence_exact_delta",
                ),
                "best_ambiguity_for_trigram_sequence_exact_delta": _peak_ambiguity(
                    rows,
                    "trigram_sequence_exact_delta",
                ),
                "best_ambiguity_for_trigram_family_delta": _peak_ambiguity(
                    rows,
                    "trigram_family_delta",
                ),
                "best_ambiguity_for_crf_family_delta": _peak_ambiguity(
                    rows,
                    "crf_family_delta",
                ),
            }
        )

    summary_csv_path = output_dir / "sequence_decoder_comparison_summary.csv"
    summary_md_path = output_dir / "sequence_decoder_comparison_summary.md"
    figure_path = output_dir / "sequence_decoder_comparison_plot.png"
    write_csv(summary_csv_path, comparison_rows)
    _write_decoder_comparison_plot(comparison_rows, figure_path)
    write_text(
        summary_md_path,
        _build_decoder_comparison_markdown(comparison_rows, failure_rows),
    )
    has_real_grouped_task = any(
        str(row["task_name"]) == "real_grouped_manifest_sequences"
        for row in comparison_rows
    )
    real_vs_synthetic_rows = [
        {
            "task_name": "real_glyph_markov_sequences",
            "evidence_scope": "synthetic_from_real",
            "real_grouped_sequence_data": False,
            "note": "Real glyph crops with generated sequence structure.",
        },
        {
            "task_name": "real_glyph_process_family_sequences",
            "evidence_scope": "synthetic_from_real",
            "real_grouped_sequence_data": False,
            "note": "Real glyph crops with generated family/rule labels.",
        },
        {
            "task_name": "real_grouped_sequence_dataset",
            "evidence_scope": (
                "preliminary_real_grouped"
                if has_real_grouped_task
                else "reported_in_separate_real_grouped_pack"
            ),
            "real_grouped_sequence_data": has_real_grouped_task,
            "note": (
                "Real grouped sequence data is included in this decoder comparison pack."
                if has_real_grouped_task
                else "Real grouped sequence evidence is reported separately under outputs/real_grouped_historical_newspapers."
            ),
        },
    ]
    write_csv(output_dir / "sequence_downstream_real_vs_synthetic_summary.csv", real_vs_synthetic_rows)
    return {
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "plot": str(figure_path),
        "real_vs_synthetic_csv": str(output_dir / "sequence_downstream_real_vs_synthetic_summary.csv"),
    }


def _write_decoder_comparison_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    filtered = [row for row in rows if row["posterior_strategy_requested"] == "calibrated_classifier"]
    tasks = sorted({str(row["task_name"]) for row in filtered})
    datasets = sorted({str(row["dataset"]) for row in filtered})
    if not filtered:
        return

    figure, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for axis, task_name in zip(axes[0], tasks):
        subset = [row for row in filtered if row["task_name"] == task_name]
        x = np.arange(len(datasets))
        width = 0.25
        by_dataset = {str(row["dataset"]): row for row in subset}
        uncertainty = [
            float(by_dataset.get(dataset, {}).get("mean_uncertainty_sequence_exact_delta") or 0.0)
            for dataset in datasets
        ]
        crf = [
            float(by_dataset.get(dataset, {}).get("mean_crf_sequence_exact_delta") or 0.0)
            for dataset in datasets
        ]
        trigram = [
            float(by_dataset.get(dataset, {}).get("mean_trigram_sequence_exact_delta") or 0.0)
            for dataset in datasets
        ]
        family = [
            float(by_dataset.get(dataset, {}).get("mean_trigram_family_delta") or 0.0)
            for dataset in datasets
        ]
        crf_family = [
            float(by_dataset.get(dataset, {}).get("mean_crf_family_delta") or 0.0)
            for dataset in datasets
        ]
        axis.bar(x - width, uncertainty, width, label="uncertainty exact delta")
        axis.bar(x, crf, width, label="CRF exact delta")
        axis.bar(x + width, trigram, width, label="trigram exact delta")
        if any(value != 0.0 for value in family):
            axis.plot(x, family, marker="o", linestyle="--", color="black", label="trigram family delta")
        if any(value != 0.0 for value in crf_family):
            axis.plot(x, crf_family, marker="s", linestyle=":", color="tab:red", label="CRF family delta")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=20, ha="right")
        axis.set_title(task_name)
        axis.set_ylabel("Mean delta")
    axes[0, -1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_decoder_comparison_markdown(
    comparison_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Sequence Decoder Comparison Summary",
        "",
        "This summary compares auditable decoder families on synthetic-from-real sequence tasks.",
        "",
        "## Aggregate Decoder Effects",
    ]
    for row in comparison_rows:
        lines.append(
            "- {dataset} / {task} ({strategy}): uncertainty exact delta = {unc_exact}, CRF exact delta = {crf_exact}, trigram exact delta = {tri_exact}, CRF family delta = {crf_family}, trigram family delta = {tri_family}, conformal trigram family delta = {conf_tri_family}.".format(
                dataset=row["dataset"],
                task=row["task_name"],
                strategy=row["posterior_strategy_requested"],
                unc_exact=_fmt(row["mean_uncertainty_sequence_exact_delta"]),
                crf_exact=_fmt(row["mean_crf_sequence_exact_delta"]),
                tri_exact=_fmt(row["mean_trigram_sequence_exact_delta"]),
                crf_family=_fmt(row["mean_crf_family_delta"]),
                tri_family=_fmt(row["mean_trigram_family_delta"]),
                conf_tri_family=_fmt(row["mean_conformal_trigram_family_delta"]),
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
