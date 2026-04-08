from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.config import DecipherLabConfig
from decipherlab.evaluation.metrics import canonical_family_name
from decipherlab.evaluation.statistics import MetricSampleGroup, bootstrap_grouped_mean, bootstrap_mean
from decipherlab.utils.io import write_csv, write_json, write_text


CI_METRICS = [
    "symbol_top1_accuracy",
    "symbol_topk_accuracy",
    "symbol_negative_log_likelihood",
    "symbol_expected_calibration_error",
    "family_topk_accuracy",
    "mean_structural_recovery_error",
]


def _condition_key(strategy: str, mode: str) -> str:
    if strategy == "cluster_distance" and mode == "fixed":
        return "A"
    if strategy == "calibrated_classifier" and mode == "fixed":
        return "B"
    if strategy == "cluster_distance" and mode == "uncertainty":
        return "C"
    return "D"


def condition_label(strategy: str, mode: str) -> str:
    prefix = _condition_key(strategy, mode)
    strategy_label = "Heuristic Posterior" if strategy == "cluster_distance" else "Calibrated Posterior"
    mode_label = "Fixed Transcript" if mode == "fixed" else "Uncertainty-Aware"
    return f"{prefix}. {mode_label} + {strategy_label}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _std(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.std(filtered))


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(right[key] - left[key])


def _family_topk_value(example_payload: dict[str, Any], top_k: int) -> float | None:
    true_family = example_payload.get("true_family")
    if true_family is None:
        return None
    canonical = canonical_family_name(true_family)
    ranked = [item["family"] for item in example_payload["ranking"]["evidences"][:top_k]]
    return float(canonical in ranked)


def _family_top1_value(example_payload: dict[str, Any]) -> float | None:
    true_family = example_payload.get("true_family")
    if true_family is None:
        return None
    canonical = canonical_family_name(true_family)
    ranked = example_payload["ranking"]["evidences"]
    if not ranked:
        return None
    return float(ranked[0]["family"] == canonical)


def _structural_error_value(example_payload: dict[str, Any]) -> float | None:
    reference = example_payload.get("reference_metrics")
    if reference is None:
        return None
    triage = example_payload["triage"]
    fields = [
        "unigram_entropy",
        "conditional_entropy",
        "index_of_coincidence",
        "repeat_rate",
        "compression_ratio",
    ]
    return sum(abs(triage[field] - reference[field]) for field in fields) / len(fields)


def _metric_group_from_cell(
    cell: dict[str, Any],
    metric_key: str,
    evaluation_top_k: int,
) -> MetricSampleGroup | None:
    values: list[float] = []
    weights: list[float] | None = []
    for example_payload in cell["example_payloads"]:
        symbol_metrics = example_payload["symbol_metrics"]
        breakdown = symbol_metrics["case_breakdown"]
        if metric_key == "symbol_top1_accuracy":
            value = symbol_metrics["top1_accuracy"]
            weight = breakdown["labeled_count"]
        elif metric_key == "symbol_topk_accuracy":
            value = symbol_metrics["topk_accuracy"]
            weight = breakdown["labeled_count"]
        elif metric_key == "symbol_negative_log_likelihood":
            value = symbol_metrics["negative_log_likelihood"]
            weight = breakdown["labeled_count"]
        elif metric_key == "symbol_expected_calibration_error":
            value = symbol_metrics["expected_calibration_error"]
            weight = breakdown["labeled_count"]
        elif metric_key == "family_topk_accuracy":
            value = _family_topk_value(example_payload, evaluation_top_k)
            weight = 1.0
        elif metric_key == "family_top1_accuracy":
            value = _family_top1_value(example_payload)
            weight = 1.0
        elif metric_key == "mean_structural_recovery_error":
            value = _structural_error_value(example_payload)
            weight = 1.0
        elif metric_key == "mean_posterior_entropy":
            value = example_payload["triage"]["mean_posterior_entropy"]
            weight = 1.0
        else:
            raise KeyError(f"Unsupported metric key: {metric_key}")
        if value is None:
            continue
        values.append(float(value))
        if weights is not None:
            weights.append(float(weight))
    if not values:
        return None
    weight_array = np.asarray(weights, dtype=float) if weights is not None else None
    if weight_array is not None and np.allclose(weight_array, 1.0):
        weight_array = None
    return MetricSampleGroup(values=np.asarray(values, dtype=float), weights=weight_array)


def _metric_groups(
    cells: list[dict[str, Any]],
    metric_key: str,
    evaluation_top_k: int,
) -> list[MetricSampleGroup]:
    groups: list[MetricSampleGroup] = []
    for cell in cells:
        group = _metric_group_from_cell(cell, metric_key=metric_key, evaluation_top_k=evaluation_top_k)
        if group is not None:
            groups.append(group)
    return groups


def build_ambiguity_sweep_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for comparison in sorted(
        comparisons,
        key=lambda item: (
            item["ambiguity_level"],
            item["seed"],
            _condition_key(item["posterior_strategy_requested"], item["posterior_mode"]),
        ),
    ):
        rows.append(
            {
                "seed": comparison["seed"],
                "ambiguity_level": comparison["ambiguity_level"],
                "condition": condition_label(comparison["posterior_strategy_requested"], comparison["posterior_mode"]),
                "posterior_mode": comparison["posterior_mode"],
                "posterior_strategy_requested": comparison["posterior_strategy_requested"],
                "posterior_strategy_actual": comparison["posterior_strategy"],
                "symbol_top1_accuracy": comparison["symbol_top1_accuracy"],
                "symbol_topk_accuracy": comparison["symbol_topk_accuracy"],
                "symbol_negative_log_likelihood": comparison["symbol_negative_log_likelihood"],
                "symbol_expected_calibration_error": comparison["symbol_expected_calibration_error"],
                "family_topk_accuracy": comparison["family_topk_accuracy"],
                "mean_structural_recovery_error": comparison["mean_structural_recovery_error"],
                "mean_posterior_entropy": comparison["mean_posterior_entropy"],
                "run_dir": comparison["run_dir"],
            }
        )
    return rows


def build_ambiguity_sweep_with_ci_rows(
    comparison_cells: list[dict[str, Any]],
    config: DecipherLabConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for cell in comparison_cells:
        key = (cell["ambiguity_level"], cell["posterior_strategy_requested"], cell["posterior_mode"])
        grouped.setdefault(key, []).append(cell)

    rows: list[dict[str, Any]] = []
    bootstrap_seed = (
        config.evaluation.bootstrap_seed
        if config.evaluation.bootstrap_seed is not None
        else config.experiment.seed + 7919
    )
    for row_index, ((ambiguity_level, strategy, mode), cells) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], _condition_key(item[0][1], item[0][2])))
    ):
        row: dict[str, Any] = {
            "ambiguity_level": ambiguity_level,
            "condition": condition_label(strategy, mode),
            "posterior_mode": mode,
            "posterior_strategy_requested": strategy,
            "posterior_strategy_actual": strategy,
            "seed_count": len({cell["seed"] for cell in cells}),
            "cell_count": len(cells),
            "run_dirs": ",".join(sorted(cell["run_dir"] for cell in cells)),
        }
        for metric_offset, metric_key in enumerate(CI_METRICS + ["mean_posterior_entropy"]):
            interval = bootstrap_grouped_mean(
                _metric_groups(cells, metric_key=metric_key, evaluation_top_k=config.evaluation.top_k),
                trials=config.evaluation.bootstrap_trials,
                confidence_level=config.evaluation.bootstrap_confidence_level,
                seed=bootstrap_seed + row_index * 101 + metric_offset,
            )
            row.update(interval.to_dict(metric_key))
        rows.append(row)
    return rows


def build_main_comparison_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for comparison in comparisons:
        key = (comparison["posterior_strategy_requested"], comparison["posterior_mode"])
        grouped.setdefault(key, []).append(comparison)
    rows: list[dict[str, Any]] = []
    for (strategy, mode), items in sorted(grouped.items(), key=lambda item: _condition_key(item[0][0], item[0][1])):
        rows.append(
            {
                "condition": condition_label(strategy, mode),
                "posterior_mode": mode,
                "posterior_strategy_requested": strategy,
                "mean_symbol_top1_accuracy": _mean([item["symbol_top1_accuracy"] for item in items]),
                "mean_symbol_topk_accuracy": _mean([item["symbol_topk_accuracy"] for item in items]),
                "mean_symbol_negative_log_likelihood": _mean([item["symbol_negative_log_likelihood"] for item in items]),
                "mean_symbol_expected_calibration_error": _mean(
                    [item["symbol_expected_calibration_error"] for item in items]
                ),
                "mean_family_topk_accuracy": _mean([item["family_topk_accuracy"] for item in items]),
                "mean_structural_recovery_error": _mean([item["mean_structural_recovery_error"] for item in items]),
                "mean_posterior_entropy": _mean([item["mean_posterior_entropy"] for item in items]),
                "seed_count": len({item["seed"] for item in items}),
                "ambiguity_levels": ",".join(
                    str(item["ambiguity_level"]) for item in sorted(items, key=lambda row: (row["ambiguity_level"], row["seed"]))
                ),
            }
        )
    return rows


def build_main_comparison_with_ci_rows(
    comparison_cells: list[dict[str, Any]],
    config: DecipherLabConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in comparison_cells:
        key = (cell["posterior_strategy_requested"], cell["posterior_mode"])
        grouped.setdefault(key, []).append(cell)

    rows: list[dict[str, Any]] = []
    bootstrap_seed = (
        config.evaluation.bootstrap_seed
        if config.evaluation.bootstrap_seed is not None
        else config.experiment.seed + 1543
    )
    for row_index, ((strategy, mode), cells) in enumerate(
        sorted(grouped.items(), key=lambda item: _condition_key(item[0][0], item[0][1]))
    ):
        row: dict[str, Any] = {
            "condition": condition_label(strategy, mode),
            "posterior_mode": mode,
            "posterior_strategy_requested": strategy,
            "seed_count": len({cell["seed"] for cell in cells}),
            "cell_count": len(cells),
            "ambiguity_levels": ",".join(str(level) for level in sorted({cell["ambiguity_level"] for cell in cells})),
        }
        metric_map = {
            "mean_symbol_top1_accuracy": "symbol_top1_accuracy",
            "mean_symbol_topk_accuracy": "symbol_topk_accuracy",
            "mean_symbol_negative_log_likelihood": "symbol_negative_log_likelihood",
            "mean_symbol_expected_calibration_error": "symbol_expected_calibration_error",
            "mean_family_topk_accuracy": "family_topk_accuracy",
            "mean_structural_recovery_error": "mean_structural_recovery_error",
            "mean_posterior_entropy": "mean_posterior_entropy",
        }
        for metric_offset, (column_name, metric_key) in enumerate(metric_map.items()):
            interval = bootstrap_grouped_mean(
                _metric_groups(cells, metric_key=metric_key, evaluation_top_k=config.evaluation.top_k),
                trials=config.evaluation.bootstrap_trials,
                confidence_level=config.evaluation.bootstrap_confidence_level,
                seed=bootstrap_seed + row_index * 101 + metric_offset,
            )
            row.update(interval.to_dict(column_name))
        rows.append(row)
    return rows


def build_calibration_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for comparison in sorted(
        comparisons,
        key=lambda item: (
            item["ambiguity_level"],
            item["seed"],
            _condition_key(item["posterior_strategy_requested"], item["posterior_mode"]),
        ),
    ):
        top1 = comparison["symbol_top1_accuracy"]
        topk = comparison["symbol_topk_accuracy"]
        rows.append(
            {
                "seed": comparison["seed"],
                "ambiguity_level": comparison["ambiguity_level"],
                "condition": condition_label(comparison["posterior_strategy_requested"], comparison["posterior_mode"]),
                "symbol_expected_calibration_error": comparison["symbol_expected_calibration_error"],
                "symbol_negative_log_likelihood": comparison["symbol_negative_log_likelihood"],
                "mean_posterior_entropy": comparison["mean_posterior_entropy"],
                "mean_correct_symbol_entropy": comparison["mean_correct_symbol_entropy"],
                "mean_incorrect_symbol_entropy": comparison["mean_incorrect_symbol_entropy"],
                "topk_minus_top1": None if top1 is None or topk is None else topk - top1,
            }
        )
    return rows


def build_seed_summary_rows(comparisons: list[dict[str, Any]], config: DecipherLabConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for comparison in comparisons:
        key = (comparison["ambiguity_level"], comparison["posterior_strategy_requested"], comparison["posterior_mode"])
        grouped.setdefault(key, []).append(comparison)

    rows: list[dict[str, Any]] = []
    bootstrap_seed = (
        config.evaluation.bootstrap_seed
        if config.evaluation.bootstrap_seed is not None
        else config.experiment.seed + 3571
    )
    metric_keys = {
        "mean_symbol_top1_accuracy": "symbol_top1_accuracy",
        "mean_symbol_topk_accuracy": "symbol_topk_accuracy",
        "mean_symbol_negative_log_likelihood": "symbol_negative_log_likelihood",
        "mean_symbol_expected_calibration_error": "symbol_expected_calibration_error",
        "mean_family_topk_accuracy": "family_topk_accuracy",
        "mean_structural_recovery_error": "mean_structural_recovery_error",
    }
    for row_index, ((ambiguity_level, strategy, mode), items) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], _condition_key(item[0][1], item[0][2])))
    ):
        row: dict[str, Any] = {
            "ambiguity_level": ambiguity_level,
            "condition": condition_label(strategy, mode),
            "posterior_mode": mode,
            "posterior_strategy_requested": strategy,
            "seed_count": len(items),
        }
        for metric_offset, (column_name, source_key) in enumerate(metric_keys.items()):
            values = [item[source_key] for item in items]
            interval = bootstrap_mean(
                values,
                trials=config.evaluation.bootstrap_trials,
                confidence_level=config.evaluation.bootstrap_confidence_level,
                seed=bootstrap_seed + row_index * 101 + metric_offset,
            )
            row.update(interval.to_dict(column_name))
            row[f"{column_name}_std"] = _std(values)
        rows.append(row)
    return rows


def build_pairwise_effect_rows(
    results_cube: dict[float, dict[str, dict[str, dict[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ambiguity_level, per_strategy in sorted(results_cube.items()):
        heuristic_fixed = per_strategy["cluster_distance"]["fixed"]["summary"]
        heuristic_uncertainty = per_strategy["cluster_distance"]["uncertainty"]["summary"]
        calibrated_fixed = per_strategy["calibrated_classifier"]["fixed"]["summary"]
        calibrated_uncertainty = per_strategy["calibrated_classifier"]["uncertainty"]["summary"]
        rows.append(
            {
                "ambiguity_level": ambiguity_level,
                "heuristic_uncertainty_top1_delta": _metric_delta(heuristic_fixed, heuristic_uncertainty, "symbol_top1_accuracy"),
                "heuristic_uncertainty_topk_delta": _metric_delta(heuristic_fixed, heuristic_uncertainty, "symbol_topk_accuracy"),
                "heuristic_uncertainty_nll_delta": _metric_delta(heuristic_fixed, heuristic_uncertainty, "symbol_negative_log_likelihood"),
                "heuristic_uncertainty_ece_delta": _metric_delta(heuristic_fixed, heuristic_uncertainty, "symbol_expected_calibration_error"),
                "calibrated_uncertainty_top1_delta": _metric_delta(calibrated_fixed, calibrated_uncertainty, "symbol_top1_accuracy"),
                "calibrated_uncertainty_topk_delta": _metric_delta(calibrated_fixed, calibrated_uncertainty, "symbol_topk_accuracy"),
                "calibrated_uncertainty_nll_delta": _metric_delta(calibrated_fixed, calibrated_uncertainty, "symbol_negative_log_likelihood"),
                "calibrated_uncertainty_ece_delta": _metric_delta(calibrated_fixed, calibrated_uncertainty, "symbol_expected_calibration_error"),
                "fixed_calibration_top1_delta": _metric_delta(heuristic_fixed, calibrated_fixed, "symbol_top1_accuracy"),
                "fixed_calibration_topk_delta": _metric_delta(heuristic_fixed, calibrated_fixed, "symbol_topk_accuracy"),
                "fixed_calibration_nll_delta": _metric_delta(heuristic_fixed, calibrated_fixed, "symbol_negative_log_likelihood"),
                "fixed_calibration_ece_delta": _metric_delta(heuristic_fixed, calibrated_fixed, "symbol_expected_calibration_error"),
                "uncertainty_calibration_top1_delta": _metric_delta(
                    heuristic_uncertainty,
                    calibrated_uncertainty,
                    "symbol_top1_accuracy",
                ),
                "uncertainty_calibration_topk_delta": _metric_delta(
                    heuristic_uncertainty,
                    calibrated_uncertainty,
                    "symbol_topk_accuracy",
                ),
                "uncertainty_calibration_nll_delta": _metric_delta(
                    heuristic_uncertainty,
                    calibrated_uncertainty,
                    "symbol_negative_log_likelihood",
                ),
                "uncertainty_calibration_ece_delta": _metric_delta(
                    heuristic_uncertainty,
                    calibrated_uncertainty,
                    "symbol_expected_calibration_error",
                ),
                "combined_top1_delta": _metric_delta(heuristic_fixed, calibrated_uncertainty, "symbol_top1_accuracy"),
                "combined_topk_delta": _metric_delta(heuristic_fixed, calibrated_uncertainty, "symbol_topk_accuracy"),
                "combined_nll_delta": _metric_delta(heuristic_fixed, calibrated_uncertainty, "symbol_negative_log_likelihood"),
                "combined_ece_delta": _metric_delta(
                    heuristic_fixed,
                    calibrated_uncertainty,
                    "symbol_expected_calibration_error",
                ),
            }
        )
    return rows


def build_pairwise_summary_rows(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in pairwise_rows:
        grouped.setdefault(row["ambiguity_level"], []).append(row)
    metric_keys = [
        "heuristic_uncertainty_top1_delta",
        "heuristic_uncertainty_topk_delta",
        "heuristic_uncertainty_nll_delta",
        "heuristic_uncertainty_ece_delta",
        "calibrated_uncertainty_top1_delta",
        "calibrated_uncertainty_topk_delta",
        "calibrated_uncertainty_nll_delta",
        "calibrated_uncertainty_ece_delta",
        "fixed_calibration_top1_delta",
        "fixed_calibration_topk_delta",
        "fixed_calibration_nll_delta",
        "fixed_calibration_ece_delta",
        "uncertainty_calibration_top1_delta",
        "uncertainty_calibration_topk_delta",
        "uncertainty_calibration_nll_delta",
        "uncertainty_calibration_ece_delta",
        "combined_top1_delta",
        "combined_topk_delta",
        "combined_nll_delta",
        "combined_ece_delta",
    ]
    rows: list[dict[str, Any]] = []
    for ambiguity_level, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "ambiguity_level": ambiguity_level,
            "seed_count": len(items),
        }
        for key in metric_keys:
            values = [item[key] for item in items]
            row[key] = _mean(values)
            row[f"{key}_std"] = _std(values)
        rows.append(row)
    return rows


def build_sequence_level_rows(
    comparison_cells: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for cell in comparison_cells:
        key = (cell["ambiguity_level"], cell["posterior_strategy_requested"], cell["posterior_mode"])
        grouped.setdefault(key, []).append(cell)

    for (ambiguity_level, strategy, mode), cells in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], _condition_key(item[0][1], item[0][2])),
    ):
        symbol_rescue_sequences = 0
        family_labeled_sequences = 0
        family_topk_correct = 0
        structural_errors: list[float] = []
        sequence_count = 0
        for cell in cells:
            for example_payload in cell["example_payloads"]:
                sequence_count += 1
                if example_payload["symbol_metrics"]["case_breakdown"]["collapse_rescued_by_topk"] > 0:
                    symbol_rescue_sequences += 1
                family_value = _family_topk_value(example_payload, top_k)
                if family_value is not None:
                    family_labeled_sequences += 1
                    family_topk_correct += int(family_value)
                structural_value = _structural_error_value(example_payload)
                if structural_value is not None:
                    structural_errors.append(structural_value)
        rows.append(
            {
                "ambiguity_level": ambiguity_level,
                "condition": condition_label(strategy, mode),
                "sequence_count": sequence_count,
                "symbol_rescue_sequences": symbol_rescue_sequences,
                "family_labeled_sequences": family_labeled_sequences,
                "family_topk_accuracy": (
                    family_topk_correct / family_labeled_sequences if family_labeled_sequences else None
                ),
                "mean_structural_recovery_error": (
                    float(np.mean(structural_errors)) if structural_errors else None
                ),
            }
        )
    return rows


def _plot_condition_metric(
    path_root: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.2))
    conditions = []
    for row in rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    for condition in conditions:
        matching = [row for row in rows if row["condition"] == condition]
        xs = [row["ambiguity_level"] for row in matching if row[metric_key] is not None]
        ys = [row[metric_key] for row in matching if row[metric_key] is not None]
        lowers = [row.get(f"{metric_key}_ci_lower") for row in matching if row[metric_key] is not None]
        uppers = [row.get(f"{metric_key}_ci_upper") for row in matching if row[metric_key] is not None]
        if xs:
            axis.plot(xs, ys, marker="o", label=condition)
            if all(value is not None for value in lowers + uppers):
                axis.fill_between(xs, lowers, uppers, alpha=0.15)
    axis.set_title(title)
    axis.set_xlabel("Ambiguity level")
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_root.with_suffix(".png"), dpi=180)
    fig.savefig(path_root.with_suffix(".pdf"))
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def _build_figure_captions() -> str:
    return "\n".join(
        [
            "# Figure Captions",
            "",
            "- `comparison_symbol_top1`: Symbol top-1 accuracy across ambiguity levels for all four baseline conditions, with bootstrap uncertainty bands when available.",
            "- `comparison_symbol_topk`: Symbol top-k accuracy across ambiguity levels for all four baseline conditions, with bootstrap uncertainty bands when available.",
            "- `comparison_symbol_nll`: Symbol negative log-likelihood across ambiguity levels for all four baseline conditions, with bootstrap uncertainty bands when available.",
            "- `comparison_symbol_ece`: Symbol expected calibration error across ambiguity levels for all four baseline conditions, with bootstrap uncertainty bands when available.",
            "- `comparison_family_topk`: Downstream family top-k accuracy across ambiguity levels for all four baseline conditions when family labels are available.",
        ]
    ) + "\n"


def _build_results_summary(
    main_ci_rows: list[dict[str, Any]],
    pairwise_summary_rows: list[dict[str, Any]],
) -> str:
    by_condition = {row["condition"]: row for row in main_ci_rows}
    baseline = by_condition.get(condition_label("cluster_distance", "fixed"))
    strongest = by_condition.get(condition_label("calibrated_classifier", "uncertainty"))
    lines = [
        "# Results Section Draft",
        "",
        "The current evidence package compares four controlled conditions: fixed vs uncertainty-aware inference crossed with heuristic vs calibrated posterior generation.",
    ]
    if baseline and strongest:
        lines.extend(
            [
                "",
                "Across the configured ambiguity sweep:",
                f"- `{baseline['condition']}` reached mean symbol top-k accuracy {_fmt(baseline['mean_symbol_topk_accuracy'])} "
                f"[{_fmt(baseline['mean_symbol_topk_accuracy_ci_lower'])}, {_fmt(baseline['mean_symbol_topk_accuracy_ci_upper'])}].",
                f"- `{strongest['condition']}` reached mean symbol top-k accuracy {_fmt(strongest['mean_symbol_topk_accuracy'])} "
                f"[{_fmt(strongest['mean_symbol_topk_accuracy_ci_lower'])}, {_fmt(strongest['mean_symbol_topk_accuracy_ci_upper'])}].",
                f"- The baseline symbol NLL was {_fmt(baseline['mean_symbol_negative_log_likelihood'])}, "
                f"while the combined condition achieved {_fmt(strongest['mean_symbol_negative_log_likelihood'])}.",
            ]
        )
    if pairwise_summary_rows:
        latest = pairwise_summary_rows[-1]
        lines.extend(
            [
                "",
                f"At the highest ambiguity level tested ({_fmt(latest['ambiguity_level'])}), the combined condition changed symbol top-k accuracy by {_fmt(latest['combined_topk_delta'])} on average across seeds.",
                "These statements are limited to the tested ambiguity protocol and available labels; they are not historical-generalization claims.",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_limitations_summary(
    main_ci_rows: list[dict[str, Any]],
    failure_payload: dict[str, Any],
    config: DecipherLabConfig,
) -> str:
    lines = [
        "# Limitations Section Draft",
        "",
        "- The current evidence supports symbol-level uncertainty retention under ambiguity, not full decipherment or historical semantic recovery.",
    ]
    if all(row.get("mean_family_topk_accuracy") in {0.0, None} for row in main_ci_rows):
        lines.append("- Downstream family-ranking gains are not established on the current protocol because family top-k accuracy remains absent or near-zero.")
    if any(row["case_type"] == "calibration_worsened_or_unstable" for row in failure_payload["summary_rows"]):
        lines.append("- Calibration is not uniformly beneficial: at least one measured condition showed worsened or unstable calibration-related behavior.")
    if len(config.experiment.seed_sweep) == 0:
        lines.append("- Robustness across multiple random seeds is available in the codebase, but this run used only a single seed.")
    lines.append("- Stronger publication evidence still requires a larger real labeled crop dataset with meaningful downstream labels.")
    return "\n".join(lines) + "\n"


def _build_experiment_metadata(
    config: DecipherLabConfig,
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    manifests = sorted({row["dataset_manifest"] for row in comparisons if row.get("dataset_manifest") is not None})
    return {
        "experiment_name": config.experiment.name,
        "base_seed": config.experiment.seed,
        "seed_sweep": config.experiment.seed_sweep,
        "dataset_source": config.dataset.source,
        "dataset_manifest": manifests[0] if manifests else None,
        "comparison_strategies": config.evaluation.comparison_strategies,
        "ambiguity_levels": sorted({row["ambiguity_level"] for row in comparisons}),
        "evaluation_top_k": config.evaluation.top_k,
        "bootstrap_trials": config.evaluation.bootstrap_trials,
        "bootstrap_confidence_level": config.evaluation.bootstrap_confidence_level,
        "train_split": config.dataset.train_split,
        "val_split": config.dataset.val_split,
        "evaluation_split": config.dataset.evaluation_split,
    }


def write_results_pack(
    run_dir: str | Path,
    config: DecipherLabConfig,
    comparisons: list[dict[str, Any]],
    comparison_cells: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    failure_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output_dir = Path(run_dir)
    ambiguity_rows = build_ambiguity_sweep_rows(comparisons)
    ambiguity_ci_rows = build_ambiguity_sweep_with_ci_rows(comparison_cells, config)
    main_rows = build_main_comparison_rows(comparisons)
    main_ci_rows = build_main_comparison_with_ci_rows(comparison_cells, config)
    seed_summary_rows = build_seed_summary_rows(comparisons, config)
    calibration_rows = build_calibration_rows(comparisons)
    pairwise_summary_rows = build_pairwise_summary_rows(pairwise_rows)
    sequence_level_rows = build_sequence_level_rows(comparison_cells, top_k=config.evaluation.top_k)

    write_json(output_dir / "main_comparison_table.json", main_rows)
    write_csv(output_dir / "main_comparison_table.csv", main_rows)
    write_json(output_dir / "main_comparison_with_ci.json", main_ci_rows)
    write_csv(output_dir / "main_comparison_with_ci.csv", main_ci_rows)
    write_text(
        output_dir / "main_comparison_table.md",
        _markdown_table(
            main_ci_rows,
            [
                "condition",
                "mean_symbol_top1_accuracy",
                "mean_symbol_top1_accuracy_ci_lower",
                "mean_symbol_top1_accuracy_ci_upper",
                "mean_symbol_topk_accuracy",
                "mean_symbol_topk_accuracy_ci_lower",
                "mean_symbol_topk_accuracy_ci_upper",
                "mean_symbol_negative_log_likelihood",
                "mean_symbol_expected_calibration_error",
                "mean_family_topk_accuracy",
            ],
        ),
    )

    write_json(output_dir / "ambiguity_sweep_table.json", ambiguity_rows)
    write_csv(output_dir / "ambiguity_sweep_table.csv", ambiguity_rows)
    write_json(output_dir / "ambiguity_sweep_with_ci.json", ambiguity_ci_rows)
    write_csv(output_dir / "ambiguity_sweep_with_ci.csv", ambiguity_ci_rows)
    write_json(output_dir / "seed_summary.json", seed_summary_rows)
    write_csv(output_dir / "seed_summary.csv", seed_summary_rows)
    write_json(output_dir / "calibration_table.json", calibration_rows)
    write_csv(output_dir / "calibration_table.csv", calibration_rows)
    write_json(output_dir / "pairwise_effect_table.json", pairwise_rows)
    write_csv(output_dir / "pairwise_effect_table.csv", pairwise_rows)
    write_json(output_dir / "pairwise_effect_summary.json", pairwise_summary_rows)
    write_csv(output_dir / "pairwise_effect_summary.csv", pairwise_summary_rows)
    write_json(output_dir / "sequence_level_summary.json", sequence_level_rows)
    write_csv(output_dir / "sequence_level_summary.csv", sequence_level_rows)
    write_json(output_dir / "failure_case_summary.json", failure_payload["summary_rows"])
    write_csv(output_dir / "failure_case_summary.csv", failure_payload["summary_rows"])
    write_json(output_dir / "failure_cases.json", failure_payload["failure_cases"])
    write_text(
        output_dir / "failure_case_summary.md",
        _markdown_table(
            failure_payload["summary_rows"],
            ["case_type", "ambiguity_level", "condition_group", "count"],
        ),
    )

    _plot_condition_metric(
        output_dir / "comparison_symbol_top1",
        ambiguity_ci_rows,
        metric_key="symbol_top1_accuracy",
        title="Symbol Top-1 Accuracy vs Ambiguity",
        ylabel="Accuracy",
    )
    _plot_condition_metric(
        output_dir / "comparison_symbol_topk",
        ambiguity_ci_rows,
        metric_key="symbol_topk_accuracy",
        title="Symbol Top-k Accuracy vs Ambiguity",
        ylabel="Accuracy",
    )
    _plot_condition_metric(
        output_dir / "comparison_symbol_nll",
        ambiguity_ci_rows,
        metric_key="symbol_negative_log_likelihood",
        title="Symbol NLL vs Ambiguity",
        ylabel="NLL",
    )
    _plot_condition_metric(
        output_dir / "comparison_symbol_ece",
        ambiguity_ci_rows,
        metric_key="symbol_expected_calibration_error",
        title="Symbol Calibration Error vs Ambiguity",
        ylabel="ECE",
    )
    _plot_condition_metric(
        output_dir / "comparison_family_topk",
        ambiguity_ci_rows,
        metric_key="family_topk_accuracy",
        title="Family Top-k Accuracy vs Ambiguity",
        ylabel="Accuracy",
    )

    write_text(output_dir / "figure_captions.md", _build_figure_captions())
    write_text(output_dir / "draft_results_subsection.md", _build_results_summary(main_ci_rows, pairwise_summary_rows))
    write_text(output_dir / "results_section_draft.md", _build_results_summary(main_ci_rows, pairwise_summary_rows))
    write_text(output_dir / "limitations_section_draft.md", _build_limitations_summary(main_ci_rows, failure_payload, config))

    metadata = _build_experiment_metadata(config, comparisons)
    write_json(output_dir / "experiment_metadata.json", metadata)
    write_text(
        output_dir / "experiment_metadata.md",
        "\n".join(
            [
                "# Experiment Metadata",
                "",
                f"- Experiment: `{metadata['experiment_name']}`",
                f"- Base seed: `{metadata['base_seed']}`",
                f"- Seed sweep: `{metadata['seed_sweep']}`",
                f"- Dataset source: `{metadata['dataset_source']}`",
                f"- Dataset manifest: `{metadata['dataset_manifest']}`",
                f"- Comparison strategies: `{', '.join(metadata['comparison_strategies'])}`",
                f"- Ambiguity levels: `{metadata['ambiguity_levels']}`",
                f"- Bootstrap trials: `{metadata['bootstrap_trials']}` at confidence `{metadata['bootstrap_confidence_level']}`",
                f"- Splits: train=`{metadata['train_split']}`, val=`{metadata['val_split']}`, test=`{metadata['evaluation_split']}`",
            ]
        )
        + "\n",
    )

    return {
        "main_rows": main_rows,
        "main_ci_rows": main_ci_rows,
        "ambiguity_rows": ambiguity_rows,
        "ambiguity_ci_rows": ambiguity_ci_rows,
        "calibration_rows": calibration_rows,
        "pairwise_rows": pairwise_rows,
        "pairwise_summary_rows": pairwise_summary_rows,
        "seed_summary_rows": seed_summary_rows,
    }
