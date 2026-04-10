from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-sequence-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.sequence.propagation import (
    ambiguity_regime_label,
    best_threshold_split,
    bootstrap_mean_ci,
    fit_regularized_logistic_regression,
)
from decipherlab.utils.io import write_csv, write_text


RUN_SPECS = [
    {
        "dataset": "omniglot",
        "task_group": "synthetic_markov",
        "cluster_pattern": "*sequence_omniglot_markov_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_omniglot_markov_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "sklearn_digits",
        "task_group": "synthetic_markov",
        "cluster_pattern": "*sequence_sklearn_digits_markov_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_sklearn_digits_markov_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "kuzushiji49",
        "task_group": "synthetic_markov",
        "cluster_pattern": "*sequence_kuzushiji_markov_cross_dataset_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_kuzushiji_markov_cross_dataset_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "omniglot",
        "task_group": "synthetic_process_family",
        "cluster_pattern": "*sequence_omniglot_process_family_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_omniglot_process_family_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "sklearn_digits",
        "task_group": "synthetic_process_family",
        "cluster_pattern": "*sequence_sklearn_digits_process_family_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_sklearn_digits_process_family_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "kuzushiji49",
        "task_group": "synthetic_process_family",
        "cluster_pattern": "*sequence_kuzushiji_process_family_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_kuzushiji_process_family_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "historical_newspapers",
        "task_group": "real_grouped_downstream_redesigned",
        "cluster_pattern": "*sequence_historical_newspapers_real_downstream_redesigned_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_historical_newspapers_real_downstream_redesigned_sequence_branch_calibrated_classifier",
    },
    {
        "dataset": "scadsai",
        "task_group": "real_grouped_downstream_redesigned",
        "cluster_pattern": "*sequence_scadsai_real_downstream_redesigned_sequence_branch_cluster_distance",
        "calibrated_pattern": "*sequence_scadsai_real_downstream_redesigned_sequence_branch_calibrated_classifier",
    },
]


def _latest_run(pattern: str) -> Path:
    candidates = sorted(Path("outputs/runs").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No runs match pattern: {pattern}")
    return candidates[-1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _render(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return _fmt(float(value))
    return str(value)


def _safe_delta(left: dict[str, str], right: dict[str, str], key: str) -> float | None:
    left_value = _to_float(left.get(key))
    right_value = _to_float(right.get(key))
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _downstream_kind(row: dict[str, str], task_group: str) -> str:
    if row.get("real_downstream_task_name"):
        return "real_downstream"
    if task_group == "synthetic_process_family":
        return "synthetic_family"
    return "none"


def _downstream_exact_value(row: dict[str, str], task_group: str) -> float | None:
    if row.get("real_downstream_task_name"):
        return _to_float(row.get("real_downstream_exact_match"))
    if task_group == "synthetic_process_family":
        return _to_float(row.get("family_identification_accuracy"))
    return None


def _downstream_partial_value(row: dict[str, str], task_group: str) -> float | None:
    if row.get("real_downstream_task_name"):
        return _to_float(row.get("real_downstream_token_accuracy"))
    if task_group == "synthetic_process_family":
        return _to_float(row.get("family_identification_topk_recovery"))
    return None


def _load_coverage_map() -> dict[tuple[str, str, str], float]:
    rows = _read_csv(Path("outputs/real_downstream_coverage_analysis.csv"))
    return {
        (row["dataset"], row["task_name"], row["coverage_metric"]): float(row["value"])
        for row in rows
    }


def _build_comparison_rows() -> list[dict[str, Any]]:
    coverage_map = _load_coverage_map()
    comparison_rows: list[dict[str, Any]] = []
    real_dataset_key_map = {
        "historical_newspapers": "historical_newspapers_real_grouped_gold",
        "scadsai": "scadsai_real_grouped",
    }
    for spec in RUN_SPECS:
        for strategy_name, pattern in [
            ("cluster_distance", spec["cluster_pattern"]),
            ("calibrated_classifier", spec["calibrated_pattern"]),
        ]:
            run_dir = _latest_run(pattern)
            example_rows = _read_csv(run_dir / "sequence_branch_examples.csv")
            dataset_summary = _read_json(run_dir / "dataset_summary.json")
            grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
            for row in example_rows:
                grouped[(row["ambiguity_level"], row["example_id"])][row["method"]] = row
            for (ambiguity_level_raw, example_id), methods in grouped.items():
                fixed = methods.get("fixed_greedy")
                uncertainty = methods.get("uncertainty_beam")
                conformal = methods.get("conformal_beam")
                if fixed is None or uncertainty is None:
                    continue
                candidates = [
                    ("uncertainty_beam", uncertainty, fixed, "raw_uncertainty"),
                ]
                if conformal is not None:
                    candidates.append(("conformal_beam", conformal, uncertainty, "conformal"))
                for method_name, method_row, baseline_row, method_family in candidates:
                    ambiguity_level = float(ambiguity_level_raw)
                    downstream_kind = _downstream_kind(method_row, spec["task_group"])
                    downstream_exact = _downstream_exact_value(method_row, spec["task_group"])
                    baseline_downstream_exact = _downstream_exact_value(baseline_row, spec["task_group"])
                    downstream_partial = _downstream_partial_value(method_row, spec["task_group"])
                    baseline_downstream_partial = _downstream_partial_value(baseline_row, spec["task_group"])
                    comparison_rows.append(
                        {
                            "dataset": spec["dataset"],
                            "task_group": spec["task_group"],
                            "example_id": example_id,
                            "run_dir": str(run_dir),
                            "synthetic_from_real": float(dataset_summary["synthetic_from_real"]),
                            "is_real_grouped": float(not dataset_summary["synthetic_from_real"]),
                            "ambiguity_level": ambiguity_level,
                            "ambiguity_regime": ambiguity_regime_label(ambiguity_level),
                            "posterior_strategy_requested": strategy_name,
                            "is_calibrated": float(strategy_name == "calibrated_classifier"),
                            "method": method_name,
                            "baseline_method": baseline_row["method"],
                            "method_family": method_family,
                            "is_conformal": float(method_family == "conformal"),
                            "sequence_length": float(method_row["sequence_length"]),
                            "symbol_topk_accuracy": _to_float(method_row["symbol_topk_accuracy"]),
                            "grouped_topk_success": _to_float(method_row["sequence_topk_recovery"]),
                            "grouped_exact_success": _to_float(method_row["sequence_exact_match"]),
                            "grouped_token_accuracy": _to_float(method_row["sequence_token_accuracy"]),
                            "downstream_kind": downstream_kind,
                            "downstream_exact_success": downstream_exact,
                            "downstream_partial_success": downstream_partial,
                            "symbol_topk_delta": _safe_delta(method_row, baseline_row, "symbol_topk_accuracy"),
                            "grouped_topk_delta": _safe_delta(method_row, baseline_row, "sequence_topk_recovery"),
                            "grouped_exact_delta": _safe_delta(method_row, baseline_row, "sequence_exact_match"),
                            "grouped_token_delta": _safe_delta(method_row, baseline_row, "sequence_token_accuracy"),
                            "downstream_exact_delta": None
                            if downstream_exact is None or baseline_downstream_exact is None
                            else float(downstream_exact - baseline_downstream_exact),
                            "downstream_partial_delta": None
                            if downstream_partial is None or baseline_downstream_partial is None
                            else float(downstream_partial - baseline_downstream_partial),
                            "symbol_rescue": float((_safe_delta(method_row, baseline_row, "symbol_topk_accuracy") or 0.0) > 0.0),
                            "grouped_topk_rescue": float((_safe_delta(method_row, baseline_row, "sequence_topk_recovery") or 0.0) > 0.0),
                            "grouped_exact_rescue": float((_safe_delta(method_row, baseline_row, "sequence_exact_match") or 0.0) > 0.0),
                            "downstream_exact_rescue": None
                            if downstream_exact is None or baseline_downstream_exact is None
                            else float(downstream_exact > baseline_downstream_exact),
                            "downstream_partial_rescue": None
                            if downstream_partial is None or baseline_downstream_partial is None
                            else float(downstream_partial > baseline_downstream_partial),
                            "prediction_set_coverage": _to_float(method_row.get("prediction_set_coverage")),
                            "prediction_set_avg_size": _to_float(method_row.get("prediction_set_avg_size")),
                            "prediction_set_singleton_rate": _to_float(method_row.get("prediction_set_singleton_rate")),
                            "prediction_set_rescue_rate": _to_float(method_row.get("prediction_set_rescue_rate")),
                            "mean_confusion_entropy": _to_float(method_row.get("mean_confusion_entropy")),
                            "mean_confusion_set_size": _to_float(method_row.get("mean_confusion_set_size")),
                            "real_downstream_bank_coverage": _to_float(method_row.get("real_downstream_bank_coverage")),
                            "dataset_support_upper_bound": coverage_map.get(
                                (
                                    real_dataset_key_map.get(spec["dataset"], spec["dataset"]),
                                    "train_supported_ngram_path",
                                    "full_path_coverage",
                                )
                            )
                            if spec["task_group"] == "real_grouped_downstream_redesigned"
                            else None,
                            "full_support_indicator": float(_to_float(method_row.get("real_downstream_exact_match_if_covered")) is not None)
                            if downstream_kind == "real_downstream"
                            else None,
                        }
                    )
    return comparison_rows


def _feature_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Propagation Features",
        "",
        "## Main Summary",
        "",
    ]
    for task_group in sorted({str(row["task_group"]) for row in rows}):
        subset = [row for row in rows if row["task_group"] == task_group]
        lines.append(
            f"- `{task_group}`: `{len(subset)}` comparison rows, symbol rescue rate `{_fmt(_mean([row['symbol_rescue'] for row in subset]))}`, grouped top-k rescue rate `{_fmt(_mean([row['grouped_topk_rescue'] for row in subset]))}`."
        )
    return "\n".join(lines) + "\n"


def _rows_to_markdown(title: str, rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = [f"# {title}", "", "## Rows", ""]
    for row in rows:
        parts = [f"`{key}`=`{_render(row.get(key))}`" for key in keys]
        lines.append(f"- " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def _fit_grouped_model(rows: list[dict[str, Any]], model_name: str, analysis_scope: str) -> list[dict[str, Any]]:
    grouped_model = fit_regularized_logistic_regression(
        rows=rows,
        target_key="grouped_topk_rescue",
        continuous_features=[
            "symbol_topk_delta",
            "ambiguity_level",
            "mean_confusion_entropy",
            "prediction_set_avg_size",
            "sequence_length",
            "real_downstream_bank_coverage",
        ],
        binary_features=["is_conformal", "is_calibrated", "synthetic_from_real", "symbol_rescue"],
        categorical_features={"dataset": ["omniglot", "sklearn_digits", "kuzushiji49", "historical_newspapers", "scadsai"]},
    )
    output_rows = grouped_model.coefficient_rows(model_name)
    for row in output_rows:
        row["analysis_scope"] = analysis_scope
        row["sample_count"] = len(rows)
    return output_rows


def _fit_downstream_model(rows: list[dict[str, Any]], model_name: str, analysis_scope: str) -> list[dict[str, Any]]:
    real_rows = [
        row
        for row in rows
        if row["downstream_kind"] == "real_downstream" and row["downstream_exact_rescue"] is not None
    ]
    if not real_rows:
        return []
    downstream_model = fit_regularized_logistic_regression(
        rows=real_rows,
        target_key="downstream_exact_rescue",
        continuous_features=[
            "ambiguity_level",
            "mean_confusion_entropy",
            "prediction_set_avg_size",
            "sequence_length",
            "real_downstream_bank_coverage",
            "grouped_topk_delta",
        ],
        binary_features=["is_conformal", "is_calibrated", "symbol_rescue", "grouped_topk_rescue"],
        categorical_features={"dataset": ["historical_newspapers", "scadsai"]},
    )
    output_rows = downstream_model.coefficient_rows(model_name)
    for row in output_rows:
        row["analysis_scope"] = analysis_scope
        row["sample_count"] = len(real_rows)
    return output_rows


def _build_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_rows: list[dict[str, Any]] = []
    model_rows.extend(_fit_grouped_model(rows, "grouped_topk_rescue_model", "all_rows"))
    for strategy_name in ["cluster_distance", "calibrated_classifier"]:
        subset = [row for row in rows if row["posterior_strategy_requested"] == strategy_name]
        model_rows.extend(
            _fit_grouped_model(
                subset,
                "grouped_topk_rescue_model",
                f"posterior={strategy_name}",
            )
        )
    model_rows.extend(_fit_downstream_model(rows, "real_downstream_exact_rescue_model", "all_rows"))
    for strategy_name in ["cluster_distance", "calibrated_classifier"]:
        subset = [row for row in rows if row["posterior_strategy_requested"] == strategy_name]
        model_rows.extend(
            _fit_downstream_model(
                subset,
                "real_downstream_exact_rescue_model",
                f"posterior={strategy_name}",
            )
        )
    return model_rows


def _threshold_rows_for_scope(
    rows: list[dict[str, Any]],
    *,
    analysis_scope: str,
    min_group_size: int,
) -> list[dict[str, Any]]:
    real_rows = [
        row
        for row in rows
        if row["downstream_kind"] == "real_downstream" and row["downstream_exact_rescue"] is not None
    ]
    threshold_rows: list[dict[str, Any]] = []
    for feature in [
        "real_downstream_bank_coverage",
        "mean_confusion_entropy",
        "prediction_set_avg_size",
        "ambiguity_level",
        "grouped_topk_delta",
    ]:
        best = best_threshold_split(real_rows, feature, "downstream_exact_rescue", min_group_size=min_group_size)
        if best is not None:
            best["analysis_scope"] = analysis_scope
            threshold_rows.append(best)
    grouped_best = best_threshold_split(rows, "mean_confusion_entropy", "grouped_topk_rescue", min_group_size=max(min_group_size, 20))
    if grouped_best is not None:
        grouped_best["feature"] = "mean_confusion_entropy_for_grouped_rescue"
        grouped_best["analysis_scope"] = analysis_scope
        threshold_rows.append(grouped_best)
    return threshold_rows


def _build_threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    threshold_rows: list[dict[str, Any]] = []
    threshold_rows.extend(_threshold_rows_for_scope(rows, analysis_scope="all_rows", min_group_size=12))
    for strategy_name in ["cluster_distance", "calibrated_classifier"]:
        subset = [row for row in rows if row["posterior_strategy_requested"] == strategy_name]
        threshold_rows.extend(
            _threshold_rows_for_scope(
                subset,
                analysis_scope=f"posterior={strategy_name}",
                min_group_size=10,
            )
        )
    return threshold_rows


def _add_threshold_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["analysis_scope"], row["feature"])].append(row)
    for (analysis_scope, feature), subset in sorted(grouped.items()):
        thresholds = [float(row["threshold"]) for row in subset]
        gap_values = [float(row["gap"]) for row in subset]
        directions = [str(row["direction"]) for row in subset]
        direction_mode = max(set(directions), key=directions.count)
        stable_rows.append(
            {
                "analysis_scope": analysis_scope,
                "feature": feature,
                "threshold": float(np.mean(thresholds)),
                "threshold_min": float(np.min(thresholds)),
                "threshold_max": float(np.max(thresholds)),
                "low_rate": float(np.mean([float(row["low_rate"]) for row in subset])),
                "high_rate": float(np.mean([float(row["high_rate"]) for row in subset])),
                "gap": float(np.mean(gap_values)),
                "direction": direction_mode,
                "direction_stability": float(np.mean([direction == direction_mode for direction in directions])),
                "low_count": int(np.mean([int(row["low_count"]) for row in subset])),
                "high_count": int(np.mean([int(row["high_count"]) for row in subset])),
            }
        )
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[str(row["feature"])].append(row)
    for feature, subset in sorted(by_feature.items()):
        thresholds = [float(row["threshold"]) for row in subset]
        gap_values = [float(row["gap"]) for row in subset]
        directions = [str(row["direction"]) for row in subset]
        direction_mode = max(set(directions), key=directions.count)
        stable_rows.append(
            {
                "analysis_scope": "across_scopes",
                "feature": feature,
                "threshold": float(np.mean(thresholds)),
                "threshold_min": float(np.min(thresholds)),
                "threshold_max": float(np.max(thresholds)),
                "low_rate": float(np.mean([float(row["low_rate"]) for row in subset])),
                "high_rate": float(np.mean([float(row["high_rate"]) for row in subset])),
                "gap": float(np.mean(gap_values)),
                "direction": direction_mode,
                "direction_stability": float(np.mean([direction == direction_mode for direction in directions])),
                "low_count": int(np.mean([int(row["low_count"]) for row in subset])),
                "high_count": int(np.mean([int(row["high_count"]) for row in subset])),
            }
        )
    return stable_rows


def _build_regime_rows(rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_rows = [
        row
        for row in rows
        if row["downstream_kind"] == "real_downstream" and row["downstream_exact_rescue"] is not None
    ]
    coverage_threshold = next(
        (
            row["threshold"]
            for row in threshold_rows
            if row["feature"] == "real_downstream_bank_coverage" and row.get("analysis_scope") == "all_rows"
        ),
        0.9,
    )
    entropy_threshold = next(
        (
            row["threshold"]
            for row in threshold_rows
            if row["feature"] == "mean_confusion_entropy" and row.get("analysis_scope") == "all_rows"
        ),
        _mean([row["mean_confusion_entropy"] for row in real_rows]) or 0.5,
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        support_regime = "high_support" if float(row["real_downstream_bank_coverage"] or 0.0) >= float(coverage_threshold) else "limited_support"
        entropy_regime = "low_entropy" if float(row["mean_confusion_entropy"] or 0.0) < float(entropy_threshold) else "high_entropy"
        grouped[(row["method_family"], support_regime, entropy_regime)].append(row)

    regime_rows: list[dict[str, Any]] = []
    for (method_family, support_regime, entropy_regime), subset in sorted(grouped.items()):
        regime_rows.append(
            {
                "method_family": method_family,
                "support_regime": support_regime,
                "entropy_regime": entropy_regime,
                "count": len(subset),
                "downstream_exact_rescue_rate": _mean([row["downstream_exact_rescue"] for row in subset]),
                "grouped_topk_rescue_rate": _mean([row["grouped_topk_rescue"] for row in subset]),
                "mean_ambiguity_level": _mean([row["ambiguity_level"] for row in subset]),
                "mean_coverage": _mean([row["real_downstream_bank_coverage"] for row in subset]),
                "mean_entropy": _mean([row["mean_confusion_entropy"] for row in subset]),
            }
        )
    return regime_rows


def _build_case_studies(rows: list[dict[str, Any]]) -> str:
    real_rows = [
        row
        for row in rows
        if row["downstream_kind"] == "real_downstream" and row["downstream_exact_rescue"] is not None
    ]

    def _pick(subset: list[dict[str, Any]], *, sort_keys: list[str], reverse: bool = True) -> dict[str, Any] | None:
        if not subset:
            return None
        return sorted(
            subset,
            key=lambda row: tuple(float(row.get(key) or 0.0) for key in sort_keys),
            reverse=reverse,
        )[0]

    studies = [
        (
            "Raw Uncertainty Downstream Rescue",
            _pick(
                [
                    row
                    for row in real_rows
                    if row["method_family"] == "raw_uncertainty" and row["downstream_exact_rescue"] == 1.0
                ],
                sort_keys=["downstream_exact_delta", "grouped_topk_delta"],
            ),
        ),
        (
            "Conformal Downstream Rescue",
            _pick(
                [
                    row
                    for row in real_rows
                    if row["method_family"] == "conformal" and row["downstream_exact_rescue"] == 1.0
                ],
                sort_keys=["downstream_exact_delta", "grouped_topk_delta"],
            ),
        ),
        (
            "Grouped Rescue Without Downstream Rescue",
            _pick(
                [
                    row
                    for row in real_rows
                    if row["grouped_topk_rescue"] == 1.0 and row["downstream_exact_rescue"] == 0.0
                ],
                sort_keys=["grouped_topk_delta", "mean_confusion_entropy"],
            ),
        ),
        (
            "Symbol Rescue Without Grouped Rescue",
            _pick(
                [row for row in rows if row["symbol_rescue"] == 1.0 and row["grouped_topk_rescue"] == 0.0],
                sort_keys=["symbol_topk_delta", "mean_confusion_entropy"],
            ),
        ),
    ]

    lines = ["# Propagation Case Studies", ""]
    for heading, row in studies:
        lines.extend([f"## {heading}", ""])
        if row is None:
            lines.extend(["- No matching example found.", ""])
            continue
        lines.extend(
            [
                f"- dataset: `{row['dataset']}`",
                f"- example_id: `{row['example_id']}`",
                f"- method_family: `{row['method_family']}`",
                f"- posterior: `{row['posterior_strategy_requested']}`",
                f"- ambiguity: `{_fmt(row['ambiguity_level'])}` ({row['ambiguity_regime']})",
                f"- sequence_length: `{_fmt(row['sequence_length'])}`",
                f"- mean_confusion_entropy: `{_fmt(row['mean_confusion_entropy'])}`",
                f"- prediction_set_avg_size: `{_fmt(row['prediction_set_avg_size'])}`",
                f"- support_coverage: `{_fmt(row['real_downstream_bank_coverage'])}`",
                f"- symbol_topk_delta: `{_fmt(row['symbol_topk_delta'])}`",
                f"- grouped_topk_delta: `{_fmt(row['grouped_topk_delta'])}`",
                f"- downstream_exact_delta: `{_fmt(row['downstream_exact_delta'])}`",
                f"- run_dir: `{row['run_dir']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_regime_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [f"{row['method_family']}:{row['support_regime']}:{row['entropy_regime']}" for row in rows]
    values = [0.0 if row["downstream_exact_rescue_rate"] is None else float(row["downstream_exact_rescue_rate"]) for row in rows]
    x = np.arange(len(labels))
    colors = ["#4c78a8" if row["method_family"] == "raw_uncertainty" else "#e45756" for row in rows]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, values, color=colors)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Downstream Exact Rescue Rate")
    ax.set_title("Propagation Regimes On Real Grouped Downstream Task")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_cross_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for dataset, subset in sorted(grouped.items()):
        symbol_subset = [row for row in subset if row["symbol_rescue"] == 1.0]
        grouped_subset = [row for row in subset if row["grouped_topk_rescue"] == 1.0]
        downstream_subset = [row for row in subset if row["downstream_exact_rescue"] is not None]
        summary_rows.append(
            {
                "dataset": dataset,
                "task_groups": ",".join(sorted({str(row["task_group"]) for row in subset})),
                "symbol_rescue_rate": _mean([row["symbol_rescue"] for row in subset]),
                "grouped_topk_rescue_rate": _mean([row["grouped_topk_rescue"] for row in subset]),
                "grouped_given_symbol_rescue_rate": _mean([row["grouped_topk_rescue"] for row in symbol_subset]),
                "downstream_exact_rescue_rate": _mean([row["downstream_exact_rescue"] for row in downstream_subset]),
                "downstream_given_grouped_rescue_rate": _mean(
                    [row["downstream_exact_rescue"] for row in downstream_subset if row["grouped_topk_rescue"] == 1.0]
                ),
                "mean_entropy": _mean([row["mean_confusion_entropy"] for row in subset]),
                "mean_set_size": _mean([row["prediction_set_avg_size"] for row in subset]),
                "mean_support": _mean([row["real_downstream_bank_coverage"] for row in subset]),
            }
        )
    return summary_rows


def main() -> None:
    propagation_rows = _build_comparison_rows()
    write_csv(Path("outputs/propagation_features.csv"), propagation_rows)
    write_text(Path("outputs/propagation_features.md"), _feature_markdown(propagation_rows))

    model_rows = _build_model_rows(propagation_rows)
    write_csv(Path("outputs/propagation_model_summary.csv"), model_rows)
    write_text(
        Path("outputs/propagation_model_summary.md"),
        _rows_to_markdown(
            "Propagation Model Summary",
            model_rows,
            ["model_name", "analysis_scope", "feature", "coefficient", "odds_ratio", "training_accuracy", "positive_rate", "sample_count"],
        ),
    )

    threshold_rows = _add_threshold_stability(_build_threshold_rows(propagation_rows))
    write_csv(Path("outputs/propagation_thresholds.csv"), threshold_rows)
    write_text(
        Path("outputs/propagation_thresholds.md"),
        _rows_to_markdown(
            "Propagation Thresholds",
            threshold_rows,
            ["analysis_scope", "feature", "threshold", "threshold_min", "threshold_max", "low_rate", "high_rate", "gap", "direction", "direction_stability", "low_count", "high_count"],
        ),
    )

    regime_rows = _build_regime_rows(propagation_rows, threshold_rows)
    write_csv(Path("outputs/propagation_regime_summary.csv"), regime_rows)
    write_text(
        Path("outputs/propagation_regime_summary.md"),
        _rows_to_markdown(
            "Propagation Regime Summary",
            regime_rows,
            ["method_family", "support_regime", "entropy_regime", "count", "downstream_exact_rescue_rate", "grouped_topk_rescue_rate", "mean_coverage", "mean_entropy"],
        ),
    )
    _build_regime_plot(regime_rows, Path("outputs/propagation_regime_plot.png"))

    cross_dataset_rows = _build_cross_dataset_rows(propagation_rows)
    write_csv(Path("outputs/propagation_cross_dataset_summary.csv"), cross_dataset_rows)
    write_text(
        Path("outputs/propagation_cross_dataset_summary.md"),
        _rows_to_markdown(
            "Propagation Cross Dataset Summary",
            cross_dataset_rows,
            ["dataset", "task_groups", "symbol_rescue_rate", "grouped_topk_rescue_rate", "grouped_given_symbol_rescue_rate", "downstream_exact_rescue_rate", "downstream_given_grouped_rescue_rate", "mean_entropy", "mean_set_size", "mean_support"],
        ),
    )
    write_text(Path("outputs/propagation_case_studies.md"), _build_case_studies(propagation_rows))

    print(
        json.dumps(
            {
                "features_csv": "outputs/propagation_features.csv",
                "model_csv": "outputs/propagation_model_summary.csv",
                "thresholds_csv": "outputs/propagation_thresholds.csv",
                "regime_csv": "outputs/propagation_regime_summary.csv",
                "cross_dataset_csv": "outputs/propagation_cross_dataset_summary.csv",
                "case_studies_md": "outputs/propagation_case_studies.md",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
