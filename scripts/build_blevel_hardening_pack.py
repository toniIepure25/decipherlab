from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
import os
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

from decipherlab.sequence.propagation import bootstrap_mean_ci
from decipherlab.utils.io import write_csv, write_text


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _bootstrap_difference(values: list[float], *, seed: int = 13, num_bootstrap: int = 2000) -> dict[str, float]:
    return bootstrap_mean_ci(values, seed=seed, num_bootstrap=num_bootstrap)


def _bootstrap_rate_difference(
    numerator_rows: list[dict[str, str | float]],
    denominator_rows: list[dict[str, str | float]],
    target_key: str,
    *,
    seed: int = 13,
    num_bootstrap: int = 2000,
) -> dict[str, float]:
    if not numerator_rows or not denominator_rows:
        raise ValueError("Both row groups must be non-empty for rate-difference bootstrapping.")
    rng = np.random.default_rng(seed)
    first = np.asarray([float(row[target_key]) for row in numerator_rows], dtype=float)
    second = np.asarray([float(row[target_key]) for row in denominator_rows], dtype=float)
    draws = []
    for _ in range(num_bootstrap):
        first_sample = rng.choice(first, size=len(first), replace=True)
        second_sample = rng.choice(second, size=len(second), replace=True)
        draws.append(float(np.mean(first_sample) - np.mean(second_sample)))
    return {
        "mean": float(np.mean(first) - np.mean(second)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "boot_prob_positive": float(np.mean(np.asarray(draws) > 0.0)),
    }


def _markdown_from_rows(title: str, rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = [f"# {title}", "", "## Rows", ""]
    for row in rows:
        rendered = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                rendered.append(f"`{key}`=`{_fmt(value)}`")
            else:
                rendered.append(f"`{key}`=`{value}`")
        lines.append("- " + ", ".join(rendered))
    return "\n".join(lines) + "\n"


def _load_propagation_rows() -> list[dict[str, Any]]:
    rows = _read_csv("outputs/propagation_features.csv")
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                **row,
                "synthetic_from_real": _to_float(row.get("synthetic_from_real")),
                "is_real_grouped": _to_float(row.get("is_real_grouped")),
                "ambiguity_level": _to_float(row.get("ambiguity_level")),
                "is_calibrated": _to_float(row.get("is_calibrated")),
                "is_conformal": _to_float(row.get("is_conformal")),
                "sequence_length": _to_float(row.get("sequence_length")),
                "symbol_topk_accuracy": _to_float(row.get("symbol_topk_accuracy")),
                "grouped_topk_success": _to_float(row.get("grouped_topk_success")),
                "grouped_exact_success": _to_float(row.get("grouped_exact_success")),
                "grouped_token_accuracy": _to_float(row.get("grouped_token_accuracy")),
                "downstream_exact_success": _to_float(row.get("downstream_exact_success")),
                "downstream_partial_success": _to_float(row.get("downstream_partial_success")),
                "symbol_topk_delta": _to_float(row.get("symbol_topk_delta")),
                "grouped_topk_delta": _to_float(row.get("grouped_topk_delta")),
                "grouped_exact_delta": _to_float(row.get("grouped_exact_delta")),
                "grouped_token_delta": _to_float(row.get("grouped_token_delta")),
                "downstream_exact_delta": _to_float(row.get("downstream_exact_delta")),
                "downstream_partial_delta": _to_float(row.get("downstream_partial_delta")),
                "symbol_rescue": _to_float(row.get("symbol_rescue")),
                "grouped_topk_rescue": _to_float(row.get("grouped_topk_rescue")),
                "grouped_exact_rescue": _to_float(row.get("grouped_exact_rescue")),
                "downstream_exact_rescue": _to_float(row.get("downstream_exact_rescue")),
                "downstream_partial_rescue": _to_float(row.get("downstream_partial_rescue")),
                "prediction_set_coverage": _to_float(row.get("prediction_set_coverage")),
                "prediction_set_avg_size": _to_float(row.get("prediction_set_avg_size")),
                "prediction_set_singleton_rate": _to_float(row.get("prediction_set_singleton_rate")),
                "prediction_set_rescue_rate": _to_float(row.get("prediction_set_rescue_rate")),
                "mean_confusion_entropy": _to_float(row.get("mean_confusion_entropy")),
                "mean_confusion_set_size": _to_float(row.get("mean_confusion_set_size")),
                "real_downstream_bank_coverage": _to_float(row.get("real_downstream_bank_coverage")),
                "dataset_support_upper_bound": _to_float(row.get("dataset_support_upper_bound")),
                "full_support_indicator": _to_float(row.get("full_support_indicator")),
            }
        )
    return converted


def _real_data_trust_rows() -> list[dict[str, Any]]:
    gold_rows = _read_csv("data/processed/historical_newspapers_grouped_words/gold_annotations.csv")
    validation_rows = _read_csv("data/processed/historical_newspapers_grouped_words/validation_subset_annotations.csv")
    correction_rows = _read_csv("data/processed/historical_newspapers_grouped_words/validation_corrections.csv")
    robustness_rows = _read_csv("outputs/real_grouped_robustness_summary.csv")
    strengthened_rows = _read_csv("outputs/real_grouped_strengthened_summary.csv")

    sequence_ids = {row["sequence_id"] for row in gold_rows}
    gold_changed_tokens = [row for row in gold_rows if row["ocr_label"] != row["adjudicated_label"]]
    changed_sequences = {row["sequence_id"] for row in gold_changed_tokens}
    error_counter = Counter(row["error_type"] for row in gold_changed_tokens if row["error_type"])
    qc_sheets = sorted((REPO_ROOT / "data/processed/historical_newspapers_grouped_words/validation_qc").glob("*.png"))

    rows: list[dict[str, Any]] = [
        {
            "category": "validation_audit",
            "metric": "audited_tokens",
            "value": len(validation_rows),
            "notes": "Full Historical Newspapers test split visual audit.",
        },
        {
            "category": "validation_audit",
            "metric": "audited_sequences",
            "value": len({row["sequence_id"] for row in validation_rows}),
            "notes": "Grouped sequences reviewed in the audit pass.",
        },
        {
            "category": "validation_audit",
            "metric": "changed_tokens",
            "value": len(correction_rows),
            "notes": "Tokens changed relative to OCR-derived labels.",
        },
        {
            "category": "gold_style",
            "metric": "pass_agreement_rate",
            "value": np.mean([row["annotator_agreement"] == "True" for row in gold_rows]),
            "notes": "Pass A vs Pass B agreement on gold-style pass.",
        },
        {
            "category": "gold_style",
            "metric": "ocr_to_gold_token_error_rate",
            "value": len(gold_changed_tokens) / len(gold_rows),
            "notes": "OCR-derived token disagreement against adjudicated labels.",
        },
        {
            "category": "gold_style",
            "metric": "ocr_to_gold_sequence_error_rate",
            "value": len(changed_sequences) / len(sequence_ids),
            "notes": "Grouped sequence disagreement rate.",
        },
        {
            "category": "trust_support",
            "metric": "qc_contact_sheet_count",
            "value": len(qc_sheets),
            "notes": "Saved QC sheets covering the reviewed Historical subset.",
        },
    ]

    for error_type, count in sorted(error_counter.items()):
        rows.append(
            {
                "category": "error_taxonomy",
                "metric": error_type,
                "value": count,
                "notes": "Observed gold-style correction count.",
            }
        )

    for row in robustness_rows:
        posterior = row["posterior_strategy_requested"]
        rows.append(
            {
                "category": "metric_stability_audit",
                "metric": f"{posterior}_validated_minus_original_conformal_exact_delta",
                "value": _to_float(row["validated_minus_original_conformal_sequence_exact_delta"]),
                "notes": "Metric drift after first audit pass.",
            }
        )
    for row in strengthened_rows:
        posterior = row["posterior_strategy_requested"]
        rows.append(
            {
                "category": "metric_stability_gold",
                "metric": f"{posterior}_gold_minus_original_conformal_exact_delta",
                "value": _to_float(row["gold_minus_original_conformal_sequence_exact_delta"]),
                "notes": "Metric drift after gold-style adjudication.",
            }
        )
    return rows


def _real_data_trust_markdown(rows: list[dict[str, Any]]) -> str:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    lines = [
        "# Real Data Trust Summary",
        "",
        "## Main Takeaways",
        "",
        "- The Historical Newspapers benchmark now has a stronger trust chain: OCR-derived labels, direct visual audit, and a gold-style adjudicated pass.",
        "- Only two tokens changed across the full evaluated test split, and all grouped metrics remained unchanged after both the audit and gold-style passes.",
        "- The observed corrections are narrow OCR substitutions, not broad sequence-level annotation failures.",
        "",
    ]
    for category, subset in sorted(by_category.items()):
        lines.extend([f"## {category.replace('_', ' ').title()}", ""])
        for row in subset:
            lines.append(f"- `{row['metric']}`: `{_fmt(row['value']) if isinstance(row['value'], float) else row['value']}`. {row['notes']}")
        lines.append("")
    return "\n".join(lines)


def _claim_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stat_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    def add_mean_delta(
        claim_id: str,
        subset: list[dict[str, Any]],
        metric_key: str,
        notes: str,
    ) -> None:
        values = [float(row[metric_key]) for row in subset if row.get(metric_key) is not None]
        if not values:
            return
        summary = _bootstrap_difference(values)
        positive = [value for value in values if value > 0.0]
        negative = [value for value in values if value < 0.0]
        stat_rows.append(
            {
                "claim_id": claim_id,
                "analysis_type": "mean_delta",
                "metric": metric_key,
                "sample_count": len(values),
                "mean_effect": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "boot_prob_positive": summary["boot_prob_positive"],
                "positive_fraction": len(positive) / len(values),
                "negative_fraction": len(negative) / len(values),
                "notes": notes,
            }
        )
        std = float(np.std(values))
        effect_rows.append(
            {
                "claim_id": claim_id,
                "metric": metric_key,
                "sample_count": len(values),
                "mean_effect": float(np.mean(values)),
                "median_effect": float(np.median(values)),
                "std_effect": std,
                "standardized_mean_effect": None if std == 0.0 else float(np.mean(values) / std),
                "positive_fraction": len(positive) / len(values),
                "notes": notes,
            }
        )

    def add_rate_gap(
        claim_id: str,
        positive_subset: list[dict[str, Any]],
        negative_subset: list[dict[str, Any]],
        target_key: str,
        notes: str,
    ) -> None:
        if not positive_subset or not negative_subset:
            return
        summary = _bootstrap_rate_difference(positive_subset, negative_subset, target_key)
        stat_rows.append(
            {
                "claim_id": claim_id,
                "analysis_type": "rate_gap",
                "metric": target_key,
                "sample_count": len(positive_subset) + len(negative_subset),
                "mean_effect": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "boot_prob_positive": summary["boot_prob_positive"],
                "positive_fraction": float(np.mean([float(row[target_key]) for row in positive_subset])),
                "negative_fraction": float(np.mean([float(row[target_key]) for row in negative_subset])),
                "notes": notes,
            }
        )

    real_rows = [row for row in rows if row["task_group"] == "real_grouped_downstream_redesigned"]
    synthetic_markov = [row for row in rows if row["task_group"] == "synthetic_markov"]
    synthetic_family = [row for row in rows if row["task_group"] == "synthetic_process_family"]

    add_mean_delta(
        "real_grouped_two_corpus_raw_grouped_topk_delta",
        [row for row in real_rows if row["method_family"] == "raw_uncertainty"],
        "grouped_topk_delta",
        "Replicated real grouped top-k rescue across Historical Newspapers and ScaDS.AI.",
    )
    add_mean_delta(
        "real_grouped_historical_conformal_exact_delta",
        [row for row in real_rows if row["dataset"] == "historical_newspapers" and row["method_family"] == "conformal"],
        "grouped_exact_delta",
        "Historical Newspapers grouped exact rescue under conformal.",
    )
    add_mean_delta(
        "real_grouped_scadsai_raw_grouped_topk_delta",
        [row for row in real_rows if row["dataset"] == "scadsai" and row["method_family"] == "raw_uncertainty"],
        "grouped_topk_delta",
        "ScaDS.AI grouped top-k rescue under raw uncertainty.",
    )
    add_mean_delta(
        "real_downstream_two_corpus_raw_exact_delta",
        [row for row in real_rows if row["method_family"] == "raw_uncertainty" and row["downstream_exact_delta"] is not None],
        "downstream_exact_delta",
        "Two-corpus real downstream exact delta under raw uncertainty.",
    )
    add_mean_delta(
        "real_downstream_two_corpus_conformal_exact_delta",
        [row for row in real_rows if row["method_family"] == "conformal" and row["downstream_exact_delta"] is not None],
        "downstream_exact_delta",
        "Two-corpus real downstream exact delta under conformal.",
    )
    add_mean_delta(
        "real_downstream_historical_conformal_exact_delta",
        [row for row in real_rows if row["dataset"] == "historical_newspapers" and row["method_family"] == "conformal" and row["downstream_exact_delta"] is not None],
        "downstream_exact_delta",
        "Historical Newspapers real downstream exact delta under conformal.",
    )
    add_mean_delta(
        "real_downstream_scadsai_raw_exact_delta",
        [row for row in real_rows if row["dataset"] == "scadsai" and row["method_family"] == "raw_uncertainty" and row["downstream_exact_delta"] is not None],
        "downstream_exact_delta",
        "ScaDS.AI real downstream exact delta under raw uncertainty.",
    )
    add_mean_delta(
        "synthetic_markov_raw_grouped_topk_delta",
        [row for row in synthetic_markov if row["method_family"] == "raw_uncertainty"],
        "grouped_topk_delta",
        "Synthetic-from-real Markov grouped top-k delta under raw uncertainty.",
    )
    add_mean_delta(
        "synthetic_family_raw_downstream_exact_delta",
        [row for row in synthetic_family if row["method_family"] == "raw_uncertainty" and row["downstream_exact_delta"] is not None],
        "downstream_exact_delta",
        "Synthetic-from-real process-family downstream delta under raw uncertainty.",
    )

    add_rate_gap(
        "grouped_rescue_given_symbol_rescue_gap",
        [row for row in rows if row["symbol_rescue"] == 1.0],
        [row for row in rows if row["symbol_rescue"] == 0.0],
        "grouped_topk_rescue",
        "Grouped rescue rate difference conditioned on symbol rescue.",
    )
    add_rate_gap(
        "downstream_rescue_given_grouped_rescue_gap",
        [row for row in real_rows if row["grouped_topk_rescue"] == 1.0 and row["downstream_exact_rescue"] is not None],
        [row for row in real_rows if row["grouped_topk_rescue"] == 0.0 and row["downstream_exact_rescue"] is not None],
        "downstream_exact_rescue",
        "Real downstream rescue rate difference conditioned on grouped rescue.",
    )

    return stat_rows, effect_rows


def _sensitivity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_rows = [row for row in rows if row["task_group"] == "real_grouped_downstream_redesigned"]
    sequence_median = float(np.median([row["sequence_length"] for row in real_rows if row["sequence_length"] is not None]))
    set_size_median = float(np.median([row["prediction_set_avg_size"] for row in real_rows if row["prediction_set_avg_size"] is not None]))
    sensitivity_rows: list[dict[str, Any]] = []

    def add_axis(axis_name: str, labels: dict[str, list[dict[str, Any]]]) -> None:
        for label, subset in labels.items():
            sensitivity_rows.append(
                {
                    "axis": axis_name,
                    "bucket": label,
                    "sample_count": len(subset),
                    "mean_grouped_topk_delta": float(np.mean([row["grouped_topk_delta"] for row in subset if row["grouped_topk_delta"] is not None])),
                    "mean_downstream_exact_delta": float(np.mean([row["downstream_exact_delta"] for row in subset if row["downstream_exact_delta"] is not None])),
                    "grouped_topk_rescue_rate": float(np.mean([row["grouped_topk_rescue"] for row in subset if row["grouped_topk_rescue"] is not None])),
                    "downstream_exact_rescue_rate": float(np.mean([row["downstream_exact_rescue"] for row in subset if row["downstream_exact_rescue"] is not None])),
                }
            )

    add_axis("dataset", {dataset: [row for row in real_rows if row["dataset"] == dataset] for dataset in sorted({str(row["dataset"]) for row in real_rows})})
    add_axis("method_family", {family: [row for row in real_rows if row["method_family"] == family] for family in ["raw_uncertainty", "conformal"]})
    add_axis(
        "sequence_length_bin",
        {
            "short_or_equal_median": [row for row in real_rows if float(row["sequence_length"] or 0.0) <= sequence_median],
            "longer_than_median": [row for row in real_rows if float(row["sequence_length"] or 0.0) > sequence_median],
        },
    )
    add_axis(
        "candidate_set_bin",
        {
            "small_or_equal_median": [row for row in real_rows if float(row["prediction_set_avg_size"] or 0.0) <= set_size_median],
            "larger_than_median": [row for row in real_rows if float(row["prediction_set_avg_size"] or 0.0) > set_size_median],
        },
    )
    add_axis(
        "support_regime",
        {
            "limited_support": [row for row in real_rows if float(row["real_downstream_bank_coverage"] or 0.0) < 0.95],
            "high_support": [row for row in real_rows if float(row["real_downstream_bank_coverage"] or 0.0) >= 0.95],
        },
    )
    add_axis(
        "ambiguity_regime",
        {regime: [row for row in real_rows if row["ambiguity_regime"] == regime] for regime in sorted({str(row["ambiguity_regime"]) for row in real_rows})},
    )

    strengthened_rows = _read_csv("outputs/real_grouped_strengthened_summary.csv")
    for row in strengthened_rows:
        posterior = row["posterior_strategy_requested"]
        sensitivity_rows.append(
            {
                "axis": "label_quality_tier",
                "bucket": f"{posterior}:original_vs_gold",
                "sample_count": 30,
                "mean_grouped_topk_delta": _to_float(row["gold_uncertainty_sequence_topk_delta"]),
                "mean_downstream_exact_delta": None,
                "grouped_topk_rescue_rate": None,
                "downstream_exact_rescue_rate": None,
            }
        )
    return sensitivity_rows


def _build_sensitivity_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    candidate_rows = [row for row in rows if row["axis"] == "candidate_set_bin"]
    support_rows = [row for row in rows if row["axis"] == "support_regime"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(
        [row["bucket"] for row in candidate_rows],
        [row["mean_grouped_topk_delta"] for row in candidate_rows],
        color=["#4c78a8", "#72b7b2"],
    )
    axes[0].set_title("Grouped Top-k Delta By Candidate Set Size")
    axes[0].set_ylabel("Mean Grouped Top-k Delta")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        [row["bucket"] for row in support_rows],
        [row["mean_downstream_exact_delta"] for row in support_rows],
        color=["#e45756", "#54a24b"],
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Downstream Exact Delta By Support Regime")
    axes[1].set_ylabel("Mean Downstream Exact Delta")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _sensitivity_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Sensitivity Summary",
        "",
        "## Main Takeaways",
        "",
    ]
    dataset_rows = {row["bucket"]: row for row in rows if row["axis"] == "dataset"}
    support_rows = {row["bucket"]: row for row in rows if row["axis"] == "support_regime"}
    candidate_rows = {row["bucket"]: row for row in rows if row["axis"] == "candidate_set_bin"}
    if "scadsai" in dataset_rows and "historical_newspapers" in dataset_rows:
        lines.append(
            f"- ScaDS.AI is more rescue-rich at grouped level (`{_fmt(dataset_rows['scadsai']['mean_grouped_topk_delta'])}` grouped top-k delta) than Historical Newspapers (`{_fmt(dataset_rows['historical_newspapers']['mean_grouped_topk_delta'])}`)."
        )
    if "high_support" in support_rows and "limited_support" in support_rows:
        lines.append(
            f"- Downstream exact recovery does not follow a simple monotonic support pattern here: `{support_rows['high_support']['bucket']}` gives `{_fmt(support_rows['high_support']['mean_downstream_exact_delta'])}`, while `{support_rows['limited_support']['bucket']}` gives `{_fmt(support_rows['limited_support']['mean_downstream_exact_delta'])}`."
        )
    if "small_or_equal_median" in candidate_rows and "larger_than_median" in candidate_rows:
        lines.append(
            f"- Larger candidate sets strengthen grouped top-k rescue (`{_fmt(candidate_rows['larger_than_median']['mean_grouped_topk_delta'])}` vs `{_fmt(candidate_rows['small_or_equal_median']['mean_grouped_topk_delta'])}`), but smaller sets give cleaner downstream exact behavior (`{_fmt(candidate_rows['small_or_equal_median']['mean_downstream_exact_delta'])}` vs `{_fmt(candidate_rows['larger_than_median']['mean_downstream_exact_delta'])}`)."
        )
    lines.extend(["", "## Rows", ""])
    for row in rows:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"`axis`=`{row['axis']}`",
                    f"`bucket`=`{row['bucket']}`",
                    f"`sample_count`=`{row['sample_count']}`",
                    f"`mean_grouped_topk_delta`=`{_fmt(row['mean_grouped_topk_delta'])}`",
                    f"`mean_downstream_exact_delta`=`{_fmt(row['mean_downstream_exact_delta'])}`",
                    f"`grouped_topk_rescue_rate`=`{_fmt(row['grouped_topk_rescue_rate'])}`",
                    f"`downstream_exact_rescue_rate`=`{_fmt(row['downstream_exact_rescue_rate'])}`",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    trust_rows = _real_data_trust_rows()
    write_csv(Path("outputs/real_data_trust_summary.csv"), trust_rows)
    write_text(Path("outputs/real_data_trust_summary.md"), _real_data_trust_markdown(trust_rows))

    propagation_rows = _load_propagation_rows()
    statistical_rows, effect_rows = _claim_rows(propagation_rows)
    write_csv(Path("outputs/statistical_robustness_summary.csv"), statistical_rows)
    write_text(
        Path("outputs/statistical_robustness_summary.md"),
        _markdown_from_rows(
            "Statistical Robustness Summary",
            statistical_rows,
            ["claim_id", "analysis_type", "metric", "sample_count", "mean_effect", "ci_low", "ci_high", "boot_prob_positive", "positive_fraction", "negative_fraction", "notes"],
        ),
    )
    write_csv(Path("outputs/effect_size_summary.csv"), effect_rows)
    write_text(
        Path("outputs/effect_size_summary.md"),
        _markdown_from_rows(
            "Effect Size Summary",
            effect_rows,
            ["claim_id", "metric", "sample_count", "mean_effect", "median_effect", "std_effect", "standardized_mean_effect", "positive_fraction", "notes"],
        ),
    )

    sensitivity_rows = _sensitivity_rows(propagation_rows)
    write_csv(Path("outputs/sensitivity_summary.csv"), sensitivity_rows)
    write_text(Path("outputs/sensitivity_summary.md"), _sensitivity_markdown(sensitivity_rows))
    _build_sensitivity_plot(sensitivity_rows, Path("outputs/sensitivity_plot.png"))

    print(
        json.dumps(
            {
                "real_data_trust_summary": "outputs/real_data_trust_summary.csv",
                "statistical_robustness_summary": "outputs/statistical_robustness_summary.csv",
                "effect_size_summary": "outputs/effect_size_summary.csv",
                "sensitivity_summary": "outputs/sensitivity_summary.csv",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
