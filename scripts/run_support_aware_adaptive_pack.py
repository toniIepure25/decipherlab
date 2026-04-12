from __future__ import annotations

import csv
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

from decipherlab.config import load_config
from decipherlab.sequence.runner import run_sequence_branch_experiment
from decipherlab.utils.io import ensure_directory, write_csv, write_text


CONFIGS = [
    ("historical_newspapers_real_grouped", Path("configs/experiments/sequence_historical_newspapers_real_grouped_gold.yaml")),
    ("historical_newspapers_real_downstream", Path("configs/experiments/sequence_historical_newspapers_real_downstream_redesigned.yaml")),
    ("scadsai_real_grouped", Path("configs/experiments/sequence_scadsai_real_grouped.yaml")),
    ("scadsai_real_downstream", Path("configs/experiments/sequence_scadsai_real_downstream_redesigned.yaml")),
]
STRATEGIES = ["cluster_distance", "calibrated_classifier"]
POLICIES = [
    "support_aware_rule",
    "support_aware_learned_gate",
    "support_aware_constrained_gate",
    "support_aware_profile_selector",
]
PROFILES = ["rescue_first", "shortlist_first"]
REVIEW_BUDGETS = [2, 3, 5]
METHOD_ORDER = [
    "fixed_greedy",
    "uncertainty_beam",
    "conformal_beam",
    "adaptive_support_beam",
    "adaptive_learned_beam",
    "adaptive_constrained_beam",
    "adaptive_profiled_beam",
    "adaptive_profile_selector_beam",
    "uncertainty_trigram_beam",
    "conformal_trigram_beam",
    "uncertainty_crf_viterbi",
    "conformal_crf_viterbi",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_to_float(row.get(key)) for row in rows]
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(np.mean(usable))


def _method_summary(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    by_method: dict[str, dict[str, float | None]] = {}
    for method in METHOD_ORDER:
        rows = [row for row in summary_rows if row["method"] == method]
        if not rows:
            continue
        by_method[method] = {
            "grouped_exact": _mean(rows, "sequence_exact_match"),
            "grouped_topk": _mean(rows, "sequence_topk_recovery"),
            "shortlist_recall_at_2": _mean(rows, "sequence_shortlist_recall_at_2"),
            "shortlist_recall_at_3": _mean(rows, "sequence_shortlist_recall_at_3"),
            "shortlist_recall_at_5": _mean(rows, "sequence_shortlist_recall_at_5"),
            "shortlist_utility": _mean(rows, "sequence_shortlist_utility"),
            "grouped_token": _mean(rows, "sequence_token_accuracy"),
            "grouped_cer": _mean(rows, "sequence_cer"),
            "downstream_exact": _mean(rows, "real_downstream_exact_match"),
            "downstream_topk": _mean(rows, "real_downstream_topk_recovery"),
            "downstream_token": _mean(rows, "real_downstream_token_accuracy"),
            "prediction_set_avg_size": _mean(rows, "prediction_set_avg_size"),
        }
    return by_method


def _review_efficiency(utility: float | None, set_size: float | None) -> float | None:
    if utility is None or set_size is None or set_size <= 0.0:
        return None
    return float(utility / set_size)


def _adaptive_usage(per_sequence_rows: list[dict[str, str]], adaptive_method: str) -> dict[str, float | None]:
    rows = [row for row in per_sequence_rows if row["method"] == adaptive_method]
    if not rows:
        return {
            "adaptive_conformal_rate": None,
            "adaptive_raw_rate": None,
            "adaptive_mean_beam_width": None,
            "adaptive_limited_support_conformal_rate": None,
            "adaptive_high_entropy_raw_rate": None,
            "adaptive_defer_rate": None,
            "adaptive_preserve_rate": None,
            "adaptive_prune_rate": None,
        }
    conformal_rows = [row for row in rows if row.get("adaptive_selected_method") == "conformal_beam"]
    raw_rows = [row for row in rows if row.get("adaptive_selected_method") == "uncertainty_beam"]
    limited_support_rows = [row for row in rows if _to_float(row.get("adaptive_limited_support")) == 1.0]
    high_entropy_rows = [row for row in rows if _to_float(row.get("adaptive_high_entropy")) == 1.0]
    beam_widths = [_to_float(row.get("adaptive_beam_width")) for row in rows]
    return {
        "adaptive_conformal_rate": len(conformal_rows) / len(rows),
        "adaptive_raw_rate": len(raw_rows) / len(rows),
        "adaptive_mean_beam_width": float(np.mean([width for width in beam_widths if width is not None])),
        "adaptive_limited_support_conformal_rate": (
            len([row for row in limited_support_rows if row.get("adaptive_selected_method") == "conformal_beam"]) / len(limited_support_rows)
            if limited_support_rows
            else None
        ),
        "adaptive_high_entropy_raw_rate": (
            len([row for row in high_entropy_rows if row.get("adaptive_selected_method") == "uncertainty_beam"]) / len(high_entropy_rows)
            if high_entropy_rows
            else None
        ),
        "adaptive_defer_rate": len([row for row in rows if _to_float(row.get("adaptive_defer_to_human")) == 1.0]) / len(rows),
        "adaptive_preserve_rate": len([row for row in rows if row.get("adaptive_control_action") == "preserve"]) / len(rows),
        "adaptive_prune_rate": len([row for row in rows if row.get("adaptive_control_action") == "prune"]) / len(rows),
    }


def _failure_reduction(per_sequence_rows: list[dict[str, str]], adaptive_method: str) -> dict[str, int]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in per_sequence_rows:
        key = (row["example_id"], row["posterior_strategy"])
        grouped.setdefault(key, {})[row["method"]] = row
    adaptive_grouped_rescue = 0
    adaptive_downstream_rescue = 0
    avoided_conformal_over_pruning = 0
    reduced_grouped_without_downstream = 0
    for methods in grouped.values():
        adaptive = methods.get(adaptive_method)
        uncertainty = methods.get("uncertainty_beam")
        conformal = methods.get("conformal_beam")
        fixed = methods.get("fixed_greedy")
        if adaptive is None or uncertainty is None or fixed is None:
            continue
        if (
            _to_float(adaptive.get("sequence_topk_recovery")) or 0.0
        ) > (_to_float(uncertainty.get("sequence_topk_recovery")) or 0.0):
            adaptive_grouped_rescue += 1
        if (
            _to_float(adaptive.get("real_downstream_exact_match")) is not None
            and _to_float(uncertainty.get("real_downstream_exact_match")) is not None
            and (_to_float(adaptive.get("real_downstream_exact_match")) or 0.0)
            > (_to_float(uncertainty.get("real_downstream_exact_match")) or 0.0)
        ):
            adaptive_downstream_rescue += 1
        if conformal is not None and (
            (_to_float(uncertainty.get("sequence_topk_recovery")) or 0.0)
            > (_to_float(fixed.get("sequence_topk_recovery")) or 0.0)
            and (_to_float(conformal.get("sequence_topk_recovery")) or 0.0)
            <= (_to_float(fixed.get("sequence_topk_recovery")) or 0.0)
            and (_to_float(adaptive.get("sequence_topk_recovery")) or 0.0)
            > (_to_float(conformal.get("sequence_topk_recovery")) or 0.0)
        ):
            avoided_conformal_over_pruning += 1
        if (
            _to_float(adaptive.get("sequence_topk_recovery")) is not None
            and _to_float(adaptive.get("real_downstream_exact_match")) is not None
            and _to_float(uncertainty.get("sequence_topk_recovery")) is not None
            and _to_float(uncertainty.get("real_downstream_exact_match")) is not None
            and (_to_float(adaptive.get("sequence_topk_recovery")) or 0.0)
            > (_to_float(uncertainty.get("sequence_topk_recovery")) or 0.0)
            and (_to_float(adaptive.get("real_downstream_exact_match")) or 0.0)
            >= (_to_float(uncertainty.get("real_downstream_exact_match")) or 0.0)
        ):
            reduced_grouped_without_downstream += 1
    return {
        "adaptive_grouped_rescue_over_uncertainty": adaptive_grouped_rescue,
        "adaptive_downstream_rescue_over_uncertainty": adaptive_downstream_rescue,
        "avoided_conformal_over_pruning_cases": avoided_conformal_over_pruning,
        "reduced_grouped_without_downstream_cases": reduced_grouped_without_downstream,
    }


def _controller_method_name(policy: str) -> str:
    if policy == "support_aware_profile_selector":
        return "adaptive_profile_selector_beam"
    if policy == "support_aware_learned_gate":
        return "adaptive_learned_beam"
    if policy == "support_aware_constrained_gate":
        return "adaptive_constrained_beam"
    if policy == "support_aware_profiled_gate":
        return "adaptive_profiled_beam"
    return "adaptive_support_beam"


def _controller_label(policy: str, operating_profile: str | None = None) -> str:
    if policy == "support_aware_profile_selector":
        return "profile_selector"
    if policy == "support_aware_profiled_gate":
        return str(operating_profile or "profiled")
    if policy == "support_aware_learned_gate":
        return "learned"
    if policy == "support_aware_constrained_gate":
        return "constrained"
    return "rule"


def _profile_budget_label(operating_profile: str, review_budget: int) -> str:
    return f"{operating_profile}@k={review_budget}"


def _budget_metric_value(row: dict[str, str], review_budget: int) -> float | None:
    return _to_float(row.get(f"sequence_shortlist_recall_at_{review_budget}"))


def _row_review_load_proxy(row: dict[str, str], review_budget: int) -> float | None:
    set_size = _to_float(row.get("prediction_set_avg_size"))
    if set_size is None:
        return None
    defer_flag = _to_float(row.get("adaptive_defer_to_human")) == 1.0
    return float(min(set_size, float(review_budget)) + (1.0 if defer_flag else 0.0))


def _row_effort_adjusted_utility(row: dict[str, str], review_budget: int) -> float | None:
    budget_value = _budget_metric_value(row, review_budget)
    review_load = _row_review_load_proxy(row, review_budget)
    if budget_value is None or review_load is None or review_load <= 0.0:
        return None
    return float(budget_value / review_load)


def _is_fragile_case(row: dict[str, str]) -> bool:
    fragile_signal_count = _to_float(row.get("adaptive_fragile_signal_count"))
    if fragile_signal_count is not None:
        return fragile_signal_count >= 2.0
    return (
        _to_float(row.get("adaptive_limited_support")) == 1.0
        or _to_float(row.get("adaptive_high_entropy")) == 1.0
        or _to_float(row.get("adaptive_diffuse_set")) == 1.0
    )


def _gate_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "adaptive_gate_diagnostics.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _build_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    label_map = {
        "historical_newspapers_real_grouped": "Hist.\ngrouped",
        "historical_newspapers_real_downstream": "Hist.\ndownstream",
        "scadsai_real_grouped": "ScaDS\n grouped",
        "scadsai_real_downstream": "ScaDS\ndownstream",
    }
    strategy_map = {"cluster_distance": "cluster", "calibrated_classifier": "cal."}
    labels = [
        f"{label_map.get(str(row['dataset_label']), str(row['dataset_label']))}\n{strategy_map.get(str(row['posterior_strategy']), str(row['posterior_strategy']))}\n{row['controller_label']}"
        for row in rows
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    grouped_adaptive = [float(row["adaptive_grouped_topk_delta"] or 0.0) for row in rows]
    grouped_uncertainty = [float(row["uncertainty_grouped_topk_delta"] or 0.0) for row in rows]
    grouped_conformal = [float(row["conformal_grouped_exact_delta"] or 0.0) for row in rows]
    axes[0].bar(x - 0.22, grouped_uncertainty, 0.22, label="Raw grouped top-k", color="#4c78a8")
    axes[0].bar(x, grouped_conformal, 0.22, label="Conformal grouped exact", color="#f58518")
    axes[0].bar(x + 0.22, grouped_adaptive, 0.22, label="Adaptive grouped top-k", color="#54a24b")
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Grouped effects vs fixed")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(frameon=False, fontsize=8)

    downstream_adaptive = [float(row["adaptive_downstream_exact_delta"] or 0.0) for row in rows]
    downstream_uncertainty = [float(row["uncertainty_downstream_exact_delta"] or 0.0) for row in rows]
    downstream_conformal = [float(row["conformal_downstream_exact_delta"] or 0.0) for row in rows]
    axes[1].bar(x - 0.22, downstream_uncertainty, 0.22, label="Raw", color="#4c78a8")
    axes[1].bar(x, downstream_conformal, 0.22, label="Conformal", color="#f58518")
    axes[1].bar(x + 0.22, downstream_adaptive, 0.22, label="Adaptive", color="#54a24b")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Downstream exact vs fixed")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    conformal_rate = [float(row["adaptive_conformal_rate"] or 0.0) for row in rows]
    beam_width = [float(row["adaptive_mean_beam_width"] or 0.0) for row in rows]
    axes[2].bar(x - 0.16, conformal_rate, 0.32, label="Conformal rate", color="#e45756")
    axes[2].plot(x + 0.16, beam_width, marker="o", color="#72b7b2", label="Mean beam width")
    axes[2].set_title("Adaptive policy behavior")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=0, labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_operating_point_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {"rule": [], "learned": [], "constrained": []}
    for row in summary_rows:
        controller_label = str(row["controller_label"])
        if controller_label not in grouped:
            continue
        grouped[controller_label].append(row)
    rows: list[dict[str, object]] = []
    for controller_label, items in grouped.items():
        if not items:
            continue
        rows.append(
            {
                "controller_label": controller_label,
                "mean_grouped_topk_delta": float(np.mean([float(item["adaptive_grouped_topk_delta"] or 0.0) for item in items])),
                "mean_downstream_exact_delta": float(np.mean([float(item["adaptive_downstream_exact_delta"] or 0.0) for item in items if item["adaptive_downstream_exact_delta"] is not None])) if any(item["adaptive_downstream_exact_delta"] is not None for item in items) else None,
                "mean_prediction_set_size": float(np.mean([float(item["adaptive_prediction_set_avg_size"] or 0.0) for item in items])),
                "mean_conformal_rate": float(np.mean([float(item["adaptive_conformal_rate"] or 0.0) for item in items])),
                "mean_beam_width": float(np.mean([float(item["adaptive_mean_beam_width"] or 0.0) for item in items])),
                "mean_vs_best_downstream_gap": float(np.mean([float(item["adaptive_vs_best_downstream_gap"] or 0.0) for item in items if item["adaptive_vs_best_downstream_gap"] is not None])) if any(item["adaptive_vs_best_downstream_gap"] is not None for item in items) else None,
            }
        )
    return rows


def _build_operating_point_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 4.2))
    colors = {"rule": "#4c78a8", "learned": "#54a24b", "constrained": "#f58518"}
    for row in rows:
        axis.scatter(
            float(row["mean_prediction_set_size"]),
            0.0 if row["mean_downstream_exact_delta"] is None else float(row["mean_downstream_exact_delta"]),
            s=120,
            color=colors.get(str(row["controller_label"]), "#999999"),
            label=str(row["controller_label"]),
        )
        axis.annotate(
            str(row["controller_label"]),
            (
                float(row["mean_prediction_set_size"]) + 0.03,
                (0.0 if row["mean_downstream_exact_delta"] is None else float(row["mean_downstream_exact_delta"])) + 0.002,
            ),
            fontsize=9,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Mean prediction-set size")
    axis.set_ylabel("Mean downstream exact delta vs fixed")
    axis.set_title("Operating point")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_shortlist_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        rows.append(
            {
                "dataset_label": row["dataset_label"],
                "posterior_strategy": row["posterior_strategy"],
                "controller_label": row["controller_label"],
                "shortlist_recall_at_2": row.get("adaptive_shortlist_recall_at_2"),
                "shortlist_recall_at_3": row.get("adaptive_shortlist_recall_at_3"),
                "shortlist_recall_at_5": row.get("adaptive_shortlist_recall_at_5"),
                "shortlist_utility": row.get("adaptive_shortlist_utility"),
                "grouped_topk_delta": row.get("adaptive_grouped_topk_delta"),
                "downstream_exact_delta": row.get("adaptive_downstream_exact_delta"),
                "prediction_set_avg_size": row.get("adaptive_prediction_set_avg_size"),
            }
        )
    return rows


def _build_shortlist_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = {"rule": [], "learned": [], "constrained": []}
    for row in rows:
        controller_label = str(row["controller_label"])
        if controller_label not in grouped:
            continue
        grouped[controller_label].append(row)
    budgets = [2, 3, 5]
    figure, axis = plt.subplots(figsize=(5.8, 4.2))
    colors = {"rule": "#4c78a8", "learned": "#54a24b", "constrained": "#f58518"}
    for controller_label, items in grouped.items():
        if not items:
            continue
        values = []
        for budget in budgets:
            key = f"shortlist_recall_at_{budget}"
            budget_values = [float(item[key] or 0.0) for item in items if item.get(key) is not None]
            values.append(float(np.mean(budget_values)) if budget_values else 0.0)
        axis.plot(
            budgets,
            values,
            marker="o",
            linewidth=2.0,
            color=colors.get(controller_label, "#999999"),
            label=controller_label,
        )
    axis.set_xticks(budgets)
    axis.set_xlabel("Review budget k")
    axis.set_ylabel("Mean shortlist recall@k")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Shortlist utility under small review budgets")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_profile_aware_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        if row["controller_policy"] != "support_aware_profiled_gate":
            continue
        rows.append(
            {
                "dataset_label": row["dataset_label"],
                "posterior_strategy": row["posterior_strategy"],
                "operating_profile": row["operating_profile"],
                "review_budget_k": row["review_budget_k"],
                "grouped_topk_delta": row["adaptive_grouped_topk_delta"],
                "grouped_exact_delta": row["adaptive_grouped_exact_delta"],
                "downstream_exact_delta": row["adaptive_downstream_exact_delta"],
                "shortlist_recall_at_2": row["adaptive_shortlist_recall_at_2"],
                "shortlist_recall_at_3": row["adaptive_shortlist_recall_at_3"],
                "shortlist_recall_at_5": row["adaptive_shortlist_recall_at_5"],
                "budgeted_recall": row.get("budgeted_recall"),
                "shortlist_utility": row["adaptive_shortlist_utility"],
                "prediction_set_avg_size": row["adaptive_prediction_set_avg_size"],
                "adaptive_conformal_rate": row["adaptive_conformal_rate"],
                "adaptive_mean_beam_width": row["adaptive_mean_beam_width"],
                "adaptive_defer_rate": row.get("adaptive_defer_rate"),
                "adaptive_vs_best_downstream_gap": row["adaptive_vs_best_downstream_gap"],
                "review_efficiency": _review_efficiency(
                    row["adaptive_shortlist_utility"],
                    row["adaptive_prediction_set_avg_size"],
                ),
            }
        )
    return rows


def _build_profile_aware_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    colors = {"rescue_first": "#2f855a", "shortlist_first": "#c05621"}
    grouped_modes: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["operating_profile"]), int(row["review_budget_k"]))
        grouped_modes.setdefault(key, []).append(row)
    for (profile, budget), items in sorted(grouped_modes.items()):
        mean_shortlist = float(np.mean([float(item["shortlist_utility"] or 0.0) for item in items]))
        mean_grouped = float(np.mean([float(item["grouped_topk_delta"] or 0.0) for item in items]))
        mean_size = float(np.mean([float(item["prediction_set_avg_size"] or 0.0) for item in items]))
        mean_budgeted = float(np.mean([float(item["budgeted_recall"] or 0.0) for item in items]))
        mean_defer = float(np.mean([float(item["adaptive_defer_rate"] or 0.0) for item in items]))
        axes[0].scatter(mean_shortlist, mean_grouped, s=135, color=colors.get(profile, "#666666"), alpha=0.9)
        axes[0].annotate(f"{profile[0].upper()}{budget}", (mean_shortlist, mean_grouped), textcoords="offset points", xytext=(5, 4), fontsize=9)
        axes[1].scatter(mean_size, mean_budgeted, s=135, color=colors.get(profile, "#666666"), alpha=0.9)
        axes[1].annotate(f"{profile[0].upper()}{budget}/d{mean_defer:.2f}", (mean_size, mean_budgeted), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[0].set_xlabel("Mean shortlist utility")
    axes[0].set_ylabel("Mean grouped top-k delta vs fixed")
    axes[0].set_title("Profile-budget operating frontier")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Mean prediction-set size")
    axes[1].set_ylabel("Mean recall@budget")
    axes[1].set_title("Budgeted review frontier")
    for axis in axes:
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _fragile_case_summary(
    per_sequence_rows: list[dict[str, str]],
    adaptive_method: str,
    review_budget: int,
) -> dict[str, float | None]:
    rows = [row for row in per_sequence_rows if row["method"] == adaptive_method]
    if not rows:
        return {
            "fragile_case_rate": None,
            "fragile_shortlist_utility": None,
            "fragile_recall_at_2": None,
            "fragile_prediction_set_avg_size": None,
            "fragile_review_efficiency": None,
            "fragile_budgeted_recall": None,
            "fragile_budget_compliance_rate": None,
            "fragile_defer_rate": None,
            "fragile_review_load_proxy": None,
            "fragile_effort_adjusted_utility": None,
        }
    fragile_rows = [row for row in rows if _is_fragile_case(row)]
    if not fragile_rows:
        return {
            "fragile_case_rate": 0.0,
            "fragile_shortlist_utility": None,
            "fragile_recall_at_2": None,
            "fragile_prediction_set_avg_size": None,
            "fragile_review_efficiency": None,
            "fragile_budgeted_recall": None,
            "fragile_budget_compliance_rate": None,
            "fragile_defer_rate": 0.0,
            "fragile_review_load_proxy": None,
            "fragile_effort_adjusted_utility": None,
        }
    fragile_shortlist_utility = _mean(fragile_rows, "sequence_shortlist_utility")
    fragile_set_size = _mean(fragile_rows, "prediction_set_avg_size")
    fragile_budget_values = [_budget_metric_value(row, review_budget) for row in fragile_rows]
    fragile_budget_recall = float(np.mean([value for value in fragile_budget_values if value is not None])) if any(value is not None for value in fragile_budget_values) else None
    fragile_review_loads = [_row_review_load_proxy(row, review_budget) for row in fragile_rows]
    fragile_review_load = float(np.mean([value for value in fragile_review_loads if value is not None])) if any(value is not None for value in fragile_review_loads) else None
    fragile_effort_adjusted = [_row_effort_adjusted_utility(row, review_budget) for row in fragile_rows]
    return {
        "fragile_case_rate": len(fragile_rows) / len(rows),
        "fragile_shortlist_utility": fragile_shortlist_utility,
        "fragile_recall_at_2": _mean(fragile_rows, "sequence_shortlist_recall_at_2"),
        "fragile_prediction_set_avg_size": fragile_set_size,
        "fragile_review_efficiency": _review_efficiency(fragile_shortlist_utility, fragile_set_size),
        "fragile_budgeted_recall": fragile_budget_recall,
        "fragile_budget_compliance_rate": len(
            [
                row
                for row in fragile_rows
                if (_to_float(row.get("prediction_set_avg_size")) or 0.0) <= float(review_budget)
                and _to_float(row.get("adaptive_defer_to_human")) != 1.0
            ]
        )
        / len(fragile_rows),
        "fragile_defer_rate": len([row for row in fragile_rows if _to_float(row.get("adaptive_defer_to_human")) == 1.0]) / len(fragile_rows),
        "fragile_review_load_proxy": fragile_review_load,
        "fragile_effort_adjusted_utility": float(np.mean([value for value in fragile_effort_adjusted if value is not None])) if any(value is not None for value in fragile_effort_adjusted) else None,
    }


def _best_baseline_shortlist(baseline_candidates: dict[str, dict[str, float | None]]) -> tuple[str, float | None]:
    method, metrics = max(
        baseline_candidates.items(),
        key=lambda item: float(item[1].get("shortlist_utility") or 0.0),
    )
    return method, metrics.get("shortlist_utility")


def _best_baseline_review_efficiency(baseline_candidates: dict[str, dict[str, float | None]]) -> tuple[str, float | None]:
    scored = []
    for method, metrics in baseline_candidates.items():
        efficiency = _review_efficiency(
            metrics.get("shortlist_utility"),
            metrics.get("prediction_set_avg_size"),
        )
        if efficiency is not None:
            scored.append((method, efficiency))
    if not scored:
        return "n/a", None
    method, efficiency = max(scored, key=lambda item: item[1])
    return method, efficiency


def _human_effort_summary(
    per_sequence_rows: list[dict[str, str]],
    method: str,
    review_budget: int,
) -> dict[str, float | None]:
    rows = [row for row in per_sequence_rows if row["method"] == method]
    if not rows:
        return {
            "budgeted_recall": None,
            "budget_compliance_rate": None,
            "defer_rate": None,
            "review_load_proxy": None,
            "effort_adjusted_utility": None,
            "rescue_per_review_item": None,
        }
    budgeted_values = [_budget_metric_value(row, review_budget) for row in rows]
    review_loads = [_row_review_load_proxy(row, review_budget) for row in rows]
    effort_values = [_row_effort_adjusted_utility(row, review_budget) for row in rows]
    rescue_values = []
    for row, review_load in zip(rows, review_loads):
        grouped_recovery = _to_float(row.get("sequence_topk_recovery"))
        if grouped_recovery is None or review_load is None or review_load <= 0.0:
            continue
        rescue_values.append(float(grouped_recovery / review_load))
    return {
        "budgeted_recall": float(np.mean([value for value in budgeted_values if value is not None])) if any(value is not None for value in budgeted_values) else None,
        "budget_compliance_rate": len(
            [
                row
                for row in rows
                if (_to_float(row.get("prediction_set_avg_size")) or 0.0) <= float(review_budget)
                and _to_float(row.get("adaptive_defer_to_human")) != 1.0
            ]
        )
        / len(rows),
        "defer_rate": len([row for row in rows if _to_float(row.get("adaptive_defer_to_human")) == 1.0]) / len(rows),
        "review_load_proxy": float(np.mean([value for value in review_loads if value is not None])) if any(value is not None for value in review_loads) else None,
        "effort_adjusted_utility": float(np.mean([value for value in effort_values if value is not None])) if any(value is not None for value in effort_values) else None,
        "rescue_per_review_item": float(np.mean(rescue_values)) if rescue_values else None,
    }


def _best_baseline_effort_adjusted_utility(
    per_sequence_rows: list[dict[str, str]],
    adaptive_method: str,
    review_budget: int,
) -> tuple[str, float | None]:
    best_method = "n/a"
    best_value = None
    for method in sorted({row["method"] for row in per_sequence_rows if row["method"] != adaptive_method}):
        metrics = _human_effort_summary(per_sequence_rows, method, review_budget)
        value = metrics["effort_adjusted_utility"]
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_method = method
            best_value = value
    return best_method, best_value


def _build_practical_utility_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 4.6))
    colors = {"rescue_first": "#f58518", "shortlist_first": "#54a24b"}
    grouped_modes: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["operating_profile"]), int(row["review_budget_k"]))
        grouped_modes.setdefault(key, []).append(row)
    for (profile, budget), items in sorted(grouped_modes.items()):
        mean_efficiency = float(np.mean([float(item["review_efficiency"] or 0.0) for item in items]))
        mean_budgeted = float(np.mean([float(item["budgeted_recall"] or 0.0) for item in items]))
        axis.scatter(
            mean_budgeted,
            mean_efficiency,
            color=colors.get(profile, "#999999"),
            s=105,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.annotate(
            f"{profile[0].upper()}{budget}",
            (mean_budgeted, mean_efficiency),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )
    axis.set_xlabel("Mean recall@budget")
    axis.set_ylabel("Mean review efficiency")
    axis.set_title("Practical utility frontier")
    axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _build_human_effort_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["operating_profile"]), int(row["review_budget_k"]))
        grouped.setdefault(key, []).append(row)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    colors = {"rescue_first": "#2f855a", "shortlist_first": "#c05621"}
    for (profile, budget), items in sorted(grouped.items()):
        mean_effort = float(np.mean([float(item["review_load_proxy"] or 0.0) for item in items]))
        mean_utility = float(np.mean([float(item["effort_adjusted_utility"] or 0.0) for item in items]))
        mean_fragile = float(np.mean([float(item["fragile_effort_adjusted_utility"] or 0.0) for item in items if item["fragile_effort_adjusted_utility"] is not None])) if any(item["fragile_effort_adjusted_utility"] is not None for item in items) else 0.0
        mean_defer = float(np.mean([float(item["defer_rate"] or 0.0) for item in items]))
        axes[0].scatter(mean_effort, mean_utility, s=130, color=colors.get(profile, "#666666"), alpha=0.9)
        axes[0].annotate(f"{profile[0].upper()}{budget}", (mean_effort, mean_utility), textcoords="offset points", xytext=(5, 4), fontsize=9)
        axes[1].scatter(mean_defer, mean_fragile, s=130, color=colors.get(profile, "#666666"), alpha=0.9)
        axes[1].annotate(f"{profile[0].upper()}{budget}", (mean_defer, mean_fragile), textcoords="offset points", xytext=(5, 4), fontsize=9)
    axes[0].set_xlabel("Mean review-load proxy")
    axes[0].set_ylabel("Mean effort-adjusted utility")
    axes[0].set_title("Human effort frontier")
    axes[1].set_xlabel("Mean defer rate")
    axes[1].set_ylabel("Mean fragile-case effort-adjusted utility")
    axes[1].set_title("Fragile-case escalation tradeoff")
    for axis in axes:
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _build_profile_regret_rows(profile_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in profile_rows:
        key = (
            str(row["dataset_label"]),
            str(row["posterior_strategy"]),
            int(row["review_budget_k"]),
        )
        grouped.setdefault(key, []).append(row)
    regret_rows: list[dict[str, object]] = []
    for (dataset_label, posterior_strategy, review_budget), items in sorted(grouped.items()):
        best_shortlist = max(float(item["adaptive_shortlist_utility"] or 0.0) for item in items)
        best_grouped = max(float(item["adaptive_grouped_topk_delta"] or 0.0) for item in items)
        downstream_values = [float(item["adaptive_downstream_exact_delta"] or 0.0) for item in items if item["adaptive_downstream_exact_delta"] is not None]
        best_downstream = max(downstream_values) if downstream_values else None
        best_budgeted_recall = max(float(item.get("budgeted_recall") or 0.0) for item in items)
        best_effort_adjusted = max(float(item.get("effort_adjusted_utility") or 0.0) for item in items)
        best_fixed_shortlist = max(float(item.get("best_baseline_shortlist_utility") or 0.0) for item in items)
        best_fixed_grouped = max(float(item.get("best_baseline_grouped_topk_delta") or 0.0) for item in items)
        fixed_downstream_values = [float(item.get("best_baseline_downstream_exact_delta") or 0.0) for item in items if item.get("best_baseline_downstream_exact_delta") is not None]
        best_fixed_downstream = max(fixed_downstream_values) if fixed_downstream_values else None
        fixed_effort_values = [float(item.get("best_fixed_effort_adjusted_utility") or 0.0) for item in items if item.get("best_fixed_effort_adjusted_utility") is not None]
        best_fixed_effort = max(fixed_effort_values) if fixed_effort_values else None
        for item in items:
            downstream_value = item["adaptive_downstream_exact_delta"]
            regret_rows.append(
                {
                    "dataset_label": dataset_label,
                    "posterior_strategy": posterior_strategy,
                    "review_budget_k": review_budget,
                    "operating_profile": item["operating_profile"],
                    "shortlist_regret_to_best_profile": best_shortlist - float(item["adaptive_shortlist_utility"] or 0.0),
                    "grouped_regret_to_best_profile": best_grouped - float(item["adaptive_grouped_topk_delta"] or 0.0),
                    "downstream_regret_to_best_profile": None
                    if best_downstream is None or downstream_value is None
                    else best_downstream - float(downstream_value or 0.0),
                    "budgeted_recall_regret_to_best_profile": best_budgeted_recall - float(item.get("budgeted_recall") or 0.0),
                    "effort_adjusted_regret_to_best_profile": best_effort_adjusted - float(item.get("effort_adjusted_utility") or 0.0),
                    "shortlist_gap_to_best_fixed": float(item["adaptive_shortlist_utility"] or 0.0) - best_fixed_shortlist,
                    "grouped_gap_to_best_fixed": float(item["adaptive_grouped_topk_delta"] or 0.0) - best_fixed_grouped,
                    "downstream_gap_to_best_fixed": None
                    if best_fixed_downstream is None or downstream_value is None
                    else float(downstream_value or 0.0) - best_fixed_downstream,
                    "effort_adjusted_gap_to_best_fixed": None
                    if best_fixed_effort is None
                    else float(item.get("effort_adjusted_utility") or 0.0) - best_fixed_effort,
                }
            )
    return regret_rows


def _selector_profile_usage(per_sequence_rows: list[dict[str, str]], adaptive_method: str) -> dict[str, float | None]:
    rows = [row for row in per_sequence_rows if row["method"] == adaptive_method]
    if not rows:
        return {
            "selector_shortlist_rate": None,
            "selector_rescue_rate": None,
            "selector_direct_defer_rate": None,
            "selector_mean_shortlist_probability": None,
        }
    shortlist_rows = [row for row in rows if row.get("adaptive_operating_profile") == "shortlist_first"]
    rescue_rows = [row for row in rows if row.get("adaptive_operating_profile") == "rescue_first"]
    probabilities = [
        _to_float(row.get("adaptive_selector_shortlist_probability"))
        for row in rows
        if _to_float(row.get("adaptive_selector_shortlist_probability")) is not None
    ]
    return {
        "selector_shortlist_rate": len(shortlist_rows) / len(rows),
        "selector_rescue_rate": len(rescue_rows) / len(rows),
        "selector_direct_defer_rate": len(
            [
                row
                for row in rows
                if row.get("adaptive_decision_reason") == "selector_uncertain_fragile_budget"
            ]
        )
        / len(rows),
        "selector_mean_shortlist_probability": float(np.mean(probabilities)) if probabilities else None,
    }


def _enrich_selector_rows(
    selector_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    lookup: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in profile_rows:
        key = (str(row["dataset_label"]), str(row["posterior_strategy"]), int(row["review_budget_k"]))
        lookup.setdefault(key, []).append(row)
    enriched: list[dict[str, object]] = []
    for row in selector_rows:
        key = (str(row["dataset_label"]), str(row["posterior_strategy"]), int(row["review_budget_k"]))
        candidates = lookup.get(key, [])
        best_effort = max((float(candidate.get("effort_adjusted_utility") or 0.0) for candidate in candidates), default=0.0)
        best_budgeted = max((float(candidate.get("budgeted_recall") or 0.0) for candidate in candidates), default=0.0)
        best_grouped = max((float(candidate.get("adaptive_grouped_topk_delta") or 0.0) for candidate in candidates), default=0.0)
        best_fragile = max((float(candidate.get("fragile_effort_adjusted_utility") or 0.0) for candidate in candidates), default=0.0)
        rescue_row = next((candidate for candidate in candidates if candidate.get("operating_profile") == "rescue_first"), None)
        shortlist_row = next((candidate for candidate in candidates if candidate.get("operating_profile") == "shortlist_first"), None)
        enriched.append(
            {
                **row,
                "effort_adjusted_regret_to_best_profile": best_effort - float(row.get("effort_adjusted_utility") or 0.0),
                "budgeted_recall_regret_to_best_profile": best_budgeted - float(row.get("budgeted_recall") or 0.0),
                "grouped_regret_to_best_profile": best_grouped - float(row.get("adaptive_grouped_topk_delta") or 0.0),
                "fragile_effort_regret_to_best_profile": best_fragile - float(row.get("fragile_effort_adjusted_utility") or 0.0),
                "gap_to_rescue_first_effort": None
                if rescue_row is None
                else float(row.get("effort_adjusted_utility") or 0.0)
                - float(rescue_row.get("effort_adjusted_utility") or 0.0),
                "gap_to_shortlist_first_effort": None
                if shortlist_row is None
                else float(row.get("effort_adjusted_utility") or 0.0)
                - float(shortlist_row.get("effort_adjusted_utility") or 0.0),
                "gap_to_rescue_first_grouped": None
                if rescue_row is None
                else float(row.get("adaptive_grouped_topk_delta") or 0.0)
                - float(rescue_row.get("adaptive_grouped_topk_delta") or 0.0),
                "gap_to_shortlist_first_grouped": None
                if shortlist_row is None
                else float(row.get("adaptive_grouped_topk_delta") or 0.0)
                - float(shortlist_row.get("adaptive_grouped_topk_delta") or 0.0),
            }
        )
    return enriched


def _build_profile_selector_plot(
    selector_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    colors = {2: "#c05621", 3: "#2b6cb0", 5: "#2f855a"}
    for row in selector_rows:
        budget = int(row["review_budget_k"])
        axes[0].scatter(
            float(row.get("effort_adjusted_utility") or 0.0),
            float(row.get("adaptive_grouped_topk_delta") or 0.0),
            s=125,
            color=colors.get(budget, "#666666"),
            alpha=0.9,
        )
        axes[0].annotate(
            f"{row['dataset_label'].split('_')[0][0].upper()}{budget}",
            (
                float(row.get("effort_adjusted_utility") or 0.0),
                float(row.get("adaptive_grouped_topk_delta") or 0.0),
            ),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
        axes[1].scatter(
            float(row.get("budgeted_recall_regret_to_best_profile") or 0.0),
            float(row.get("effort_adjusted_regret_to_best_profile") or 0.0),
            s=125,
            color=colors.get(budget, "#666666"),
            alpha=0.9,
        )
        axes[1].annotate(
            f"{row['posterior_strategy'][0:3]}{budget}",
            (
                float(row.get("budgeted_recall_regret_to_best_profile") or 0.0),
                float(row.get("effort_adjusted_regret_to_best_profile") or 0.0),
            ),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[0].set_xlabel("Effort-adjusted utility")
    axes[0].set_ylabel("Grouped top-k delta vs fixed")
    axes[0].set_title("Selector operating point")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Budgeted-recall regret to best profile")
    axes[1].set_ylabel("Effort-adjusted regret to best profile")
    axes[1].set_title("Selector gap to profile oracle")
    for axis in axes:
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    output_root = ensure_directory("paper/icdar_das/paper_2/experiments")
    summary_rows: list[dict[str, object]] = []
    profile_summary_rows: list[dict[str, object]] = []
    selector_summary_rows: list[dict[str, object]] = []
    practical_utility_rows: list[dict[str, object]] = []
    human_effort_rows: list[dict[str, object]] = []
    fragile_case_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    for dataset_label, config_path in CONFIGS:
        base_config = load_config(config_path)
        for policy in POLICIES:
            adaptive_config = base_config.model_copy(
                update={
                    "experiment": base_config.experiment.model_copy(
                        update={
                            "name": f"{base_config.experiment.name}_{policy}",
                            "notes": (
                                f"{base_config.experiment.notes or ''} "
                                f"Adaptive decoder pack for paper_2 using {policy}."
                            ).strip(),
                        }
                    ),
                    "adaptive_decoding": base_config.adaptive_decoding.model_copy(
                        update={"enabled": True, "policy": policy}
                    ),
                }
            )
            for strategy in STRATEGIES:
                result = run_sequence_branch_experiment(adaptive_config, strategy_override=strategy)
                summary = _read_csv(Path(result["run_dir"]) / "sequence_branch_summary.csv")
                per_sequence = _read_csv(Path(result["run_dir"]) / "sequence_branch_examples.csv")
                by_method = _method_summary(summary)
                fixed = by_method["fixed_greedy"]
                uncertainty = by_method.get("uncertainty_beam", {})
                conformal = by_method.get("conformal_beam", {})
                adaptive_method = _controller_method_name(policy)
                adaptive = by_method.get(adaptive_method, {})
                baseline_candidates = {
                    method: metrics
                    for method, metrics in by_method.items()
                    if method != adaptive_method
                }
                best_grouped_method = max(
                    baseline_candidates.items(),
                    key=lambda item: float(item[1].get("grouped_topk") or 0.0),
                )[0]
                best_downstream_method = max(
                    baseline_candidates.items(),
                    key=lambda item: float(item[1].get("downstream_exact") or 0.0),
                )[0]
                usage = _adaptive_usage(per_sequence, adaptive_method)
                failure = _failure_reduction(per_sequence, adaptive_method)
                best_shortlist_method, best_shortlist_utility = _best_baseline_shortlist(baseline_candidates)
                best_review_efficiency_method, best_review_efficiency = _best_baseline_review_efficiency(
                    baseline_candidates
                )
                row_payload = {
                    "dataset_label": dataset_label,
                    "posterior_strategy": strategy,
                    "controller_policy": policy,
                    "controller_label": _controller_label(policy),
                    "operating_profile": None,
                    "review_budget_k": adaptive_config.adaptive_decoding.review_budget_k,
                    "run_dir": str(result["run_dir"]),
                    "best_baseline_grouped_method": best_grouped_method,
                    "best_baseline_downstream_method": best_downstream_method,
                    "best_baseline_shortlist_method": best_shortlist_method,
                    "best_baseline_shortlist_utility": best_shortlist_utility,
                    "best_baseline_review_efficiency_method": best_review_efficiency_method,
                    "best_baseline_review_efficiency": best_review_efficiency,
                    "best_baseline_grouped_topk_delta": None
                    if baseline_candidates[best_grouped_method].get("grouped_topk") is None
                    else float((baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0) - (fixed.get("grouped_topk") or 0.0)),
                    "best_baseline_downstream_exact_delta": None
                    if baseline_candidates[best_downstream_method].get("downstream_exact") is None
                    else float((baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0) - (fixed.get("downstream_exact") or 0.0)),
                    "adaptive_grouped_topk_delta": None if adaptive.get("grouped_topk") is None else float(adaptive["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                    "adaptive_grouped_exact_delta": None if adaptive.get("grouped_exact") is None else float(adaptive["grouped_exact"] - (fixed.get("grouped_exact") or 0.0)),
                    "adaptive_downstream_exact_delta": None if adaptive.get("downstream_exact") is None else float(adaptive["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)),
                    "adaptive_downstream_token_delta": None if adaptive.get("downstream_token") is None else float(adaptive["downstream_token"] - (fixed.get("downstream_token") or 0.0)),
                    "adaptive_vs_uncertainty_grouped_topk_delta": None if adaptive.get("grouped_topk") is None else float(adaptive["grouped_topk"] - (uncertainty.get("grouped_topk") or 0.0)),
                    "adaptive_vs_uncertainty_downstream_exact_delta": None if adaptive.get("downstream_exact") is None else float(adaptive["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)),
                    "adaptive_vs_conformal_downstream_exact_delta": None if adaptive.get("downstream_exact") is None else float(adaptive["downstream_exact"] - (conformal.get("downstream_exact") or 0.0)),
                    "adaptive_vs_best_grouped_gap": None if adaptive.get("grouped_topk") is None else float(adaptive["grouped_topk"] - (baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0)),
                    "adaptive_vs_best_downstream_gap": None if adaptive.get("downstream_exact") is None else float(adaptive["downstream_exact"] - (baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0)),
                    "uncertainty_grouped_topk_delta": None if uncertainty.get("grouped_topk") is None else float(uncertainty["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                    "conformal_grouped_exact_delta": None if conformal.get("grouped_exact") is None else float(conformal["grouped_exact"] - (uncertainty.get("grouped_exact") or 0.0)),
                    "uncertainty_downstream_exact_delta": None if uncertainty.get("downstream_exact") is None else float(uncertainty["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)),
                    "conformal_downstream_exact_delta": None if conformal.get("downstream_exact") is None else float(conformal["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)),
                    "adaptive_shortlist_recall_at_2": adaptive.get("shortlist_recall_at_2"),
                    "adaptive_shortlist_recall_at_3": adaptive.get("shortlist_recall_at_3"),
                    "adaptive_shortlist_recall_at_5": adaptive.get("shortlist_recall_at_5"),
                    "adaptive_shortlist_utility": adaptive.get("shortlist_utility"),
                    "adaptive_prediction_set_avg_size": adaptive.get("prediction_set_avg_size"),
                    "budgeted_recall": None,
                    "budget_compliance_rate": None,
                    "review_load_proxy": None,
                    "effort_adjusted_utility": None,
                    "rescue_per_review_item": None,
                    "best_fixed_effort_adjusted_utility": None,
                    "selector_shortlist_rate": None,
                    "selector_rescue_rate": None,
                    "selector_direct_defer_rate": None,
                    "selector_mean_shortlist_probability": None,
                    "fragile_case_rate": None,
                    "fragile_shortlist_utility": None,
                    "fragile_recall_at_2": None,
                    "fragile_prediction_set_avg_size": None,
                    "fragile_review_efficiency": None,
                    "fragile_budgeted_recall": None,
                    "fragile_budget_compliance_rate": None,
                    "fragile_defer_rate": None,
                    "fragile_review_load_proxy": None,
                    "fragile_effort_adjusted_utility": None,
                    **usage,
                    **failure,
                }
                summary_rows.append(row_payload)
                for gate_row in _gate_rows(Path(result["run_dir"])):
                    gate_rows.append(
                        {
                            "dataset_label": dataset_label,
                            "posterior_strategy": strategy,
                            "controller_policy": policy,
                            "controller_label": _controller_label(policy),
                            "operating_profile": None,
                            "review_budget_k": adaptive_config.adaptive_decoding.review_budget_k,
                            **gate_row,
                        }
                    )

        for profile in PROFILES:
            for review_budget in REVIEW_BUDGETS:
                adaptive_config = base_config.model_copy(
                    update={
                        "experiment": base_config.experiment.model_copy(
                            update={
                                "name": (
                                    f"{base_config.experiment.name}_support_aware_profiled_gate_"
                                    f"{profile}_k{review_budget}"
                                ),
                                "notes": (
                                    f"{base_config.experiment.notes or ''} "
                                    f"Profile-aware adaptive decoder pack for paper_2 using {profile} at budget {review_budget}."
                                ).strip(),
                            }
                        ),
                        "adaptive_decoding": base_config.adaptive_decoding.model_copy(
                            update={
                                "enabled": True,
                                "policy": "support_aware_profiled_gate",
                                "operating_profile": profile,
                                "review_budget_k": review_budget,
                            }
                        ),
                    }
                )
                for strategy in STRATEGIES:
                    result = run_sequence_branch_experiment(adaptive_config, strategy_override=strategy)
                    summary = _read_csv(Path(result["run_dir"]) / "sequence_branch_summary.csv")
                    per_sequence = _read_csv(Path(result["run_dir"]) / "sequence_branch_examples.csv")
                    by_method = _method_summary(summary)
                    fixed = by_method["fixed_greedy"]
                    uncertainty = by_method.get("uncertainty_beam", {})
                    conformal = by_method.get("conformal_beam", {})
                    adaptive_method = _controller_method_name("support_aware_profiled_gate")
                    adaptive = by_method.get(adaptive_method, {})
                    baseline_candidates = {
                        method: metrics
                        for method, metrics in by_method.items()
                        if method != adaptive_method
                    }
                    best_grouped_method = max(
                        baseline_candidates.items(),
                        key=lambda item: float(item[1].get("grouped_topk") or 0.0),
                    )[0]
                    best_downstream_method = max(
                        baseline_candidates.items(),
                        key=lambda item: float(item[1].get("downstream_exact") or 0.0),
                    )[0]
                    usage = _adaptive_usage(per_sequence, adaptive_method)
                    failure = _failure_reduction(per_sequence, adaptive_method)
                    fragile = _fragile_case_summary(per_sequence, adaptive_method, review_budget)
                    human_effort = _human_effort_summary(per_sequence, adaptive_method, review_budget)
                    best_shortlist_method, best_shortlist_utility = _best_baseline_shortlist(
                        baseline_candidates
                    )
                    best_review_efficiency_method, best_review_efficiency = (
                        _best_baseline_review_efficiency(baseline_candidates)
                    )
                    best_fixed_effort_method, best_fixed_effort_utility = (
                        _best_baseline_effort_adjusted_utility(
                            per_sequence,
                            adaptive_method,
                            review_budget,
                        )
                    )
                    profile_row = {
                        "dataset_label": dataset_label,
                        "posterior_strategy": strategy,
                        "controller_policy": "support_aware_profiled_gate",
                        "controller_label": _profile_budget_label(profile, review_budget),
                        "operating_profile": profile,
                        "review_budget_k": review_budget,
                        "run_dir": str(result["run_dir"]),
                        "best_baseline_grouped_method": best_grouped_method,
                        "best_baseline_downstream_method": best_downstream_method,
                        "best_baseline_shortlist_method": best_shortlist_method,
                        "best_baseline_shortlist_utility": best_shortlist_utility,
                        "best_baseline_review_efficiency_method": best_review_efficiency_method,
                        "best_baseline_review_efficiency": best_review_efficiency,
                        "best_baseline_grouped_topk_delta": None
                        if baseline_candidates[best_grouped_method].get("grouped_topk") is None
                        else float(
                            (baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0)
                            - (fixed.get("grouped_topk") or 0.0)
                        ),
                        "best_baseline_downstream_exact_delta": None
                        if baseline_candidates[best_downstream_method].get("downstream_exact") is None
                        else float(
                            (baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0)
                            - (fixed.get("downstream_exact") or 0.0)
                        ),
                        "adaptive_grouped_topk_delta": None
                        if adaptive.get("grouped_topk") is None
                        else float(adaptive["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                        "adaptive_grouped_exact_delta": None
                        if adaptive.get("grouped_exact") is None
                        else float(adaptive["grouped_exact"] - (fixed.get("grouped_exact") or 0.0)),
                        "adaptive_downstream_exact_delta": None
                        if adaptive.get("downstream_exact") is None
                        else float(adaptive["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)),
                        "adaptive_downstream_token_delta": None
                        if adaptive.get("downstream_token") is None
                        else float(adaptive["downstream_token"] - (fixed.get("downstream_token") or 0.0)),
                        "adaptive_vs_uncertainty_grouped_topk_delta": None
                        if adaptive.get("grouped_topk") is None
                        else float(adaptive["grouped_topk"] - (uncertainty.get("grouped_topk") or 0.0)),
                        "adaptive_vs_uncertainty_downstream_exact_delta": None
                        if adaptive.get("downstream_exact") is None
                        else float(
                            adaptive["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)
                        ),
                        "adaptive_vs_conformal_downstream_exact_delta": None
                        if adaptive.get("downstream_exact") is None
                        else float(
                            adaptive["downstream_exact"] - (conformal.get("downstream_exact") or 0.0)
                        ),
                        "adaptive_vs_best_grouped_gap": None
                        if adaptive.get("grouped_topk") is None
                        else float(
                            adaptive["grouped_topk"]
                            - (baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0)
                        ),
                        "adaptive_vs_best_downstream_gap": None
                        if adaptive.get("downstream_exact") is None
                        else float(
                            adaptive["downstream_exact"]
                            - (baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0)
                        ),
                        "uncertainty_grouped_topk_delta": None
                        if uncertainty.get("grouped_topk") is None
                        else float(uncertainty["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                        "conformal_grouped_exact_delta": None
                        if conformal.get("grouped_exact") is None
                        else float(
                            conformal["grouped_exact"] - (uncertainty.get("grouped_exact") or 0.0)
                        ),
                        "uncertainty_downstream_exact_delta": None
                        if uncertainty.get("downstream_exact") is None
                        else float(
                            uncertainty["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)
                        ),
                        "conformal_downstream_exact_delta": None
                        if conformal.get("downstream_exact") is None
                        else float(
                            conformal["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)
                        ),
                        "adaptive_shortlist_recall_at_2": adaptive.get("shortlist_recall_at_2"),
                        "adaptive_shortlist_recall_at_3": adaptive.get("shortlist_recall_at_3"),
                        "adaptive_shortlist_recall_at_5": adaptive.get("shortlist_recall_at_5"),
                        "adaptive_shortlist_utility": adaptive.get("shortlist_utility"),
                        "adaptive_prediction_set_avg_size": adaptive.get("prediction_set_avg_size"),
                        "budgeted_recall": human_effort["budgeted_recall"],
                        "budget_compliance_rate": human_effort["budget_compliance_rate"],
                        "review_load_proxy": human_effort["review_load_proxy"],
                        "effort_adjusted_utility": human_effort["effort_adjusted_utility"],
                        "rescue_per_review_item": human_effort["rescue_per_review_item"],
                        "best_fixed_effort_adjusted_utility": best_fixed_effort_utility,
                        "selector_shortlist_rate": None,
                        "selector_rescue_rate": None,
                        "selector_direct_defer_rate": None,
                        "selector_mean_shortlist_probability": None,
                        **usage,
                        **failure,
                        **fragile,
                    }
                    summary_rows.append(profile_row)
                    profile_summary_rows.append(profile_row)
                    practical_utility_rows.append(
                        {
                            "dataset_label": dataset_label,
                            "posterior_strategy": strategy,
                            "operating_profile": profile,
                            "review_budget_k": review_budget,
                            "shortlist_utility": adaptive.get("shortlist_utility"),
                            "prediction_set_avg_size": adaptive.get("prediction_set_avg_size"),
                            "review_efficiency": _review_efficiency(
                                adaptive.get("shortlist_utility"),
                                adaptive.get("prediction_set_avg_size"),
                            ),
                            "shortlist_recall_at_2": adaptive.get("shortlist_recall_at_2"),
                            "shortlist_recall_at_3": adaptive.get("shortlist_recall_at_3"),
                            "shortlist_recall_at_5": adaptive.get("shortlist_recall_at_5"),
                            "fragile_case_rate": fragile["fragile_case_rate"],
                            "fragile_shortlist_utility": fragile["fragile_shortlist_utility"],
                            "fragile_recall_at_2": fragile["fragile_recall_at_2"],
                            "fragile_prediction_set_avg_size": fragile["fragile_prediction_set_avg_size"],
                            "fragile_review_efficiency": fragile["fragile_review_efficiency"],
                            "defer_rate": usage["adaptive_defer_rate"],
                            "budgeted_recall": human_effort["budgeted_recall"],
                            "budget_compliance_rate": human_effort["budget_compliance_rate"],
                            "review_load_proxy": human_effort["review_load_proxy"],
                            "effort_adjusted_utility": human_effort["effort_adjusted_utility"],
                            "rescue_per_review_item": human_effort["rescue_per_review_item"],
                            "best_fixed_effort_adjusted_utility": best_fixed_effort_utility,
                            "grouped_topk_delta": profile_row["adaptive_grouped_topk_delta"],
                            "downstream_exact_delta": profile_row["adaptive_downstream_exact_delta"],
                            "shortlist_gap_to_best_fixed": None
                            if best_shortlist_utility is None or adaptive.get("shortlist_utility") is None
                            else float((adaptive.get("shortlist_utility") or 0.0) - best_shortlist_utility),
                            "review_efficiency_gap_to_best_fixed": None
                            if best_review_efficiency is None
                            else float(
                                (
                                    _review_efficiency(
                                        adaptive.get("shortlist_utility"),
                                        adaptive.get("prediction_set_avg_size"),
                                    )
                                    or 0.0
                                )
                                - best_review_efficiency
                            ),
                            "effort_adjusted_gap_to_best_fixed": None
                            if best_fixed_effort_utility is None
                            else float(
                                (human_effort["effort_adjusted_utility"] or 0.0)
                                - best_fixed_effort_utility
                            ),
                            "best_fixed_shortlist_method": best_shortlist_method,
                            "best_fixed_review_efficiency_method": best_review_efficiency_method,
                            "best_fixed_effort_method": best_fixed_effort_method,
                        }
                    )
                    human_effort_rows.append(
                        {
                            "dataset_label": dataset_label,
                            "posterior_strategy": strategy,
                            "operating_profile": profile,
                            "review_budget_k": review_budget,
                            "budgeted_recall": human_effort["budgeted_recall"],
                            "budget_compliance_rate": human_effort["budget_compliance_rate"],
                            "defer_rate": human_effort["defer_rate"],
                            "review_load_proxy": human_effort["review_load_proxy"],
                            "effort_adjusted_utility": human_effort["effort_adjusted_utility"],
                            "rescue_per_review_item": human_effort["rescue_per_review_item"],
                            "best_fixed_effort_adjusted_utility": best_fixed_effort_utility,
                            "best_fixed_effort_method": best_fixed_effort_method,
                            "effort_adjusted_gap_to_best_fixed": None
                            if best_fixed_effort_utility is None
                            else float(
                                (human_effort["effort_adjusted_utility"] or 0.0)
                                - best_fixed_effort_utility
                            ),
                            "grouped_topk_delta": profile_row["adaptive_grouped_topk_delta"],
                            "downstream_exact_delta": profile_row["adaptive_downstream_exact_delta"],
                            "fragile_budgeted_recall": fragile["fragile_budgeted_recall"],
                            "fragile_budget_compliance_rate": fragile["fragile_budget_compliance_rate"],
                            "fragile_defer_rate": fragile["fragile_defer_rate"],
                            "fragile_review_load_proxy": fragile["fragile_review_load_proxy"],
                            "fragile_effort_adjusted_utility": fragile["fragile_effort_adjusted_utility"],
                        }
                    )
                    fragile_case_rows.append(
                        {
                            "dataset_label": dataset_label,
                            "posterior_strategy": strategy,
                            "operating_profile": profile,
                            "review_budget_k": review_budget,
                            "fragile_case_rate": fragile["fragile_case_rate"],
                            "fragile_shortlist_utility": fragile["fragile_shortlist_utility"],
                            "fragile_budgeted_recall": fragile["fragile_budgeted_recall"],
                            "fragile_budget_compliance_rate": fragile["fragile_budget_compliance_rate"],
                            "fragile_defer_rate": fragile["fragile_defer_rate"],
                            "fragile_review_load_proxy": fragile["fragile_review_load_proxy"],
                            "fragile_effort_adjusted_utility": fragile["fragile_effort_adjusted_utility"],
                            "grouped_topk_delta": profile_row["adaptive_grouped_topk_delta"],
                            "downstream_exact_delta": profile_row["adaptive_downstream_exact_delta"],
                        }
                    )
                    for gate_row in _gate_rows(Path(result["run_dir"])):
                        gate_rows.append(
                            {
                                "dataset_label": dataset_label,
                                "posterior_strategy": strategy,
                                "controller_policy": "support_aware_profiled_gate",
                                "controller_label": _profile_budget_label(profile, review_budget),
                                "operating_profile": profile,
                                "review_budget_k": review_budget,
                                **gate_row,
                            }
                        )

        for review_budget in REVIEW_BUDGETS:
            adaptive_config = base_config.model_copy(
                update={
                    "experiment": base_config.experiment.model_copy(
                        update={
                            "name": (
                                f"{base_config.experiment.name}_support_aware_profile_selector"
                                f"_k{review_budget}"
                            ),
                            "notes": (
                                f"{base_config.experiment.notes or ''} "
                                f"Profile-selection adaptive decoder pack for paper_2 at budget {review_budget}."
                            ).strip(),
                        }
                    ),
                    "adaptive_decoding": base_config.adaptive_decoding.model_copy(
                        update={
                            "enabled": True,
                            "policy": "support_aware_profile_selector",
                            "review_budget_k": review_budget,
                        }
                    ),
                }
            )
            for strategy in STRATEGIES:
                result = run_sequence_branch_experiment(adaptive_config, strategy_override=strategy)
                summary = _read_csv(Path(result["run_dir"]) / "sequence_branch_summary.csv")
                per_sequence = _read_csv(Path(result["run_dir"]) / "sequence_branch_examples.csv")
                by_method = _method_summary(summary)
                fixed = by_method["fixed_greedy"]
                uncertainty = by_method.get("uncertainty_beam", {})
                conformal = by_method.get("conformal_beam", {})
                adaptive_method = _controller_method_name("support_aware_profile_selector")
                adaptive = by_method.get(adaptive_method, {})
                baseline_candidates = {
                    method: metrics
                    for method, metrics in by_method.items()
                    if method != adaptive_method
                }
                best_grouped_method = max(
                    baseline_candidates.items(),
                    key=lambda item: float(item[1].get("grouped_topk") or 0.0),
                )[0]
                best_downstream_method = max(
                    baseline_candidates.items(),
                    key=lambda item: float(item[1].get("downstream_exact") or 0.0),
                )[0]
                usage = _adaptive_usage(per_sequence, adaptive_method)
                selector_usage = _selector_profile_usage(per_sequence, adaptive_method)
                failure = _failure_reduction(per_sequence, adaptive_method)
                fragile = _fragile_case_summary(per_sequence, adaptive_method, review_budget)
                human_effort = _human_effort_summary(per_sequence, adaptive_method, review_budget)
                best_shortlist_method, best_shortlist_utility = _best_baseline_shortlist(
                    baseline_candidates
                )
                best_review_efficiency_method, best_review_efficiency = (
                    _best_baseline_review_efficiency(baseline_candidates)
                )
                best_fixed_effort_method, best_fixed_effort_utility = (
                    _best_baseline_effort_adjusted_utility(
                        per_sequence,
                        adaptive_method,
                        review_budget,
                    )
                )
                selector_row = {
                    "dataset_label": dataset_label,
                    "posterior_strategy": strategy,
                    "controller_policy": "support_aware_profile_selector",
                    "controller_label": f"profile_selector@k={review_budget}",
                    "operating_profile": "auto_selector",
                    "review_budget_k": review_budget,
                    "run_dir": str(result["run_dir"]),
                    "best_baseline_grouped_method": best_grouped_method,
                    "best_baseline_downstream_method": best_downstream_method,
                    "best_baseline_shortlist_method": best_shortlist_method,
                    "best_baseline_shortlist_utility": best_shortlist_utility,
                    "best_baseline_review_efficiency_method": best_review_efficiency_method,
                    "best_baseline_review_efficiency": best_review_efficiency,
                    "best_baseline_grouped_topk_delta": None
                    if baseline_candidates[best_grouped_method].get("grouped_topk") is None
                    else float(
                        (baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0)
                        - (fixed.get("grouped_topk") or 0.0)
                    ),
                    "best_baseline_downstream_exact_delta": None
                    if baseline_candidates[best_downstream_method].get("downstream_exact") is None
                    else float(
                        (baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0)
                        - (fixed.get("downstream_exact") or 0.0)
                    ),
                    "adaptive_grouped_topk_delta": None
                    if adaptive.get("grouped_topk") is None
                    else float(adaptive["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                    "adaptive_grouped_exact_delta": None
                    if adaptive.get("grouped_exact") is None
                    else float(adaptive["grouped_exact"] - (fixed.get("grouped_exact") or 0.0)),
                    "adaptive_downstream_exact_delta": None
                    if adaptive.get("downstream_exact") is None
                    else float(adaptive["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)),
                    "adaptive_downstream_token_delta": None
                    if adaptive.get("downstream_token") is None
                    else float(adaptive["downstream_token"] - (fixed.get("downstream_token") or 0.0)),
                    "adaptive_vs_uncertainty_grouped_topk_delta": None
                    if adaptive.get("grouped_topk") is None
                    else float(adaptive["grouped_topk"] - (uncertainty.get("grouped_topk") or 0.0)),
                    "adaptive_vs_uncertainty_downstream_exact_delta": None
                    if adaptive.get("downstream_exact") is None
                    else float(
                        adaptive["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)
                    ),
                    "adaptive_vs_conformal_downstream_exact_delta": None
                    if adaptive.get("downstream_exact") is None
                    else float(
                        adaptive["downstream_exact"] - (conformal.get("downstream_exact") or 0.0)
                    ),
                    "adaptive_vs_best_grouped_gap": None
                    if adaptive.get("grouped_topk") is None
                    else float(
                        adaptive["grouped_topk"]
                        - (baseline_candidates[best_grouped_method].get("grouped_topk") or 0.0)
                    ),
                    "adaptive_vs_best_downstream_gap": None
                    if adaptive.get("downstream_exact") is None
                    else float(
                        adaptive["downstream_exact"]
                        - (baseline_candidates[best_downstream_method].get("downstream_exact") or 0.0)
                    ),
                    "uncertainty_grouped_topk_delta": None
                    if uncertainty.get("grouped_topk") is None
                    else float(uncertainty["grouped_topk"] - (fixed.get("grouped_topk") or 0.0)),
                    "conformal_grouped_exact_delta": None
                    if conformal.get("grouped_exact") is None
                    else float(
                        conformal["grouped_exact"] - (uncertainty.get("grouped_exact") or 0.0)
                    ),
                    "uncertainty_downstream_exact_delta": None
                    if uncertainty.get("downstream_exact") is None
                    else float(
                        uncertainty["downstream_exact"] - (fixed.get("downstream_exact") or 0.0)
                    ),
                    "conformal_downstream_exact_delta": None
                    if conformal.get("downstream_exact") is None
                    else float(
                        conformal["downstream_exact"] - (uncertainty.get("downstream_exact") or 0.0)
                    ),
                    "adaptive_shortlist_recall_at_2": adaptive.get("shortlist_recall_at_2"),
                    "adaptive_shortlist_recall_at_3": adaptive.get("shortlist_recall_at_3"),
                    "adaptive_shortlist_recall_at_5": adaptive.get("shortlist_recall_at_5"),
                    "adaptive_shortlist_utility": adaptive.get("shortlist_utility"),
                    "adaptive_prediction_set_avg_size": adaptive.get("prediction_set_avg_size"),
                    "budgeted_recall": human_effort["budgeted_recall"],
                    "budget_compliance_rate": human_effort["budget_compliance_rate"],
                    "review_load_proxy": human_effort["review_load_proxy"],
                    "effort_adjusted_utility": human_effort["effort_adjusted_utility"],
                    "rescue_per_review_item": human_effort["rescue_per_review_item"],
                    "best_fixed_effort_adjusted_utility": best_fixed_effort_utility,
                    **usage,
                    **selector_usage,
                    **failure,
                    **fragile,
                }
                summary_rows.append(selector_row)
                selector_summary_rows.append(selector_row)
                for gate_row in _gate_rows(Path(result["run_dir"])):
                    gate_rows.append(
                        {
                            "dataset_label": dataset_label,
                            "posterior_strategy": strategy,
                            "controller_policy": "support_aware_profile_selector",
                            "controller_label": f"profile_selector@k={review_budget}",
                            "operating_profile": "selector",
                            "review_budget_k": review_budget,
                            **gate_row,
                        }
                    )

    csv_path = output_root / "adaptive_decoder_summary.csv"
    md_path = output_root / "adaptive_decoder_summary.md"
    plot_path = output_root / "adaptive_decoder_plot.png"
    upgraded_csv_path = output_root / "upgraded_adaptive_summary.csv"
    upgraded_md_path = output_root / "upgraded_adaptive_summary.md"
    upgraded_plot_path = output_root / "upgraded_adaptive_plot.png"
    operating_csv_path = output_root / "operating_point_summary.csv"
    operating_md_path = output_root / "operating_point_summary.md"
    operating_plot_path = output_root / "operating_point_plot.png"
    shortlist_csv_path = output_root / "shortlist_utility_summary.csv"
    shortlist_md_path = output_root / "shortlist_utility_summary.md"
    shortlist_plot_path = output_root / "shortlist_utility_plot.png"
    profile_csv_path = output_root / "profile_aware_summary.csv"
    profile_md_path = output_root / "profile_aware_summary.md"
    profile_plot_path = output_root / "profile_aware_plot.png"
    practical_csv_path = output_root / "practical_utility_summary.csv"
    practical_md_path = output_root / "practical_utility_summary.md"
    practical_plot_path = output_root / "practical_utility_plot.png"
    human_effort_csv_path = output_root / "human_effort_summary.csv"
    human_effort_md_path = output_root / "human_effort_summary.md"
    human_effort_plot_path = output_root / "human_effort_plot.png"
    fragile_case_csv_path = output_root / "fragile_case_summary.csv"
    fragile_case_md_path = output_root / "fragile_case_summary.md"
    regret_csv_path = output_root / "profile_regret_summary.csv"
    regret_md_path = output_root / "profile_regret_summary.md"
    selector_csv_path = output_root / "profile_selector_summary.csv"
    selector_md_path = output_root / "profile_selector_summary.md"
    selector_plot_path = output_root / "profile_selector_plot.png"
    operator_effort_csv_path = output_root / "operator_effort_dominance.csv"
    operator_effort_md_path = output_root / "operator_effort_dominance.md"
    fragile_selector_csv_path = output_root / "fragile_case_selector_summary.csv"
    fragile_selector_md_path = output_root / "fragile_case_selector_summary.md"
    gate_csv_path = output_root / "learned_gate_coefficients.csv"
    write_csv(csv_path, summary_rows)
    write_csv(upgraded_csv_path, summary_rows)
    _build_plot(summary_rows, plot_path)
    _build_plot(summary_rows, upgraded_plot_path)
    operating_rows = _build_operating_point_rows(summary_rows)
    write_csv(operating_csv_path, operating_rows)
    _build_operating_point_plot(operating_rows, operating_plot_path)
    shortlist_rows = _build_shortlist_rows(summary_rows)
    write_csv(shortlist_csv_path, shortlist_rows)
    _build_shortlist_plot(shortlist_rows, shortlist_plot_path)
    profile_rows = _build_profile_aware_rows(profile_summary_rows)
    write_csv(profile_csv_path, profile_rows)
    _build_profile_aware_plot(profile_rows, profile_plot_path)
    write_csv(practical_csv_path, practical_utility_rows)
    _build_practical_utility_plot(practical_utility_rows, practical_plot_path)
    write_csv(human_effort_csv_path, human_effort_rows)
    _build_human_effort_plot(human_effort_rows, human_effort_plot_path)
    write_csv(fragile_case_csv_path, fragile_case_rows)
    regret_rows = _build_profile_regret_rows(profile_summary_rows)
    write_csv(regret_csv_path, regret_rows)
    selector_rows = _enrich_selector_rows(selector_summary_rows, profile_summary_rows)
    write_csv(selector_csv_path, selector_rows)
    _build_profile_selector_plot(selector_rows, selector_plot_path)
    operator_effort_rows = [
        {
            "dataset_label": row["dataset_label"],
            "posterior_strategy": row["posterior_strategy"],
            "review_budget_k": row["review_budget_k"],
            "effort_adjusted_utility": row["effort_adjusted_utility"],
            "review_load_proxy": row["review_load_proxy"],
            "rescue_per_review_item": row["rescue_per_review_item"],
            "selector_shortlist_rate": row.get("selector_shortlist_rate"),
            "selector_rescue_rate": row.get("selector_rescue_rate"),
            "selector_direct_defer_rate": row.get("selector_direct_defer_rate"),
            "effort_adjusted_regret_to_best_profile": row.get("effort_adjusted_regret_to_best_profile"),
            "gap_to_rescue_first_effort": row.get("gap_to_rescue_first_effort"),
            "gap_to_shortlist_first_effort": row.get("gap_to_shortlist_first_effort"),
            "effort_adjusted_gap_to_best_fixed": None
            if row.get("best_fixed_effort_adjusted_utility") is None
            else float(row.get("effort_adjusted_utility") or 0.0)
            - float(row.get("best_fixed_effort_adjusted_utility") or 0.0),
        }
        for row in selector_rows
    ]
    write_csv(operator_effort_csv_path, operator_effort_rows)
    fragile_selector_rows = [
        {
            "dataset_label": row["dataset_label"],
            "posterior_strategy": row["posterior_strategy"],
            "review_budget_k": row["review_budget_k"],
            "fragile_case_rate": row.get("fragile_case_rate"),
            "fragile_budgeted_recall": row.get("fragile_budgeted_recall"),
            "fragile_defer_rate": row.get("fragile_defer_rate"),
            "fragile_review_load_proxy": row.get("fragile_review_load_proxy"),
            "fragile_effort_adjusted_utility": row.get("fragile_effort_adjusted_utility"),
            "fragile_effort_regret_to_best_profile": row.get("fragile_effort_regret_to_best_profile"),
            "selector_shortlist_rate": row.get("selector_shortlist_rate"),
            "selector_rescue_rate": row.get("selector_rescue_rate"),
            "selector_direct_defer_rate": row.get("selector_direct_defer_rate"),
            "gap_to_rescue_first_grouped": row.get("gap_to_rescue_first_grouped"),
        }
        for row in selector_rows
    ]
    write_csv(fragile_selector_csv_path, fragile_selector_rows)
    write_csv(gate_csv_path, gate_rows)
    lines = [
        "# Support-Aware Adaptive Decoder Summary",
        "",
        "This pack evaluates lightweight adaptive controllers that switch between raw uncertainty beam decoding and conformal beam decoding, and adjust beam width using support features identified in paper_1.",
        "",
        "## Results",
        "",
    ]
    for row in summary_rows:
        lines.append(
            (
                f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['controller_label']}`: adaptive grouped top-k delta `{_fmt(row['adaptive_grouped_topk_delta'])}`, "
                f"adaptive downstream exact delta `{_fmt(row['adaptive_downstream_exact_delta'])}`, "
                f"adaptive vs raw downstream exact delta `{_fmt(row['adaptive_vs_uncertainty_downstream_exact_delta'])}`, "
                f"adaptive vs best downstream gap `{_fmt(row['adaptive_vs_best_downstream_gap'])}`, "
                f"conformal selection rate `{_fmt(row['adaptive_conformal_rate'])}`, "
                f"mean beam width `{_fmt(row['adaptive_mean_beam_width'])}`, "
                f"best baseline downstream method `{row['best_baseline_downstream_method']}`."
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The adaptive controller is worthwhile only if it improves grouped or downstream recovery without turning into another opaque decoder family.",
            "- The constrained gate is worthwhile only if it preserves grouped rescue while improving the operating point or downstream behavior.",
        ]
    )
    write_text(md_path, "\n".join(lines) + "\n")
    upgraded_lines = [
        "# Upgraded Adaptive Decoder Summary",
        "",
        "This pack compares the rule controller, the unconstrained learned gate, and the constrained learned gate on the real grouped and real downstream evaluation bed.",
        "",
        "## Results",
        "",
    ]
    for row in summary_rows:
        upgraded_lines.append(
            (
                f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['controller_label']}`: grouped top-k delta `{_fmt(row['adaptive_grouped_topk_delta'])}`, "
                f"downstream exact delta `{_fmt(row['adaptive_downstream_exact_delta'])}`, "
                f"gap to best downstream baseline `{_fmt(row['adaptive_vs_best_downstream_gap'])}`, "
                f"set size `{_fmt(row['adaptive_prediction_set_avg_size'])}`, "
                f"conformal rate `{_fmt(row['adaptive_conformal_rate'])}`, "
                f"beam width `{_fmt(row['adaptive_mean_beam_width'])}`."
            )
        )
    upgraded_lines.extend(
        [
            "",
            "## Operating-point takeaway",
            "",
            "- `rule` remains the most rescue-preserving controller on average.",
            "- `learned` is the most compact but can become too conformal-heavy.",
            "- `constrained` is intended to preserve more grouped rescue while staying more efficient than the rule policy.",
        ]
    )
    write_text(upgraded_md_path, "\n".join(upgraded_lines) + "\n")
    operating_lines = [
        "# Operating Point Summary",
        "",
        "This summary compares controller operating points rather than only raw accuracy.",
        "",
    ]
    for row in operating_rows:
        operating_lines.append(
            f"- `{row['controller_label']}`: mean grouped top-k delta `{_fmt(row['mean_grouped_topk_delta'])}`, mean downstream exact delta `{_fmt(row['mean_downstream_exact_delta'])}`, mean set size `{_fmt(row['mean_prediction_set_size'])}`, mean conformal rate `{_fmt(row['mean_conformal_rate'])}`, mean beam width `{_fmt(row['mean_beam_width'])}`."
        )
    write_text(operating_md_path, "\n".join(operating_lines) + "\n")
    shortlist_lines = [
        "# Shortlist Utility Summary",
        "",
        "This summary measures how often the correct grouped transcript remains within small operator review budgets.",
        "",
    ]
    for row in shortlist_rows:
        shortlist_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['controller_label']}`: recall@2 `{_fmt(row['shortlist_recall_at_2'])}`, recall@3 `{_fmt(row['shortlist_recall_at_3'])}`, recall@5 `{_fmt(row['shortlist_recall_at_5'])}`, shortlist utility `{_fmt(row['shortlist_utility'])}`, set size `{_fmt(row['prediction_set_avg_size'])}`."
        )
    write_text(shortlist_md_path, "\n".join(shortlist_lines) + "\n")
    profile_lines = [
        "# Profile-Aware Interactive Decoding Summary",
        "",
        "This summary compares explicit operating profiles rather than forcing a single adaptive policy to serve every archival workflow.",
        "",
        "## Results",
        "",
    ]
    for row in profile_rows:
        profile_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['operating_profile']}` / `k={row['review_budget_k']}`: grouped top-k delta `{_fmt(row['grouped_topk_delta'])}`, downstream exact delta `{_fmt(row['downstream_exact_delta'])}`, recall@budget `{_fmt(row['budgeted_recall'])}`, shortlist utility `{_fmt(row['shortlist_utility'])}`, set size `{_fmt(row['prediction_set_avg_size'])}`, defer rate `{_fmt(row['adaptive_defer_rate'])}`."
        )
    profile_lines.extend(
        [
            "",
            "## Operating-profile takeaway",
            "",
            "- `shortlist_first` is the compact verification profile: it improves budgeted shortlist utility and usually returns smaller candidate sets.",
            "- `rescue_first` is the fragile-manuscript profile: it protects grouped rescue and is more willing to preserve candidates or defer when pruning is risky.",
        ]
    )
    write_text(profile_md_path, "\n".join(profile_lines) + "\n")
    practical_lines = [
        "# Practical Utility Summary",
        "",
        "This summary focuses on operator-facing usefulness: shortlist utility, review efficiency, budgeted recall, and fragile-case behavior under real archival review budgets.",
        "",
    ]
    for row in practical_utility_rows:
        practical_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['operating_profile']}` / `k={row['review_budget_k']}`: shortlist utility `{_fmt(row['shortlist_utility'])}`, review efficiency `{_fmt(row['review_efficiency'])}`, recall@budget `{_fmt(row['budgeted_recall'])}`, budget compliance `{_fmt(row['budget_compliance_rate'])}`, defer rate `{_fmt(row['defer_rate'])}`, fragile utility `{_fmt(row['fragile_shortlist_utility'])}`, gap to best fixed shortlist utility `{_fmt(row['shortlist_gap_to_best_fixed'])}`."
        )
    write_text(practical_md_path, "\n".join(practical_lines) + "\n")
    human_effort_lines = [
        "# Human Effort Summary",
        "",
        "This summary treats the adaptive controller as an operator-facing verification component and measures budgeted recall, review-load proxy, defer rate, and effort-adjusted utility.",
        "",
    ]
    for row in human_effort_rows:
        human_effort_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['operating_profile']}` / `k={row['review_budget_k']}`: recall@budget `{_fmt(row['budgeted_recall'])}`, review-load proxy `{_fmt(row['review_load_proxy'])}`, effort-adjusted utility `{_fmt(row['effort_adjusted_utility'])}`, rescue per review item `{_fmt(row['rescue_per_review_item'])}`, defer rate `{_fmt(row['defer_rate'])}`, gap to best fixed effort utility `{_fmt(row['effort_adjusted_gap_to_best_fixed'])}`."
        )
    write_text(human_effort_md_path, "\n".join(human_effort_lines) + "\n")
    fragile_lines = [
        "# Fragile-Case Summary",
        "",
        "This summary isolates high-risk grouped examples with multiple fragility signals to show how the profile-aware system behaves when historical verification is hardest.",
        "",
    ]
    for row in fragile_case_rows:
        fragile_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `{row['operating_profile']}` / `k={row['review_budget_k']}`: fragile case rate `{_fmt(row['fragile_case_rate'])}`, fragile shortlist utility `{_fmt(row['fragile_shortlist_utility'])}`, fragile recall@budget `{_fmt(row['fragile_budgeted_recall'])}`, fragile defer rate `{_fmt(row['fragile_defer_rate'])}`, fragile review-load proxy `{_fmt(row['fragile_review_load_proxy'])}`, fragile effort-adjusted utility `{_fmt(row['fragile_effort_adjusted_utility'])}`."
        )
    write_text(fragile_case_md_path, "\n".join(fragile_lines) + "\n")
    regret_lines = [
        "# Profile Regret Summary",
        "",
        "This summary measures how far each operating profile is from the best profile in hindsight for each corpus/strategy/budget regime, and how each profile compares with the strongest fixed baseline.",
        "",
    ]
    by_profile: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in regret_rows:
        by_profile.setdefault((str(row["operating_profile"]), int(row["review_budget_k"])), []).append(row)
    for (profile, review_budget), items in sorted(by_profile.items()):
        shortlist_regret = float(np.mean([float(item["shortlist_regret_to_best_profile"] or 0.0) for item in items]))
        grouped_regret = float(np.mean([float(item["grouped_regret_to_best_profile"] or 0.0) for item in items]))
        downstream_values = [float(item["downstream_regret_to_best_profile"] or 0.0) for item in items if item["downstream_regret_to_best_profile"] is not None]
        downstream_regret = float(np.mean(downstream_values)) if downstream_values else None
        budgeted_regret = float(np.mean([float(item["budgeted_recall_regret_to_best_profile"] or 0.0) for item in items]))
        effort_regret = float(np.mean([float(item["effort_adjusted_regret_to_best_profile"] or 0.0) for item in items]))
        shortlist_gap_fixed = float(np.mean([float(item["shortlist_gap_to_best_fixed"] or 0.0) for item in items]))
        grouped_gap_fixed = float(np.mean([float(item["grouped_gap_to_best_fixed"] or 0.0) for item in items]))
        downstream_gap_values = [float(item["downstream_gap_to_best_fixed"] or 0.0) for item in items if item["downstream_gap_to_best_fixed"] is not None]
        downstream_gap_fixed = float(np.mean(downstream_gap_values)) if downstream_gap_values else None
        effort_gap_values = [float(item["effort_adjusted_gap_to_best_fixed"] or 0.0) for item in items if item["effort_adjusted_gap_to_best_fixed"] is not None]
        effort_gap_fixed = float(np.mean(effort_gap_values)) if effort_gap_values else None
        regret_lines.append(
            f"- `{profile}` / `k={review_budget}`: mean shortlist regret `{_fmt(shortlist_regret)}`, mean budgeted-recall regret `{_fmt(budgeted_regret)}`, mean grouped regret `{_fmt(grouped_regret)}`, mean downstream regret `{_fmt(downstream_regret)}`, mean effort-adjusted regret `{_fmt(effort_regret)}`, mean shortlist gap to best fixed `{_fmt(shortlist_gap_fixed)}`, mean grouped gap to best fixed `{_fmt(grouped_gap_fixed)}`, mean downstream gap to best fixed `{_fmt(downstream_gap_fixed)}`, mean effort gap to best fixed `{_fmt(effort_gap_fixed)}`."
        )
    write_text(regret_md_path, "\n".join(regret_lines) + "\n")
    selector_lines = [
        "# Profile Selector Summary",
        "",
        "This summary evaluates the lightweight profile selector that chooses between `rescue_first` and `shortlist_first` under explicit review budgets and fragile-case signals.",
        "",
    ]
    for row in selector_rows:
        selector_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `k={row['review_budget_k']}`: effort-adjusted utility `{_fmt(row['effort_adjusted_utility'])}`, grouped top-k delta `{_fmt(row['adaptive_grouped_topk_delta'])}`, recall@budget `{_fmt(row['budgeted_recall'])}`, shortlist rate `{_fmt(row.get('selector_shortlist_rate'))}`, rescue rate `{_fmt(row.get('selector_rescue_rate'))}`, direct defer rate `{_fmt(row.get('selector_direct_defer_rate'))}`, effort regret to best profile `{_fmt(row.get('effort_adjusted_regret_to_best_profile'))}`, gap to best fixed effort `{_fmt(None if row.get('best_fixed_effort_adjusted_utility') is None else float(row.get('effort_adjusted_utility') or 0.0) - float(row.get('best_fixed_effort_adjusted_utility') or 0.0))}`."
        )
    write_text(selector_md_path, "\n".join(selector_lines) + "\n")
    operator_effort_lines = [
        "# Operator Effort Dominance",
        "",
        "This summary compares the selector's operator-facing efficiency with explicit profiles and the best fixed strategy under the same review budgets.",
        "",
    ]
    for row in operator_effort_rows:
        operator_effort_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `k={row['review_budget_k']}`: effort-adjusted utility `{_fmt(row['effort_adjusted_utility'])}`, review-load proxy `{_fmt(row['review_load_proxy'])}`, rescue per review item `{_fmt(row['rescue_per_review_item'])}`, shortlist rate `{_fmt(row['selector_shortlist_rate'])}`, rescue rate `{_fmt(row['selector_rescue_rate'])}`, selector defer rate `{_fmt(row['selector_direct_defer_rate'])}`, effort regret to best profile `{_fmt(row['effort_adjusted_regret_to_best_profile'])}`, effort gap to best fixed `{_fmt(row['effort_adjusted_gap_to_best_fixed'])}`."
        )
    write_text(operator_effort_md_path, "\n".join(operator_effort_lines) + "\n")
    fragile_selector_lines = [
        "# Fragile-Case Selector Summary",
        "",
        "This summary isolates fragile cases to show whether the selector routes toward rescue-preserving behavior or defer recommendation when historical verification is hardest.",
        "",
    ]
    for row in fragile_selector_rows:
        fragile_selector_lines.append(
            f"- `{row['dataset_label']}` / `{row['posterior_strategy']}` / `k={row['review_budget_k']}`: fragile case rate `{_fmt(row['fragile_case_rate'])}`, fragile recall@budget `{_fmt(row['fragile_budgeted_recall'])}`, fragile defer rate `{_fmt(row['fragile_defer_rate'])}`, fragile effort-adjusted utility `{_fmt(row['fragile_effort_adjusted_utility'])}`, fragile effort regret to best profile `{_fmt(row['fragile_effort_regret_to_best_profile'])}`, shortlist rate `{_fmt(row['selector_shortlist_rate'])}`, rescue rate `{_fmt(row['selector_rescue_rate'])}`, selector defer rate `{_fmt(row['selector_direct_defer_rate'])}`."
        )
    write_text(fragile_selector_md_path, "\n".join(fragile_selector_lines) + "\n")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "plot": str(plot_path),
                "upgraded_csv": str(upgraded_csv_path),
                "upgraded_markdown": str(upgraded_md_path),
                "upgraded_plot": str(upgraded_plot_path),
                "shortlist_csv": str(shortlist_csv_path),
                "shortlist_markdown": str(shortlist_md_path),
                "shortlist_plot": str(shortlist_plot_path),
                "profile_csv": str(profile_csv_path),
                "profile_markdown": str(profile_md_path),
                "profile_plot": str(profile_plot_path),
                "practical_csv": str(practical_csv_path),
                "practical_markdown": str(practical_md_path),
                "practical_plot": str(practical_plot_path),
                "human_effort_csv": str(human_effort_csv_path),
                "human_effort_markdown": str(human_effort_md_path),
                "human_effort_plot": str(human_effort_plot_path),
                "fragile_case_csv": str(fragile_case_csv_path),
                "fragile_case_markdown": str(fragile_case_md_path),
                "regret_csv": str(regret_csv_path),
                "regret_markdown": str(regret_md_path),
                "selector_csv": str(selector_csv_path),
                "selector_markdown": str(selector_md_path),
                "selector_plot": str(selector_plot_path),
                "operator_effort_csv": str(operator_effort_csv_path),
                "operator_effort_markdown": str(operator_effort_md_path),
                "fragile_selector_csv": str(fragile_selector_csv_path),
                "fragile_selector_markdown": str(fragile_selector_md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
