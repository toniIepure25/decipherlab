from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_sequence_failure_cases(
    per_sequence_rows: list[dict[str, Any]],
    high_ambiguity_threshold: float = 0.45,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[float, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_sequence_rows:
        key = (
            float(row["ambiguity_level"]),
            str(row.get("posterior_strategy", "unknown")),
            str(row["example_id"]),
        )
        grouped[key][str(row["method"])] = row

    cases: list[dict[str, Any]] = []
    for (ambiguity_level, posterior_strategy, example_id), methods in sorted(grouped.items()):
        fixed = methods.get("fixed_greedy")
        uncertainty = methods.get("uncertainty_beam")
        conformal = methods.get("conformal_beam")
        if fixed is None or uncertainty is None:
            continue

        def add(case_type: str, source_method: str, target_method: str | None = None) -> None:
            cases.append(
                {
                    "case_type": case_type,
                    "ambiguity_level": ambiguity_level,
                    "posterior_strategy": posterior_strategy,
                    "example_id": example_id,
                    "source_method": source_method,
                    "target_method": target_method,
                }
            )

        if (
            uncertainty["symbol_topk_accuracy"] > fixed["symbol_topk_accuracy"]
            and uncertainty["sequence_topk_recovery"] <= fixed["sequence_topk_recovery"]
        ):
            add("symbol_rescue_without_sequence_rescue", "uncertainty_beam", "fixed_greedy")
        if (
            uncertainty["sequence_topk_recovery"] > fixed["sequence_topk_recovery"]
            or uncertainty["sequence_exact_match"] > fixed["sequence_exact_match"]
        ):
            add("sequence_rescue_uncertainty_beam", "uncertainty_beam", "fixed_greedy")
        if (
            uncertainty["symbol_topk_accuracy"] >= fixed["symbol_topk_accuracy"]
            and uncertainty["sequence_token_accuracy"] < fixed["sequence_token_accuracy"]
        ):
            add("decoder_trapped_by_transition_prior", "uncertainty_beam", "fixed_greedy")
        if ambiguity_level >= high_ambiguity_threshold and (
            uncertainty["prediction_set_avg_size"] is not None
            and uncertainty["prediction_set_avg_size"] >= 3.0
            and uncertainty["sequence_topk_recovery"] <= fixed["sequence_topk_recovery"]
        ):
            add("high_ambiguity_diffuse_uncertainty", "uncertainty_beam", "fixed_greedy")

        if conformal is None:
            continue
        if (
            conformal["prediction_set_coverage"] < uncertainty["prediction_set_coverage"]
            and conformal["sequence_topk_recovery"] < uncertainty["sequence_topk_recovery"]
        ):
            add("conformal_over_pruning", "conformal_beam", "uncertainty_beam")
        if (
            conformal["prediction_set_avg_size"] >= uncertainty["prediction_set_avg_size"]
            and conformal["sequence_topk_recovery"] <= uncertainty["sequence_topk_recovery"]
        ):
            add("conformal_over_expansion", "conformal_beam", "uncertainty_beam")
        if (
            uncertainty["sequence_topk_recovery"] > fixed["sequence_topk_recovery"]
            and conformal["sequence_topk_recovery"] <= fixed["sequence_topk_recovery"]
        ):
            add("sequence_rescue_only_uncertainty_beam", "uncertainty_beam", "conformal_beam")

    grouped_counts: dict[tuple[str, float, str], int] = defaultdict(int)
    for case in cases:
        grouped_counts[(case["case_type"], case["ambiguity_level"], case["posterior_strategy"])] += 1
    summary_rows = [
        {
            "case_type": case_type,
            "ambiguity_level": ambiguity_level,
            "posterior_strategy": posterior_strategy,
            "count": count,
        }
        for (case_type, ambiguity_level, posterior_strategy), count in sorted(grouped_counts.items())
    ]
    return cases, summary_rows
