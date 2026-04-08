from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def build_pairwise_effect_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[(float(row["ambiguity_level"]), str(row.get("posterior_strategy_requested", row.get("posterior_strategy", "unknown"))))][
            str(row["method"])
        ] = row

    pairwise_rows: list[dict[str, Any]] = []
    for (ambiguity_level, posterior_strategy), methods in sorted(grouped.items()):
        fixed = methods.get("fixed_greedy")
        uncertainty = methods.get("uncertainty_beam")
        conformal = methods.get("conformal_beam")
        if fixed is None or uncertainty is None:
            continue

        def _delta(left: dict[str, Any] | None, right: dict[str, Any] | None, key: str) -> float | None:
            if left is None or right is None:
                return None
            left_value = left.get(key)
            right_value = right.get(key)
            if left_value is None or right_value is None:
                return None
            return float(left_value) - float(right_value)

        pairwise_rows.append(
            {
                "ambiguity_level": ambiguity_level,
                "ambiguity_regime": ambiguity_regime_label(ambiguity_level),
                "posterior_strategy_requested": posterior_strategy,
                "uncertainty_sequence_exact_delta": _delta(uncertainty, fixed, "sequence_exact_match"),
                "uncertainty_sequence_topk_delta": _delta(uncertainty, fixed, "sequence_topk_recovery"),
                "uncertainty_token_accuracy_delta": _delta(uncertainty, fixed, "sequence_token_accuracy"),
                "uncertainty_cer_improvement": _delta(fixed, uncertainty, "sequence_cer"),
                "uncertainty_symbol_topk_delta": _delta(uncertainty, fixed, "symbol_topk_accuracy"),
                "conformal_sequence_exact_delta": _delta(conformal, uncertainty, "sequence_exact_match"),
                "conformal_sequence_topk_delta": _delta(conformal, uncertainty, "sequence_topk_recovery"),
                "conformal_token_accuracy_delta": _delta(conformal, uncertainty, "sequence_token_accuracy"),
                "conformal_cer_improvement": _delta(uncertainty, conformal, "sequence_cer"),
                "conformal_coverage_delta": _delta(conformal, uncertainty, "prediction_set_coverage"),
                "conformal_set_size_delta": _delta(conformal, uncertainty, "prediction_set_avg_size"),
                "conformal_family_delta": _delta(
                    conformal,
                    uncertainty,
                    "family_identification_accuracy",
                ),
                "family_signal_available": float(
                    any(method.get("family_identification_accuracy") is not None for method in methods.values())
                ),
            }
        )
    return pairwise_rows


def ambiguity_regime_label(ambiguity_level: float) -> str:
    if ambiguity_level <= 0.15:
        return "low"
    if ambiguity_level < 0.45:
        return "medium"
    return "high"


def build_ambiguity_regime_rows(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        grouped[(row["posterior_strategy_requested"], row["ambiguity_regime"])].append(row)

    regime_rows: list[dict[str, Any]] = []
    for (posterior_strategy, regime), rows in sorted(grouped.items()):
        regime_rows.append(
            {
                "posterior_strategy_requested": posterior_strategy,
                "ambiguity_regime": regime,
                "mean_uncertainty_sequence_exact_delta": float(
                    np.mean([row["uncertainty_sequence_exact_delta"] for row in rows if row["uncertainty_sequence_exact_delta"] is not None])
                )
                if any(row["uncertainty_sequence_exact_delta"] is not None for row in rows)
                else None,
                "mean_uncertainty_sequence_topk_delta": float(
                    np.mean([row["uncertainty_sequence_topk_delta"] for row in rows if row["uncertainty_sequence_topk_delta"] is not None])
                )
                if any(row["uncertainty_sequence_topk_delta"] is not None for row in rows)
                else None,
                "mean_conformal_sequence_exact_delta": float(
                    np.mean([row["conformal_sequence_exact_delta"] for row in rows if row["conformal_sequence_exact_delta"] is not None])
                )
                if any(row["conformal_sequence_exact_delta"] is not None for row in rows)
                else None,
                "mean_conformal_coverage_delta": float(
                    np.mean([row["conformal_coverage_delta"] for row in rows if row["conformal_coverage_delta"] is not None])
                )
                if any(row["conformal_coverage_delta"] is not None for row in rows)
                else None,
            }
        )
    return regime_rows


def summarize_best_regime(pairwise_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_strategy: dict[str, dict[str, Any]] = {}
    for posterior_strategy in sorted({row["posterior_strategy_requested"] for row in pairwise_rows}):
        rows = [row for row in pairwise_rows if row["posterior_strategy_requested"] == posterior_strategy]
        rows = [row for row in rows if row["uncertainty_sequence_exact_delta"] is not None]
        if not rows:
            continue
        best = max(rows, key=lambda row: float(row["uncertainty_sequence_exact_delta"]))
        best_by_strategy[posterior_strategy] = {
            "ambiguity_level": best["ambiguity_level"],
            "ambiguity_regime": best["ambiguity_regime"],
            "uncertainty_sequence_exact_delta": best["uncertainty_sequence_exact_delta"],
            "uncertainty_sequence_topk_delta": best["uncertainty_sequence_topk_delta"],
        }
    return best_by_strategy
