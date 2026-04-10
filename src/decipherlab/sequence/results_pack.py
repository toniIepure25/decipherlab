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
        trigram = methods.get("uncertainty_trigram_beam")
        conformal_trigram = methods.get("conformal_trigram_beam")
        crf = methods.get("uncertainty_crf_viterbi")
        conformal_crf = methods.get("conformal_crf_viterbi")
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
                "uncertainty_family_delta": _delta(
                    uncertainty,
                    fixed,
                    "family_identification_accuracy",
                ),
                "uncertainty_downstream_exact_delta": _delta(uncertainty, fixed, "real_downstream_exact_match"),
                "uncertainty_downstream_topk_delta": _delta(uncertainty, fixed, "real_downstream_topk_recovery"),
                "uncertainty_downstream_token_accuracy_delta": _delta(uncertainty, fixed, "real_downstream_token_accuracy"),
                "uncertainty_downstream_cer_improvement": _delta(fixed, uncertainty, "real_downstream_cer"),
                "conformal_sequence_exact_delta": _delta(conformal, uncertainty, "sequence_exact_match"),
                "conformal_sequence_topk_delta": _delta(conformal, uncertainty, "sequence_topk_recovery"),
                "conformal_token_accuracy_delta": _delta(conformal, uncertainty, "sequence_token_accuracy"),
                "conformal_cer_improvement": _delta(uncertainty, conformal, "sequence_cer"),
                "conformal_downstream_exact_delta": _delta(conformal, uncertainty, "real_downstream_exact_match"),
                "conformal_downstream_topk_delta": _delta(conformal, uncertainty, "real_downstream_topk_recovery"),
                "conformal_downstream_token_accuracy_delta": _delta(conformal, uncertainty, "real_downstream_token_accuracy"),
                "conformal_downstream_cer_improvement": _delta(uncertainty, conformal, "real_downstream_cer"),
                "conformal_coverage_delta": _delta(conformal, uncertainty, "prediction_set_coverage"),
                "conformal_set_size_delta": _delta(conformal, uncertainty, "prediction_set_avg_size"),
                "conformal_family_delta": _delta(
                    conformal,
                    uncertainty,
                    "family_identification_accuracy",
                ),
                "trigram_sequence_exact_delta": _delta(trigram, uncertainty, "sequence_exact_match"),
                "trigram_sequence_topk_delta": _delta(trigram, uncertainty, "sequence_topk_recovery"),
                "trigram_token_accuracy_delta": _delta(trigram, uncertainty, "sequence_token_accuracy"),
                "trigram_cer_improvement": _delta(uncertainty, trigram, "sequence_cer"),
                "trigram_family_delta": _delta(
                    trigram,
                    uncertainty,
                    "family_identification_accuracy",
                ),
                "crf_sequence_exact_delta": _delta(crf, uncertainty, "sequence_exact_match"),
                "crf_sequence_topk_delta": _delta(crf, uncertainty, "sequence_topk_recovery"),
                "crf_token_accuracy_delta": _delta(crf, uncertainty, "sequence_token_accuracy"),
                "crf_cer_improvement": _delta(uncertainty, crf, "sequence_cer"),
                "crf_family_delta": _delta(
                    crf,
                    uncertainty,
                    "family_identification_accuracy",
                ),
                "conformal_trigram_sequence_exact_delta": _delta(
                    conformal_trigram,
                    trigram,
                    "sequence_exact_match",
                ),
                "conformal_trigram_sequence_topk_delta": _delta(
                    conformal_trigram,
                    trigram,
                    "sequence_topk_recovery",
                ),
                "conformal_trigram_coverage_delta": _delta(
                    conformal_trigram,
                    trigram,
                    "prediction_set_coverage",
                ),
                "conformal_trigram_set_size_delta": _delta(
                    conformal_trigram,
                    trigram,
                    "prediction_set_avg_size",
                ),
                "conformal_trigram_family_delta": _delta(
                    conformal_trigram,
                    trigram,
                    "family_identification_accuracy",
                ),
                "conformal_crf_sequence_exact_delta": _delta(
                    conformal_crf,
                    crf,
                    "sequence_exact_match",
                ),
                "conformal_crf_sequence_topk_delta": _delta(
                    conformal_crf,
                    crf,
                    "sequence_topk_recovery",
                ),
                "conformal_crf_coverage_delta": _delta(
                    conformal_crf,
                    crf,
                    "prediction_set_coverage",
                ),
                "conformal_crf_set_size_delta": _delta(
                    conformal_crf,
                    crf,
                    "prediction_set_avg_size",
                ),
                "conformal_crf_family_delta": _delta(
                    conformal_crf,
                    crf,
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
                "mean_uncertainty_downstream_exact_delta": float(
                    np.mean([row["uncertainty_downstream_exact_delta"] for row in rows if row["uncertainty_downstream_exact_delta"] is not None])
                )
                if any(row["uncertainty_downstream_exact_delta"] is not None for row in rows)
                else None,
                "mean_uncertainty_downstream_topk_delta": float(
                    np.mean([row["uncertainty_downstream_topk_delta"] for row in rows if row["uncertainty_downstream_topk_delta"] is not None])
                )
                if any(row["uncertainty_downstream_topk_delta"] is not None for row in rows)
                else None,
                "mean_conformal_downstream_exact_delta": float(
                    np.mean([row["conformal_downstream_exact_delta"] for row in rows if row["conformal_downstream_exact_delta"] is not None])
                )
                if any(row["conformal_downstream_exact_delta"] is not None for row in rows)
                else None,
                "mean_conformal_coverage_delta": float(
                    np.mean([row["conformal_coverage_delta"] for row in rows if row["conformal_coverage_delta"] is not None])
                )
                if any(row["conformal_coverage_delta"] is not None for row in rows)
                else None,
                "mean_conformal_downstream_exact_delta": float(
                    np.mean([row["conformal_downstream_exact_delta"] for row in rows if row["conformal_downstream_exact_delta"] is not None])
                )
                if any(row["conformal_downstream_exact_delta"] is not None for row in rows)
                else None,
                "mean_uncertainty_family_delta": float(
                    np.mean([row["uncertainty_family_delta"] for row in rows if row["uncertainty_family_delta"] is not None])
                )
                if any(row["uncertainty_family_delta"] is not None for row in rows)
                else None,
                "mean_conformal_family_delta": float(
                    np.mean([row["conformal_family_delta"] for row in rows if row["conformal_family_delta"] is not None])
                )
                if any(row["conformal_family_delta"] is not None for row in rows)
                else None,
                "mean_trigram_sequence_exact_delta": float(
                    np.mean([row["trigram_sequence_exact_delta"] for row in rows if row["trigram_sequence_exact_delta"] is not None])
                )
                if any(row["trigram_sequence_exact_delta"] is not None for row in rows)
                else None,
                "mean_trigram_family_delta": float(
                    np.mean([row["trigram_family_delta"] for row in rows if row["trigram_family_delta"] is not None])
                )
                if any(row["trigram_family_delta"] is not None for row in rows)
                else None,
                "mean_crf_sequence_exact_delta": float(
                    np.mean([row["crf_sequence_exact_delta"] for row in rows if row["crf_sequence_exact_delta"] is not None])
                )
                if any(row["crf_sequence_exact_delta"] is not None for row in rows)
                else None,
                "mean_crf_family_delta": float(
                    np.mean([row["crf_family_delta"] for row in rows if row["crf_family_delta"] is not None])
                )
                if any(row["crf_family_delta"] is not None for row in rows)
                else None,
                "mean_conformal_trigram_family_delta": float(
                    np.mean([row["conformal_trigram_family_delta"] for row in rows if row["conformal_trigram_family_delta"] is not None])
                )
                if any(row["conformal_trigram_family_delta"] is not None for row in rows)
                else None,
                "mean_conformal_crf_family_delta": float(
                    np.mean([row["conformal_crf_family_delta"] for row in rows if row["conformal_crf_family_delta"] is not None])
                )
                if any(row["conformal_crf_family_delta"] is not None for row in rows)
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
