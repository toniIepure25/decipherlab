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
        trigram = methods.get("uncertainty_trigram_beam")
        conformal_trigram = methods.get("conformal_trigram_beam")
        crf = methods.get("uncertainty_crf_viterbi")
        conformal_crf = methods.get("conformal_crf_viterbi")
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
            uncertainty.get("real_downstream_exact_match") is not None
            and fixed.get("real_downstream_exact_match") is not None
            and uncertainty["symbol_topk_accuracy"] > fixed["symbol_topk_accuracy"]
            and uncertainty["real_downstream_exact_match"] <= fixed["real_downstream_exact_match"]
        ):
            add("symbol_rescue_without_downstream_recovery", "uncertainty_beam", "fixed_greedy")
        if (
            uncertainty.get("real_downstream_topk_recovery") is not None
            and uncertainty.get("sequence_topk_recovery") is not None
            and fixed.get("real_downstream_topk_recovery") is not None
            and uncertainty.get("real_downstream_exact_match") is not None
            and fixed.get("real_downstream_exact_match") is not None
            and uncertainty["sequence_topk_recovery"] > fixed["sequence_topk_recovery"]
            and uncertainty["real_downstream_exact_match"] <= fixed["real_downstream_exact_match"]
        ):
            add("grouped_topk_rescue_without_downstream_exact", "uncertainty_beam", "fixed_greedy")
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
        if (
            uncertainty.get("family_identification_accuracy") is not None
            and fixed.get("family_identification_accuracy") is not None
            and uncertainty["family_identification_accuracy"] > fixed["family_identification_accuracy"]
            and uncertainty["sequence_exact_match"] <= fixed["sequence_exact_match"]
        ):
            add("family_rescue_without_sequence_rescue", "uncertainty_beam", "fixed_greedy")
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
        if (
            conformal.get("real_downstream_exact_match") is not None
            and uncertainty.get("real_downstream_exact_match") is not None
            and conformal["real_downstream_exact_match"] > uncertainty["real_downstream_exact_match"]
        ):
            add("conformal_downstream_rescue", "conformal_beam", "uncertainty_beam")

        if trigram is not None:
            if trigram["sequence_exact_match"] > uncertainty["sequence_exact_match"]:
                add("trigram_sequence_rescue", "uncertainty_trigram_beam", "uncertainty_beam")
            if (
                trigram.get("family_identification_accuracy") is not None
                and uncertainty.get("family_identification_accuracy") is not None
                and trigram["family_identification_accuracy"] > uncertainty["family_identification_accuracy"]
                and trigram["sequence_exact_match"] <= uncertainty["sequence_exact_match"]
            ):
                add("family_rescue_only_trigram", "uncertainty_trigram_beam", "uncertainty_beam")
            if (
                trigram["sequence_token_accuracy"] < uncertainty["sequence_token_accuracy"]
                and (
                    trigram.get("family_identification_accuracy") is None
                    or uncertainty.get("family_identification_accuracy") is None
                    or trigram["family_identification_accuracy"] <= uncertainty["family_identification_accuracy"]
                )
            ):
                add("trigram_amplifies_bad_prior", "uncertainty_trigram_beam", "uncertainty_beam")

        if trigram is not None and conformal_trigram is not None:
            if (
                conformal_trigram["prediction_set_coverage"] < trigram["prediction_set_coverage"]
                and conformal_trigram["family_identification_accuracy"] is not None
                and trigram["family_identification_accuracy"] is not None
                and conformal_trigram["family_identification_accuracy"] > trigram["family_identification_accuracy"]
            ):
                add("conformal_trigram_family_rescue", "conformal_trigram_beam", "uncertainty_trigram_beam")
            if (
                trigram["sequence_topk_recovery"] > fixed["sequence_topk_recovery"]
                and conformal_trigram["sequence_topk_recovery"] <= fixed["sequence_topk_recovery"]
            ):
                add("sequence_rescue_only_trigram_not_conformal", "uncertainty_trigram_beam", "conformal_trigram_beam")

        if crf is not None:
            if crf["sequence_exact_match"] > uncertainty["sequence_exact_match"]:
                add("crf_sequence_rescue", "uncertainty_crf_viterbi", "uncertainty_beam")
            if (
                crf.get("family_identification_accuracy") is not None
                and uncertainty.get("family_identification_accuracy") is not None
                and crf["family_identification_accuracy"] > uncertainty["family_identification_accuracy"]
                and crf["sequence_exact_match"] <= uncertainty["sequence_exact_match"]
            ):
                add("family_rescue_only_crf", "uncertainty_crf_viterbi", "uncertainty_beam")
            if (
                crf["sequence_token_accuracy"] < uncertainty["sequence_token_accuracy"]
                and (
                    crf.get("family_identification_accuracy") is None
                    or uncertainty.get("family_identification_accuracy") is None
                    or crf["family_identification_accuracy"] <= uncertainty["family_identification_accuracy"]
                )
            ):
                add("crf_amplifies_bad_prior", "uncertainty_crf_viterbi", "uncertainty_beam")

        if crf is not None and conformal_crf is not None:
            if (
                conformal_crf["prediction_set_coverage"] < crf["prediction_set_coverage"]
                and conformal_crf["family_identification_accuracy"] is not None
                and crf["family_identification_accuracy"] is not None
                and conformal_crf["family_identification_accuracy"] > crf["family_identification_accuracy"]
            ):
                add("conformal_crf_family_rescue", "conformal_crf_viterbi", "uncertainty_crf_viterbi")
            if (
                crf["sequence_topk_recovery"] > fixed["sequence_topk_recovery"]
                and conformal_crf["sequence_topk_recovery"] <= fixed["sequence_topk_recovery"]
            ):
                add("sequence_rescue_only_crf_not_conformal", "uncertainty_crf_viterbi", "conformal_crf_viterbi")

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
