from __future__ import annotations

import math
from typing import Any

from decipherlab.evaluation.metrics import canonical_family_name


def _family_topk_correct(example_payload: dict[str, Any], top_k: int) -> float | None:
    true_family = example_payload.get("true_family")
    if true_family is None:
        return None
    canonical = canonical_family_name(true_family)
    ranked = [item["family"] for item in example_payload["ranking"]["evidences"][:top_k]]
    return float(canonical in ranked)


def _structural_error(example_payload: dict[str, Any]) -> float | None:
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


def _example_lookup(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {payload["example_id"]: payload for payload in result["example_payloads"]}


def _condition_label(strategy: str, mode: str) -> str:
    strategy_label = "heuristic" if strategy == "cluster_distance" else "calibrated"
    return f"{mode}_{strategy_label}"


def analyze_failure_cases(
    results_cube: dict[float, dict[str, dict[str, dict[str, Any]]]],
    top_k: int,
    overdiffuse_entropy_ratio: float,
) -> dict[str, Any]:
    failure_cases: list[dict[str, Any]] = []

    for ambiguity_level, per_strategy in sorted(results_cube.items()):
        for strategy, per_mode in per_strategy.items():
            fixed = _example_lookup(per_mode["fixed"])
            uncertainty = _example_lookup(per_mode["uncertainty"])
            for example_id in sorted(set(fixed) & set(uncertainty)):
                fixed_payload = fixed[example_id]
                uncertainty_payload = uncertainty[example_id]
                fixed_symbol_topk = fixed_payload["symbol_metrics"]["topk_accuracy"]
                uncertainty_symbol_topk = uncertainty_payload["symbol_metrics"]["topk_accuracy"]
                if (
                    fixed_symbol_topk is not None
                    and uncertainty_symbol_topk is not None
                    and uncertainty_symbol_topk > fixed_symbol_topk
                ):
                    fixed_family = _family_topk_correct(fixed_payload, top_k)
                    uncertainty_family = _family_topk_correct(uncertainty_payload, top_k)
                    fixed_struct = _structural_error(fixed_payload)
                    uncertainty_struct = _structural_error(uncertainty_payload)
                    downstream_improved = False
                    if fixed_family is not None and uncertainty_family is not None:
                        downstream_improved = uncertainty_family > fixed_family
                    elif fixed_struct is not None and uncertainty_struct is not None:
                        downstream_improved = uncertainty_struct < fixed_struct
                    if not downstream_improved:
                        failure_cases.append(
                            {
                                "case_type": "uncertainty_helped_symbols_not_downstream",
                                "ambiguity_level": ambiguity_level,
                                "condition_group": strategy,
                                "example_id": example_id,
                                "symbol_topk_delta": uncertainty_symbol_topk - fixed_symbol_topk,
                                "fixed_family_topk": fixed_family,
                                "uncertainty_family_topk": uncertainty_family,
                                "fixed_structural_error": fixed_struct,
                                "uncertainty_structural_error": uncertainty_struct,
                            }
                        )

        strategies = sorted(per_strategy)
        if {"cluster_distance", "calibrated_classifier"}.issubset(strategies):
            for mode in ("fixed", "uncertainty"):
                heuristic = _example_lookup(per_strategy["cluster_distance"][mode])
                calibrated = _example_lookup(per_strategy["calibrated_classifier"][mode])
                for example_id in sorted(set(heuristic) & set(calibrated)):
                    heuristic_payload = heuristic[example_id]
                    calibrated_payload = calibrated[example_id]
                    heuristic_ece = heuristic_payload["symbol_metrics"]["expected_calibration_error"]
                    calibrated_ece = calibrated_payload["symbol_metrics"]["expected_calibration_error"]
                    heuristic_nll = heuristic_payload["symbol_metrics"]["negative_log_likelihood"]
                    calibrated_nll = calibrated_payload["symbol_metrics"]["negative_log_likelihood"]
                    if (
                        heuristic_ece is not None
                        and calibrated_ece is not None
                        and calibrated_ece > heuristic_ece + 0.02
                    ) or (
                        heuristic_nll is not None
                        and calibrated_nll is not None
                        and calibrated_nll > heuristic_nll + 0.2
                    ):
                        failure_cases.append(
                            {
                                "case_type": "calibration_worsened_or_unstable",
                                "ambiguity_level": ambiguity_level,
                                "condition_group": mode,
                                "example_id": example_id,
                                "heuristic_symbol_ece": heuristic_ece,
                                "calibrated_symbol_ece": calibrated_ece,
                                "heuristic_symbol_nll": heuristic_nll,
                                "calibrated_symbol_nll": calibrated_nll,
                                "heuristic_entropy": heuristic_payload["triage"]["mean_posterior_entropy"],
                                "calibrated_entropy": calibrated_payload["triage"]["mean_posterior_entropy"],
                            }
                        )

        for strategy, per_mode in per_strategy.items():
            for mode, result in per_mode.items():
                condition = _condition_label(strategy, mode)
                for payload in result["example_payloads"]:
                    topk = len(payload["posterior"]["candidate_ids"][0]) if payload["posterior"]["candidate_ids"] else 1
                    max_entropy = math.log(max(topk, 1))
                    entropy_threshold = overdiffuse_entropy_ratio * max_entropy
                    mean_entropy = payload["triage"]["mean_posterior_entropy"]
                    top1_accuracy = payload["symbol_metrics"]["top1_accuracy"]
                    topk_accuracy = payload["symbol_metrics"]["topk_accuracy"]
                    breakdown = payload["symbol_metrics"]["case_breakdown"]
                    if mean_entropy > entropy_threshold and top1_accuracy is not None and (
                        top1_accuracy < 0.5 or (topk_accuracy is not None and topk_accuracy > top1_accuracy)
                    ):
                        failure_cases.append(
                            {
                                "case_type": "overdiffuse_posterior",
                                "ambiguity_level": ambiguity_level,
                                "condition_group": condition,
                                "example_id": payload["example_id"],
                                "mean_entropy": mean_entropy,
                                "entropy_threshold": entropy_threshold,
                                "symbol_top1_accuracy": top1_accuracy,
                                "symbol_topk_accuracy": topk_accuracy,
                            }
                        )
                    if breakdown["collapse_rescued_by_topk"] > 0:
                        failure_cases.append(
                            {
                                "case_type": "top1_collapse_but_topk_rescue",
                                "ambiguity_level": ambiguity_level,
                                "condition_group": condition,
                                "example_id": payload["example_id"],
                                "collapse_rescued_by_topk": breakdown["collapse_rescued_by_topk"],
                                "missing_from_topk": breakdown["missing_from_topk"],
                                "labeled_count": breakdown["labeled_count"],
                            }
                        )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, str], int] = {}
    for case in failure_cases:
        key = (case["case_type"], case["ambiguity_level"], case["condition_group"])
        grouped[key] = grouped.get(key, 0) + 1
    for (case_type, ambiguity_level, condition_group), count in sorted(grouped.items()):
        summary_rows.append(
            {
                "case_type": case_type,
                "ambiguity_level": ambiguity_level,
                "condition_group": condition_group,
                "count": count,
            }
        )

    return {
        "failure_cases": failure_cases,
        "summary_rows": summary_rows,
    }
