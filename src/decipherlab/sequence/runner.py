from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from decipherlab.config import DecipherLabConfig, dump_config, load_config
from decipherlab.decoding.beam_search import (
    BeamDecodingResult,
    BigramTransitionModel,
    TrigramTransitionModel,
    beam_decode_confusion_network,
    greedy_decode_confusion_network,
    trigram_beam_decode_confusion_network,
)
from decipherlab.decoding.crf import crf_viterbi_decode_confusion_network
from decipherlab.evaluation.metrics import (
    symbol_case_breakdown,
    symbol_expected_calibration_error,
    symbol_negative_log_likelihood,
    symbol_top_k_accuracy,
)
from decipherlab.glyphs.features import extract_feature_matrix
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.models import DatasetCollection, SequenceExample, TranscriptionPosterior
from decipherlab.risk_control.conformal import SplitConformalSetPredictor, summarize_prediction_sets
from decipherlab.sequence.benchmark import build_real_glyph_sequence_benchmark
from decipherlab.sequence.adaptive_decoder import (
    AdaptiveDecodingDecision,
    build_support_snapshot,
    decide_support_aware_method,
    resolve_operating_profile,
    support_feature_row,
)
from decipherlab.sequence.family_identification import (
    ProcessFamilyClassifier,
    family_identification_payload,
)
from decipherlab.sequence.failure_analysis import build_sequence_failure_cases
from decipherlab.sequence.learned_gate import fit_binary_logistic_gate
from decipherlab.sequence.metrics import sequence_metric_bundle
from decipherlab.sequence.profile_selector import (
    build_profile_selector_feature_row,
    select_profile,
    selector_feature_names,
)
from decipherlab.sequence.real_downstream import build_real_downstream_resource, downstream_payload
from decipherlab.sequence.results_pack import (
    build_ambiguity_regime_rows,
    build_pairwise_effect_rows,
    summarize_best_regime,
)
from decipherlab.structured_uncertainty.confusion_network import (
    confusion_network_to_posterior,
    posterior_to_confusion_network,
)
from decipherlab.transcription.model import fit_posterior_model
from decipherlab.transcription.posterior import split_posterior_by_lengths
from decipherlab.utils.io import write_csv, write_json, write_text
from decipherlab.utils.logging import configure_logging
from decipherlab.utils.runtime import prepare_run_context
from decipherlab.vision.corruption import apply_ambiguity_to_examples


def _load_manifest_dataset(config: DecipherLabConfig) -> DatasetCollection:
    if config.dataset.source != "manifest" or config.dataset.manifest_path is None:
        raise ValueError("Sequence branch v1 expects a manifest-backed glyph crop dataset.")
    if config.dataset.manifest_format != "glyph_crop":
        raise ValueError("Sequence branch v1 requires dataset.manifest_format='glyph_crop'.")
    return load_glyph_crop_manifest_dataset(config.dataset.manifest_path, dataset_config=config.dataset)


def _flatten_glyphs(examples: list[SequenceExample]) -> list:
    return [glyph for example in examples for glyph in example.glyphs]


def _weighted_average(values: list[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def _sequence_labels(examples: list[SequenceExample]) -> list[list[str | None]]:
    return [example.observed_symbols for example in examples]


def _decode_method_result(
    method: str,
    posterior: TranscriptionPosterior,
    network,
    bigram_transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    config: DecipherLabConfig,
    beam_width_override: int | None = None,
) -> tuple[BeamDecodingResult, TranscriptionPosterior, Any]:
    beam_width = config.decoding.beam_width if beam_width_override is None else beam_width_override
    if method == "fixed_greedy":
        fixed = posterior.collapsed()
        fixed_network = posterior_to_confusion_network(
            fixed,
            config.structured_uncertainty.model_copy(update={"max_candidates_per_position": 1}),
        )
        return greedy_decode_confusion_network(fixed_network), fixed, fixed_network
    if method == "uncertainty_beam":
        return (
            beam_decode_confusion_network(
                network=network,
                transition_model=bigram_transition_model,
                beam_width=beam_width,
                lm_weight=config.decoding.lm_weight,
                top_k_sequences=config.decoding.top_k_sequences,
                length_normalize=config.decoding.length_normalize,
            ),
            posterior,
            network,
        )
    if method == "uncertainty_trigram_beam":
        if trigram_transition_model is None:
            raise ValueError("Trigram decoding requested but no trigram transition model is available.")
        return (
            trigram_beam_decode_confusion_network(
                network=network,
                transition_model=trigram_transition_model,
                beam_width=beam_width,
                lm_weight=config.decoding.lm_weight
                if config.decoding.trigram_lm_weight is None
                else config.decoding.trigram_lm_weight,
                top_k_sequences=config.decoding.top_k_sequences,
                length_normalize=config.decoding.length_normalize,
            ),
            posterior,
            network,
        )
    if method == "uncertainty_crf_viterbi":
        return (
            crf_viterbi_decode_confusion_network(
                network=network,
                transition_model=bigram_transition_model,
                lm_weight=config.decoding.lm_weight,
            ),
            posterior,
            network,
        )
    raise ValueError(f"Unsupported decoding method: {method}")


def _decode_conformal_result(
    method: str,
    posterior: TranscriptionPosterior,
    network,
    conformal_predictor: SplitConformalSetPredictor,
    bigram_transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    config: DecipherLabConfig,
    beam_width_override: int | None = None,
) -> tuple[BeamDecodingResult, TranscriptionPosterior, Any]:
    beam_width = config.decoding.beam_width if beam_width_override is None else beam_width_override
    active_network = conformal_predictor.apply(network, config.risk_control)
    active_posterior = confusion_network_to_posterior(active_network)
    if method == "conformal_beam":
        decoded = beam_decode_confusion_network(
            network=active_network,
            transition_model=bigram_transition_model,
            beam_width=beam_width,
            lm_weight=config.decoding.lm_weight,
            top_k_sequences=config.decoding.top_k_sequences,
            length_normalize=config.decoding.length_normalize,
        )
    elif method == "conformal_crf_viterbi":
        decoded = crf_viterbi_decode_confusion_network(
            network=active_network,
            transition_model=bigram_transition_model,
            lm_weight=config.decoding.lm_weight,
        )
    elif method == "conformal_trigram_beam":
        if trigram_transition_model is None:
            raise ValueError("Conformal trigram decoding requested without a trigram model.")
        decoded = trigram_beam_decode_confusion_network(
            network=active_network,
            transition_model=trigram_transition_model,
            beam_width=beam_width,
            lm_weight=config.decoding.lm_weight
            if config.decoding.trigram_lm_weight is None
            else config.decoding.trigram_lm_weight,
            top_k_sequences=config.decoding.top_k_sequences,
            length_normalize=config.decoding.length_normalize,
        )
    else:
        raise ValueError(f"Unsupported conformal decoding method: {method}")
    return decoded, active_posterior, active_network


def _adaptive_utility(
    sequence_metrics: dict[str, float],
    downstream_metrics: dict[str, Any],
    prediction_set_avg_size: float | None,
    *,
    prediction_set_penalty: float,
    shortlist_utility_weight: float,
    review_budget_k: int,
) -> float:
    budgeted_shortlist_recall = float(
        sequence_metrics.get(f"sequence_shortlist_recall_at_{review_budget_k}", 0.0)
    )
    score = (
        1.5 * float(sequence_metrics["sequence_exact_match"])
        + 1.0 * float(sequence_metrics["sequence_topk_recovery"])
        + shortlist_utility_weight * float(sequence_metrics.get("sequence_shortlist_utility", 0.0))
        + 0.5 * budgeted_shortlist_recall
        + 0.25 * float(sequence_metrics["sequence_token_accuracy"])
        - 0.15 * float(sequence_metrics["sequence_cer"])
    )
    if downstream_metrics.get("real_downstream_exact_match") is not None:
        score += (
            4.0 * float(downstream_metrics["real_downstream_exact_match"])
            + 1.5 * float(downstream_metrics["real_downstream_topk_recovery"] or 0.0)
            + 0.5 * float(downstream_metrics["real_downstream_token_accuracy"] or 0.0)
            - 0.25 * float(downstream_metrics["real_downstream_cer"] or 0.0)
        )
    if prediction_set_avg_size is not None:
        score -= prediction_set_penalty * float(prediction_set_avg_size)
    return float(score)


def _downstream_exact_value(downstream_metrics: dict[str, Any]) -> float | None:
    value = downstream_metrics.get("real_downstream_exact_match")
    return None if value is None else float(value)


def _adaptive_policy_branch(config: DecipherLabConfig) -> str | None:
    policy = config.adaptive_decoding.policy
    if policy == "support_aware_learned_gate":
        return "shortlist_first"
    if policy == "support_aware_constrained_gate":
        return "rescue_first"
    if policy == "support_aware_profiled_gate":
        return config.adaptive_decoding.operating_profile
    if policy == "support_aware_profile_selector":
        return "profile_selector"
    return None


def _adaptive_method_name(config: DecipherLabConfig) -> str:
    policy = config.adaptive_decoding.policy
    if policy == "support_aware_constrained_gate":
        return "adaptive_constrained_beam"
    if policy == "support_aware_learned_gate":
        return "adaptive_learned_beam"
    if policy == "support_aware_profiled_gate":
        return "adaptive_profiled_beam"
    if policy == "support_aware_profile_selector":
        return "adaptive_profile_selector_beam"
    return "adaptive_support_beam"


def _delegated_policy_for_profile(profile_mode: str) -> str:
    if profile_mode == "rescue_first":
        return "support_aware_constrained_gate"
    if profile_mode == "shortlist_first":
        return "support_aware_learned_gate"
    raise ValueError(f"Unsupported profile mode: {profile_mode}")


def _decode_profiled_adaptive_result(
    *,
    snapshot,
    posterior_strategy: str,
    posterior: TranscriptionPosterior,
    network,
    conformal_predictor: SplitConformalSetPredictor | None,
    transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    config: DecipherLabConfig,
    conformal_gate: Any,
    wide_gate: Any,
    profile_mode: str,
    profile_reason: str,
) -> tuple[AdaptiveDecodingDecision, BeamDecodingResult, TranscriptionPosterior, Any, float | None, float | None]:
    feature_row = support_feature_row(snapshot)
    review_budget = config.adaptive_decoding.review_budget_k
    budget_tight = review_budget <= config.adaptive_decoding.tight_review_budget_threshold
    conformal_probability = None if conformal_gate is None else float(conformal_gate.predict_proba(feature_row))
    wide_probability = None if wide_gate is None else float(wide_gate.predict_proba(feature_row))
    policy_is_constrained = profile_mode == "rescue_first"
    conformal_threshold = (
        config.adaptive_decoding.constrained_conformal_threshold
        if policy_is_constrained
        else config.adaptive_decoding.learned_gate_decision_threshold
    )
    wide_threshold = (
        config.adaptive_decoding.constrained_wide_threshold
        if policy_is_constrained
        else config.adaptive_decoding.learned_gate_decision_threshold
    )
    if budget_tight:
        if policy_is_constrained:
            conformal_threshold = min(
                0.95,
                conformal_threshold + config.adaptive_decoding.budget_threshold_delta,
            )
            wide_threshold = max(
                0.05,
                wide_threshold - (0.5 * config.adaptive_decoding.budget_threshold_delta),
            )
        else:
            conformal_threshold = max(
                0.05,
                conformal_threshold - config.adaptive_decoding.budget_threshold_delta,
            )
            wide_threshold = min(
                0.95,
                wide_threshold + config.adaptive_decoding.budget_threshold_delta,
            )
    choose_conformal = (
        conformal_predictor is not None
        and conformal_probability is not None
        and conformal_probability >= conformal_threshold
    )
    low_entropy = bool(snapshot.mean_confusion_entropy <= config.adaptive_decoding.conformal_entropy_threshold)
    high_entropy = bool(snapshot.mean_confusion_entropy >= config.adaptive_decoding.raw_entropy_threshold)
    compact_set = bool(snapshot.mean_confusion_set_size <= config.adaptive_decoding.conformal_set_size_threshold)
    diffuse_set = bool(snapshot.mean_confusion_set_size >= config.adaptive_decoding.raw_set_size_threshold)
    limited_support = bool(feature_row["limited_support"])
    fragile_signal_count = (
        int(snapshot.mean_confusion_entropy >= config.adaptive_decoding.defer_entropy_threshold)
        + int(snapshot.mean_confusion_set_size >= config.adaptive_decoding.defer_set_size_threshold)
        + int(
            0 < snapshot.length_support_count <= config.adaptive_decoding.defer_length_support_threshold
        )
    )
    if policy_is_constrained:
        conformal_regime_ok = (
            not high_entropy
            and not diffuse_set
            and (
                compact_set
                or limited_support
                or posterior_strategy == "calibrated_classifier"
            )
        )
        choose_conformal = choose_conformal and conformal_regime_ok
    defer_to_human = (
        config.adaptive_decoding.enable_defer_to_human
        and budget_tight
        and fragile_signal_count >= config.adaptive_decoding.defer_min_fragile_signals
    )
    if defer_to_human:
        beam_width = (
            max(config.decoding.beam_width, config.adaptive_decoding.wide_beam_width)
            if policy_is_constrained or high_entropy or diffuse_set
            else config.decoding.beam_width
        )
        adaptive_decision = AdaptiveDecodingDecision(
            selected_method="uncertainty_beam",
            beam_width=beam_width,
            decision_reason=(
                "profile_rescue_defer_budget" if policy_is_constrained else "profile_shortlist_defer_budget"
            ),
            control_action="defer",
            limited_support=limited_support,
            low_entropy=low_entropy,
            high_entropy=high_entropy,
            compact_set=compact_set,
            diffuse_set=diffuse_set,
            defer_to_human=True,
            review_budget=review_budget,
            budget_tight=budget_tight,
            fragile_signal_count=fragile_signal_count,
            operating_profile=profile_mode,
            profile_reason=profile_reason,
        )
        decoded, active_posterior, active_network = _decode_method_result(
            method="uncertainty_beam",
            posterior=posterior,
            network=network,
            bigram_transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
            beam_width_override=beam_width,
        )
    elif choose_conformal:
        adaptive_decision = AdaptiveDecodingDecision(
            selected_method="conformal_beam",
            beam_width=min(config.decoding.beam_width, config.adaptive_decoding.narrow_beam_width),
            decision_reason=(
                "profile_rescue_conformal" if policy_is_constrained else "profile_shortlist_conformal"
            ),
            control_action="prune",
            limited_support=limited_support,
            low_entropy=low_entropy,
            high_entropy=high_entropy,
            compact_set=compact_set,
            diffuse_set=diffuse_set,
            defer_to_human=False,
            review_budget=review_budget,
            budget_tight=budget_tight,
            fragile_signal_count=fragile_signal_count,
            operating_profile=profile_mode,
            profile_reason=profile_reason,
        )
        decoded, active_posterior, active_network = _decode_conformal_result(
            method="conformal_beam",
            posterior=posterior,
            network=network,
            conformal_predictor=conformal_predictor,
            bigram_transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
            beam_width_override=adaptive_decision.beam_width,
        )
    else:
        use_wide = wide_probability is not None and wide_probability >= wide_threshold
        if policy_is_constrained and (high_entropy or diffuse_set):
            use_wide = True
        beam_width = (
            max(config.decoding.beam_width, config.adaptive_decoding.wide_beam_width)
            if use_wide
            else config.decoding.beam_width
        )
        adaptive_decision = AdaptiveDecodingDecision(
            selected_method="uncertainty_beam",
            beam_width=beam_width,
            decision_reason=(
                "profile_rescue_raw_wide"
                if use_wide and policy_is_constrained
                else "profile_rescue_raw_default"
                if policy_is_constrained
                else "profile_shortlist_raw_wide"
                if use_wide
                else "profile_shortlist_raw_default"
            ),
            control_action="preserve",
            limited_support=limited_support,
            low_entropy=low_entropy,
            high_entropy=high_entropy,
            compact_set=compact_set,
            diffuse_set=diffuse_set,
            defer_to_human=False,
            review_budget=review_budget,
            budget_tight=budget_tight,
            fragile_signal_count=fragile_signal_count,
            operating_profile=profile_mode,
            profile_reason=profile_reason,
        )
        decoded, active_posterior, active_network = _decode_method_result(
            method="uncertainty_beam",
            posterior=posterior,
            network=network,
            bigram_transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
            beam_width_override=beam_width,
        )
    return (
        adaptive_decision,
        decoded,
        active_posterior,
        active_network,
        conformal_probability,
        wide_probability,
    )


def _train_learned_adaptive_gates(
    *,
    validation_examples: list[SequenceExample],
    validation_posteriors: list[TranscriptionPosterior],
    validation_networks: list[Any],
    posterior_strategy: str,
    conformal_predictor: SplitConformalSetPredictor | None,
    transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    downstream_resource: Any,
    config: DecipherLabConfig,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    gate_rows: list[dict[str, Any]] = []
    for example, posterior, network in zip(validation_examples, validation_posteriors, validation_networks):
        snapshot = build_support_snapshot(
            network=network,
            posterior_strategy=posterior_strategy,
            sequence_length=example.sequence_length,
            downstream_resource=downstream_resource,
            conformal_available=conformal_predictor is not None,
        )
        features = support_feature_row(snapshot)

        raw_default_decoded, raw_default_posterior, raw_default_network = _decode_method_result(
            method="uncertainty_beam",
            posterior=posterior,
            network=network,
            bigram_transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
        )
        raw_wide_decoded, raw_wide_posterior, raw_wide_network = _decode_method_result(
            method="uncertainty_beam",
            posterior=posterior,
            network=network,
            bigram_transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
            beam_width_override=max(config.decoding.beam_width, config.adaptive_decoding.wide_beam_width),
        )
        raw_default_set = summarize_prediction_sets([raw_default_network], [example.observed_symbols])
        raw_wide_set = summarize_prediction_sets([raw_wide_network], [example.observed_symbols])
        raw_default_downstream = (
            downstream_payload(
                method="uncertainty_beam",
                decoded=raw_default_decoded,
                posterior=raw_default_posterior,
                truth=example.observed_symbols,
                downstream_resource=downstream_resource,
                real_downstream_config=config.real_downstream,
                decoding_config=config.decoding,
                bigram_transition_model=transition_model,
                trigram_transition_model=trigram_transition_model,
            )
            if downstream_resource is not None
            else {"real_downstream_exact_match": None, "real_downstream_topk_recovery": None, "real_downstream_token_accuracy": None, "real_downstream_cer": None}
        )
        raw_wide_downstream = (
            downstream_payload(
                method="uncertainty_beam",
                decoded=raw_wide_decoded,
                posterior=raw_wide_posterior,
                truth=example.observed_symbols,
                downstream_resource=downstream_resource,
                real_downstream_config=config.real_downstream,
                decoding_config=config.decoding,
                bigram_transition_model=transition_model,
                trigram_transition_model=trigram_transition_model,
            )
            if downstream_resource is not None
            else {"real_downstream_exact_match": None, "real_downstream_topk_recovery": None, "real_downstream_token_accuracy": None, "real_downstream_cer": None}
        )
        raw_default_metrics = sequence_metric_bundle(raw_default_decoded, [symbol for symbol in example.observed_symbols if symbol is not None])
        raw_default_score = _adaptive_utility(
            raw_default_metrics,
            raw_default_downstream,
            raw_default_set["prediction_set_avg_size"],
            prediction_set_penalty=config.adaptive_decoding.learned_gate_prediction_set_penalty,
            shortlist_utility_weight=config.adaptive_decoding.shortlist_utility_weight,
            review_budget_k=config.adaptive_decoding.review_budget_k,
        )
        raw_wide_metrics = sequence_metric_bundle(raw_wide_decoded, [symbol for symbol in example.observed_symbols if symbol is not None])
        raw_wide_score = _adaptive_utility(
            raw_wide_metrics,
            raw_wide_downstream,
            raw_wide_set["prediction_set_avg_size"],
            prediction_set_penalty=config.adaptive_decoding.learned_gate_prediction_set_penalty,
            shortlist_utility_weight=config.adaptive_decoding.shortlist_utility_weight,
            review_budget_k=config.adaptive_decoding.review_budget_k,
        )

        conformal_score = None
        conformal_metrics = None
        conformal_set = None
        conformal_downstream = None
        if conformal_predictor is not None:
            conformal_decoded, conformal_posterior, conformal_network = _decode_conformal_result(
                method="conformal_beam",
                posterior=posterior,
                network=network,
                conformal_predictor=conformal_predictor,
                bigram_transition_model=transition_model,
                trigram_transition_model=trigram_transition_model,
                config=config,
                beam_width_override=min(config.decoding.beam_width, config.adaptive_decoding.narrow_beam_width),
            )
            conformal_set = summarize_prediction_sets([conformal_network], [example.observed_symbols])
            conformal_downstream = (
                downstream_payload(
                    method="conformal_beam",
                    decoded=conformal_decoded,
                    posterior=conformal_posterior,
                    truth=example.observed_symbols,
                    downstream_resource=downstream_resource,
                    real_downstream_config=config.real_downstream,
                    decoding_config=config.decoding,
                    bigram_transition_model=transition_model,
                    trigram_transition_model=trigram_transition_model,
                )
                if downstream_resource is not None
                else {"real_downstream_exact_match": None, "real_downstream_topk_recovery": None, "real_downstream_token_accuracy": None, "real_downstream_cer": None}
            )
            conformal_metrics = sequence_metric_bundle(conformal_decoded, [symbol for symbol in example.observed_symbols if symbol is not None])
            conformal_score = _adaptive_utility(
                conformal_metrics,
                conformal_downstream,
                conformal_set["prediction_set_avg_size"],
                prediction_set_penalty=config.adaptive_decoding.learned_gate_prediction_set_penalty,
                shortlist_utility_weight=config.adaptive_decoding.shortlist_utility_weight,
                review_budget_k=config.adaptive_decoding.review_budget_k,
            )

        raw_best_metrics = raw_default_metrics
        raw_best_downstream = raw_default_downstream
        raw_best_set = raw_default_set
        raw_best_score = raw_default_score
        if raw_wide_score > raw_best_score:
            raw_best_metrics = raw_wide_metrics
            raw_best_downstream = raw_wide_downstream
            raw_best_set = raw_wide_set
            raw_best_score = raw_wide_score

        if config.adaptive_decoding.policy == "support_aware_constrained_gate":
            conformal_target = None
            if conformal_score is not None and conformal_metrics is not None and conformal_set is not None and conformal_downstream is not None:
                grouped_guardrail = (
                    float(conformal_metrics["sequence_topk_recovery"])
                    >= float(raw_best_metrics["sequence_topk_recovery"]) - config.adaptive_decoding.constrained_max_grouped_topk_drop
                )
                grouped_exact_guardrail = (
                    float(conformal_metrics["sequence_exact_match"])
                    >= float(raw_best_metrics["sequence_exact_match"]) - config.adaptive_decoding.constrained_max_grouped_exact_drop
                )
                downstream_guardrail = True
                raw_best_downstream_exact = _downstream_exact_value(raw_best_downstream)
                conformal_downstream_exact = _downstream_exact_value(conformal_downstream)
                if raw_best_downstream_exact is not None and conformal_downstream_exact is not None:
                    downstream_guardrail = (
                        conformal_downstream_exact
                        >= raw_best_downstream_exact + config.adaptive_decoding.constrained_min_downstream_exact_gain
                    )
                utility_guardrail = conformal_score >= raw_best_score + config.adaptive_decoding.constrained_min_utility_margin
                conformal_target = float(
                    grouped_guardrail and grouped_exact_guardrail and downstream_guardrail and utility_guardrail
                )

            wide_target = float(
                raw_wide_score >= raw_default_score + config.adaptive_decoding.constrained_min_utility_margin
                and float(raw_wide_metrics["sequence_topk_recovery"])
                >= float(raw_default_metrics["sequence_topk_recovery"])
                + config.adaptive_decoding.constrained_min_wide_grouped_topk_gain
                and float(raw_wide_set["prediction_set_avg_size"])
                <= float(raw_default_set["prediction_set_avg_size"])
                + config.adaptive_decoding.constrained_max_wide_set_size_inflation
            )
        else:
            conformal_target = (
                None
                if conformal_score is None
                else float(conformal_score > max(raw_default_score, raw_wide_score))
            )
            wide_target = float(raw_wide_score > raw_default_score)

        gate_rows.append(
            {
                **features,
                "prefer_conformal": conformal_target,
                "prefer_wide_beam": wide_target,
            }
        )

    continuous_features = [
        "mean_confusion_entropy",
        "mean_confusion_set_size",
        "sequence_length",
        "length_support_count",
    ]
    binary_features = ["limited_support", "is_calibrated", "conformal_available"]
    conformal_gate = fit_binary_logistic_gate(
        gate_rows,
        target_key="prefer_conformal",
        continuous_features=continuous_features,
        binary_features=binary_features,
        learning_rate=config.adaptive_decoding.learned_gate_learning_rate,
        steps=config.adaptive_decoding.learned_gate_steps,
        l2_penalty=config.adaptive_decoding.learned_gate_l2_penalty,
    )
    wide_gate = fit_binary_logistic_gate(
        gate_rows,
        target_key="prefer_wide_beam",
        continuous_features=continuous_features,
        binary_features=binary_features,
        learning_rate=config.adaptive_decoding.learned_gate_learning_rate,
        steps=config.adaptive_decoding.learned_gate_steps,
        l2_penalty=config.adaptive_decoding.learned_gate_l2_penalty,
    )
    diagnostics = [
        {
            "target_name": row["target_name"],
            "feature": row["feature"],
            "coefficient": row["coefficient"],
            "odds_ratio": row["odds_ratio"],
            "training_accuracy": row["training_accuracy"],
            "positive_rate": row["positive_rate"],
        }
        for row in conformal_gate.coefficient_rows() + wide_gate.coefficient_rows()
    ]
    return conformal_gate, wide_gate, diagnostics


def _selector_effort_adjusted_utility(
    *,
    sequence_metrics: dict[str, float],
    prediction_set_avg_size: float | None,
    review_budget_k: int,
    defer_to_human: bool,
) -> float:
    budgeted_recall = float(sequence_metrics.get(f"sequence_shortlist_recall_at_{review_budget_k}", 0.0))
    if prediction_set_avg_size is None:
        return 0.0
    review_load = min(float(prediction_set_avg_size), float(review_budget_k)) + (
        1.0 if defer_to_human else 0.0
    )
    if review_load <= 0.0:
        return 0.0
    return float(budgeted_recall / review_load)


def _train_profile_selector(
    *,
    validation_examples: list[SequenceExample],
    validation_posteriors: list[TranscriptionPosterior],
    validation_networks: list[Any],
    posterior_strategy: str,
    conformal_predictor: SplitConformalSetPredictor | None,
    transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    downstream_resource: Any,
    config: DecipherLabConfig,
    profile_gates: dict[str, tuple[Any, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    selector_rows: list[dict[str, Any]] = []
    for example, posterior, network in zip(validation_examples, validation_posteriors, validation_networks):
        snapshot = build_support_snapshot(
            network=network,
            posterior_strategy=posterior_strategy,
            sequence_length=example.sequence_length,
            downstream_resource=downstream_resource,
            conformal_available=conformal_predictor is not None,
        )
        profile_outputs: dict[str, dict[str, Any]] = {}
        for profile_mode in ("rescue_first", "shortlist_first"):
            conformal_gate, wide_gate = profile_gates[profile_mode]
            (
                adaptive_decision,
                decoded,
                active_posterior,
                active_network,
                conformal_probability,
                wide_probability,
            ) = _decode_profiled_adaptive_result(
                snapshot=snapshot,
                posterior_strategy=posterior_strategy,
                posterior=posterior,
                network=network,
                conformal_predictor=conformal_predictor,
                transition_model=transition_model,
                trigram_transition_model=trigram_transition_model,
                config=config,
                conformal_gate=conformal_gate,
                wide_gate=wide_gate,
                profile_mode=profile_mode,
                profile_reason="selector_training_profile",
            )
            sequence_metrics = sequence_metric_bundle(
                decoded,
                [symbol for symbol in example.observed_symbols if symbol is not None],
            )
            set_metrics = summarize_prediction_sets([active_network], [example.observed_symbols])
            downstream_metrics = (
                downstream_payload(
                    method="adaptive_profiled_beam",
                    decoded=decoded,
                    posterior=active_posterior,
                    truth=example.observed_symbols,
                    downstream_resource=downstream_resource,
                    real_downstream_config=config.real_downstream,
                    decoding_config=config.decoding,
                    bigram_transition_model=transition_model,
                    trigram_transition_model=trigram_transition_model,
                )
                if downstream_resource is not None
                else {"real_downstream_exact_match": None}
            )
            profile_outputs[profile_mode] = {
                "adaptive_decision": adaptive_decision,
                "sequence_metrics": sequence_metrics,
                "set_metrics": set_metrics,
                "downstream_metrics": downstream_metrics,
                "conformal_probability": conformal_probability,
                "wide_probability": wide_probability,
            }

        rescue_output = profile_outputs["rescue_first"]
        shortlist_output = profile_outputs["shortlist_first"]
        rescue_effort = _selector_effort_adjusted_utility(
            sequence_metrics=rescue_output["sequence_metrics"],
            prediction_set_avg_size=rescue_output["set_metrics"]["prediction_set_avg_size"],
            review_budget_k=config.adaptive_decoding.review_budget_k,
            defer_to_human=bool(rescue_output["adaptive_decision"].defer_to_human),
        )
        shortlist_effort = _selector_effort_adjusted_utility(
            sequence_metrics=shortlist_output["sequence_metrics"],
            prediction_set_avg_size=shortlist_output["set_metrics"]["prediction_set_avg_size"],
            review_budget_k=config.adaptive_decoding.review_budget_k,
            defer_to_human=bool(shortlist_output["adaptive_decision"].defer_to_human),
        )
        rescue_grouped = float(rescue_output["sequence_metrics"]["sequence_topk_recovery"])
        shortlist_grouped = float(shortlist_output["sequence_metrics"]["sequence_topk_recovery"])
        rescue_downstream = _downstream_exact_value(rescue_output["downstream_metrics"])
        shortlist_downstream = _downstream_exact_value(shortlist_output["downstream_metrics"])
        shortlist_allowed = (
            shortlist_grouped
            >= rescue_grouped - config.adaptive_decoding.selector_max_grouped_topk_drop
        ) and (
            rescue_downstream is None
            or shortlist_downstream is None
            or shortlist_downstream
            >= rescue_downstream + config.adaptive_decoding.selector_min_downstream_exact_gain
        )
        prefer_shortlist = float(
            shortlist_allowed
            and shortlist_effort >= rescue_effort + config.adaptive_decoding.selector_min_effort_margin
        )
        selector_row = build_profile_selector_feature_row(
            snapshot=snapshot,
            review_budget=config.adaptive_decoding.review_budget_k,
            rescue_decision=rescue_output["adaptive_decision"],
            shortlist_decision=shortlist_output["adaptive_decision"],
            rescue_conformal_probability=rescue_output["conformal_probability"],
            rescue_wide_probability=rescue_output["wide_probability"],
            shortlist_conformal_probability=shortlist_output["conformal_probability"],
            shortlist_wide_probability=shortlist_output["wide_probability"],
        )
        selector_rows.append(
            {
                **selector_row,
                "prefer_shortlist_profile": prefer_shortlist,
            }
        )

    continuous_features, binary_features = selector_feature_names()
    selector_gate = fit_binary_logistic_gate(
        selector_rows,
        target_key="prefer_shortlist_profile",
        continuous_features=continuous_features,
        binary_features=binary_features,
        learning_rate=config.adaptive_decoding.selector_learning_rate,
        steps=config.adaptive_decoding.selector_steps,
        l2_penalty=config.adaptive_decoding.selector_l2_penalty,
    )
    diagnostics = [
        {
            "target_name": row["target_name"],
            "feature": row["feature"],
            "coefficient": row["coefficient"],
            "odds_ratio": row["odds_ratio"],
            "training_accuracy": row["training_accuracy"],
            "positive_rate": row["positive_rate"],
        }
        for row in selector_gate.coefficient_rows()
    ]
    return selector_gate, diagnostics


def _decode_profile_selector_result(
    *,
    snapshot,
    posterior_strategy: str,
    posterior: TranscriptionPosterior,
    network,
    conformal_predictor: SplitConformalSetPredictor | None,
    transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
    config: DecipherLabConfig,
    profile_gates: dict[str, tuple[Any, Any]],
    selector_gate: Any,
) -> tuple[AdaptiveDecodingDecision, BeamDecodingResult, TranscriptionPosterior, Any, float | None, float | None, float]:
    profile_outputs: dict[str, tuple[AdaptiveDecodingDecision, BeamDecodingResult, TranscriptionPosterior, Any, float | None, float | None]] = {}
    for profile_mode in ("rescue_first", "shortlist_first"):
        conformal_gate, wide_gate = profile_gates[profile_mode]
        profile_outputs[profile_mode] = _decode_profiled_adaptive_result(
            snapshot=snapshot,
            posterior_strategy=posterior_strategy,
            posterior=posterior,
            network=network,
            conformal_predictor=conformal_predictor,
            transition_model=transition_model,
            trigram_transition_model=trigram_transition_model,
            config=config,
            conformal_gate=conformal_gate,
            wide_gate=wide_gate,
            profile_mode=profile_mode,
            profile_reason="selector_candidate_profile",
        )
    rescue_output = profile_outputs["rescue_first"]
    shortlist_output = profile_outputs["shortlist_first"]
    selector_features = build_profile_selector_feature_row(
        snapshot=snapshot,
        review_budget=config.adaptive_decoding.review_budget_k,
        rescue_decision=rescue_output[0],
        shortlist_decision=shortlist_output[0],
        rescue_conformal_probability=rescue_output[4],
        rescue_wide_probability=rescue_output[5],
        shortlist_conformal_probability=shortlist_output[4],
        shortlist_wide_probability=shortlist_output[5],
    )
    shortlist_probability = float(selector_gate.predict_proba(selector_features))
    selector_decision = select_profile(
        shortlist_probability=shortlist_probability,
        snapshot=snapshot,
        rescue_decision=rescue_output[0],
        shortlist_decision=shortlist_output[0],
        config=config.adaptive_decoding,
    )
    selected_output = profile_outputs[selector_decision.selected_profile]
    adaptive_decision = replace(
        selected_output[0],
        operating_profile=selector_decision.selected_profile,
        profile_reason=selector_decision.decision_reason,
        decision_reason=selector_decision.decision_reason
        if selector_decision.direct_defer
        else selected_output[0].decision_reason,
        control_action="defer" if selector_decision.direct_defer else selected_output[0].control_action,
        defer_to_human=selector_decision.direct_defer or selected_output[0].defer_to_human,
    )
    return (
        adaptive_decision,
        selected_output[1],
        selected_output[2],
        selected_output[3],
        selected_output[4],
        selected_output[5],
        shortlist_probability,
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["ambiguity_level"], row["method"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (ambiguity_level, method), items in sorted(grouped.items()):
        symbol_top1_values = [(item["symbol_top1_accuracy"], item["labeled_symbol_count"]) for item in items]
        symbol_topk_values = [(item["symbol_topk_accuracy"], item["labeled_symbol_count"]) for item in items]
        symbol_nll_values = [(item["symbol_negative_log_likelihood"], item["labeled_symbol_count"]) for item in items]
        symbol_ece_values = [(item["symbol_expected_calibration_error"], item["labeled_symbol_count"]) for item in items]
        summary_rows.append(
            {
                "ambiguity_level": ambiguity_level,
                "method": method,
                "sequence_exact_match": float(np.mean([item["sequence_exact_match"] for item in items])),
                "sequence_token_accuracy": float(np.mean([item["sequence_token_accuracy"] for item in items])),
                "sequence_topk_recovery": float(np.mean([item["sequence_topk_recovery"] for item in items])),
                "sequence_shortlist_recall_at_2": float(np.mean([item["sequence_shortlist_recall_at_2"] for item in items])),
                "sequence_shortlist_recall_at_3": float(np.mean([item["sequence_shortlist_recall_at_3"] for item in items])),
                "sequence_shortlist_recall_at_5": float(np.mean([item["sequence_shortlist_recall_at_5"] for item in items])),
                "sequence_shortlist_utility": float(np.mean([item["sequence_shortlist_utility"] for item in items])),
                "sequence_cer": float(np.mean([item["sequence_cer"] for item in items])),
                "symbol_top1_accuracy": _weighted_average(symbol_top1_values),
                "symbol_topk_accuracy": _weighted_average(symbol_topk_values),
                "symbol_negative_log_likelihood": _weighted_average(symbol_nll_values),
                "symbol_expected_calibration_error": _weighted_average(symbol_ece_values),
                "prediction_set_coverage": float(np.mean([item["prediction_set_coverage"] for item in items]))
                if any(item["prediction_set_coverage"] is not None for item in items)
                else None,
                "prediction_set_avg_size": float(np.mean([item["prediction_set_avg_size"] for item in items if item["prediction_set_avg_size"] is not None]))
                if any(item["prediction_set_avg_size"] is not None for item in items)
                else None,
                "prediction_set_singleton_rate": float(np.mean([item["prediction_set_singleton_rate"] for item in items if item["prediction_set_singleton_rate"] is not None]))
                if any(item["prediction_set_singleton_rate"] is not None for item in items)
                else None,
                "prediction_set_rescue_rate": float(np.mean([item["prediction_set_rescue_rate"] for item in items if item["prediction_set_rescue_rate"] is not None]))
                if any(item["prediction_set_rescue_rate"] is not None for item in items)
                else None,
                "family_identification_accuracy": float(np.mean([item["family_identification_accuracy"] for item in items if item["family_identification_accuracy"] is not None]))
                if any(item["family_identification_accuracy"] is not None for item in items)
                else None,
                "family_identification_topk_recovery": float(np.mean([item["family_identification_topk_recovery"] for item in items if item["family_identification_topk_recovery"] is not None]))
                if any(item["family_identification_topk_recovery"] is not None for item in items)
                else None,
                "real_downstream_bank_coverage": float(np.mean([item["real_downstream_bank_coverage"] for item in items if item["real_downstream_bank_coverage"] is not None]))
                if any(item.get("real_downstream_bank_coverage") is not None for item in items)
                else None,
                "real_downstream_bank_size": float(np.mean([item["real_downstream_bank_size"] for item in items if item["real_downstream_bank_size"] is not None]))
                if any(item.get("real_downstream_bank_size") is not None for item in items)
                else None,
                "real_downstream_exact_match": float(np.mean([item["real_downstream_exact_match"] for item in items if item["real_downstream_exact_match"] is not None]))
                if any(item.get("real_downstream_exact_match") is not None for item in items)
                else None,
                "real_downstream_topk_recovery": float(np.mean([item["real_downstream_topk_recovery"] for item in items if item["real_downstream_topk_recovery"] is not None]))
                if any(item.get("real_downstream_topk_recovery") is not None for item in items)
                else None,
                "real_downstream_token_accuracy": float(np.mean([item["real_downstream_token_accuracy"] for item in items if item["real_downstream_token_accuracy"] is not None]))
                if any(item.get("real_downstream_token_accuracy") is not None for item in items)
                else None,
                "real_downstream_cer": float(np.mean([item["real_downstream_cer"] for item in items if item["real_downstream_cer"] is not None]))
                if any(item.get("real_downstream_cer") is not None for item in items)
                else None,
                "real_downstream_exact_match_if_covered": float(np.mean([item["real_downstream_exact_match_if_covered"] for item in items if item["real_downstream_exact_match_if_covered"] is not None]))
                if any(item.get("real_downstream_exact_match_if_covered") is not None for item in items)
                else None,
                "real_downstream_topk_recovery_if_covered": float(np.mean([item["real_downstream_topk_recovery_if_covered"] for item in items if item["real_downstream_topk_recovery_if_covered"] is not None]))
                if any(item.get("real_downstream_topk_recovery_if_covered") is not None for item in items)
                else None,
                "sequence_count": len(items),
                "labeled_symbol_count": int(sum(item["labeled_symbol_count"] for item in items)),
            }
        )
    return summary_rows


def _build_report(summary_rows: list[dict[str, Any]], benchmark_metadata: dict[str, Any]) -> str:
    synthetic_from_real = bool(benchmark_metadata["synthetic_from_real"])
    lines = [
        "# Sequence Branch Report",
        "",
        "This report belongs to the sequence-focused branch and should not be merged into the workshop paper claims.",
        "",
        "## Benchmark",
        f"- Source dataset: `{benchmark_metadata['source_dataset_name']}`",
        f"- Task name: `{benchmark_metadata['task_name']}`",
        f"- Synthetic-from-real: `{synthetic_from_real}`",
        f"- Selected symbols: `{len(benchmark_metadata['selected_symbols'])}`",
        f"- Sequence length: `{benchmark_metadata['sequence_length']}`",
        f"- Real downstream task: `{benchmark_metadata.get('real_downstream_task_name', 'disabled')}`",
        "",
        "## Aggregated Results",
        "",
        "| Ambiguity | Method | Seq exact | Seq token | Seq top-k | Downstream exact | Downstream top-k | Seq CER | Symbol top-k | Set coverage | Family acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {ambiguity:.2f} | {method} | {exact:.3f} | {token:.3f} | {topk:.3f} | {downstream_exact} | {downstream_topk} | {cer:.3f} | {symbol_topk:.3f} | {coverage} | {family_acc} |".format(
                ambiguity=row["ambiguity_level"],
                method=row["method"],
                exact=row["sequence_exact_match"],
                token=row["sequence_token_accuracy"],
                topk=row["sequence_topk_recovery"],
                downstream_exact="n/a" if row["real_downstream_exact_match"] is None else f"{row['real_downstream_exact_match']:.3f}",
                downstream_topk="n/a" if row["real_downstream_topk_recovery"] is None else f"{row['real_downstream_topk_recovery']:.3f}",
                cer=row["sequence_cer"],
                symbol_topk=row["symbol_topk_accuracy"] or 0.0,
                coverage="n/a" if row["prediction_set_coverage"] is None else f"{row['prediction_set_coverage']:.3f}",
                family_acc="n/a" if row["family_identification_accuracy"] is None else f"{row['family_identification_accuracy']:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            (
                "- Sequence-level claims in this branch are synthetic-from-real: real glyph crops are used, but transition structure is generated."
                if synthetic_from_real
                else "- Sequence-level claims here come from a real grouped manifest with OCR-derived token labels; they should be treated as preliminary grouped evidence."
            ),
            "- Positive results here would support structured use of uncertainty under ambiguity, not semantic decipherment.",
        ]
    )
    return "\n".join(lines)


def run_sequence_branch_experiment(
    config: DecipherLabConfig | str | Path,
    strategy_override: str | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not resolved.sequence_benchmark.enabled:
        raise ValueError("sequence_benchmark.enabled must be true for the sequence branch.")
    if not resolved.structured_uncertainty.enabled or not resolved.decoding.enabled:
        raise ValueError("structured_uncertainty.enabled and decoding.enabled must both be true.")

    posterior_strategy = resolved.posterior.strategy if strategy_override is None else strategy_override
    run_context = prepare_run_context(resolved, suffix=f"sequence_branch_{posterior_strategy}")
    logger = configure_logging(run_context.run_dir)
    dump_config(resolved, run_context.run_dir / "config.yaml")

    logger.info("Loading source manifest dataset for sequence benchmark generation.")
    source_dataset = _load_manifest_dataset(resolved)
    benchmark = build_real_glyph_sequence_benchmark(
        source_dataset=source_dataset,
        config=resolved.sequence_benchmark,
        seed=resolved.experiment.seed,
        source_train_split=resolved.dataset.train_split,
        source_val_split=resolved.dataset.val_split,
        source_test_split=resolved.dataset.evaluation_split,
    )
    write_json(run_context.run_dir / "benchmark_summary.json", benchmark.to_dict())

    train_examples = benchmark.dataset.get_split(resolved.dataset.train_split)
    validation_examples = benchmark.dataset.get_split(resolved.dataset.val_split)
    evaluation_examples = benchmark.dataset.get_split(resolved.dataset.evaluation_split)
    train_glyphs = _flatten_glyphs(train_examples)
    validation_glyphs = _flatten_glyphs(validation_examples)
    if not train_glyphs or not validation_glyphs or not evaluation_examples:
        raise ValueError("Sequence branch requires non-empty train, val, and test synthetic-from-real splits.")

    train_features = extract_feature_matrix(train_glyphs, downsample=resolved.vision.feature_downsample)
    validation_features = extract_feature_matrix(validation_glyphs, downsample=resolved.vision.feature_downsample)
    posterior_model = fit_posterior_model(
        train_features=train_features,
        train_labels=[glyph.true_symbol for glyph in train_glyphs],
        validation_features=validation_features,
        validation_labels=[glyph.true_symbol for glyph in validation_glyphs],
        posterior_config=resolved.posterior.model_copy(update={"strategy": posterior_strategy}),
        vision_config=resolved.vision,
        seed=resolved.experiment.seed,
    )
    transition_model = BigramTransitionModel.fit(
        sequences=[[symbol for symbol in example.observed_symbols if symbol is not None] for example in train_examples],
        smoothing=resolved.decoding.transition_smoothing,
    )
    trigram_transition_model = (
        TrigramTransitionModel.fit(
            sequences=[[symbol for symbol in example.observed_symbols if symbol is not None] for example in train_examples],
            smoothing=resolved.decoding.transition_smoothing,
        )
        if "trigram_beam" in resolved.decoding.decoder_variants
        else None
    )
    family_classifier = ProcessFamilyClassifier.fit(
        train_examples,
        smoothing=resolved.decoding.transition_smoothing,
    )
    downstream_resource = (
        build_real_downstream_resource(train_examples, resolved.real_downstream)
        if resolved.real_downstream.enabled and benchmark.metadata["task_name"] == "real_grouped_manifest_sequences"
        else None
    )

    per_sequence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    adaptive_gate_diagnostics: list[dict[str, Any]] = []
    for ambiguity_index, ambiguity_level in enumerate(resolved.evaluation.ambiguity_levels):
        logger.info("Running sequence branch at ambiguity %.2f.", ambiguity_level)
        ambiguous_validation = apply_ambiguity_to_examples(
            validation_examples,
            ambiguity_level=ambiguity_level,
            seed=resolved.experiment.seed + 5000 + ambiguity_index,
        )
        ambiguous_evaluation = apply_ambiguity_to_examples(
            evaluation_examples,
            ambiguity_level=ambiguity_level,
            seed=resolved.experiment.seed + 8000 + ambiguity_index,
        )
        validation_features_ambiguous = extract_feature_matrix(
            _flatten_glyphs(ambiguous_validation),
            downsample=resolved.vision.feature_downsample,
        )
        evaluation_features = extract_feature_matrix(
            _flatten_glyphs(ambiguous_evaluation),
            downsample=resolved.vision.feature_downsample,
        )
        validation_posterior = posterior_model.predict_posterior(
            validation_features_ambiguous,
            top_k=resolved.posterior.top_k,
            floor_probability=resolved.posterior.floor_probability,
        )
        evaluation_posterior = posterior_model.predict_posterior(
            evaluation_features,
            top_k=resolved.posterior.top_k,
            floor_probability=resolved.posterior.floor_probability,
        )
        validation_posteriors = split_posterior_by_lengths(
            validation_posterior,
            [example.sequence_length for example in ambiguous_validation],
        )
        evaluation_posteriors = split_posterior_by_lengths(
            evaluation_posterior,
            [example.sequence_length for example in ambiguous_evaluation],
        )
        validation_networks = [
            posterior_to_confusion_network(posterior, resolved.structured_uncertainty)
            for posterior in validation_posteriors
        ]
        evaluation_networks = [
            posterior_to_confusion_network(posterior, resolved.structured_uncertainty)
            for posterior in evaluation_posteriors
        ]

        conformal_predictor = None
        if resolved.risk_control.enabled:
            conformal_predictor = SplitConformalSetPredictor.fit(
                validation_networks,
                _sequence_labels(validation_examples),
                resolved.risk_control,
            )
        learned_conformal_gate = None
        learned_wide_gate = None
        profile_gates: dict[str, tuple[Any, Any]] = {}
        selector_gate = None
        if resolved.adaptive_decoding.enabled:
            adaptive_branch = _adaptive_policy_branch(resolved)
            profiles_to_train: list[str] = []
            if adaptive_branch in {"shortlist_first", "rescue_first"}:
                profiles_to_train = [adaptive_branch]
            elif adaptive_branch == "profile_selector":
                profiles_to_train = ["rescue_first", "shortlist_first"]
            for profile_mode in profiles_to_train:
                training_config = resolved.model_copy(
                    update={
                        "adaptive_decoding": resolved.adaptive_decoding.model_copy(
                            update={"policy": _delegated_policy_for_profile(profile_mode)}
                        )
                    }
                )
                learned_conformal_gate, learned_wide_gate, gate_rows = _train_learned_adaptive_gates(
                    validation_examples=validation_examples,
                    validation_posteriors=validation_posteriors,
                    validation_networks=validation_networks,
                    posterior_strategy=posterior_model.strategy,
                    conformal_predictor=conformal_predictor,
                    transition_model=transition_model,
                    trigram_transition_model=trigram_transition_model,
                    downstream_resource=downstream_resource,
                    config=training_config,
                )
                profile_gates[profile_mode] = (learned_conformal_gate, learned_wide_gate)
                for row in gate_rows:
                    adaptive_gate_diagnostics.append(
                        {
                            "ambiguity_level": ambiguity_level,
                            "posterior_strategy": posterior_model.strategy,
                            "operating_profile": profile_mode,
                            **row,
                        }
                    )
            if adaptive_branch in {"shortlist_first", "rescue_first"}:
                learned_conformal_gate, learned_wide_gate = profile_gates[adaptive_branch]
            if resolved.adaptive_decoding.policy == "support_aware_profile_selector":
                selector_gate, selector_rows = _train_profile_selector(
                    validation_examples=validation_examples,
                    validation_posteriors=validation_posteriors,
                    validation_networks=validation_networks,
                    posterior_strategy=posterior_model.strategy,
                    conformal_predictor=conformal_predictor,
                    transition_model=transition_model,
                    trigram_transition_model=trigram_transition_model,
                    downstream_resource=downstream_resource,
                    config=resolved,
                    profile_gates=profile_gates,
                )
                for row in selector_rows:
                    adaptive_gate_diagnostics.append(
                        {
                            "ambiguity_level": ambiguity_level,
                            "posterior_strategy": posterior_model.strategy,
                            "operating_profile": "selector",
                            **row,
                        }
                    )
        methods = ["fixed_greedy"]
        if "bigram_beam" in resolved.decoding.decoder_variants:
            methods.append("uncertainty_beam")
            if conformal_predictor is not None:
                methods.append("conformal_beam")
            if resolved.adaptive_decoding.enabled:
                methods.append(_adaptive_method_name(resolved))
        if "trigram_beam" in resolved.decoding.decoder_variants:
            methods.append("uncertainty_trigram_beam")
            if conformal_predictor is not None:
                methods.append("conformal_trigram_beam")
        if "crf_viterbi" in resolved.decoding.decoder_variants:
            methods.append("uncertainty_crf_viterbi")
            if conformal_predictor is not None:
                methods.append("conformal_crf_viterbi")

        for example, posterior, network in zip(ambiguous_evaluation, evaluation_posteriors, evaluation_networks):
            for method in methods:
                active_network = network
                active_posterior = posterior
                adaptive_decision: AdaptiveDecodingDecision | None = None
                conformal_probability: float | None = None
                wide_probability: float | None = None
                selector_probability: float | None = None
                if method in {"conformal_beam", "conformal_trigram_beam", "conformal_crf_viterbi"}:
                    decoded, active_posterior, active_network = _decode_conformal_result(
                        method=method,
                        posterior=posterior,
                        network=network,
                        conformal_predictor=conformal_predictor,
                        bigram_transition_model=transition_model,
                        trigram_transition_model=trigram_transition_model,
                        config=resolved,
                    )
                elif method == "adaptive_support_beam":
                    adaptive_decision = decide_support_aware_method(
                        build_support_snapshot(
                            network=network,
                            posterior_strategy=posterior_model.strategy,
                            sequence_length=example.sequence_length,
                            downstream_resource=downstream_resource,
                            conformal_available=conformal_predictor is not None,
                        ),
                        resolved.adaptive_decoding,
                        default_beam_width=resolved.decoding.beam_width,
                    )
                    if adaptive_decision.selected_method == "conformal_beam":
                        decoded, active_posterior, active_network = _decode_conformal_result(
                            method="conformal_beam",
                            posterior=posterior,
                            network=network,
                            conformal_predictor=conformal_predictor,
                            bigram_transition_model=transition_model,
                            trigram_transition_model=trigram_transition_model,
                            config=resolved,
                            beam_width_override=adaptive_decision.beam_width,
                        )
                    else:
                        decoded, active_posterior, active_network = _decode_method_result(
                            method="uncertainty_beam",
                            posterior=posterior,
                            network=network,
                            bigram_transition_model=transition_model,
                            trigram_transition_model=trigram_transition_model,
                            config=resolved,
                            beam_width_override=adaptive_decision.beam_width,
                        )
                elif method in {
                    "adaptive_learned_beam",
                    "adaptive_constrained_beam",
                    "adaptive_profiled_beam",
                    "adaptive_profile_selector_beam",
                }:
                    snapshot = build_support_snapshot(
                        network=network,
                        posterior_strategy=posterior_model.strategy,
                        sequence_length=example.sequence_length,
                        downstream_resource=downstream_resource,
                        conformal_available=conformal_predictor is not None,
                    )
                    if method == "adaptive_profiled_beam":
                        profile_mode, profile_reason = resolve_operating_profile(resolved.adaptive_decoding)
                        (
                            adaptive_decision,
                            decoded,
                            active_posterior,
                            active_network,
                            conformal_probability,
                            wide_probability,
                        ) = _decode_profiled_adaptive_result(
                            snapshot=snapshot,
                            posterior_strategy=posterior_model.strategy,
                            posterior=posterior,
                            network=network,
                            conformal_predictor=conformal_predictor,
                            transition_model=transition_model,
                            trigram_transition_model=trigram_transition_model,
                            config=resolved,
                            conformal_gate=learned_conformal_gate,
                            wide_gate=learned_wide_gate,
                            profile_mode=profile_mode,
                            profile_reason=profile_reason,
                        )
                    elif method == "adaptive_profile_selector_beam":
                        (
                            adaptive_decision,
                            decoded,
                            active_posterior,
                            active_network,
                            conformal_probability,
                            wide_probability,
                            selector_probability,
                        ) = _decode_profile_selector_result(
                            snapshot=snapshot,
                            posterior_strategy=posterior_model.strategy,
                            posterior=posterior,
                            network=network,
                            conformal_predictor=conformal_predictor,
                            transition_model=transition_model,
                            trigram_transition_model=trigram_transition_model,
                            config=resolved,
                            profile_gates=profile_gates,
                            selector_gate=selector_gate,
                        )
                    else:
                        profile_mode = "rescue_first" if method == "adaptive_constrained_beam" else "shortlist_first"
                        profile_reason = "legacy_policy"
                        (
                            adaptive_decision,
                            decoded,
                            active_posterior,
                            active_network,
                            conformal_probability,
                            wide_probability,
                        ) = _decode_profiled_adaptive_result(
                            snapshot=snapshot,
                            posterior_strategy=posterior_model.strategy,
                            posterior=posterior,
                            network=network,
                            conformal_predictor=conformal_predictor,
                            transition_model=transition_model,
                            trigram_transition_model=trigram_transition_model,
                            config=resolved,
                            conformal_gate=learned_conformal_gate,
                            wide_gate=learned_wide_gate,
                            profile_mode=profile_mode,
                            profile_reason=profile_reason,
                        )
                else:
                    decoded, active_posterior, active_network = _decode_method_result(
                        method=method,
                        posterior=posterior,
                        network=network,
                        bigram_transition_model=transition_model,
                        trigram_transition_model=trigram_transition_model,
                        config=resolved,
                    )
                sequence_metrics = sequence_metric_bundle(
                    decoded,
                    [symbol for symbol in example.observed_symbols if symbol is not None],
                )
                family_metrics = family_identification_payload(
                    family_classifier,
                    decoded.best.symbols if decoded.sequences else [],
                    example.family,
                )
                downstream_metrics = (
                    downstream_payload(
                        method=method,
                        decoded=decoded,
                        posterior=active_posterior,
                        truth=example.observed_symbols,
                        downstream_resource=downstream_resource,
                        real_downstream_config=resolved.real_downstream,
                        decoding_config=resolved.decoding,
                        bigram_transition_model=transition_model,
                        trigram_transition_model=trigram_transition_model,
                    )
                    if downstream_resource is not None
                    else {
                        "real_downstream_supported": False,
                        "real_downstream_task_name": None,
                        "real_downstream_bank_size": None,
                        "real_downstream_bank_coverage": None,
                        "real_downstream_exact_match": None,
                        "real_downstream_topk_recovery": None,
                        "real_downstream_token_accuracy": None,
                        "real_downstream_cer": None,
                        "real_downstream_exact_match_if_covered": None,
                        "real_downstream_topk_recovery_if_covered": None,
                        "real_downstream_best_transcript": [],
                    }
                )
                set_metrics = summarize_prediction_sets([active_network], [example.observed_symbols])
                symbol_top1 = symbol_top_k_accuracy(active_posterior, example.observed_symbols, top_k=1)
                symbol_topk = symbol_top_k_accuracy(active_posterior, example.observed_symbols, top_k=resolved.evaluation.top_k)
                symbol_nll = symbol_negative_log_likelihood(active_posterior, example.observed_symbols)
                symbol_ece = symbol_expected_calibration_error(active_posterior, example.observed_symbols)
                per_sequence_rows.append(
                    {
                        "ambiguity_level": ambiguity_level,
                        "example_id": example.example_id,
                        "method": method,
                        "posterior_strategy": posterior_model.strategy,
                        "posterior_strategy_requested": posterior_strategy,
                        "task_name": benchmark.metadata["task_name"],
                        "dataset_name": benchmark.metadata["source_dataset_name"],
                        "true_family": example.family,
                        "sequence_length": example.sequence_length,
                        **sequence_metrics,
                        "symbol_top1_accuracy": symbol_top1[0],
                        "symbol_topk_accuracy": symbol_topk[0],
                        "symbol_negative_log_likelihood": symbol_nll[0],
                        "symbol_expected_calibration_error": symbol_ece[0],
                        "prediction_set_coverage": set_metrics["prediction_set_coverage"],
                        "prediction_set_avg_size": set_metrics["prediction_set_avg_size"],
                        "prediction_set_singleton_rate": set_metrics["prediction_set_singleton_rate"],
                        "prediction_set_rescue_rate": set_metrics["prediction_set_rescue_rate"],
                        "family_identification_accuracy": family_metrics["family_identification_accuracy"],
                        "family_identification_topk_recovery": family_metrics["family_identification_topk_recovery"],
                        "predicted_family": family_metrics["predicted_family"],
                        "real_downstream_supported": downstream_metrics["real_downstream_supported"],
                        "real_downstream_task_name": downstream_metrics["real_downstream_task_name"],
                        "real_downstream_bank_size": downstream_metrics["real_downstream_bank_size"],
                        "real_downstream_bank_coverage": downstream_metrics["real_downstream_bank_coverage"],
                        "real_downstream_exact_match": downstream_metrics["real_downstream_exact_match"],
                        "real_downstream_topk_recovery": downstream_metrics["real_downstream_topk_recovery"],
                        "real_downstream_token_accuracy": downstream_metrics["real_downstream_token_accuracy"],
                        "real_downstream_cer": downstream_metrics["real_downstream_cer"],
                        "real_downstream_exact_match_if_covered": downstream_metrics["real_downstream_exact_match_if_covered"],
                        "real_downstream_topk_recovery_if_covered": downstream_metrics["real_downstream_topk_recovery_if_covered"],
                        "labeled_symbol_count": symbol_top1[1],
                        "mean_confusion_entropy": active_network.mean_entropy(),
                        "mean_confusion_set_size": active_network.average_set_size(),
                        "support_length_count": 0 if downstream_resource is None else downstream_resource.metadata.get("length_support_counts", {}).get(str(example.sequence_length), 0),
                        "adaptive_selected_method": None if adaptive_decision is None else adaptive_decision.selected_method,
                        "adaptive_beam_width": None if adaptive_decision is None else adaptive_decision.beam_width,
                        "adaptive_decision_reason": None if adaptive_decision is None else adaptive_decision.decision_reason,
                        "adaptive_control_action": None if adaptive_decision is None else adaptive_decision.control_action,
                        "adaptive_defer_to_human": None if adaptive_decision is None else float(adaptive_decision.defer_to_human),
                        "adaptive_review_budget": None if adaptive_decision is None else float(adaptive_decision.review_budget),
                        "adaptive_budget_tight": None if adaptive_decision is None else float(adaptive_decision.budget_tight),
                        "adaptive_fragile_signal_count": None if adaptive_decision is None else float(adaptive_decision.fragile_signal_count),
                        "adaptive_operating_profile": None if adaptive_decision is None else adaptive_decision.operating_profile,
                        "adaptive_profile_reason": None if adaptive_decision is None else adaptive_decision.profile_reason,
                        "adaptive_limited_support": None if adaptive_decision is None else float(adaptive_decision.limited_support),
                        "adaptive_low_entropy": None if adaptive_decision is None else float(adaptive_decision.low_entropy),
                        "adaptive_high_entropy": None if adaptive_decision is None else float(adaptive_decision.high_entropy),
                        "adaptive_compact_set": None if adaptive_decision is None else float(adaptive_decision.compact_set),
                        "adaptive_diffuse_set": None if adaptive_decision is None else float(adaptive_decision.diffuse_set),
                        "adaptive_gate_conformal_probability": conformal_probability,
                        "adaptive_gate_wide_probability": wide_probability,
                        "adaptive_selector_shortlist_probability": selector_probability,
                        "case_breakdown": symbol_case_breakdown(
                            active_posterior,
                            example.observed_symbols,
                            top_k=resolved.evaluation.top_k,
                        ),
                        "decoded_best_sequence": decoded.best.symbols if decoded.sequences else [],
                    }
                )
        summary_rows.extend(_aggregate_rows([row for row in per_sequence_rows if row["ambiguity_level"] == ambiguity_level]))

    for row in summary_rows:
        row["posterior_strategy_requested"] = posterior_strategy
        row["posterior_strategy"] = posterior_model.strategy
        row["task_name"] = benchmark.metadata["task_name"]
        row["dataset_name"] = benchmark.metadata["source_dataset_name"]

    pairwise_rows = build_pairwise_effect_rows(summary_rows)
    ambiguity_regime_rows = build_ambiguity_regime_rows(pairwise_rows)
    best_regime = summarize_best_regime(pairwise_rows)
    failure_cases, failure_summary = build_sequence_failure_cases(per_sequence_rows)
    dataset_summary = {
        "dataset_name": benchmark.metadata["source_dataset_name"],
        "task_name": benchmark.metadata["task_name"],
        "sequence_length": benchmark.dataset.metadata["sequence_length"],
        "selected_symbol_count": len(benchmark.alphabet),
        "sequence_counts": benchmark.dataset.metadata["sequence_counts"],
        "source_manifest": benchmark.metadata["source_manifest"],
        "synthetic_from_real": benchmark.metadata["synthetic_from_real"],
        "family_signal_available": family_classifier is not None,
        "real_downstream_enabled": downstream_resource is not None,
        "real_downstream_task_name": None if downstream_resource is None else resolved.real_downstream.task_name,
        "real_downstream_bank_length_count": 0
        if downstream_resource is None
        else int(downstream_resource.metadata.get("length_count", downstream_resource.metadata.get("inventory_size", 0))),
        "best_ambiguity_regime_by_strategy": best_regime,
    }

    write_json(run_context.run_dir / "sequence_branch_summary.json", summary_rows)
    write_csv(run_context.run_dir / "sequence_branch_summary.csv", summary_rows)
    write_json(run_context.run_dir / "sequence_branch_examples.json", per_sequence_rows)
    write_csv(
        run_context.run_dir / "sequence_branch_examples.csv",
        [
            {key: value for key, value in row.items() if key not in {"case_breakdown", "decoded_best_sequence"}}
            for row in per_sequence_rows
        ],
    )
    write_json(run_context.run_dir / "sequence_pairwise_effects.json", pairwise_rows)
    write_csv(run_context.run_dir / "sequence_pairwise_effects.csv", pairwise_rows)
    write_json(run_context.run_dir / "sequence_ambiguity_regime_table.json", ambiguity_regime_rows)
    write_csv(run_context.run_dir / "sequence_ambiguity_regime_table.csv", ambiguity_regime_rows)
    write_json(run_context.run_dir / "sequence_failure_cases.json", failure_cases)
    write_json(run_context.run_dir / "sequence_failure_summary.json", failure_summary)
    write_csv(run_context.run_dir / "sequence_failure_summary.csv", failure_summary)
    write_json(run_context.run_dir / "dataset_summary.json", dataset_summary)
    if adaptive_gate_diagnostics:
        write_json(run_context.run_dir / "adaptive_gate_diagnostics.json", adaptive_gate_diagnostics)
    write_text(
        run_context.run_dir / "dataset_summary.md",
        "\n".join(
            [
                "# Sequence Dataset Summary",
                "",
                f"- Source dataset: `{dataset_summary['dataset_name']}`",
                f"- Task name: `{dataset_summary['task_name']}`",
                f"- Sequence length: `{dataset_summary['sequence_length']}`",
                f"- Selected symbol count: `{dataset_summary['selected_symbol_count']}`",
                f"- Train/val/test sequences: `{dataset_summary['sequence_counts']}`",
                f"- Synthetic-from-real: `{dataset_summary['synthetic_from_real']}`",
                f"- Family signal available: `{dataset_summary['family_signal_available']}`",
                f"- Source manifest: `{dataset_summary['source_manifest']}`",
            ]
        ),
    )
    write_text(
        run_context.run_dir / "sequence_branch_report.md",
        _build_report(summary_rows, benchmark.dataset.metadata),
    )
    return {
        "run_dir": run_context.run_dir,
        "summary_rows": summary_rows,
        "per_sequence_rows": per_sequence_rows,
        "pairwise_rows": pairwise_rows,
        "ambiguity_regime_rows": ambiguity_regime_rows,
        "failure_summary": failure_summary,
        "benchmark": benchmark.to_dict(),
        "posterior_model": posterior_model.to_dict(),
        "transition_model": transition_model.diagnostics,
        "adaptive_gate_diagnostics": adaptive_gate_diagnostics,
    }
