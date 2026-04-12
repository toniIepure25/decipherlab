from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decipherlab.config import AdaptiveDecodingConfig
from decipherlab.sequence.adaptive_decoder import (
    AdaptiveDecodingDecision,
    AdaptiveSupportSnapshot,
    support_feature_row,
)


@dataclass(frozen=True)
class ProfileSelectorDecision:
    selected_profile: str
    shortlist_probability: float
    decision_reason: str
    direct_defer: bool
    probability_margin: float


def build_profile_selector_feature_row(
    *,
    snapshot: AdaptiveSupportSnapshot,
    review_budget: int,
    rescue_decision: AdaptiveDecodingDecision,
    shortlist_decision: AdaptiveDecodingDecision,
    rescue_conformal_probability: float | None,
    rescue_wide_probability: float | None,
    shortlist_conformal_probability: float | None,
    shortlist_wide_probability: float | None,
) -> dict[str, float]:
    row = support_feature_row(snapshot)
    row.update(
        {
            "review_budget": float(review_budget),
            "budget_tight": float(review_budget <= 3),
            "fragile_signal_count": float(
                max(rescue_decision.fragile_signal_count, shortlist_decision.fragile_signal_count)
            ),
            "rescue_conformal_probability": float(rescue_conformal_probability or 0.0),
            "rescue_wide_probability": float(rescue_wide_probability or 0.0),
            "shortlist_conformal_probability": float(shortlist_conformal_probability or 0.0),
            "shortlist_wide_probability": float(shortlist_wide_probability or 0.0),
            "rescue_budget_tight": float(rescue_decision.budget_tight),
            "shortlist_budget_tight": float(shortlist_decision.budget_tight),
            "rescue_defer_candidate": float(rescue_decision.defer_to_human),
            "shortlist_defer_candidate": float(shortlist_decision.defer_to_human),
            "rescue_preserve_candidate": float(rescue_decision.control_action == "preserve"),
            "shortlist_prune_candidate": float(shortlist_decision.control_action == "prune"),
        }
    )
    return row


def select_profile(
    *,
    shortlist_probability: float,
    snapshot: AdaptiveSupportSnapshot,
    rescue_decision: AdaptiveDecodingDecision,
    shortlist_decision: AdaptiveDecodingDecision,
    config: AdaptiveDecodingConfig,
) -> ProfileSelectorDecision:
    selected_profile = (
        "shortlist_first"
        if shortlist_probability >= config.selector_decision_threshold
        else "rescue_first"
    )
    probability_margin = abs(shortlist_probability - config.selector_decision_threshold)
    fragile_signal_count = max(
        rescue_decision.fragile_signal_count,
        shortlist_decision.fragile_signal_count,
    )
    direct_defer = bool(
        config.selector_enable_direct_defer
        and config.enable_defer_to_human
        and config.review_budget_k <= config.tight_review_budget_threshold
        and fragile_signal_count >= config.defer_min_fragile_signals
        and probability_margin <= config.selector_defer_margin
    )
    if direct_defer:
        decision_reason = "selector_uncertain_fragile_budget"
    elif selected_profile == "shortlist_first":
        decision_reason = "selector_prefers_shortlist_profile"
    else:
        decision_reason = "selector_prefers_rescue_profile"
    return ProfileSelectorDecision(
        selected_profile=selected_profile,
        shortlist_probability=float(shortlist_probability),
        decision_reason=decision_reason,
        direct_defer=direct_defer,
        probability_margin=float(probability_margin),
    )


def selector_feature_names() -> tuple[list[str], list[str]]:
    continuous_features = [
        "mean_confusion_entropy",
        "mean_confusion_set_size",
        "sequence_length",
        "length_support_count",
        "review_budget",
        "fragile_signal_count",
        "rescue_conformal_probability",
        "rescue_wide_probability",
        "shortlist_conformal_probability",
        "shortlist_wide_probability",
    ]
    binary_features = [
        "limited_support",
        "is_calibrated",
        "conformal_available",
        "budget_tight",
        "rescue_budget_tight",
        "shortlist_budget_tight",
        "rescue_defer_candidate",
        "shortlist_defer_candidate",
        "rescue_preserve_candidate",
        "shortlist_prune_candidate",
    ]
    return continuous_features, binary_features
