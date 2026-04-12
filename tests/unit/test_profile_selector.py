from __future__ import annotations

from decipherlab.config import AdaptiveDecodingConfig
from decipherlab.sequence.adaptive_decoder import AdaptiveDecodingDecision, AdaptiveSupportSnapshot
from decipherlab.sequence.profile_selector import (
    build_profile_selector_feature_row,
    select_profile,
)


def _decision(*, profile: str, action: str, defer: bool = False, fragile_signal_count: int = 0) -> AdaptiveDecodingDecision:
    return AdaptiveDecodingDecision(
        selected_method="uncertainty_beam",
        beam_width=8,
        decision_reason="test",
        control_action=action,
        limited_support=False,
        low_entropy=False,
        high_entropy=False,
        compact_set=False,
        diffuse_set=False,
        defer_to_human=defer,
        review_budget=2,
        budget_tight=True,
        fragile_signal_count=fragile_signal_count,
        operating_profile=profile,
        profile_reason="test_profile",
    )


def test_profile_selector_feature_row_includes_budget_and_profile_signals():
    row = build_profile_selector_feature_row(
        snapshot=AdaptiveSupportSnapshot(
            mean_confusion_entropy=1.1,
            mean_confusion_set_size=2.6,
            sequence_length=5,
            posterior_strategy="calibrated_classifier",
            length_support_count=2,
            conformal_available=True,
        ),
        review_budget=3,
        rescue_decision=_decision(profile="rescue_first", action="preserve", fragile_signal_count=2),
        shortlist_decision=_decision(profile="shortlist_first", action="prune", fragile_signal_count=1),
        rescue_conformal_probability=0.2,
        rescue_wide_probability=0.8,
        shortlist_conformal_probability=0.7,
        shortlist_wide_probability=0.3,
    )

    assert row["review_budget"] == 3.0
    assert row["fragile_signal_count"] == 2.0
    assert row["rescue_preserve_candidate"] == 1.0
    assert row["shortlist_prune_candidate"] == 1.0
    assert row["shortlist_conformal_probability"] == 0.7


def test_profile_selector_prefers_shortlist_but_can_defer_when_uncertain_and_fragile():
    config = AdaptiveDecodingConfig(
        enabled=True,
        policy="support_aware_profile_selector",
        review_budget_k=2,
        selector_decision_threshold=0.5,
        selector_defer_margin=0.08,
        defer_min_fragile_signals=2,
    )

    decision = select_profile(
        shortlist_probability=0.52,
        snapshot=AdaptiveSupportSnapshot(
            mean_confusion_entropy=1.3,
            mean_confusion_set_size=3.4,
            sequence_length=6,
            posterior_strategy="cluster_distance",
            length_support_count=1,
            conformal_available=True,
        ),
        rescue_decision=_decision(profile="rescue_first", action="preserve", fragile_signal_count=2),
        shortlist_decision=_decision(profile="shortlist_first", action="prune", fragile_signal_count=2),
        config=config,
    )

    assert decision.selected_profile == "shortlist_first"
    assert decision.direct_defer is True
    assert decision.decision_reason == "selector_uncertain_fragile_budget"
