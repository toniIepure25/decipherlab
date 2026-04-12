from __future__ import annotations

from decipherlab.config import AdaptiveDecodingConfig
from decipherlab.sequence.adaptive_decoder import (
    AdaptiveSupportSnapshot,
    decide_support_aware_method,
    resolve_operating_profile,
)


def test_support_aware_rule_prefers_conformal_in_low_entropy_limited_support():
    decision = decide_support_aware_method(
        AdaptiveSupportSnapshot(
            mean_confusion_entropy=0.2,
            mean_confusion_set_size=1.5,
            sequence_length=4,
            posterior_strategy="calibrated_classifier",
            length_support_count=2,
            conformal_available=True,
        ),
        AdaptiveDecodingConfig(enabled=True),
        default_beam_width=8,
    )

    assert decision.selected_method == "conformal_beam"
    assert decision.beam_width == 4
    assert decision.decision_reason == "low_entropy_conformal"
    assert decision.control_action == "prune"
    assert decision.review_budget == 3


def test_support_aware_rule_prefers_raw_uncertainty_in_high_entropy_regime():
    decision = decide_support_aware_method(
        AdaptiveSupportSnapshot(
            mean_confusion_entropy=1.4,
            mean_confusion_set_size=4.2,
            sequence_length=4,
            posterior_strategy="cluster_distance",
            length_support_count=8,
            conformal_available=True,
        ),
        AdaptiveDecodingConfig(enabled=True),
        default_beam_width=8,
    )

    assert decision.selected_method == "uncertainty_beam"
    assert decision.beam_width == 12
    assert decision.decision_reason == "high_entropy_preserve"
    assert decision.control_action == "preserve"


def test_profiled_policy_exposes_explicit_operating_profiles():
    config = AdaptiveDecodingConfig(enabled=True, policy="support_aware_profiled_gate", operating_profile="shortlist_first")

    profile_mode, profile_reason = resolve_operating_profile(config)

    assert profile_mode == "shortlist_first"
    assert profile_reason == "explicit_shortlist_first"
