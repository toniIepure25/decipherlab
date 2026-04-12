from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decipherlab.config import AdaptiveDecodingConfig
from decipherlab.sequence.real_downstream import SupportedNGramInventory, TranscriptBank
from decipherlab.structured_uncertainty.confusion_network import ConfusionNetwork


@dataclass(frozen=True)
class AdaptiveSupportSnapshot:
    mean_confusion_entropy: float
    mean_confusion_set_size: float
    sequence_length: int
    posterior_strategy: str
    length_support_count: int
    conformal_available: bool


@dataclass(frozen=True)
class AdaptiveDecodingDecision:
    selected_method: str
    beam_width: int
    decision_reason: str
    control_action: str
    limited_support: bool
    low_entropy: bool
    high_entropy: bool
    compact_set: bool
    diffuse_set: bool
    defer_to_human: bool = False
    review_budget: int = 3
    budget_tight: bool = False
    fragile_signal_count: int = 0
    operating_profile: str = "fixed_policy"
    profile_reason: str = "fixed_policy"

    def to_row(self) -> dict[str, Any]:
        return {
            "adaptive_selected_method": self.selected_method,
            "adaptive_beam_width": self.beam_width,
            "adaptive_decision_reason": self.decision_reason,
            "adaptive_control_action": self.control_action,
            "adaptive_defer_to_human": float(self.defer_to_human),
            "adaptive_review_budget": float(self.review_budget),
            "adaptive_budget_tight": float(self.budget_tight),
            "adaptive_fragile_signal_count": float(self.fragile_signal_count),
            "adaptive_operating_profile": self.operating_profile,
            "adaptive_profile_reason": self.profile_reason,
            "adaptive_limited_support": float(self.limited_support),
            "adaptive_low_entropy": float(self.low_entropy),
            "adaptive_high_entropy": float(self.high_entropy),
            "adaptive_compact_set": float(self.compact_set),
            "adaptive_diffuse_set": float(self.diffuse_set),
        }


def support_feature_row(snapshot: AdaptiveSupportSnapshot) -> dict[str, float]:
    return {
        "mean_confusion_entropy": float(snapshot.mean_confusion_entropy),
        "mean_confusion_set_size": float(snapshot.mean_confusion_set_size),
        "sequence_length": float(snapshot.sequence_length),
        "length_support_count": float(snapshot.length_support_count),
        "limited_support": float(0 < snapshot.length_support_count <= 4),
        "is_calibrated": float(snapshot.posterior_strategy == "calibrated_classifier"),
        "conformal_available": float(snapshot.conformal_available),
    }


def length_support_count(
    downstream_resource: TranscriptBank | SupportedNGramInventory | None,
    sequence_length: int,
) -> int:
    if downstream_resource is None:
        return 0
    counts = downstream_resource.metadata.get("length_support_counts", {})
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(str(sequence_length), 0) or 0)


def build_support_snapshot(
    *,
    network: ConfusionNetwork,
    posterior_strategy: str,
    sequence_length: int,
    downstream_resource: TranscriptBank | SupportedNGramInventory | None,
    conformal_available: bool,
) -> AdaptiveSupportSnapshot:
    return AdaptiveSupportSnapshot(
        mean_confusion_entropy=float(network.mean_entropy()),
        mean_confusion_set_size=float(network.average_set_size()),
        sequence_length=int(sequence_length),
        posterior_strategy=str(posterior_strategy),
        length_support_count=length_support_count(downstream_resource, sequence_length),
        conformal_available=bool(conformal_available),
    )


def decide_support_aware_method(
    snapshot: AdaptiveSupportSnapshot,
    config: AdaptiveDecodingConfig,
    *,
    default_beam_width: int,
) -> AdaptiveDecodingDecision:
    if config.policy != "support_aware_rule":
        raise ValueError(f"Unsupported adaptive decoding policy: {config.policy}")

    low_entropy = snapshot.mean_confusion_entropy <= config.conformal_entropy_threshold
    high_entropy = snapshot.mean_confusion_entropy >= config.raw_entropy_threshold
    compact_set = snapshot.mean_confusion_set_size <= config.conformal_set_size_threshold
    useful_to_prune = snapshot.mean_confusion_set_size >= config.minimum_set_size_for_conformal
    diffuse_set = snapshot.mean_confusion_set_size >= config.raw_set_size_threshold
    limited_support = 0 < snapshot.length_support_count <= config.low_support_length_count_threshold
    fragile_signal_count = int(high_entropy) + int(diffuse_set) + int(limited_support)
    budget_tight = config.review_budget_k <= config.tight_review_budget_threshold

    selected_method = "uncertainty_beam"
    beam_width = default_beam_width
    decision_reason = "default_uncertainty"
    control_action = "preserve"

    if high_entropy or diffuse_set:
        selected_method = "uncertainty_beam"
        beam_width = max(default_beam_width, config.wide_beam_width)
        decision_reason = "high_entropy_preserve"
        control_action = "preserve"
    elif (
        snapshot.conformal_available
        and low_entropy
        and compact_set
        and useful_to_prune
        and (
            limited_support
            or (
                config.prefer_conformal_for_calibrated
                and snapshot.posterior_strategy == "calibrated_classifier"
            )
        )
    ):
        selected_method = "conformal_beam"
        beam_width = min(default_beam_width, config.narrow_beam_width)
        decision_reason = "low_entropy_conformal"
        control_action = "prune"

    return AdaptiveDecodingDecision(
        selected_method=selected_method,
        beam_width=beam_width,
        decision_reason=decision_reason,
        control_action=control_action,
        operating_profile="rule",
        profile_reason="rule_policy",
        limited_support=limited_support,
        low_entropy=low_entropy,
        high_entropy=high_entropy,
        compact_set=compact_set,
        diffuse_set=diffuse_set,
        defer_to_human=False,
        review_budget=config.review_budget_k,
        budget_tight=budget_tight,
        fragile_signal_count=fragile_signal_count,
    )


def resolve_operating_profile(
    config: AdaptiveDecodingConfig,
) -> tuple[str, str]:
    if config.policy != "support_aware_profiled_gate":
        raise ValueError(f"Unsupported profiled policy: {config.policy}")
    if config.operating_profile == "rescue_first":
        return "rescue_first", "explicit_rescue_first"
    if config.operating_profile == "shortlist_first":
        return "shortlist_first", "explicit_shortlist_first"
    raise ValueError(f"Unsupported operating profile: {config.operating_profile}")
