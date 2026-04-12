from __future__ import annotations

from decipherlab.decoding.beam_search import BeamDecodingResult, DecodedSequence
from decipherlab.sequence.metrics import (
    sequence_metric_bundle,
    sequence_top_k_recovery_at_budget,
    shortlist_utility_score,
)


def _result() -> BeamDecodingResult:
    return BeamDecodingResult(
        method="uncertainty_beam",
        sequences=[
            DecodedSequence(["A", "B"], total_score=0.0, visual_score=0.0, structural_score=0.0),
            DecodedSequence(["A", "C"], total_score=-0.2, visual_score=-0.1, structural_score=-0.1),
            DecodedSequence(["B", "C"], total_score=-0.3, visual_score=-0.2, structural_score=-0.1),
        ],
    )


def test_sequence_top_k_recovery_at_budget_respects_shortlist_size():
    result = _result()

    assert sequence_top_k_recovery_at_budget(result, ["A", "B"], budget=1) == 1.0
    assert sequence_top_k_recovery_at_budget(result, ["A", "C"], budget=1) == 0.0
    assert sequence_top_k_recovery_at_budget(result, ["A", "C"], budget=2) == 1.0
    assert sequence_top_k_recovery_at_budget(result, ["B", "C"], budget=2) == 0.0
    assert sequence_top_k_recovery_at_budget(result, ["B", "C"], budget=3) == 1.0


def test_shortlist_utility_score_prioritizes_small_budget_recall():
    result = _result()

    assert shortlist_utility_score(result, ["A", "B"]) == 1.0
    assert shortlist_utility_score(result, ["B", "C"]) == 0.5


def test_sequence_metric_bundle_includes_shortlist_metrics():
    metrics = sequence_metric_bundle(_result(), ["B", "C"])

    assert metrics["sequence_shortlist_recall_at_2"] == 0.0
    assert metrics["sequence_shortlist_recall_at_3"] == 1.0
    assert metrics["sequence_shortlist_recall_at_5"] == 1.0
    assert metrics["sequence_shortlist_utility"] == 0.5
