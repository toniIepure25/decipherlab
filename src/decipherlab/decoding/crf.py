from __future__ import annotations

from typing import Any

import numpy as np

from decipherlab.decoding.beam_search import BeamDecodingResult, BigramTransitionModel, DecodedSequence
from decipherlab.structured_uncertainty.confusion_network import ConfusionNetwork


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if np.isneginf(maximum):
        return maximum
    shifted = np.asarray([value - maximum for value in values], dtype=float)
    return float(maximum + np.log(np.sum(np.exp(shifted))))


def crf_viterbi_decode_confusion_network(
    network: ConfusionNetwork,
    transition_model: BigramTransitionModel,
    lm_weight: float,
) -> BeamDecodingResult:
    """Decode a confusion network with exact dynamic programming.

    This is a CRF-style baseline rather than a trained CRF proper: unary potentials
    come directly from the visual posterior log-probabilities, while pairwise
    transition potentials come from smoothed count-derived bigram log-probabilities.
    The value of this decoder is exact inference over those explicit factors.
    """

    if not network.positions:
        return BeamDecodingResult(sequences=[], method="uncertainty_crf_viterbi")

    first_position = network.positions[0]
    best_scores: dict[str, float] = {}
    best_visual_scores: dict[str, float] = {}
    best_structural_scores: dict[str, float] = {}
    forward_scores: dict[str, float] = {}
    backpointers: list[dict[str, str | None]] = []

    first_backpointers: dict[str, str | None] = {}
    for candidate, log_probability in zip(first_position.candidate_ids, first_position.log_probabilities):
        visual_score = float(log_probability)
        structural_score = lm_weight * transition_model.log_start(candidate)
        total_score = visual_score + structural_score
        best_scores[candidate] = total_score
        best_visual_scores[candidate] = visual_score
        best_structural_scores[candidate] = structural_score
        forward_scores[candidate] = total_score
        first_backpointers[candidate] = None
    backpointers.append(first_backpointers)

    for position in network.positions[1:]:
        current_scores: dict[str, float] = {}
        current_visual_scores: dict[str, float] = {}
        current_structural_scores: dict[str, float] = {}
        current_forward_scores: dict[str, float] = {}
        current_backpointers: dict[str, str | None] = {}
        for candidate, log_probability in zip(position.candidate_ids, position.log_probabilities):
            visual_increment = float(log_probability)
            predecessor_scores: list[tuple[str, float, float]] = []
            forward_candidates: list[float] = []
            for previous_candidate, previous_score in best_scores.items():
                structural_increment = lm_weight * transition_model.log_transition(previous_candidate, candidate)
                predecessor_scores.append((previous_candidate, previous_score + structural_increment, structural_increment))
                forward_candidates.append(forward_scores[previous_candidate] + structural_increment)

            if not predecessor_scores:
                continue
            best_previous, best_total_without_unary, best_structural_increment = max(
                predecessor_scores,
                key=lambda item: item[1],
            )
            current_scores[candidate] = best_total_without_unary + visual_increment
            current_visual_scores[candidate] = best_visual_scores[best_previous] + visual_increment
            current_structural_scores[candidate] = (
                best_structural_scores[best_previous] + best_structural_increment
            )
            current_forward_scores[candidate] = visual_increment + _logsumexp(forward_candidates)
            current_backpointers[candidate] = best_previous
        best_scores = current_scores
        best_visual_scores = current_visual_scores
        best_structural_scores = current_structural_scores
        forward_scores = current_forward_scores
        backpointers.append(current_backpointers)

    best_final_candidate = max(best_scores.items(), key=lambda item: item[1])[0]
    best_sequence = [best_final_candidate]
    current = best_final_candidate
    for position_index in range(len(network.positions) - 1, 0, -1):
        previous = backpointers[position_index][current]
        if previous is None:
            break
        best_sequence.append(previous)
        current = previous
    best_sequence.reverse()

    decoded = DecodedSequence(
        symbols=best_sequence,
        total_score=best_scores[best_final_candidate],
        visual_score=best_visual_scores[best_final_candidate],
        structural_score=best_structural_scores[best_final_candidate],
    )
    log_partition = _logsumexp(list(forward_scores.values()))
    return BeamDecodingResult(
        sequences=[decoded],
        method="uncertainty_crf_viterbi",
        diagnostics={
            "lm_weight": float(lm_weight),
            "exact_dynamic_programming": 1.0,
            "log_partition": float(log_partition),
            "decoder_family": 3.0,
        },
    )
