from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decipherlab.structured_uncertainty.confusion_network import ConfusionNetwork


@dataclass(frozen=True)
class DecodedSequence:
    symbols: list[str]
    total_score: float
    visual_score: float
    structural_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "total_score": self.total_score,
            "visual_score": self.visual_score,
            "structural_score": self.structural_score,
        }


@dataclass
class BeamDecodingResult:
    sequences: list[DecodedSequence]
    method: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> DecodedSequence:
        return self.sequences[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "diagnostics": self.diagnostics,
        }


@dataclass
class BigramTransitionModel:
    symbols: list[str]
    start_log_probabilities: dict[str, float]
    transition_log_probabilities: dict[str, dict[str, float]]
    diagnostics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        sequences: list[list[str]],
        smoothing: float,
    ) -> "BigramTransitionModel":
        support = sorted({symbol for sequence in sequences for symbol in sequence})
        if not support:
            raise ValueError("Transition model fitting requires at least one labeled sequence.")
        start_counts = {symbol: smoothing for symbol in support}
        transition_counts = {
            symbol: {other: smoothing for other in support}
            for symbol in support
        }
        for sequence in sequences:
            if not sequence:
                continue
            start_counts[sequence[0]] += 1.0
            for left, right in zip(sequence[:-1], sequence[1:]):
                transition_counts[left][right] += 1.0
        start_total = sum(start_counts.values())
        start_log_probabilities = {
            symbol: float(np.log(count / start_total))
            for symbol, count in start_counts.items()
        }
        transition_log_probabilities: dict[str, dict[str, float]] = {}
        for symbol, counts in transition_counts.items():
            total = sum(counts.values())
            transition_log_probabilities[symbol] = {
                other: float(np.log(count / total))
                for other, count in counts.items()
            }
        return cls(
            symbols=support,
            start_log_probabilities=start_log_probabilities,
            transition_log_probabilities=transition_log_probabilities,
            diagnostics={
                "support_size": float(len(support)),
                "training_sequence_count": float(len(sequences)),
                "smoothing": float(smoothing),
            },
        )

    def log_start(self, symbol: str) -> float:
        return self.start_log_probabilities.get(symbol, float(np.log(1.0e-8)))

    def log_transition(self, previous: str, current: str) -> float:
        return self.transition_log_probabilities.get(previous, {}).get(current, float(np.log(1.0e-8)))


def _rank_key(score: float, length: int, length_normalize: bool) -> float:
    if not length_normalize or length <= 0:
        return score
    return score / length


def greedy_decode_confusion_network(network: ConfusionNetwork) -> BeamDecodingResult:
    symbols = [position.candidate_ids[0] for position in network.positions]
    visual_score = float(np.sum([position.log_probabilities[0] for position in network.positions])) if network.positions else 0.0
    decoded = DecodedSequence(
        symbols=symbols,
        total_score=visual_score,
        visual_score=visual_score,
        structural_score=0.0,
    )
    return BeamDecodingResult(
        sequences=[decoded],
        method="fixed_greedy",
        diagnostics={"beam_width": 1.0},
    )


def beam_decode_confusion_network(
    network: ConfusionNetwork,
    transition_model: BigramTransitionModel,
    beam_width: int,
    lm_weight: float,
    top_k_sequences: int,
    length_normalize: bool,
) -> BeamDecodingResult:
    if not network.positions:
        return BeamDecodingResult(sequences=[], method="uncertainty_beam")

    beams: list[DecodedSequence] = []
    first_position = network.positions[0]
    for candidate, log_probability in zip(first_position.candidate_ids, first_position.log_probabilities):
        structural_score = lm_weight * transition_model.log_start(candidate)
        visual_score = float(log_probability)
        beams.append(
            DecodedSequence(
                symbols=[candidate],
                total_score=visual_score + structural_score,
                visual_score=visual_score,
                structural_score=structural_score,
            )
        )
    beams = sorted(
        beams,
        key=lambda beam: _rank_key(beam.total_score, len(beam.symbols), length_normalize),
        reverse=True,
    )[:beam_width]

    for position in network.positions[1:]:
        expanded: list[DecodedSequence] = []
        for beam in beams:
            previous = beam.symbols[-1]
            for candidate, log_probability in zip(position.candidate_ids, position.log_probabilities):
                structural_increment = lm_weight * transition_model.log_transition(previous, candidate)
                visual_increment = float(log_probability)
                expanded.append(
                    DecodedSequence(
                        symbols=beam.symbols + [candidate],
                        total_score=beam.total_score + visual_increment + structural_increment,
                        visual_score=beam.visual_score + visual_increment,
                        structural_score=beam.structural_score + structural_increment,
                    )
                )
        beams = sorted(
            expanded,
            key=lambda beam: _rank_key(beam.total_score, len(beam.symbols), length_normalize),
            reverse=True,
        )[:beam_width]
    ranked = sorted(
        beams,
        key=lambda beam: _rank_key(beam.total_score, len(beam.symbols), length_normalize),
        reverse=True,
    )[:top_k_sequences]
    return BeamDecodingResult(
        sequences=ranked,
        method="uncertainty_beam",
        diagnostics={
            "beam_width": float(beam_width),
            "lm_weight": float(lm_weight),
            "top_k_sequences": float(top_k_sequences),
        },
    )
