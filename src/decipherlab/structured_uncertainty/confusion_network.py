from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decipherlab.config import StructuredUncertaintyConfig
from decipherlab.models import TranscriptionPosterior


@dataclass(frozen=True)
class ConfusionNetworkPosition:
    position: int
    candidate_ids: list[str]
    log_probabilities: np.ndarray
    source_entropy: float
    retained_probability_mass: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.log_probabilities.ndim != 1:
            raise ValueError("Confusion network position probabilities must be 1D.")
        if len(self.candidate_ids) != len(self.log_probabilities):
            raise ValueError("Candidate ids and log-probabilities must align per position.")

    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_probabilities)

    def entropy(self) -> float:
        probabilities = self.probabilities()
        return float(-np.sum(probabilities * self.log_probabilities))

    def to_distribution(self) -> dict[str, float]:
        return {
            candidate: float(probability)
            for candidate, probability in zip(self.candidate_ids, self.probabilities())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "candidate_ids": self.candidate_ids,
            "probabilities": self.probabilities().tolist(),
            "source_entropy": self.source_entropy,
            "retained_probability_mass": self.retained_probability_mass,
            "metadata": self.metadata,
        }


@dataclass
class ConfusionNetwork:
    positions: list[ConfusionNetworkPosition]
    boundary_probabilities: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def hard_sequence(self) -> list[str]:
        return [position.candidate_ids[0] for position in self.positions]

    def entropy_per_position(self) -> np.ndarray:
        return np.asarray([position.entropy() for position in self.positions], dtype=float)

    def mean_entropy(self) -> float:
        if not self.positions:
            return 0.0
        return float(np.mean(self.entropy_per_position()))

    def average_set_size(self) -> float:
        if not self.positions:
            return 0.0
        return float(np.mean([len(position.candidate_ids) for position in self.positions]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": [position.to_dict() for position in self.positions],
            "boundary_probabilities": None
            if self.boundary_probabilities is None
            else self.boundary_probabilities.tolist(),
            "metadata": self.metadata,
            "mean_entropy": self.mean_entropy(),
            "average_set_size": self.average_set_size(),
        }


def posterior_to_confusion_network(
    posterior: TranscriptionPosterior,
    config: StructuredUncertaintyConfig,
) -> ConfusionNetwork:
    positions: list[ConfusionNetworkPosition] = []
    source_entropies = posterior.entropy_per_position()
    for index, (row_candidates, row_log_probabilities, source_entropy) in enumerate(
        zip(posterior.candidate_ids, posterior.log_probabilities, source_entropies)
    ):
        row_probabilities = np.exp(row_log_probabilities)
        cumulative_mass = 0.0
        kept_candidates: list[str] = []
        kept_probabilities: list[float] = []
        for candidate, probability in zip(row_candidates, row_probabilities):
            if probability < config.min_probability and kept_candidates:
                continue
            kept_candidates.append(candidate)
            kept_probabilities.append(float(probability))
            cumulative_mass += float(probability)
            if (
                cumulative_mass >= config.cumulative_probability_mass
                or len(kept_candidates) >= config.max_candidates_per_position
            ):
                break
        if not kept_candidates and config.include_top1_fallback and row_candidates:
            kept_candidates = [row_candidates[0]]
            kept_probabilities = [float(row_probabilities[0])]
            cumulative_mass = kept_probabilities[0]
        normalized = np.asarray(kept_probabilities, dtype=float)
        normalized /= np.sum(normalized)
        positions.append(
            ConfusionNetworkPosition(
                position=index,
                candidate_ids=kept_candidates,
                log_probabilities=np.log(normalized),
                source_entropy=float(source_entropy),
                retained_probability_mass=float(cumulative_mass),
                metadata={"source_candidate_count": len(row_candidates)},
            )
        )
    return ConfusionNetwork(
        positions=positions,
        boundary_probabilities=posterior.boundary_probabilities,
        metadata={
            "representation": config.representation,
            "max_candidates_per_position": config.max_candidates_per_position,
            "cumulative_probability_mass": config.cumulative_probability_mass,
        },
    )


def confusion_network_to_posterior(network: ConfusionNetwork) -> TranscriptionPosterior:
    if not network.positions:
        return TranscriptionPosterior(candidate_ids=[], log_probabilities=np.empty((0, 0), dtype=float))
    max_width = max(len(position.candidate_ids) for position in network.positions)
    padded_candidates: list[list[str]] = []
    padded_log_probabilities = np.full((len(network.positions), max_width), fill_value=np.log(1.0e-8), dtype=float)
    for row_index, position in enumerate(network.positions):
        padding = [f"__pad_{row_index}_{column_index}" for column_index in range(max_width - len(position.candidate_ids))]
        padded_candidates.append(list(position.candidate_ids) + padding)
        padded_log_probabilities[row_index, : len(position.candidate_ids)] = position.log_probabilities
        if len(position.candidate_ids) < max_width:
            probabilities = np.exp(padded_log_probabilities[row_index])
            probabilities /= np.sum(probabilities)
            padded_log_probabilities[row_index] = np.log(probabilities)
    return TranscriptionPosterior(
        candidate_ids=padded_candidates,
        log_probabilities=padded_log_probabilities,
        boundary_probabilities=network.boundary_probabilities,
    )
