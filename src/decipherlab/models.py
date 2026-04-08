from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


@dataclass(frozen=True)
class GlyphCrop:
    position: int
    image: np.ndarray
    true_symbol: str | None
    variant_id: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class SequenceExample:
    example_id: str
    family: str | None
    glyphs: list[GlyphCrop]
    plaintext: str | None
    observed_symbols: list[str | None]
    split: str = "unassigned"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_length(self) -> int:
        return len(self.glyphs)

    @property
    def has_symbol_labels(self) -> bool:
        return all(symbol is not None for symbol in self.observed_symbols)

    @property
    def labeled_symbol_count(self) -> int:
        return sum(symbol is not None for symbol in self.observed_symbols)


@dataclass(frozen=True)
class DatasetCollection:
    dataset_name: str
    examples: list[SequenceExample]
    manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_names(self) -> list[str]:
        return sorted({example.split for example in self.examples})

    def get_split(self, split: str) -> list[SequenceExample]:
        if split == "all":
            return list(self.examples)
        return [example for example in self.examples if example.split == split]

    def count_examples(self, split: str = "all") -> int:
        return len(self.get_split(split))

    def labeled_examples(self, split: str = "all") -> list[SequenceExample]:
        return [example for example in self.get_split(split) if example.has_symbol_labels]

    def families_available(self, split: str = "all") -> bool:
        return any(example.family is not None for example in self.get_split(split))


@dataclass
class GlyphClusterResult:
    labels: np.ndarray
    inventory: list[str]
    centroids: np.ndarray
    feature_matrix: np.ndarray
    estimated_cluster_count: int
    silhouette_score: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_cluster_count": self.estimated_cluster_count,
            "inventory": list(self.inventory),
            "silhouette_score": self.silhouette_score,
            "diagnostics": self.diagnostics,
        }


@dataclass
class TranscriptionPosterior:
    candidate_ids: list[list[str]]
    log_probabilities: np.ndarray
    boundary_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != self.log_probabilities.shape[0]:
            raise ValueError("Candidate rows and log-probabilities must align.")
        if self.log_probabilities.ndim != 2:
            raise ValueError("log_probabilities must be a 2D array.")

    @classmethod
    def from_scores(
        cls,
        support: list[str],
        scores: np.ndarray,
        top_k: int,
        floor_probability: float = 1.0e-6,
        boundary_probabilities: np.ndarray | None = None,
    ) -> "TranscriptionPosterior":
        top_k = min(top_k, scores.shape[1])
        top_indices = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]
        row_indices = np.arange(scores.shape[0])[:, None]
        top_scores = scores[row_indices, top_indices]
        top_probabilities = _stable_softmax(top_scores)
        top_probabilities = np.clip(top_probabilities, floor_probability, None)
        top_probabilities /= np.sum(top_probabilities, axis=1, keepdims=True)
        candidate_ids = [[support[index] for index in row] for row in top_indices]
        return cls(
            candidate_ids=candidate_ids,
            log_probabilities=np.log(top_probabilities),
            boundary_probabilities=boundary_probabilities,
        )

    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_probabilities)

    def hard_sequence(self) -> list[str]:
        return [candidates[0] for candidates in self.candidate_ids]

    def entropy_per_position(self) -> np.ndarray:
        probabilities = self.probabilities()
        return -np.sum(probabilities * self.log_probabilities, axis=1)

    def mean_entropy(self) -> float:
        return float(np.mean(self.entropy_per_position()))

    def support(self) -> list[str]:
        seen: list[str] = []
        for row in self.candidate_ids:
            for candidate in row:
                if candidate not in seen:
                    seen.append(candidate)
        return seen

    def iter_position_distributions(self) -> list[dict[str, float]]:
        probabilities = self.probabilities()
        distributions: list[dict[str, float]] = []
        for row_candidates, row_probabilities in zip(self.candidate_ids, probabilities):
            distributions.append(
                {
                    candidate: float(probability)
                    for candidate, probability in zip(row_candidates, row_probabilities)
                }
            )
        return distributions

    def collapsed(self) -> "TranscriptionPosterior":
        hard_ids = [[row[0]] for row in self.candidate_ids]
        hard_log_probabilities = np.zeros((len(hard_ids), 1), dtype=float)
        return TranscriptionPosterior(
            candidate_ids=hard_ids,
            log_probabilities=hard_log_probabilities,
            boundary_probabilities=self.boundary_probabilities,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": self.candidate_ids,
            "log_probabilities": self.log_probabilities.tolist(),
            "boundary_probabilities": None
            if self.boundary_probabilities is None
            else self.boundary_probabilities.tolist(),
            "hard_sequence": self.hard_sequence(),
            "mean_entropy": self.mean_entropy(),
        }


@dataclass
class TriageReport:
    family: str
    sequence_length: int
    alphabet_size_estimate: int
    unigram_entropy: float
    conditional_entropy: float
    index_of_coincidence: float
    repeat_rate: float
    compression_ratio: float
    adjacency_density: float
    mean_posterior_entropy: float
    null_repeat_rate: float
    null_compression_ratio: float
    routing_scores: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "sequence_length": self.sequence_length,
            "alphabet_size_estimate": self.alphabet_size_estimate,
            "unigram_entropy": self.unigram_entropy,
            "conditional_entropy": self.conditional_entropy,
            "index_of_coincidence": self.index_of_coincidence,
            "repeat_rate": self.repeat_rate,
            "compression_ratio": self.compression_ratio,
            "adjacency_density": self.adjacency_density,
            "mean_posterior_entropy": self.mean_posterior_entropy,
            "null_repeat_rate": self.null_repeat_rate,
            "null_compression_ratio": self.null_compression_ratio,
            "routing_scores": self.routing_scores,
            "diagnostics": self.diagnostics,
        }


@dataclass
class HypothesisEvidence:
    family: str
    score: float
    probability: float
    rationale: list[str]
    diagnostics: dict[str, float]
    caveats: list[str] = field(default_factory=list)
    heuristic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "score": self.score,
            "probability": self.probability,
            "rationale": self.rationale,
            "diagnostics": self.diagnostics,
            "caveats": self.caveats,
            "heuristic": self.heuristic,
        }


@dataclass
class HypothesisRanking:
    evidences: list[HypothesisEvidence]

    @property
    def best(self) -> HypothesisEvidence:
        return self.evidences[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_family": self.best.family,
            "evidences": [evidence.to_dict() for evidence in self.evidences],
        }
