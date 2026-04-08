from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np

from decipherlab.config import RiskControlConfig
from decipherlab.structured_uncertainty.confusion_network import (
    ConfusionNetwork,
    ConfusionNetworkPosition,
)


def _labeled_probability_rows(
    networks: list[ConfusionNetwork],
    labels: list[list[str | None]],
) -> list[float]:
    retained_probabilities: list[float] = []
    for network, row_labels in zip(networks, labels):
        for position, true_symbol in zip(network.positions, row_labels):
            if true_symbol is None:
                continue
            retained_probabilities.append(position.to_distribution().get(true_symbol, 0.0))
    return retained_probabilities


@dataclass
class SplitConformalSetPredictor:
    alpha: float
    threshold_probability: float
    quantile_score: float
    calibration_sample_count: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        networks: list[ConfusionNetwork],
        labels: list[list[str | None]],
        config: RiskControlConfig,
    ) -> "SplitConformalSetPredictor":
        retained_probabilities = _labeled_probability_rows(networks, labels)
        if not retained_probabilities:
            raise ValueError("Split conformal fitting requires at least one labeled validation position.")
        nonconformity = np.asarray([1.0 - probability for probability in retained_probabilities], dtype=float)
        quantile_rank = ceil((len(nonconformity) + 1) * (1.0 - config.alpha))
        quantile_rank = min(len(nonconformity), max(1, quantile_rank))
        quantile_score = float(np.sort(nonconformity)[quantile_rank - 1])
        threshold_probability = max(0.0, 1.0 - quantile_score)
        return cls(
            alpha=config.alpha,
            threshold_probability=threshold_probability,
            quantile_score=quantile_score,
            calibration_sample_count=len(retained_probabilities),
            diagnostics={
                "alpha": config.alpha,
                "threshold_probability": threshold_probability,
                "quantile_score": quantile_score,
            },
        )

    def apply(
        self,
        network: ConfusionNetwork,
        config: RiskControlConfig,
    ) -> ConfusionNetwork:
        filtered_positions: list[ConfusionNetworkPosition] = []
        for position in network.positions:
            probabilities = position.probabilities()
            keep_indices = [index for index, probability in enumerate(probabilities) if probability >= self.threshold_probability]
            if config.max_set_size is not None and len(keep_indices) > config.max_set_size:
                keep_indices = keep_indices[: config.max_set_size]
            if len(keep_indices) < config.min_set_size:
                keep_indices = list(range(min(config.min_set_size, len(position.candidate_ids))))
            if not keep_indices and config.include_top1_fallback and position.candidate_ids:
                keep_indices = [0]
            retained_candidates = [position.candidate_ids[index] for index in keep_indices]
            retained_probabilities = probabilities[np.asarray(keep_indices, dtype=int)]
            retained_probabilities = retained_probabilities / np.sum(retained_probabilities)
            filtered_positions.append(
                ConfusionNetworkPosition(
                    position=position.position,
                    candidate_ids=retained_candidates,
                    log_probabilities=np.log(retained_probabilities),
                    source_entropy=position.source_entropy,
                    retained_probability_mass=float(np.sum(probabilities[np.asarray(keep_indices, dtype=int)])),
                    metadata=position.metadata
                    | {
                        "conformal_threshold_probability": self.threshold_probability,
                        "filtered_candidate_count": len(retained_candidates),
                    },
                )
            )
        return ConfusionNetwork(
            positions=filtered_positions,
            boundary_probabilities=network.boundary_probabilities,
            metadata=network.metadata
            | {
                "risk_control_method": "split_conformal",
                "threshold_probability": self.threshold_probability,
                "alpha": self.alpha,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold_probability": self.threshold_probability,
            "quantile_score": self.quantile_score,
            "calibration_sample_count": self.calibration_sample_count,
            "diagnostics": self.diagnostics,
        }


def summarize_prediction_sets(
    networks: list[ConfusionNetwork],
    labels: list[list[str | None]],
) -> dict[str, float | None]:
    total = 0
    covered = 0
    singleton = 0
    rescue = 0
    set_sizes: list[int] = []
    for network, row_labels in zip(networks, labels):
        for position, true_symbol in zip(network.positions, row_labels):
            if true_symbol is None:
                continue
            total += 1
            set_sizes.append(len(position.candidate_ids))
            if len(position.candidate_ids) == 1:
                singleton += 1
            if true_symbol in position.candidate_ids:
                covered += 1
                if position.candidate_ids[0] != true_symbol:
                    rescue += 1
    if total == 0:
        return {
            "prediction_set_coverage": None,
            "prediction_set_avg_size": None,
            "prediction_set_singleton_rate": None,
            "prediction_set_rescue_rate": None,
        }
    return {
        "prediction_set_coverage": covered / total,
        "prediction_set_avg_size": float(np.mean(set_sizes)) if set_sizes else None,
        "prediction_set_singleton_rate": singleton / total,
        "prediction_set_rescue_rate": rescue / total,
    }
