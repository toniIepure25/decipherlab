from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from decipherlab.decoding.beam_search import BigramTransitionModel
from decipherlab.models import SequenceExample


@dataclass
class ProcessFamilyClassifier:
    family_models: dict[str, BigramTransitionModel]

    @classmethod
    def fit(
        cls,
        examples: list[SequenceExample],
        smoothing: float,
    ) -> "ProcessFamilyClassifier | None":
        grouped: dict[str, list[list[str]]] = {}
        for example in examples:
            if example.family is None or not example.has_symbol_labels:
                continue
            grouped.setdefault(example.family, []).append(
                [symbol for symbol in example.observed_symbols if symbol is not None]
            )
        if len(grouped) < 2:
            return None
        return cls(
            family_models={
                family: BigramTransitionModel.fit(sequences=sequences, smoothing=smoothing)
                for family, sequences in grouped.items()
            }
        )

    def score_sequence(self, sequence: list[str], model: BigramTransitionModel) -> float:
        if not sequence:
            return float("-inf")
        score = model.log_start(sequence[0])
        for left, right in zip(sequence[:-1], sequence[1:]):
            score += model.log_transition(left, right)
        return float(score / len(sequence))

    def predict(self, sequence: list[str]) -> tuple[str, dict[str, float]]:
        scores = {
            family: self.score_sequence(sequence, model)
            for family, model in self.family_models.items()
        }
        predicted = max(scores.items(), key=lambda item: item[1])[0]
        return predicted, scores


def family_identification_payload(
    classifier: ProcessFamilyClassifier | None,
    decoded_sequence: list[str],
    true_family: str | None,
) -> dict[str, Any]:
    if classifier is None or true_family is None:
        return {
            "family_identification_supported": False,
            "predicted_family": None,
            "family_identification_accuracy": None,
            "family_scores": {},
        }
    predicted_family, scores = classifier.predict(decoded_sequence)
    return {
        "family_identification_supported": True,
        "predicted_family": predicted_family,
        "family_identification_accuracy": float(predicted_family == true_family),
        "family_scores": scores,
    }
