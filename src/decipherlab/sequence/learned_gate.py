from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class LearnedGateModel:
    target_name: str
    continuous_features: list[str]
    binary_features: list[str]
    means: list[float]
    stds: list[float]
    coefficients: list[float]
    intercept: float
    training_accuracy: float
    positive_rate: float

    def predict_proba(self, row: dict[str, float | int | bool | None]) -> float:
        values: list[float] = []
        for feature, mean, std in zip(self.continuous_features, self.means, self.stds):
            value = float(row.get(feature, 0.0) or 0.0)
            values.append(0.0 if std == 0.0 else (value - mean) / std)
        for feature in self.binary_features:
            values.append(float(row.get(feature, 0.0) or 0.0))
        design = np.asarray([1.0] + values, dtype=float)
        weights = np.asarray([self.intercept] + self.coefficients, dtype=float)
        return float(_sigmoid(design @ weights))

    def coefficient_rows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "target_name": self.target_name,
                "feature": "intercept",
                "coefficient": self.intercept,
                "odds_ratio": float(np.exp(self.intercept)),
                "training_accuracy": self.training_accuracy,
                "positive_rate": self.positive_rate,
            }
        ]
        for feature, coefficient in zip(self.continuous_features + self.binary_features, self.coefficients):
            rows.append(
                {
                    "target_name": self.target_name,
                    "feature": feature,
                    "coefficient": coefficient,
                    "odds_ratio": float(np.exp(coefficient)),
                    "training_accuracy": self.training_accuracy,
                    "positive_rate": self.positive_rate,
                }
            )
        return rows


@dataclass(frozen=True)
class ConstantGateModel:
    target_name: str
    probability: float
    positive_rate: float

    def predict_proba(self, row: dict[str, float | int | bool | None]) -> float:
        _ = row
        return self.probability

    def coefficient_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "target_name": self.target_name,
                "feature": "constant",
                "coefficient": 0.0,
                "odds_ratio": 1.0,
                "training_accuracy": max(self.probability, 1.0 - self.probability),
                "positive_rate": self.positive_rate,
            }
        ]


def fit_binary_logistic_gate(
    rows: list[dict[str, Any]],
    *,
    target_key: str,
    continuous_features: list[str],
    binary_features: list[str],
    learning_rate: float,
    steps: int,
    l2_penalty: float,
) -> LearnedGateModel | ConstantGateModel:
    usable = [row for row in rows if row.get(target_key) is not None]
    if not usable:
        raise ValueError(f"No usable rows for target {target_key}.")

    y = np.asarray([float(row[target_key]) for row in usable], dtype=float)
    positive_rate = float(np.mean(y))
    if positive_rate in {0.0, 1.0}:
        return ConstantGateModel(
            target_name=target_key,
            probability=positive_rate,
            positive_rate=positive_rate,
        )

    feature_columns: list[np.ndarray] = []
    means: list[float] = []
    stds: list[float] = []
    for feature in continuous_features:
        values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in usable], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        means.append(mean)
        stds.append(std)
        feature_columns.append(np.zeros_like(values) if std == 0.0 else (values - mean) / std)
    for feature in binary_features:
        values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in usable], dtype=float)
        feature_columns.append(values)

    x = np.column_stack(feature_columns) if feature_columns else np.zeros((len(usable), 0), dtype=float)
    design = np.concatenate([np.ones((len(usable), 1), dtype=float), x], axis=1)
    weights = np.zeros(design.shape[1], dtype=float)
    regularizer = np.ones_like(weights) * l2_penalty
    regularizer[0] = 0.0
    for _ in range(steps):
        probabilities = _sigmoid(design @ weights)
        gradient = (design.T @ (probabilities - y)) / len(y)
        gradient += regularizer * weights / len(y)
        weights -= learning_rate * gradient

    predictions = (_sigmoid(design @ weights) >= 0.5).astype(float)
    accuracy = float(np.mean(predictions == y))
    return LearnedGateModel(
        target_name=target_key,
        continuous_features=continuous_features,
        binary_features=binary_features,
        means=means,
        stds=stds,
        coefficients=list(weights[1:]),
        intercept=float(weights[0]),
        training_accuracy=accuracy,
        positive_rate=positive_rate,
    )
