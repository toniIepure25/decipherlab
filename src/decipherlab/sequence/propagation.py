from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LogisticModelSummary:
    feature_names: list[str]
    coefficients: list[float]
    intercept: float
    training_accuracy: float
    positive_rate: float

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = [
            {
                "model_name": model_name,
                "feature": "intercept",
                "coefficient": self.intercept,
                "odds_ratio": float(np.exp(self.intercept)),
                "training_accuracy": self.training_accuracy,
                "positive_rate": self.positive_rate,
            }
        ]
        for feature, coefficient in zip(self.feature_names, self.coefficients):
            rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "coefficient": coefficient,
                    "odds_ratio": float(np.exp(coefficient)),
                    "training_accuracy": self.training_accuracy,
                    "positive_rate": self.positive_rate,
                }
            )
        return rows


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_regularized_logistic_regression(
    rows: list[dict[str, Any]],
    target_key: str,
    continuous_features: list[str],
    binary_features: list[str],
    categorical_features: dict[str, list[str]],
    learning_rate: float = 0.2,
    steps: int = 4000,
    l2_penalty: float = 1.0e-2,
) -> LogisticModelSummary:
    usable = [row for row in rows if row.get(target_key) is not None]
    if not usable:
        raise ValueError(f"No usable rows for target {target_key}.")

    feature_names: list[str] = []
    columns: list[np.ndarray] = []
    for feature in continuous_features:
        values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in usable], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        standardized = np.zeros_like(values) if std == 0.0 else (values - mean) / std
        feature_names.append(feature)
        columns.append(standardized)
    for feature in binary_features:
        values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in usable], dtype=float)
        feature_names.append(feature)
        columns.append(values)
    for feature, levels in categorical_features.items():
        for level in levels[1:]:
            values = np.asarray(
                [1.0 if row.get(feature) == level else 0.0 for row in usable],
                dtype=float,
            )
            feature_names.append(f"{feature}={level}")
            columns.append(values)

    x = np.column_stack(columns) if columns else np.zeros((len(usable), 0), dtype=float)
    intercept = np.ones((len(usable), 1), dtype=float)
    design = np.concatenate([intercept, x], axis=1)
    y = np.asarray([float(row[target_key]) for row in usable], dtype=float)

    weights = np.zeros(design.shape[1], dtype=float)
    regularizer = np.ones_like(weights) * l2_penalty
    regularizer[0] = 0.0
    for _ in range(steps):
        probabilities = sigmoid(design @ weights)
        gradient = (design.T @ (probabilities - y)) / len(y)
        gradient += regularizer * weights / len(y)
        weights -= learning_rate * gradient

    predictions = (sigmoid(design @ weights) >= 0.5).astype(float)
    accuracy = float(np.mean(predictions == y))
    return LogisticModelSummary(
        feature_names=feature_names,
        coefficients=list(weights[1:]),
        intercept=float(weights[0]),
        training_accuracy=accuracy,
        positive_rate=float(np.mean(y)),
    )


def best_threshold_split(
    rows: list[dict[str, Any]],
    feature_key: str,
    target_key: str,
    min_group_size: int = 8,
) -> dict[str, Any] | None:
    usable = [
        row
        for row in rows
        if row.get(feature_key) is not None and row.get(target_key) is not None
    ]
    if len(usable) < min_group_size * 2:
        return None
    values = sorted({float(row[feature_key]) for row in usable})
    best: dict[str, Any] | None = None
    for threshold in values:
        low = [row for row in usable if float(row[feature_key]) < threshold]
        high = [row for row in usable if float(row[feature_key]) >= threshold]
        if len(low) < min_group_size or len(high) < min_group_size:
            continue
        low_rate = float(np.mean([float(row[target_key]) for row in low]))
        high_rate = float(np.mean([float(row[target_key]) for row in high]))
        gap = abs(high_rate - low_rate)
        candidate = {
            "feature": feature_key,
            "threshold": float(threshold),
            "low_count": len(low),
            "high_count": len(high),
            "low_rate": low_rate,
            "high_rate": high_rate,
            "gap": gap,
            "direction": "higher_is_better" if high_rate >= low_rate else "lower_is_better",
        }
        if best is None or candidate["gap"] > best["gap"]:
            best = candidate
    return best


def ambiguity_regime_label(ambiguity_level: float) -> str:
    if ambiguity_level <= 0.15:
        return "low"
    if ambiguity_level < 0.45:
        return "medium"
    return "high"


def bootstrap_mean_ci(
    values: list[float | int],
    *,
    seed: int = 13,
    num_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap_mean_ci requires at least one value.")
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(num_bootstrap):
        bootstrap_sample = rng.choice(sample, size=len(sample), replace=True)
        draws.append(float(np.mean(bootstrap_sample)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(sample)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "boot_prob_positive": float(np.mean(np.asarray(draws) > 0.0)),
    }
