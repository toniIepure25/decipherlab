from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    bootstrap_std: float | None
    sample_count: int
    group_count: int

    def to_dict(self, prefix: str) -> dict[str, float | int | None]:
        return {
            prefix: self.point_estimate,
            f"{prefix}_ci_lower": self.ci_lower,
            f"{prefix}_ci_upper": self.ci_upper,
            f"{prefix}_bootstrap_std": self.bootstrap_std,
            f"{prefix}_sample_count": self.sample_count,
            f"{prefix}_group_count": self.group_count,
        }


@dataclass(frozen=True)
class MetricSampleGroup:
    values: np.ndarray
    weights: np.ndarray | None = None


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    if values.size == 0:
        return None
    if weights is None:
        return float(np.mean(values))
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return float(np.sum(values * weights) / total)


def bootstrap_mean(
    values: list[float | None],
    trials: int,
    confidence_level: float,
    seed: int,
) -> BootstrapInterval:
    filtered = np.asarray([value for value in values if value is not None], dtype=float)
    if filtered.size == 0:
        return BootstrapInterval(None, None, None, None, 0, 0)
    point_estimate = float(np.mean(filtered))
    if trials <= 0 or filtered.size == 1:
        return BootstrapInterval(point_estimate, point_estimate, point_estimate, 0.0, int(filtered.size), 1)

    rng = np.random.default_rng(seed)
    estimates = np.empty(trials, dtype=float)
    for trial_index in range(trials):
        sample = rng.choice(filtered, size=filtered.size, replace=True)
        estimates[trial_index] = float(np.mean(sample))
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        point_estimate=point_estimate,
        ci_lower=float(np.quantile(estimates, alpha)),
        ci_upper=float(np.quantile(estimates, 1.0 - alpha)),
        bootstrap_std=float(np.std(estimates)),
        sample_count=int(filtered.size),
        group_count=1,
    )


def bootstrap_grouped_mean(
    groups: list[MetricSampleGroup],
    trials: int,
    confidence_level: float,
    seed: int,
) -> BootstrapInterval:
    filtered_groups = [group for group in groups if group.values.size > 0]
    if not filtered_groups:
        return BootstrapInterval(None, None, None, None, 0, 0)

    point_estimates = [
        _weighted_mean(group.values, group.weights)
        for group in filtered_groups
    ]
    point_estimates = [value for value in point_estimates if value is not None]
    if not point_estimates:
        return BootstrapInterval(None, None, None, None, 0, len(filtered_groups))

    point_estimate = float(np.mean(point_estimates))
    sample_count = int(sum(group.values.size for group in filtered_groups))
    if trials <= 0:
        return BootstrapInterval(
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            bootstrap_std=0.0,
            sample_count=sample_count,
            group_count=len(filtered_groups),
        )

    rng = np.random.default_rng(seed)
    estimates = np.empty(trials, dtype=float)
    for trial_index in range(trials):
        group_estimates: list[float] = []
        for group in filtered_groups:
            indices = rng.integers(0, group.values.size, size=group.values.size)
            sample_values = group.values[indices]
            sample_weights = None if group.weights is None else group.weights[indices]
            estimate = _weighted_mean(sample_values, sample_weights)
            if estimate is not None:
                group_estimates.append(estimate)
        estimates[trial_index] = float(np.mean(group_estimates))
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        point_estimate=point_estimate,
        ci_lower=float(np.quantile(estimates, alpha)),
        ci_upper=float(np.quantile(estimates, 1.0 - alpha)),
        bootstrap_std=float(np.std(estimates)),
        sample_count=sample_count,
        group_count=len(filtered_groups),
    )
