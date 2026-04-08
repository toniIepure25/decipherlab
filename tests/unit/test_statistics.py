from __future__ import annotations

from decipherlab.evaluation.statistics import MetricSampleGroup, bootstrap_grouped_mean, bootstrap_mean

import numpy as np


def test_bootstrap_mean_returns_deterministic_interval() -> None:
    interval = bootstrap_mean([0.2, 0.4, 0.6, 0.8], trials=100, confidence_level=0.95, seed=7)
    assert interval.point_estimate == 0.5
    assert interval.ci_lower is not None
    assert interval.ci_upper is not None
    assert interval.ci_lower <= interval.point_estimate <= interval.ci_upper
    assert interval.sample_count == 4


def test_bootstrap_grouped_mean_handles_weighted_groups() -> None:
    groups = [
        MetricSampleGroup(values=np.asarray([1.0, 0.0]), weights=np.asarray([2.0, 1.0])),
        MetricSampleGroup(values=np.asarray([0.5, 0.5]), weights=None),
    ]
    interval = bootstrap_grouped_mean(groups, trials=100, confidence_level=0.9, seed=3)
    assert interval.point_estimate is not None
    assert interval.ci_lower is not None
    assert interval.ci_upper is not None
    assert interval.group_count == 2
    assert interval.sample_count == 4
