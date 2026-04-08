from __future__ import annotations

import numpy as np

from decipherlab.config import RiskControlConfig, StructuredUncertaintyConfig
from decipherlab.models import TranscriptionPosterior
from decipherlab.risk_control.conformal import SplitConformalSetPredictor, summarize_prediction_sets
from decipherlab.structured_uncertainty.confusion_network import posterior_to_confusion_network


def test_confusion_network_and_split_conformal_retain_rescue_cases():
    posterior = TranscriptionPosterior(
        candidate_ids=[["a", "b", "c"], ["b", "a", "c"]],
        log_probabilities=np.log(
            np.asarray(
                [
                    [0.55, 0.35, 0.10],
                    [0.52, 0.28, 0.20],
                ],
                dtype=float,
            )
        ),
    )
    network = posterior_to_confusion_network(
        posterior,
        StructuredUncertaintyConfig(
            enabled=True,
            max_candidates_per_position=3,
            cumulative_probability_mass=0.95,
            min_probability=0.05,
        ),
    )
    predictor = SplitConformalSetPredictor.fit(
        [network],
        [["b", "b"]],
        RiskControlConfig(enabled=True, alpha=0.2, min_set_size=1, max_set_size=3),
    )
    filtered = predictor.apply(network, RiskControlConfig(enabled=True, alpha=0.2, min_set_size=1, max_set_size=3))
    summary = summarize_prediction_sets([filtered], [["b", "b"]])

    assert predictor.threshold_probability > 0.0
    assert summary["prediction_set_coverage"] == 1.0
    assert summary["prediction_set_avg_size"] >= 1.0
    assert summary["prediction_set_rescue_rate"] > 0.0
