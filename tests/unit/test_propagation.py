from __future__ import annotations

from decipherlab.sequence.propagation import (
    ambiguity_regime_label,
    best_threshold_split,
    bootstrap_mean_ci,
    fit_regularized_logistic_regression,
)


def test_fit_regularized_logistic_regression_learns_positive_signal():
    rows = [
        {"feature": 0.0, "flag": 0.0, "target": 0.0},
        {"feature": 0.2, "flag": 0.0, "target": 0.0},
        {"feature": 1.0, "flag": 1.0, "target": 1.0},
        {"feature": 1.2, "flag": 1.0, "target": 1.0},
    ]
    model = fit_regularized_logistic_regression(
        rows=rows,
        target_key="target",
        continuous_features=["feature"],
        binary_features=["flag"],
        categorical_features={},
        steps=1500,
    )

    coefficient_map = dict(zip(model.feature_names, model.coefficients))
    assert model.training_accuracy >= 0.75
    assert coefficient_map["feature"] > 0.0
    assert coefficient_map["flag"] > 0.0


def test_best_threshold_split_finds_high_rate_region():
    rows = [
        {"support": 0.1, "target": 0.0},
        {"support": 0.2, "target": 0.0},
        {"support": 0.3, "target": 0.0},
        {"support": 0.4, "target": 0.0},
        {"support": 0.5, "target": 0.0},
        {"support": 0.6, "target": 1.0},
        {"support": 0.7, "target": 1.0},
        {"support": 0.8, "target": 1.0},
        {"support": 0.9, "target": 1.0},
        {"support": 1.0, "target": 1.0},
        {"support": 1.1, "target": 1.0},
        {"support": 1.2, "target": 1.0},
        {"support": 1.3, "target": 1.0},
        {"support": 1.4, "target": 1.0},
        {"support": 1.5, "target": 1.0},
        {"support": 1.6, "target": 1.0},
    ]
    threshold = best_threshold_split(rows, "support", "target", min_group_size=4)

    assert threshold is not None
    assert threshold["high_rate"] > threshold["low_rate"]


def test_ambiguity_regime_label_thresholds():
    assert ambiguity_regime_label(0.0) == "low"
    assert ambiguity_regime_label(0.3) == "medium"
    assert ambiguity_regime_label(0.45) == "high"


def test_bootstrap_mean_ci_tracks_positive_effect():
    summary = bootstrap_mean_ci([0.2, 0.3, 0.1, 0.4], seed=7, num_bootstrap=400)

    assert summary["mean"] > 0.0
    assert summary["ci_low"] > 0.0
    assert summary["boot_prob_positive"] > 0.95
