from __future__ import annotations

from decipherlab.sequence.learned_gate import fit_binary_logistic_gate


def test_fit_binary_logistic_gate_predicts_higher_probability_for_positive_region():
    rows = [
        {"entropy": 0.1, "set_size": 1.0, "limited_support": 1.0, "target": 1.0},
        {"entropy": 0.2, "set_size": 1.2, "limited_support": 1.0, "target": 1.0},
        {"entropy": 1.1, "set_size": 4.0, "limited_support": 0.0, "target": 0.0},
        {"entropy": 1.3, "set_size": 4.5, "limited_support": 0.0, "target": 0.0},
    ]
    model = fit_binary_logistic_gate(
        rows,
        target_key="target",
        continuous_features=["entropy", "set_size"],
        binary_features=["limited_support"],
        learning_rate=0.2,
        steps=2000,
        l2_penalty=1.0e-2,
    )

    positive = model.predict_proba({"entropy": 0.15, "set_size": 1.1, "limited_support": 1.0})
    negative = model.predict_proba({"entropy": 1.2, "set_size": 4.2, "limited_support": 0.0})

    assert positive > 0.5
    assert negative < 0.5
