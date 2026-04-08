from __future__ import annotations

from decipherlab.evaluation.runner import run_ablation_suite


def test_uncertainty_ablation_reports_comparable_metrics_and_tradeoffs(small_config) -> None:
    result = run_ablation_suite(small_config)
    comparisons = result["comparisons"]
    assert len(comparisons) == 8
    by_noise = {}
    for comparison in comparisons:
        by_noise.setdefault(comparison["ambiguity_level"], {})[
            (comparison["posterior_strategy_requested"], comparison["posterior_mode"])
        ] = comparison

    noisy = by_noise[0.35]
    assert set(noisy) == {
        ("calibrated_classifier", "fixed"),
        ("calibrated_classifier", "uncertainty"),
        ("cluster_distance", "fixed"),
        ("cluster_distance", "uncertainty"),
    }
    assert "mean_structural_recovery_error" in noisy[("calibrated_classifier", "uncertainty")]
    assert "expected_calibration_error" in noisy[("cluster_distance", "fixed")]
    assert noisy[("calibrated_classifier", "uncertainty")]["mean_posterior_entropy"] > 0.0
    assert result["pairwise_rows"]
    assert (result["run_dir"] / "main_comparison_table.csv").exists()
    assert (result["run_dir"] / "main_comparison_with_ci.csv").exists()
    assert (result["run_dir"] / "ambiguity_sweep_with_ci.csv").exists()
    assert (result["run_dir"] / "seed_summary.csv").exists()
    assert (result["run_dir"] / "failure_case_summary.csv").exists()
    assert (result["run_dir"] / "comparison_symbol_topk.png").exists()
