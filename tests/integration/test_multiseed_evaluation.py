from __future__ import annotations

from decipherlab.evaluation.runner import run_ablation_suite


def test_multiseed_evaluation_writes_seed_aggregates(small_config) -> None:
    small_config.experiment.seed_sweep = [17]
    small_config.evaluation.ambiguity_levels = [0.0]
    result = run_ablation_suite(small_config)
    assert len(result["comparisons"]) == 8
    assert all("seed" in row for row in result["comparisons"])
    assert (result["run_dir"] / "seed_summary.csv").exists()
    assert (result["run_dir"] / "pairwise_effect_summary.csv").exists()
    assert (result["run_dir"] / "results_section_draft.md").exists()
    assert (result["run_dir"] / "limitations_section_draft.md").exists()
