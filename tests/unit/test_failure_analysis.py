from __future__ import annotations

from decipherlab.evaluation.runner import run_ablation_suite


def test_failure_analysis_emits_structured_summary(small_config) -> None:
    result = run_ablation_suite(small_config)
    assert "failure_cases" in result["failure_payload"]
    assert "summary_rows" in result["failure_payload"]
