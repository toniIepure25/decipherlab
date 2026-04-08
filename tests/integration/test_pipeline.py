from __future__ import annotations

import json

from decipherlab.pipeline import run_pipeline


def test_full_pipeline_writes_expected_artifacts(small_config) -> None:
    result = run_pipeline(small_config, posterior_mode="uncertainty", suffix="integration")
    run_dir = result["run_dir"]
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "example_results.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "family_probabilities.png").exists()
    assert (run_dir / "posterior_entropy.png").exists()
    assert (run_dir / "posterior_model.json").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["example_count"] == 4
    assert 0.0 <= metrics["family_top1_accuracy"] <= 1.0
    assert metrics["symbol_top1_accuracy"] is not None
