from __future__ import annotations

import json

from decipherlab.pipeline import run_pipeline

from tests.helpers import build_test_config, create_real_manifest_fixture


def test_pipeline_runs_on_real_manifest_adapter(tmp_path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    config = build_test_config(tmp_path, noise_std=0.1)
    config.dataset.source = "manifest"
    config.dataset.manifest_path = manifest_path
    config.dataset.manifest_format = "glyph_crop"
    config.dataset.train_split = "train"
    config.dataset.val_split = "val"
    config.dataset.evaluation_split = "test"

    result = run_pipeline(config, posterior_mode="uncertainty", suffix="real_manifest", ambiguity_level=0.2)
    run_dir = result["run_dir"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_name"] == "fixture_real_manifest"
    assert metrics["symbol_top1_accuracy"] is not None
    assert metrics["labeled_symbol_count"] > 0
    assert (run_dir / "posterior_model.json").exists()
