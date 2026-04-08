from __future__ import annotations

from pathlib import Path

import yaml

from decipherlab.ingest.sklearn_digits import (
    build_sklearn_digits_manifest,
    format_sklearn_digits_integration_note,
    summarize_sklearn_digits_local_artifacts,
)


def test_build_sklearn_digits_manifest_and_note(tmp_path) -> None:
    dataset_root = tmp_path / "digits"
    manifest_path = dataset_root / "manifest.yaml"

    manifest = build_sklearn_digits_manifest(
        output_dir=dataset_root,
        manifest_path=manifest_path,
        train_count_per_class=5,
        val_count_per_class=2,
        split_seed=17,
    )

    assert manifest.dataset_name == "sklearn_digits_crops"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1797
    assert payload["metadata"]["split_counts"]["train"] == 50
    assert payload["metadata"]["split_counts"]["val"] == 20
    assert payload["metadata"]["split_counts"]["test"] == 1727

    summary = summarize_sklearn_digits_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    assert summary["total_footprint"] > 0
    assert summary["manifest_summary"]["label_count"] == 10

    note = format_sklearn_digits_integration_note(dataset_root=dataset_root, manifest_path=manifest_path)
    assert "Additional network download: `0` bytes" in note
    assert "- `1797` total labeled crops" in note
    assert "- `10` digit classes" in note
