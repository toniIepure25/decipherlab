from __future__ import annotations

from pathlib import Path

import yaml

from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.ingest.validation import summarize_glyph_crop_manifest

from tests.helpers import build_test_config, create_real_manifest_fixture


def test_manifest_validation_summary_includes_split_and_group_counts(tmp_path: Path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = GlyphCropManifest.model_validate(payload)
    config = build_test_config(tmp_path)
    summary = summarize_glyph_crop_manifest(manifest, manifest_path=manifest_path, dataset_config=config.dataset)
    assert summary["split_sequence_counts"]["train"] == 2
    assert summary["split_group_counts"]["train"] >= 1
    assert "warnings" in summary


def test_manifest_loader_attaches_validation_report(tmp_path: Path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    config = build_test_config(tmp_path)
    config.dataset.min_symbol_instances_per_train_class_warning = 3
    dataset = load_glyph_crop_manifest_dataset(manifest_path, dataset_config=config.dataset)
    report = dataset.metadata["validation_report"]
    assert report["dataset_name"] == "fixture_real_manifest"
    assert report["warning_count"] >= 1
