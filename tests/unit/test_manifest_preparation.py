from __future__ import annotations

import csv
import json

from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.ingest.preparation import build_glyph_crop_manifest_from_table

import yaml

from tests.helpers import create_real_manifest_fixture


def test_build_manifest_from_csv_table_roundtrips(tmp_path) -> None:
    source_manifest = create_real_manifest_fixture(tmp_path)
    payload = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    records_csv = tmp_path / "records.csv"
    fieldnames = [
        "sequence_id",
        "position",
        "image_path",
        "split",
        "example_id",
        "group_id",
        "family",
        "transcription",
        "metadata",
    ]
    with records_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload["records"]:
            writer.writerow(record | {"metadata": json.dumps(record.get("metadata", {}))})

    built_manifest = tmp_path / "built_manifest.yaml"
    manifest = build_glyph_crop_manifest_from_table(
        records_path=records_csv,
        output_path=built_manifest,
        dataset_name="built_fixture_manifest",
        image_root=source_manifest.parent,
    )
    assert manifest.dataset_name == "built_fixture_manifest"
    dataset = load_glyph_crop_manifest_dataset(built_manifest)
    assert dataset.dataset_name == "built_fixture_manifest"
    assert dataset.count_examples("train") == 2
