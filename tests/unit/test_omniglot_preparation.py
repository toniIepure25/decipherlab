from __future__ import annotations

from pathlib import Path

import yaml

from decipherlab.ingest.omniglot import format_omniglot_integration_note, summarize_omniglot_local_artifacts


def test_omniglot_artifact_summary_and_note_include_manifest_stats(tmp_path) -> None:
    dataset_root = tmp_path / "omniglot"
    extracted_root = dataset_root / "extracted" / "images_background" / "Alphabet" / "character01"
    extracted_root.mkdir(parents=True, exist_ok=True)

    (dataset_root / "images_background.zip").write_bytes(b"bg")
    (dataset_root / "images_evaluation.zip").write_bytes(b"ev")
    (extracted_root / "sample.png").write_bytes(b"png")

    manifest_path = dataset_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "omniglot_fixture",
                "unit_type": "glyph_crop",
                "metadata": {"split_seed": 23},
                "records": [
                    {
                        "sequence_id": "Alphabet__character01__sample_train",
                        "position": 0,
                        "image_path": "extracted/images_background/Alphabet/character01/sample.png",
                        "split": "train",
                        "example_id": "train_sample",
                        "group_id": "Alphabet",
                        "transcription": "Alphabet__character01",
                        "family": None,
                        "metadata": {"source_split": "images_background"},
                    },
                    {
                        "sequence_id": "Alphabet__character01__sample_val",
                        "position": 0,
                        "image_path": "extracted/images_background/Alphabet/character01/sample.png",
                        "split": "val",
                        "example_id": "val_sample",
                        "group_id": "Alphabet",
                        "transcription": "Alphabet__character01",
                        "family": None,
                        "metadata": {"source_split": "images_background"},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = summarize_omniglot_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    assert summary["total_footprint"] > 0
    assert summary["extracted_png_size"] == 3
    manifest_summary = summary["manifest_summary"]
    assert manifest_summary["record_count"] == 2
    assert manifest_summary["split_counts"] == {"train": 1, "val": 1}
    assert manifest_summary["label_count"] == 1
    assert manifest_summary["group_count"] == 1

    note = format_omniglot_integration_note(dataset_root=dataset_root, manifest_path=manifest_path)
    assert "Full download strategy: **full dataset**" in note
    assert "`2` total labeled crops" in note
    assert "- `1` train examples" in note
    assert "- `1` val examples" in note
    assert "- split seed: `23`" in note
