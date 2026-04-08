from __future__ import annotations

from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset

from tests.helpers import create_real_manifest_fixture


def test_real_manifest_adapter_groups_sequences_and_splits(tmp_path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    dataset = load_glyph_crop_manifest_dataset(manifest_path)
    assert dataset.dataset_name == "fixture_real_manifest"
    assert sorted(dataset.split_names()) == ["test", "train", "val"]
    test_examples = dataset.get_split("test")
    assert len(test_examples) == 1
    assert test_examples[0].sequence_length == 8
    assert test_examples[0].has_symbol_labels
    assert test_examples[0].glyphs[0].source_path is not None
