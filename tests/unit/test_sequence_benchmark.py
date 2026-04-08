from __future__ import annotations

from decipherlab.config import SequenceBenchmarkConfig
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.sequence.family_identification import ProcessFamilyClassifier, family_identification_payload
from decipherlab.sequence.benchmark import build_real_glyph_sequence_benchmark
from tests.helpers import build_test_config, create_real_manifest_fixture


def test_real_glyph_sequence_benchmark_is_deterministic_and_split_aware(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path)
    dataset_config = build_test_config(tmp_path).dataset.model_copy(
        update={"source": "manifest", "manifest_path": manifest_path, "generate_if_missing": False}
    )
    dataset = load_glyph_crop_manifest_dataset(manifest_path, dataset_config=dataset_config)
    benchmark_config = SequenceBenchmarkConfig(
        enabled=True,
        selected_symbol_count=4,
        min_instances_per_symbol=2,
        train_sequences=6,
        val_sequences=3,
        test_sequences=3,
        sequence_length=6,
        group_count=2,
    )

    bundle_a = build_real_glyph_sequence_benchmark(dataset, benchmark_config, seed=19)
    bundle_b = build_real_glyph_sequence_benchmark(dataset, benchmark_config, seed=19)

    assert bundle_a.alphabet == bundle_b.alphabet
    assert bundle_a.dataset.count_examples("train") == 6
    assert bundle_a.dataset.count_examples("val") == 3
    assert bundle_a.dataset.count_examples("test") == 3
    assert all(example.sequence_length == 6 for example in bundle_a.dataset.examples)
    assert bundle_a.dataset.metadata["synthetic_from_real"] is True
    assert bundle_a.transition_matrix == bundle_b.transition_matrix


def test_process_family_sequence_benchmark_emits_family_labels_and_supports_classifier(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path)
    dataset_config = build_test_config(tmp_path).dataset.model_copy(
        update={"source": "manifest", "manifest_path": manifest_path, "generate_if_missing": False}
    )
    dataset = load_glyph_crop_manifest_dataset(manifest_path, dataset_config=dataset_config)
    benchmark_config = SequenceBenchmarkConfig(
        enabled=True,
        task_name="real_glyph_process_family_sequences",
        selected_symbol_count=4,
        min_instances_per_symbol=2,
        train_sequences=9,
        val_sequences=3,
        test_sequences=3,
        sequence_length=6,
        group_count=2,
    )

    bundle = build_real_glyph_sequence_benchmark(dataset, benchmark_config, seed=11)
    classifier = ProcessFamilyClassifier.fit(bundle.dataset.get_split("train"), smoothing=0.1)

    assert classifier is not None
    assert all(example.family is not None for example in bundle.dataset.examples)
    payload = family_identification_payload(
        classifier,
        [symbol for symbol in bundle.dataset.get_split("test")[0].observed_symbols if symbol is not None],
        bundle.dataset.get_split("test")[0].family,
    )
    assert payload["family_identification_supported"] is True
