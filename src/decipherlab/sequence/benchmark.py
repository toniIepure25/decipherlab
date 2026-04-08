from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from decipherlab.config import SequenceBenchmarkConfig
from decipherlab.models import DatasetCollection, GlyphCrop, SequenceExample


@dataclass(frozen=True)
class SequenceBenchmarkBundle:
    dataset: DatasetCollection
    alphabet: list[str]
    label_groups: dict[str, str]
    transition_matrix: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset.dataset_name,
            "alphabet": self.alphabet,
            "label_groups": self.label_groups,
            "transition_matrix": self.transition_matrix,
            "metadata": self.metadata,
        }


def _glyph_label_pool(
    dataset: DatasetCollection,
) -> dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]]:
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]] = {}
    for example in dataset.examples:
        split_pool = pools.setdefault(example.split, {})
        for glyph in example.glyphs:
            if glyph.true_symbol is None:
                continue
            split_pool.setdefault(glyph.true_symbol, []).append((glyph, example))
    return pools


def _select_labels(
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]],
    config: SequenceBenchmarkConfig,
    source_splits: tuple[str, str, str],
) -> list[str]:
    required_splits = set(source_splits)
    if not required_splits.issubset(pools):
        raise ValueError("Sequence benchmark generation requires train, val, and test source splits.")
    eligible: list[tuple[str, int]] = []
    source_train_split, source_val_split, source_test_split = source_splits
    for label in sorted(
        set(pools[source_train_split]) & set(pools[source_val_split]) & set(pools[source_test_split])
    ):
        counts = [len(pools[split].get(label, [])) for split in source_splits]
        if min(counts) >= config.min_instances_per_symbol:
            eligible.append((label, sum(counts)))
    if len(eligible) < config.selected_symbol_count:
        raise ValueError(
            "Not enough labels satisfy the sequence benchmark coverage constraints: "
            f"required {config.selected_symbol_count}, found {len(eligible)}."
        )
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _ in eligible[: config.selected_symbol_count]]


def _build_transition_matrix(
    labels: list[str],
    label_groups: dict[str, str],
    config: SequenceBenchmarkConfig,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    transition_matrix: dict[str, dict[str, float]] = {}
    for left in labels:
        row: dict[str, float] = {}
        for right in labels:
            if left == right:
                base = config.self_transition_bias
            elif label_groups[left] == label_groups[right]:
                base = config.within_group_bias
            else:
                base = config.cross_group_bias
            base += float(rng.uniform(0.0, config.transition_noise))
            row[right] = base
        total = sum(row.values())
        transition_matrix[left] = {right: value / total for right, value in row.items()}
    return transition_matrix


def _sample_symbol_sequence(
    labels: list[str],
    transition_matrix: dict[str, dict[str, float]],
    length: int,
    rng: np.random.Generator,
) -> list[str]:
    current = labels[int(rng.integers(0, len(labels)))]
    sequence = [current]
    for _ in range(length - 1):
        row = transition_matrix[current]
        candidates = list(row.keys())
        probabilities = np.asarray([row[label] for label in candidates], dtype=float)
        current = str(rng.choice(candidates, p=probabilities))
        sequence.append(current)
    return sequence


def _sample_alternating_sequence(
    labels: list[str],
    label_groups: dict[str, str],
    length: int,
    rng: np.random.Generator,
) -> list[str]:
    groups: dict[str, list[str]] = {}
    for label in labels:
        groups.setdefault(label_groups[label], []).append(label)
    ordered_groups = sorted(groups)
    current_group_index = int(rng.integers(0, len(ordered_groups)))
    sequence: list[str] = []
    for position in range(length):
        group = ordered_groups[(current_group_index + position) % len(ordered_groups)]
        sequence.append(str(rng.choice(groups[group])))
    return sequence


def _sample_motif_repeat_sequence(
    labels: list[str],
    length: int,
    motif_length: int,
    motif_noise_probability: float,
    rng: np.random.Generator,
) -> list[str]:
    motif = [str(rng.choice(labels)) for _ in range(motif_length)]
    sequence: list[str] = []
    for position in range(length):
        candidate = motif[position % motif_length]
        if rng.random() < motif_noise_probability:
            candidate = str(rng.choice(labels))
        sequence.append(candidate)
    return sequence


def _materialize_sequence(
    split: str,
    sequence_index: int,
    label_sequence: list[str],
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]],
    label_groups: dict[str, str],
    sample_with_replacement: bool,
    rng: np.random.Generator,
    source_dataset_name: str,
    family: str | None = None,
) -> SequenceExample:
    glyphs: list[GlyphCrop] = []
    source_sequence_ids: list[str] = []
    source_group_ids: list[str] = []
    label_draw_counts: dict[str, int] = {}
    label_permutations: dict[str, np.ndarray] = {}
    for position, label in enumerate(label_sequence):
        candidates = pools[split][label]
        if sample_with_replacement or len(candidates) == 1:
            glyph, source_example = candidates[int(rng.integers(0, len(candidates)))]
        else:
            draw_count = label_draw_counts.get(label, 0)
            permutation = label_permutations.setdefault(label, rng.permutation(len(candidates)))
            glyph, source_example = candidates[int(permutation[draw_count % len(candidates)])]
            label_draw_counts[label] = draw_count + 1
        source_sequence_ids.append(source_example.example_id)
        group_id = source_example.metadata.get("group_id")
        if group_id is not None:
            source_group_ids.append(str(group_id))
        glyphs.append(
            replace(
                glyph,
                position=position,
                true_symbol=label,
                variant_id=f"{glyph.variant_id or glyph.source_path or label}_seq_{split}_{sequence_index:04d}_{position:02d}",
            )
        )
    return SequenceExample(
        example_id=f"{split}_seq_{sequence_index:04d}",
        family=family,
        glyphs=glyphs,
        plaintext=None,
        observed_symbols=list(label_sequence),
        split=split,
        metadata={
            "synthetic_sequence_task": True,
            "task_name": "real_glyph_markov_sequences" if family is None else "real_glyph_process_family_sequences",
            "source_dataset_name": source_dataset_name,
            "label_groups": [label_groups[label] for label in label_sequence],
            "source_sequence_ids": source_sequence_ids,
            "source_group_ids": sorted(set(source_group_ids)),
        },
    )


def _split_count_map(config: SequenceBenchmarkConfig) -> dict[str, int]:
    return {
        "train": config.train_sequences,
        "val": config.val_sequences,
        "test": config.test_sequences,
    }


def _source_pools_for_splits(
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]],
    source_train_split: str,
    source_val_split: str,
    source_test_split: str,
) -> dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]]:
    return {
        "train": pools[source_train_split],
        "val": pools[source_val_split],
        "test": pools[source_test_split],
    }


def _build_markov_sequence_examples(
    labels: list[str],
    label_groups: dict[str, str],
    transition_matrix: dict[str, dict[str, float]],
    config: SequenceBenchmarkConfig,
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]],
    rng: np.random.Generator,
    source_dataset_name: str,
) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    for split, count in _split_count_map(config).items():
        for sequence_index in range(count):
            label_sequence = _sample_symbol_sequence(labels, transition_matrix, config.sequence_length, rng)
            examples.append(
                _materialize_sequence(
                    split=split,
                    sequence_index=sequence_index,
                    label_sequence=label_sequence,
                    pools=pools,
                    label_groups=label_groups,
                    sample_with_replacement=config.sample_with_replacement,
                    rng=rng,
                    source_dataset_name=source_dataset_name,
                )
            )
    return examples


def _build_process_family_sequence_examples(
    labels: list[str],
    label_groups: dict[str, str],
    transition_matrix: dict[str, dict[str, float]],
    config: SequenceBenchmarkConfig,
    pools: dict[str, dict[str, list[tuple[GlyphCrop, SequenceExample]]]],
    rng: np.random.Generator,
    source_dataset_name: str,
) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    families = list(config.process_families)
    split_counts = _split_count_map(config)
    for split, total_count in split_counts.items():
        base_count = total_count // len(families)
        remainder = total_count % len(families)
        counts = {
            family: base_count + (1 if family_index < remainder else 0)
            for family_index, family in enumerate(families)
        }
        sequence_index = 0
        for family in families:
            for _ in range(counts[family]):
                if family == "sticky_markov":
                    label_sequence = _sample_symbol_sequence(labels, transition_matrix, config.sequence_length, rng)
                elif family == "alternating_markov":
                    label_sequence = _sample_alternating_sequence(
                        labels,
                        label_groups,
                        config.sequence_length,
                        rng,
                    )
                elif family == "motif_repeat":
                    label_sequence = _sample_motif_repeat_sequence(
                        labels,
                        config.sequence_length,
                        config.motif_length,
                        config.motif_noise_probability,
                        rng,
                    )
                else:
                    raise ValueError(f"Unsupported process family: {family}")
                examples.append(
                    _materialize_sequence(
                        split=split,
                        sequence_index=sequence_index,
                        label_sequence=label_sequence,
                        pools=pools,
                        label_groups=label_groups,
                        sample_with_replacement=config.sample_with_replacement,
                        rng=rng,
                        source_dataset_name=source_dataset_name,
                        family=family,
                    )
                )
                sequence_index += 1
    return examples


def build_real_glyph_sequence_benchmark(
    source_dataset: DatasetCollection,
    config: SequenceBenchmarkConfig,
    seed: int,
    source_train_split: str = "train",
    source_val_split: str = "val",
    source_test_split: str = "test",
) -> SequenceBenchmarkBundle:
    if not config.enabled:
        raise ValueError("Sequence benchmark generation is disabled in the current config.")
    pools = _glyph_label_pool(source_dataset)
    labels = _select_labels(
        pools,
        config,
        source_splits=(source_train_split, source_val_split, source_test_split),
    )
    label_groups = {
        label: f"group_{index % config.group_count:02d}"
        for index, label in enumerate(labels)
    }
    rng = np.random.default_rng(seed)
    transition_matrix = _build_transition_matrix(labels, label_groups, config, rng)
    split_counts = _split_count_map(config)
    source_pools = _source_pools_for_splits(
        pools,
        source_train_split=source_train_split,
        source_val_split=source_val_split,
        source_test_split=source_test_split,
    )
    if config.task_name == "real_glyph_markov_sequences":
        examples = _build_markov_sequence_examples(
            labels=labels,
            label_groups=label_groups,
            transition_matrix=transition_matrix,
            config=config,
            pools=source_pools,
            rng=rng,
            source_dataset_name=source_dataset.dataset_name,
        )
    elif config.task_name == "real_glyph_process_family_sequences":
        examples = _build_process_family_sequence_examples(
            labels=labels,
            label_groups=label_groups,
            transition_matrix=transition_matrix,
            config=config,
            pools=source_pools,
            rng=rng,
            source_dataset_name=source_dataset.dataset_name,
        )
    else:
        raise ValueError(f"Unsupported sequence benchmark task: {config.task_name}")
    dataset = DatasetCollection(
        dataset_name=f"{source_dataset.dataset_name}_sequence_branch",
        examples=examples,
        manifest_path=source_dataset.manifest_path,
        metadata=source_dataset.metadata
        | {
            "synthetic_from_real": True,
            "source_dataset_name": source_dataset.dataset_name,
            "task_name": config.task_name,
            "selected_symbols": labels,
            "label_groups": label_groups,
            "sequence_length": config.sequence_length,
            "sequence_counts": split_counts,
            "process_families": list(config.process_families),
        },
    )
    return SequenceBenchmarkBundle(
        dataset=dataset,
        alphabet=labels,
        label_groups=label_groups,
        transition_matrix=transition_matrix,
        metadata={
            "source_dataset_name": source_dataset.dataset_name,
            "source_manifest": source_dataset.manifest_path,
            "synthetic_from_real": True,
            "benchmark_note": (
                "Sequence-level tasks are synthetic and built from real glyph crops. "
                "They support downstream evaluation without implying semantic decipherment."
            ),
            "task_name": config.task_name,
            "selected_symbols": labels,
            "sequence_length": config.sequence_length,
        },
    )
