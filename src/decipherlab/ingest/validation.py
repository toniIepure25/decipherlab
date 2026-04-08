from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from decipherlab.config import DatasetConfig
from decipherlab.ingest.schema import GlyphCropManifest


def summarize_glyph_crop_manifest(
    manifest: GlyphCropManifest,
    manifest_path: str | Path,
    dataset_config: DatasetConfig | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    split_sequence_counts: Counter[str] = Counter()
    split_record_counts: Counter[str] = Counter()
    split_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_symbol_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_labeled_positions: Counter[str] = Counter()
    split_total_positions: Counter[str] = Counter()
    train_symbol_counts: Counter[str] = Counter()
    train_symbol_sequence_counts: dict[str, set[str]] = defaultdict(set)
    split_group_counts: dict[str, set[str]] = defaultdict(set)
    split_grouped_sequence_counts: dict[str, set[str]] = defaultdict(set)
    split_sequence_ids: dict[str, set[str]] = defaultdict(set)
    missing_images: list[str] = []
    warnings: list[str] = []

    for record in manifest.records:
        split_total_positions[record.split] += 1
        split_sequence_counts[record.split] += 0
        split_record_counts[record.split] += 1
        split_sequence_counts[record.split] += int(record.position == 0)
        if record.position == 0:
            split_sequence_ids[record.split].add(record.sequence_id)
        if record.family is not None:
            split_family_counts[record.split][record.family] += int(record.position == 0)
        if record.transcription is not None:
            split_symbol_counts[record.split][record.transcription] += 1
            split_labeled_positions[record.split] += 1
        if record.split == "train" and record.transcription is not None:
            train_symbol_counts[record.transcription] += 1
            train_symbol_sequence_counts[record.transcription].add(record.sequence_id)
        if record.group_id is not None:
            split_group_counts[record.split].add(record.group_id)
            split_grouped_sequence_counts[record.split].add(record.sequence_id)
        image_path = manifest_file.parent / record.image_path
        if not image_path.exists():
            missing_images.append(str(image_path))

    split_sequence_fractions = {
        split: count / max(sum(split_sequence_counts.values()), 1)
        for split, count in sorted(split_sequence_counts.items())
    }
    split_label_coverage = {
        split: split_labeled_positions[split] / split_total_positions[split]
        for split in sorted(split_total_positions)
        if split_total_positions[split] > 0
    }
    split_group_coverage = {
        split: len(split_grouped_sequence_counts[split]) / max(len(split_sequence_ids[split]), 1)
        for split in sorted(split_sequence_ids)
    }
    requested_splits = set()
    if dataset_config is not None:
        requested_splits = {
            dataset_config.train_split,
            dataset_config.val_split,
            dataset_config.evaluation_split,
        }
        available_splits = set(split_sequence_counts)
        missing_requested = sorted(split for split in requested_splits if split not in available_splits)
        if missing_requested:
            warnings.append(
                "Requested splits missing from manifest: " + ", ".join(missing_requested) + "."
            )

        threshold = dataset_config.min_sequences_per_split_warning
        for split, count in sorted(split_sequence_counts.items()):
            if count < threshold:
                warnings.append(
                    f"Split '{split}' has only {count} sequence(s); this is below the warning threshold of {threshold}."
                )

        symbol_threshold = dataset_config.min_symbol_instances_per_train_class_warning
        underpowered_symbols = sorted(
            symbol
            for symbol, sequence_ids in train_symbol_sequence_counts.items()
            if len(sequence_ids) < symbol_threshold
        )
        if underpowered_symbols:
            warnings.append(
                "Train symbol classes below the recommended sequence-count threshold "
                f"({symbol_threshold}): {', '.join(underpowered_symbols)}."
            )

        family_threshold = dataset_config.min_family_instances_per_split_warning
        for split, counts in sorted(split_family_counts.items()):
            low_families = sorted(family for family, count in counts.items() if count < family_threshold)
            if low_families:
                warnings.append(
                    f"Split '{split}' has family labels below the warning threshold of {family_threshold}: "
                    + ", ".join(low_families)
                    + "."
                )

        evaluation_symbols = set(split_symbol_counts.get(dataset_config.evaluation_split, Counter()))
        train_symbols = set(split_symbol_counts.get(dataset_config.train_split, Counter()))
        unseen_eval_symbols = sorted(evaluation_symbols - train_symbols)
        if unseen_eval_symbols:
            warnings.append(
                f"Evaluation split '{dataset_config.evaluation_split}' contains symbols not seen in train: "
                + ", ".join(unseen_eval_symbols)
                + "."
            )

    if missing_images:
        warnings.append(f"Missing image files detected: {len(missing_images)}")

    return {
        "dataset_name": manifest.dataset_name,
        "unit_type": manifest.unit_type,
        "record_count": len(manifest.records),
        "sequence_count": int(sum(split_sequence_counts.values())),
        "split_sequence_counts": dict(sorted(split_sequence_counts.items())),
        "split_sequence_fractions": split_sequence_fractions,
        "split_record_counts": dict(sorted(split_record_counts.items())),
        "split_label_coverage": split_label_coverage,
        "split_group_coverage": split_group_coverage,
        "split_group_counts": {
            split: len(group_ids) for split, group_ids in sorted(split_group_counts.items())
        },
        "split_grouped_sequence_counts": {
            split: len(sequence_ids) for split, sequence_ids in sorted(split_grouped_sequence_counts.items())
        },
        "split_symbol_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_symbol_counts.items())
        },
        "train_symbol_counts": dict(sorted(train_symbol_counts.items())),
        "train_symbol_sequence_counts": {
            symbol: len(sequence_ids) for symbol, sequence_ids in sorted(train_symbol_sequence_counts.items())
        },
        "split_family_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_family_counts.items())
        },
        "missing_images": missing_images[:50],
        "warning_count": len(warnings),
        "requested_splits": sorted(requested_splits),
        "warnings": warnings,
    }


def format_manifest_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dataset Integration Summary",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Unit type: `{summary['unit_type']}`",
        f"- Sequences: `{summary['sequence_count']}`",
        f"- Records: `{summary['record_count']}`",
        "",
        "## Split Composition",
        "| Split | Sequences | Fraction | Label Coverage | Groups | Grouped Sequence Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in sorted(summary["split_sequence_counts"]):
        lines.append(
            "| {split} | {sequences} | {fraction:.3f} | {label_coverage:.3f} | {groups} | {group_coverage:.3f} |".format(
                split=split,
                sequences=summary["split_sequence_counts"].get(split, 0),
                fraction=summary["split_sequence_fractions"].get(split, 0.0),
                label_coverage=summary["split_label_coverage"].get(split, 0.0),
                groups=summary["split_group_counts"].get(split, 0),
                group_coverage=summary["split_group_coverage"].get(split, 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Class Coverage",
            f"- Train symbol classes: `{len(summary['train_symbol_counts'])}`",
            f"- Requested splits: `{summary['requested_splits']}`",
        ]
    )
    if summary["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.extend(["", "## Warnings", "- None."])
    return "\n".join(lines) + "\n"
