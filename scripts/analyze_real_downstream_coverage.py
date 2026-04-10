from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.config import load_config
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.sequence.benchmark import build_real_glyph_sequence_benchmark
from decipherlab.sequence.real_downstream import build_real_downstream_resource
from decipherlab.utils.io import write_csv, write_text


CONFIGS = [
    (
        "historical_newspapers_real_grouped_gold",
        Path("configs/experiments/sequence_historical_newspapers_real_downstream.yaml"),
        Path("configs/experiments/sequence_historical_newspapers_real_downstream_redesigned.yaml"),
    ),
    (
        "scadsai_real_grouped",
        Path("configs/experiments/sequence_scadsai_real_downstream.yaml"),
        Path("configs/experiments/sequence_scadsai_real_downstream_redesigned.yaml"),
    ),
]


def _ngram_tokens(transcript: tuple[str, ...], order: int) -> list[tuple[str, ...]]:
    if len(transcript) < order:
        return []
    return [tuple(transcript[index : index + order]) for index in range(len(transcript) - order + 1)]


def main() -> None:
    rows: list[dict[str, object]] = []
    for dataset_label, old_config_path, new_config_path in CONFIGS:
        old_config = load_config(old_config_path)
        new_config = load_config(new_config_path)
        source_dataset = load_glyph_crop_manifest_dataset(old_config.dataset.manifest_path, dataset_config=old_config.dataset)
        benchmark = build_real_glyph_sequence_benchmark(
            source_dataset=source_dataset,
            config=old_config.sequence_benchmark,
            seed=old_config.experiment.seed,
            source_train_split=old_config.dataset.train_split,
            source_val_split=old_config.dataset.val_split,
            source_test_split=old_config.dataset.evaluation_split,
        )
        train_examples = benchmark.dataset.get_split(old_config.dataset.train_split)
        test_examples = benchmark.dataset.get_split(old_config.dataset.evaluation_split)

        transcript_bank = build_real_downstream_resource(train_examples, old_config.real_downstream)
        ngram_inventory = build_real_downstream_resource(train_examples, new_config.real_downstream)

        old_candidate_sizes: list[int] = []
        old_exact_hits = 0
        old_valid = 0
        for example in test_examples:
            truth = tuple(symbol for symbol in example.observed_symbols if symbol is not None)
            candidates = transcript_bank.candidates_for_length(len(truth))
            old_candidate_sizes.append(len(candidates))
            if candidates:
                old_valid += 1
            if any(entry.transcript == truth for entry in candidates):
                old_exact_hits += 1

        new_supported_counts: list[int] = []
        new_total_counts: list[int] = []
        new_valid = 0
        new_full = 0
        for example in test_examples:
            truth = tuple(symbol for symbol in example.observed_symbols if symbol is not None)
            all_ngrams = _ngram_tokens(truth, new_config.real_downstream.ngram_order)
            supported = [ngram for ngram in all_ngrams if ngram in ngram_inventory.inventory]
            new_supported_counts.append(len(supported))
            new_total_counts.append(len(all_ngrams))
            if len(supported) >= new_config.real_downstream.min_supported_ngrams:
                new_valid += 1
            if all_ngrams and len(supported) == len(all_ngrams):
                new_full += 1

        rows.extend(
            [
                {
                    "dataset": dataset_label,
                    "task_name": "train_transcript_bank",
                    "coverage_metric": "valid_candidate_rate",
                    "value": old_valid / len(test_examples),
                    "note": "Fraction of test examples with any same-length train transcript candidate.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_transcript_bank",
                    "coverage_metric": "exact_target_coverage",
                    "value": old_exact_hits / len(test_examples),
                    "note": "Fraction of test examples whose exact transcript appears in the train bank.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_transcript_bank",
                    "coverage_metric": "approximate_exact_upper_bound",
                    "value": old_exact_hits / len(test_examples),
                    "note": "Under exact transcript-bank decoding, exact recovery cannot exceed exact train/test overlap.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_transcript_bank",
                    "coverage_metric": "mean_candidate_size",
                    "value": sum(old_candidate_sizes) / len(old_candidate_sizes),
                    "note": "Average number of same-length train transcript candidates per test example.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "valid_candidate_rate",
                    "value": new_valid / len(test_examples),
                    "note": "Fraction of test examples with at least one train-supported n-gram in the gold path.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "full_path_coverage",
                    "value": new_full / len(test_examples),
                    "note": "Fraction of test examples whose full gold n-gram path is supported by train n-grams.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "mean_supported_fraction",
                    "value": sum(
                        supported / total for supported, total in zip(new_supported_counts, new_total_counts) if total > 0
                    )
                    / sum(1 for total in new_total_counts if total > 0),
                    "note": "Average fraction of each gold n-gram path covered by the train n-gram inventory.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "approximate_exact_upper_bound",
                    "value": new_full / len(test_examples),
                    "note": "Exact n-gram-path recovery cannot exceed the fraction with full train-supported n-gram coverage.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "mean_candidate_size",
                    "value": sum(new_supported_counts) / len(new_supported_counts),
                    "note": "Average number of supported gold n-grams per test example.",
                },
                {
                    "dataset": dataset_label,
                    "task_name": "train_supported_ngram_path",
                    "coverage_metric": "train_inventory_size",
                    "value": float(len(ngram_inventory.inventory)),
                    "note": "Distinct train n-gram inventory size used by the redesigned task.",
                },
            ]
        )

    csv_path = Path("outputs/real_downstream_coverage_analysis.csv")
    md_path = Path("outputs/real_downstream_coverage_analysis.md")
    write_csv(csv_path, rows)

    lines = [
        "# Real Downstream Coverage Analysis",
        "",
        "## Main Finding",
        "",
        "- Exact train-transcript-bank recovery is coverage-limited on both real grouped corpora.",
        "- Train-supported n-gram-path recovery materially improves coverage on both corpora, especially on ScaDS.AI.",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['dataset']}` / `{row['task_name']}` / `{row['coverage_metric']}`: `{float(row['value']):.3f}`. {row['note']}"
        )
    write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
