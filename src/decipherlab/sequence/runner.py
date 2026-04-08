from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from decipherlab.config import DecipherLabConfig, dump_config, load_config
from decipherlab.decoding.beam_search import (
    BeamDecodingResult,
    BigramTransitionModel,
    beam_decode_confusion_network,
    greedy_decode_confusion_network,
)
from decipherlab.evaluation.metrics import (
    symbol_case_breakdown,
    symbol_expected_calibration_error,
    symbol_negative_log_likelihood,
    symbol_top_k_accuracy,
)
from decipherlab.glyphs.features import extract_feature_matrix
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.models import DatasetCollection, SequenceExample, TranscriptionPosterior
from decipherlab.risk_control.conformal import SplitConformalSetPredictor, summarize_prediction_sets
from decipherlab.sequence.benchmark import build_real_glyph_sequence_benchmark
from decipherlab.sequence.family_identification import (
    ProcessFamilyClassifier,
    family_identification_payload,
)
from decipherlab.sequence.failure_analysis import build_sequence_failure_cases
from decipherlab.sequence.metrics import sequence_metric_bundle
from decipherlab.sequence.results_pack import (
    build_ambiguity_regime_rows,
    build_pairwise_effect_rows,
    summarize_best_regime,
)
from decipherlab.structured_uncertainty.confusion_network import (
    confusion_network_to_posterior,
    posterior_to_confusion_network,
)
from decipherlab.transcription.model import fit_posterior_model
from decipherlab.transcription.posterior import split_posterior_by_lengths
from decipherlab.utils.io import write_csv, write_json, write_text
from decipherlab.utils.logging import configure_logging
from decipherlab.utils.runtime import prepare_run_context
from decipherlab.vision.corruption import apply_ambiguity_to_examples


def _load_manifest_dataset(config: DecipherLabConfig) -> DatasetCollection:
    if config.dataset.source != "manifest" or config.dataset.manifest_path is None:
        raise ValueError("Sequence branch v1 expects a manifest-backed glyph crop dataset.")
    if config.dataset.manifest_format != "glyph_crop":
        raise ValueError("Sequence branch v1 requires dataset.manifest_format='glyph_crop'.")
    return load_glyph_crop_manifest_dataset(config.dataset.manifest_path, dataset_config=config.dataset)


def _flatten_glyphs(examples: list[SequenceExample]) -> list:
    return [glyph for example in examples for glyph in example.glyphs]


def _weighted_average(values: list[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def _sequence_labels(examples: list[SequenceExample]) -> list[list[str | None]]:
    return [example.observed_symbols for example in examples]


def _decode_method_result(
    method: str,
    posterior: TranscriptionPosterior,
    network,
    transition_model: BigramTransitionModel,
    config: DecipherLabConfig,
) -> tuple[BeamDecodingResult, TranscriptionPosterior, Any]:
    if method == "fixed_greedy":
        fixed = posterior.collapsed()
        fixed_network = posterior_to_confusion_network(
            fixed,
            config.structured_uncertainty.model_copy(update={"max_candidates_per_position": 1}),
        )
        return greedy_decode_confusion_network(fixed_network), fixed, fixed_network
    if method == "uncertainty_beam":
        return (
            beam_decode_confusion_network(
                network=network,
                transition_model=transition_model,
                beam_width=config.decoding.beam_width,
                lm_weight=config.decoding.lm_weight,
                top_k_sequences=config.decoding.top_k_sequences,
                length_normalize=config.decoding.length_normalize,
            ),
            posterior,
            network,
        )
    raise ValueError(f"Unsupported decoding method: {method}")


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["ambiguity_level"], row["method"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (ambiguity_level, method), items in sorted(grouped.items()):
        symbol_top1_values = [(item["symbol_top1_accuracy"], item["labeled_symbol_count"]) for item in items]
        symbol_topk_values = [(item["symbol_topk_accuracy"], item["labeled_symbol_count"]) for item in items]
        symbol_nll_values = [(item["symbol_negative_log_likelihood"], item["labeled_symbol_count"]) for item in items]
        symbol_ece_values = [(item["symbol_expected_calibration_error"], item["labeled_symbol_count"]) for item in items]
        summary_rows.append(
            {
                "ambiguity_level": ambiguity_level,
                "method": method,
                "sequence_exact_match": float(np.mean([item["sequence_exact_match"] for item in items])),
                "sequence_token_accuracy": float(np.mean([item["sequence_token_accuracy"] for item in items])),
                "sequence_topk_recovery": float(np.mean([item["sequence_topk_recovery"] for item in items])),
                "sequence_cer": float(np.mean([item["sequence_cer"] for item in items])),
                "symbol_top1_accuracy": _weighted_average(symbol_top1_values),
                "symbol_topk_accuracy": _weighted_average(symbol_topk_values),
                "symbol_negative_log_likelihood": _weighted_average(symbol_nll_values),
                "symbol_expected_calibration_error": _weighted_average(symbol_ece_values),
                "prediction_set_coverage": float(np.mean([item["prediction_set_coverage"] for item in items]))
                if any(item["prediction_set_coverage"] is not None for item in items)
                else None,
                "prediction_set_avg_size": float(np.mean([item["prediction_set_avg_size"] for item in items if item["prediction_set_avg_size"] is not None]))
                if any(item["prediction_set_avg_size"] is not None for item in items)
                else None,
                "prediction_set_singleton_rate": float(np.mean([item["prediction_set_singleton_rate"] for item in items if item["prediction_set_singleton_rate"] is not None]))
                if any(item["prediction_set_singleton_rate"] is not None for item in items)
                else None,
                "prediction_set_rescue_rate": float(np.mean([item["prediction_set_rescue_rate"] for item in items if item["prediction_set_rescue_rate"] is not None]))
                if any(item["prediction_set_rescue_rate"] is not None for item in items)
                else None,
                "family_identification_accuracy": float(np.mean([item["family_identification_accuracy"] for item in items if item["family_identification_accuracy"] is not None]))
                if any(item["family_identification_accuracy"] is not None for item in items)
                else None,
                "sequence_count": len(items),
                "labeled_symbol_count": int(sum(item["labeled_symbol_count"] for item in items)),
            }
        )
    return summary_rows


def _build_report(summary_rows: list[dict[str, Any]], benchmark_metadata: dict[str, Any]) -> str:
    lines = [
        "# Sequence Branch Report",
        "",
        "This report belongs to the sequence-focused branch and should not be merged into the workshop paper claims.",
        "",
        "## Benchmark",
        f"- Source dataset: `{benchmark_metadata['source_dataset_name']}`",
        f"- Synthetic-from-real task: `{benchmark_metadata['task_name']}`",
        f"- Selected symbols: `{len(benchmark_metadata['selected_symbols'])}`",
        f"- Sequence length: `{benchmark_metadata['sequence_length']}`",
        "",
        "## Aggregated Results",
        "",
        "| Ambiguity | Method | Seq exact | Seq token | Seq top-k | Seq CER | Symbol top-k | Set coverage | Set size | Family acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {ambiguity:.2f} | {method} | {exact:.3f} | {token:.3f} | {topk:.3f} | {cer:.3f} | {symbol_topk:.3f} | {coverage} | {set_size} | {family_acc} |".format(
                ambiguity=row["ambiguity_level"],
                method=row["method"],
                exact=row["sequence_exact_match"],
                token=row["sequence_token_accuracy"],
                topk=row["sequence_topk_recovery"],
                cer=row["sequence_cer"],
                symbol_topk=row["symbol_topk_accuracy"] or 0.0,
                coverage="n/a" if row["prediction_set_coverage"] is None else f"{row['prediction_set_coverage']:.3f}",
                set_size="n/a" if row["prediction_set_avg_size"] is None else f"{row['prediction_set_avg_size']:.3f}",
                family_acc="n/a" if row["family_identification_accuracy"] is None else f"{row['family_identification_accuracy']:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "- Sequence-level claims in this branch are synthetic-from-real: real glyph crops are used, but transition structure is generated.",
            "- Positive results here would support structured use of uncertainty under ambiguity, not semantic decipherment.",
        ]
    )
    return "\n".join(lines)


def run_sequence_branch_experiment(
    config: DecipherLabConfig | str | Path,
    strategy_override: str | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not resolved.sequence_benchmark.enabled:
        raise ValueError("sequence_benchmark.enabled must be true for the sequence branch.")
    if not resolved.structured_uncertainty.enabled or not resolved.decoding.enabled:
        raise ValueError("structured_uncertainty.enabled and decoding.enabled must both be true.")

    run_context = prepare_run_context(resolved, suffix="sequence_branch")
    logger = configure_logging(run_context.run_dir)
    dump_config(resolved, run_context.run_dir / "config.yaml")

    logger.info("Loading source manifest dataset for sequence benchmark generation.")
    source_dataset = _load_manifest_dataset(resolved)
    benchmark = build_real_glyph_sequence_benchmark(
        source_dataset=source_dataset,
        config=resolved.sequence_benchmark,
        seed=resolved.experiment.seed,
        source_train_split=resolved.dataset.train_split,
        source_val_split=resolved.dataset.val_split,
        source_test_split=resolved.dataset.evaluation_split,
    )
    write_json(run_context.run_dir / "benchmark_summary.json", benchmark.to_dict())

    train_examples = benchmark.dataset.get_split(resolved.dataset.train_split)
    validation_examples = benchmark.dataset.get_split(resolved.dataset.val_split)
    evaluation_examples = benchmark.dataset.get_split(resolved.dataset.evaluation_split)
    train_glyphs = _flatten_glyphs(train_examples)
    validation_glyphs = _flatten_glyphs(validation_examples)
    if not train_glyphs or not validation_glyphs or not evaluation_examples:
        raise ValueError("Sequence branch requires non-empty train, val, and test synthetic-from-real splits.")

    train_features = extract_feature_matrix(train_glyphs, downsample=resolved.vision.feature_downsample)
    validation_features = extract_feature_matrix(validation_glyphs, downsample=resolved.vision.feature_downsample)
    posterior_strategy = resolved.posterior.strategy if strategy_override is None else strategy_override
    posterior_model = fit_posterior_model(
        train_features=train_features,
        train_labels=[glyph.true_symbol for glyph in train_glyphs],
        validation_features=validation_features,
        validation_labels=[glyph.true_symbol for glyph in validation_glyphs],
        posterior_config=resolved.posterior.model_copy(update={"strategy": posterior_strategy}),
        vision_config=resolved.vision,
        seed=resolved.experiment.seed,
    )
    transition_model = BigramTransitionModel.fit(
        sequences=[[symbol for symbol in example.observed_symbols if symbol is not None] for example in train_examples],
        smoothing=resolved.decoding.transition_smoothing,
    )
    family_classifier = ProcessFamilyClassifier.fit(
        train_examples,
        smoothing=resolved.decoding.transition_smoothing,
    )

    per_sequence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for ambiguity_index, ambiguity_level in enumerate(resolved.evaluation.ambiguity_levels):
        logger.info("Running sequence branch at ambiguity %.2f.", ambiguity_level)
        ambiguous_validation = apply_ambiguity_to_examples(
            validation_examples,
            ambiguity_level=ambiguity_level,
            seed=resolved.experiment.seed + 5000 + ambiguity_index,
        )
        ambiguous_evaluation = apply_ambiguity_to_examples(
            evaluation_examples,
            ambiguity_level=ambiguity_level,
            seed=resolved.experiment.seed + 8000 + ambiguity_index,
        )
        validation_features_ambiguous = extract_feature_matrix(
            _flatten_glyphs(ambiguous_validation),
            downsample=resolved.vision.feature_downsample,
        )
        evaluation_features = extract_feature_matrix(
            _flatten_glyphs(ambiguous_evaluation),
            downsample=resolved.vision.feature_downsample,
        )
        validation_posterior = posterior_model.predict_posterior(
            validation_features_ambiguous,
            top_k=resolved.posterior.top_k,
            floor_probability=resolved.posterior.floor_probability,
        )
        evaluation_posterior = posterior_model.predict_posterior(
            evaluation_features,
            top_k=resolved.posterior.top_k,
            floor_probability=resolved.posterior.floor_probability,
        )
        validation_posteriors = split_posterior_by_lengths(
            validation_posterior,
            [example.sequence_length for example in ambiguous_validation],
        )
        evaluation_posteriors = split_posterior_by_lengths(
            evaluation_posterior,
            [example.sequence_length for example in ambiguous_evaluation],
        )
        validation_networks = [
            posterior_to_confusion_network(posterior, resolved.structured_uncertainty)
            for posterior in validation_posteriors
        ]
        evaluation_networks = [
            posterior_to_confusion_network(posterior, resolved.structured_uncertainty)
            for posterior in evaluation_posteriors
        ]

        conformal_predictor = None
        if resolved.risk_control.enabled:
            conformal_predictor = SplitConformalSetPredictor.fit(
                validation_networks,
                _sequence_labels(validation_examples),
                resolved.risk_control,
            )
        methods = ["fixed_greedy", "uncertainty_beam"]
        if conformal_predictor is not None:
            methods.append("conformal_beam")

        for example, posterior, network in zip(ambiguous_evaluation, evaluation_posteriors, evaluation_networks):
            for method in methods:
                active_network = network
                active_posterior = posterior
                if method == "conformal_beam":
                    active_network = conformal_predictor.apply(network, resolved.risk_control)
                    active_posterior = confusion_network_to_posterior(active_network)
                    decoded = beam_decode_confusion_network(
                        network=active_network,
                        transition_model=transition_model,
                        beam_width=resolved.decoding.beam_width,
                        lm_weight=resolved.decoding.lm_weight,
                        top_k_sequences=resolved.decoding.top_k_sequences,
                        length_normalize=resolved.decoding.length_normalize,
                    )
                else:
                    decoded, active_posterior, active_network = _decode_method_result(
                        method=method,
                        posterior=posterior,
                        network=network,
                        transition_model=transition_model,
                        config=resolved,
                    )
                sequence_metrics = sequence_metric_bundle(
                    decoded,
                    [symbol for symbol in example.observed_symbols if symbol is not None],
                )
                family_metrics = family_identification_payload(
                    family_classifier,
                    decoded.best.symbols if decoded.sequences else [],
                    example.family,
                )
                set_metrics = summarize_prediction_sets([active_network], [example.observed_symbols])
                symbol_top1 = symbol_top_k_accuracy(active_posterior, example.observed_symbols, top_k=1)
                symbol_topk = symbol_top_k_accuracy(active_posterior, example.observed_symbols, top_k=resolved.evaluation.top_k)
                symbol_nll = symbol_negative_log_likelihood(active_posterior, example.observed_symbols)
                symbol_ece = symbol_expected_calibration_error(active_posterior, example.observed_symbols)
                per_sequence_rows.append(
                    {
                        "ambiguity_level": ambiguity_level,
                        "example_id": example.example_id,
                        "method": method,
                        "posterior_strategy": posterior_model.strategy,
                        "posterior_strategy_requested": posterior_strategy,
                        "task_name": benchmark.metadata["task_name"],
                        "dataset_name": benchmark.metadata["source_dataset_name"],
                        **sequence_metrics,
                        "symbol_top1_accuracy": symbol_top1[0],
                        "symbol_topk_accuracy": symbol_topk[0],
                        "symbol_negative_log_likelihood": symbol_nll[0],
                        "symbol_expected_calibration_error": symbol_ece[0],
                        "prediction_set_coverage": set_metrics["prediction_set_coverage"],
                        "prediction_set_avg_size": set_metrics["prediction_set_avg_size"],
                        "prediction_set_singleton_rate": set_metrics["prediction_set_singleton_rate"],
                        "prediction_set_rescue_rate": set_metrics["prediction_set_rescue_rate"],
                        "family_identification_accuracy": family_metrics["family_identification_accuracy"],
                        "predicted_family": family_metrics["predicted_family"],
                        "labeled_symbol_count": symbol_top1[1],
                        "mean_confusion_entropy": active_network.mean_entropy(),
                        "mean_confusion_set_size": active_network.average_set_size(),
                        "case_breakdown": symbol_case_breakdown(
                            active_posterior,
                            example.observed_symbols,
                            top_k=resolved.evaluation.top_k,
                        ),
                        "decoded_best_sequence": decoded.best.symbols if decoded.sequences else [],
                    }
                )
        summary_rows.extend(_aggregate_rows([row for row in per_sequence_rows if row["ambiguity_level"] == ambiguity_level]))

    for row in summary_rows:
        row["posterior_strategy_requested"] = posterior_strategy
        row["posterior_strategy"] = posterior_model.strategy
        row["task_name"] = benchmark.metadata["task_name"]
        row["dataset_name"] = benchmark.metadata["source_dataset_name"]

    pairwise_rows = build_pairwise_effect_rows(summary_rows)
    ambiguity_regime_rows = build_ambiguity_regime_rows(pairwise_rows)
    best_regime = summarize_best_regime(pairwise_rows)
    failure_cases, failure_summary = build_sequence_failure_cases(per_sequence_rows)
    dataset_summary = {
        "dataset_name": benchmark.metadata["source_dataset_name"],
        "task_name": benchmark.metadata["task_name"],
        "sequence_length": benchmark.dataset.metadata["sequence_length"],
        "selected_symbol_count": len(benchmark.alphabet),
        "sequence_counts": benchmark.dataset.metadata["sequence_counts"],
        "source_manifest": benchmark.metadata["source_manifest"],
        "synthetic_from_real": benchmark.metadata["synthetic_from_real"],
        "family_signal_available": family_classifier is not None,
        "best_ambiguity_regime_by_strategy": best_regime,
    }

    write_json(run_context.run_dir / "sequence_branch_summary.json", summary_rows)
    write_csv(run_context.run_dir / "sequence_branch_summary.csv", summary_rows)
    write_json(run_context.run_dir / "sequence_branch_examples.json", per_sequence_rows)
    write_csv(
        run_context.run_dir / "sequence_branch_examples.csv",
        [
            {key: value for key, value in row.items() if key not in {"case_breakdown", "decoded_best_sequence"}}
            for row in per_sequence_rows
        ],
    )
    write_json(run_context.run_dir / "sequence_pairwise_effects.json", pairwise_rows)
    write_csv(run_context.run_dir / "sequence_pairwise_effects.csv", pairwise_rows)
    write_json(run_context.run_dir / "sequence_ambiguity_regime_table.json", ambiguity_regime_rows)
    write_csv(run_context.run_dir / "sequence_ambiguity_regime_table.csv", ambiguity_regime_rows)
    write_json(run_context.run_dir / "sequence_failure_cases.json", failure_cases)
    write_json(run_context.run_dir / "sequence_failure_summary.json", failure_summary)
    write_csv(run_context.run_dir / "sequence_failure_summary.csv", failure_summary)
    write_json(run_context.run_dir / "dataset_summary.json", dataset_summary)
    write_text(
        run_context.run_dir / "dataset_summary.md",
        "\n".join(
            [
                "# Sequence Dataset Summary",
                "",
                f"- Source dataset: `{dataset_summary['dataset_name']}`",
                f"- Task name: `{dataset_summary['task_name']}`",
                f"- Sequence length: `{dataset_summary['sequence_length']}`",
                f"- Selected symbol count: `{dataset_summary['selected_symbol_count']}`",
                f"- Train/val/test sequences: `{dataset_summary['sequence_counts']}`",
                f"- Synthetic-from-real: `{dataset_summary['synthetic_from_real']}`",
                f"- Family signal available: `{dataset_summary['family_signal_available']}`",
                f"- Source manifest: `{dataset_summary['source_manifest']}`",
            ]
        ),
    )
    write_text(
        run_context.run_dir / "sequence_branch_report.md",
        _build_report(summary_rows, benchmark.dataset.metadata),
    )
    return {
        "run_dir": run_context.run_dir,
        "summary_rows": summary_rows,
        "per_sequence_rows": per_sequence_rows,
        "pairwise_rows": pairwise_rows,
        "ambiguity_regime_rows": ambiguity_regime_rows,
        "failure_summary": failure_summary,
        "benchmark": benchmark.to_dict(),
        "posterior_model": posterior_model.to_dict(),
        "transition_model": transition_model.diagnostics,
    }
