from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from decipherlab.benchmarks.synthetic import generate_synthetic_dataset, save_synthetic_dataset
from decipherlab.config import DecipherLabConfig, dump_config, load_config
from decipherlab.evaluation.metrics import (
    clustering_ari,
    structural_recovery_error,
    summarize_rankings,
    symbol_case_breakdown,
    symbol_entropy_by_correctness,
    symbol_expected_calibration_error,
    symbol_negative_log_likelihood,
    symbol_top_k_accuracy,
)
from decipherlab.glyphs.clustering import cluster_feature_matrix
from decipherlab.glyphs.features import extract_feature_matrix
from decipherlab.hypotheses.scorers import rank_hypotheses
from decipherlab.ingest.manifest import (
    load_glyph_crop_manifest_dataset,
    load_synthetic_manifest_dataset,
)
from decipherlab.models import DatasetCollection, SequenceExample
from decipherlab.scoring.reporting import write_run_artifacts
from decipherlab.structure.triage import analyze_posterior, sequence_metrics_from_symbols
from decipherlab.transcription.model import fit_posterior_model
from decipherlab.transcription.posterior import split_posterior_by_lengths
from decipherlab.utils.logging import configure_logging
from decipherlab.utils.runtime import RunContext, prepare_run_context
from decipherlab.vision.corruption import apply_ambiguity_to_examples


PosteriorMode = Literal["uncertainty", "fixed"]


def _load_dataset(config: DecipherLabConfig, run_dir: Path) -> tuple[DatasetCollection, Path | None]:
    if config.dataset.source == "manifest":
        if config.dataset.manifest_path is None:
            raise ValueError("manifest_path is required when dataset.source='manifest'.")
        manifest_path = Path(config.dataset.manifest_path)
        if config.dataset.manifest_format == "glyph_crop":
            return load_glyph_crop_manifest_dataset(manifest_path, dataset_config=config.dataset), manifest_path
        return load_synthetic_manifest_dataset(manifest_path), manifest_path

    bundle = generate_synthetic_dataset(config.synthetic, seed=config.experiment.seed)
    manifest_path = save_synthetic_dataset(
        bundle,
        output_dir=run_dir / "dataset",
        dataset_name=config.experiment.name,
        seed=config.experiment.seed,
    )
    dataset = DatasetCollection(
        dataset_name=bundle.dataset.dataset_name,
        examples=bundle.dataset.examples,
        manifest_path=str(manifest_path),
        metadata=bundle.dataset.metadata,
    )
    return dataset, manifest_path


def _weighted_average(values: list[tuple[float | None, int]]) -> float | None:
    weighted_sum = 0.0
    count = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        count += weight
    if count == 0:
        return None
    return weighted_sum / count


def _resolve_split(
    dataset: DatasetCollection,
    split: str,
    fallback: str,
    strict: bool = False,
) -> list[SequenceExample]:
    examples = dataset.get_split(split)
    if examples:
        return examples
    if strict:
        available = ", ".join(dataset.split_names()) or "none"
        raise ValueError(f"Requested split '{split}' not found. Available splits: {available}.")
    return dataset.get_split(fallback)


def _labeled_glyph_targets(examples: list[SequenceExample]) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    labels: list[str] = []
    flat_index = 0
    for example in examples:
        for glyph in example.glyphs:
            if glyph.true_symbol is not None:
                indices.append(flat_index)
                labels.append(glyph.true_symbol)
            flat_index += 1
    return indices, labels


def execute_pipeline(
    config: DecipherLabConfig,
    run_context: RunContext,
    posterior_mode: PosteriorMode,
    ambiguity_level: float = 0.0,
    strategy_override: str | None = None,
) -> dict[str, object]:
    logger = configure_logging(run_context.run_dir)
    dump_config(config, run_context.run_dir / "config.yaml")
    logger.info("Loading or generating dataset.")
    dataset, manifest_path = _load_dataset(config, run_context.run_dir)

    strict_splits = config.dataset.source == "manifest"
    train_examples = _resolve_split(dataset, config.dataset.train_split, fallback="all", strict=strict_splits)
    validation_examples = _resolve_split(
        dataset,
        config.dataset.val_split,
        fallback=config.dataset.train_split,
        strict=strict_splits,
    )
    evaluation_examples = _resolve_split(
        dataset,
        config.dataset.evaluation_split,
        fallback="all",
        strict=strict_splits,
    )
    evaluation_examples = apply_ambiguity_to_examples(
        evaluation_examples,
        ambiguity_level=ambiguity_level,
        seed=config.experiment.seed + 10000,
    )

    train_glyphs = [glyph for example in train_examples for glyph in example.glyphs]
    validation_glyphs = [glyph for example in validation_examples for glyph in example.glyphs]
    evaluation_glyphs = [glyph for example in evaluation_examples for glyph in example.glyphs]
    if not train_glyphs:
        raise ValueError("Posterior model training requires at least one training glyph.")
    if not evaluation_glyphs:
        raise ValueError("Evaluation requires at least one glyph in the evaluation split.")

    train_features = extract_feature_matrix(train_glyphs, downsample=config.vision.feature_downsample)
    validation_features = (
        extract_feature_matrix(validation_glyphs, downsample=config.vision.feature_downsample)
        if validation_glyphs
        else np.empty((0, train_features.shape[1]))
    )
    evaluation_features = extract_feature_matrix(evaluation_glyphs, downsample=config.vision.feature_downsample)

    posterior_strategy = config.posterior.strategy if strategy_override is None else strategy_override
    posterior_model = fit_posterior_model(
        train_features=train_features,
        train_labels=[glyph.true_symbol for glyph in train_glyphs],
        validation_features=validation_features,
        validation_labels=[glyph.true_symbol for glyph in validation_glyphs],
        posterior_config=config.posterior.model_copy(update={"strategy": posterior_strategy}),
        vision_config=config.vision,
        seed=config.experiment.seed,
    )
    posterior = posterior_model.predict_posterior(
        evaluation_features,
        top_k=config.posterior.top_k,
        floor_probability=config.posterior.floor_probability,
    )
    logger.info(
        "Posterior model %s fitted with support size %s.",
        posterior_model.strategy,
        len(posterior_model.support),
    )

    labeled_indices, true_symbols = _labeled_glyph_targets(evaluation_examples)
    logger.info("Clustering %s evaluation glyph crops for inventory diagnostics.", len(evaluation_glyphs))
    cluster_result = cluster_feature_matrix(
        feature_matrix=evaluation_features,
        estimate_clusters=config.vision.estimate_clusters,
        min_clusters=config.vision.min_clusters,
        max_clusters=config.vision.max_clusters,
        seed=config.experiment.seed,
    )
    example_posteriors = split_posterior_by_lengths(
        posterior.collapsed() if posterior_mode == "fixed" else posterior,
        lengths=[example.sequence_length for example in evaluation_examples],
    )

    example_payloads: list[dict[str, object]] = []
    rankings = []
    structural_errors: list[float] = []
    symbol_top1_values: list[tuple[float | None, int]] = []
    symbol_topk_values: list[tuple[float | None, int]] = []
    symbol_nll_values: list[tuple[float | None, int]] = []
    symbol_ece_values: list[tuple[float | None, int]] = []
    correct_entropy_values: list[float] = []
    incorrect_entropy_values: list[float] = []
    labeled_symbol_count = 0

    for index, (example, example_posterior) in enumerate(zip(evaluation_examples, example_posteriors)):
        triage = analyze_posterior(
            family=example.family or "unknown",
            posterior=example_posterior,
            repeat_ngram_sizes=config.triage.repeat_ngram_sizes,
            shuffled_null_trials=config.triage.shuffled_null_trials,
            rng=np.random.default_rng(config.experiment.seed + index),
        )
        ranking = rank_hypotheses(triage, families=config.hypotheses.families)
        reference = (
            sequence_metrics_from_symbols(
                [symbol for symbol in example.observed_symbols if symbol is not None],
                config.triage.repeat_ngram_sizes,
            )
            if example.has_symbol_labels
            else None
        )
        if reference is not None:
            structural_errors.append(structural_recovery_error(triage, reference))
        rankings.append(ranking)
        symbol_top1 = symbol_top_k_accuracy(example_posterior, example.observed_symbols, top_k=1)
        symbol_topk = symbol_top_k_accuracy(example_posterior, example.observed_symbols, top_k=config.evaluation.top_k)
        symbol_nll = symbol_negative_log_likelihood(example_posterior, example.observed_symbols)
        symbol_ece = symbol_expected_calibration_error(example_posterior, example.observed_symbols)
        correct_entropy, incorrect_entropy = symbol_entropy_by_correctness(example_posterior, example.observed_symbols)
        case_breakdown = symbol_case_breakdown(
            example_posterior,
            example.observed_symbols,
            top_k=config.evaluation.top_k,
        )
        symbol_top1_values.append(symbol_top1)
        symbol_topk_values.append(symbol_topk)
        symbol_nll_values.append(symbol_nll)
        symbol_ece_values.append(symbol_ece)
        labeled_symbol_count += symbol_top1[1]
        if correct_entropy is not None:
            correct_entropy_values.append(correct_entropy)
        if incorrect_entropy is not None:
            incorrect_entropy_values.append(incorrect_entropy)
        example_payloads.append(
            {
                "example_id": example.example_id,
                "true_family": example.family,
                "posterior_mode": posterior_mode,
                "split": example.split,
                "triage": triage.to_dict(),
                "ranking": ranking.to_dict(),
                "ranking_object": ranking,
                "posterior": example_posterior.to_dict(),
                "reference_metrics": reference,
                "symbol_metrics": {
                    "top1_accuracy": symbol_top1[0],
                    "topk_accuracy": symbol_topk[0],
                    "negative_log_likelihood": symbol_nll[0],
                    "expected_calibration_error": symbol_ece[0],
                    "correct_entropy": correct_entropy,
                    "incorrect_entropy": incorrect_entropy,
                    "case_breakdown": case_breakdown,
                },
            }
        )

    summary = summarize_rankings(rankings, [example.family for example in evaluation_examples], config.evaluation.top_k)
    summary.update(
        {
            "dataset_name": dataset.dataset_name,
            "example_count": len(evaluation_examples),
            "train_example_count": len(train_examples),
            "validation_example_count": len(validation_examples),
            "glyph_clustering_ari": clustering_ari(
                true_symbols,
                cluster_result.labels[np.asarray(labeled_indices, dtype=int)],
            )
            if true_symbols
            else None,
            "mean_structural_recovery_error": float(np.mean(structural_errors)) if structural_errors else None,
            "mean_posterior_entropy": float(
                np.mean([payload["triage"]["mean_posterior_entropy"] for payload in example_payloads])
            )
            if example_payloads
            else 0.0,
            "symbol_top1_accuracy": _weighted_average(symbol_top1_values),
            "symbol_topk_accuracy": _weighted_average(symbol_topk_values),
            "symbol_negative_log_likelihood": _weighted_average(symbol_nll_values),
            "symbol_expected_calibration_error": _weighted_average(symbol_ece_values),
            "mean_correct_symbol_entropy": float(np.mean(correct_entropy_values)) if correct_entropy_values else None,
            "mean_incorrect_symbol_entropy": float(np.mean(incorrect_entropy_values))
            if incorrect_entropy_values
            else None,
            "labeled_symbol_count": labeled_symbol_count,
            "posterior_mode": posterior_mode,
            "posterior_strategy_requested": posterior_strategy,
            "posterior_strategy": posterior_model.strategy,
            "ambiguity_level": ambiguity_level,
            "evaluation_top_k": config.evaluation.top_k,
            "dataset_manifest": None if manifest_path is None else str(manifest_path),
            "evaluation_split": config.dataset.evaluation_split,
        }
    )

    write_run_artifacts(
        run_dir=run_context.run_dir,
        summary=summary,
        example_payloads=example_payloads,
        cluster_payload=cluster_result.to_dict(),
        posterior_payload=posterior_model.to_dict(),
    )
    logger.info("Completed run in %s", run_context.run_dir)
    return {
        "run_dir": run_context.run_dir,
        "summary": summary,
        "example_payloads": example_payloads,
        "cluster_result": cluster_result,
        "posterior_model": posterior_model,
    }


def run_pipeline(
    config: DecipherLabConfig | str | Path,
    posterior_mode: PosteriorMode = "uncertainty",
    suffix: str | None = None,
    ambiguity_level: float = 0.0,
    strategy_override: str | None = None,
) -> dict[str, object]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    context = prepare_run_context(resolved, suffix=suffix if suffix is not None else posterior_mode)
    return execute_pipeline(
        resolved,
        run_context=context,
        posterior_mode=posterior_mode,
        ambiguity_level=ambiguity_level,
        strategy_override=strategy_override,
    )
