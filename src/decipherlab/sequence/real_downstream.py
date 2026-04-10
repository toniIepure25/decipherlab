from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from typing import Any

from decipherlab.config import DecodingConfig, RealDownstreamConfig
from decipherlab.decoding.beam_search import (
    BeamDecodingResult,
    BigramTransitionModel,
    DecodedSequence,
    TrigramTransitionModel,
)
from decipherlab.models import SequenceExample, TranscriptionPosterior
from decipherlab.sequence.metrics import sequence_metric_bundle


@dataclass(frozen=True)
class TranscriptBankEntry:
    transcript: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class TranscriptBank:
    by_length: dict[int, list[TranscriptBankEntry]]
    metadata: dict[str, Any]

    def candidates_for_length(self, length: int) -> list[TranscriptBankEntry]:
        return self.by_length.get(length, [])


@dataclass(frozen=True)
class SupportedNGramInventory:
    order: int
    inventory: set[tuple[str, ...]]
    metadata: dict[str, Any]


def build_transcript_bank(
    examples: list[SequenceExample],
    config: RealDownstreamConfig,
) -> TranscriptBank:
    counts_by_length: dict[int, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for example in examples:
        transcript = tuple(symbol for symbol in example.observed_symbols if symbol is not None)
        if transcript:
            counts_by_length[len(transcript)][transcript] += 1
    return TranscriptBank(
        by_length={
            length: [
                TranscriptBankEntry(transcript=transcript, count=count)
                for transcript, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
                if count >= config.min_frequency
            ]
            for length, counter in counts_by_length.items()
        },
        metadata={
            "task_name": config.task_name,
            "length_count": len(counts_by_length),
            "min_frequency": config.min_frequency,
            "transcript_top_k": config.transcript_top_k,
        },
    )


def _ngram_tokens(transcript: tuple[str, ...], order: int) -> list[tuple[str, ...]]:
    if len(transcript) < order:
        return []
    return [tuple(transcript[index : index + order]) for index in range(len(transcript) - order + 1)]


def build_supported_ngram_inventory(
    examples: list[SequenceExample],
    config: RealDownstreamConfig,
) -> SupportedNGramInventory:
    order = config.ngram_order
    inventory: set[tuple[str, ...]] = set()
    for example in examples:
        transcript = tuple(symbol for symbol in example.observed_symbols if symbol is not None)
        inventory.update(_ngram_tokens(transcript, order))
    return SupportedNGramInventory(
        order=order,
        inventory=inventory,
        metadata={
            "task_name": config.task_name,
            "ngram_order": order,
            "inventory_size": len(inventory),
            "min_supported_ngrams": config.min_supported_ngrams,
        },
    )


def build_real_downstream_resource(
    examples: list[SequenceExample],
    config: RealDownstreamConfig,
) -> TranscriptBank | SupportedNGramInventory:
    if config.task_name == "train_transcript_bank":
        return build_transcript_bank(examples, config)
    if config.task_name in {"train_supported_ngram_path", "train_supported_trigram_path"}:
        return build_supported_ngram_inventory(examples, config)
    raise ValueError(f"Unsupported real downstream task: {config.task_name}")


def _visual_log_score(
    posterior: TranscriptionPosterior,
    transcript: tuple[str, ...],
    missing_log_probability: float,
) -> float:
    fallback = float(log(missing_log_probability))
    return float(
        sum(
            distribution.get(symbol, fallback)
            for distribution, symbol in zip(posterior.iter_position_distributions(), transcript)
        )
    )


def _bigram_score(
    transcript: tuple[str, ...],
    transition_model: BigramTransitionModel,
    lm_weight: float,
) -> float:
    if not transcript:
        return 0.0
    total = lm_weight * transition_model.log_start(transcript[0])
    for left, right in zip(transcript[:-1], transcript[1:]):
        total += lm_weight * transition_model.log_transition(left, right)
    return float(total)


def _trigram_score(
    transcript: tuple[str, ...],
    transition_model: TrigramTransitionModel,
    lm_weight: float,
) -> float:
    if not transcript:
        return 0.0
    if len(transcript) == 1:
        return float(lm_weight * transition_model.log_start(transcript[0]))
    total = lm_weight * transition_model.log_start(transcript[0])
    total += lm_weight * transition_model.log_second(transcript[0], transcript[1])
    for left, center, right in zip(transcript[:-2], transcript[1:-1], transcript[2:]):
        total += lm_weight * transition_model.log_transition(left, center, right)
    return float(total)


def _structural_score(
    method: str,
    transcript: tuple[str, ...],
    decoding_config: DecodingConfig,
    bigram_transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
) -> float:
    if method == "fixed_greedy":
        return 0.0
    if method in {"uncertainty_trigram_beam", "conformal_trigram_beam"} and trigram_transition_model is not None:
        weight = (
            decoding_config.lm_weight
            if decoding_config.trigram_lm_weight is None
            else decoding_config.trigram_lm_weight
        )
        return _trigram_score(transcript, trigram_transition_model, weight)
    return _bigram_score(transcript, bigram_transition_model, decoding_config.lm_weight)


def _ngram_label(ngram: tuple[str, ...]) -> str:
    return "|".join(ngram)


def _supported_ngram_path(
    transcript: tuple[str, ...],
    inventory: SupportedNGramInventory,
) -> tuple[list[str], int]:
    all_ngrams = _ngram_tokens(transcript, inventory.order)
    supported = [ngram for ngram in all_ngrams if ngram in inventory.inventory]
    return [_ngram_label(ngram) for ngram in supported], len(all_ngrams)


def downstream_transcript_bank_payload(
    method: str,
    posterior: TranscriptionPosterior,
    truth: list[str | None],
    transcript_bank: TranscriptBank,
    real_downstream_config: RealDownstreamConfig,
    decoding_config: DecodingConfig,
    bigram_transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
) -> dict[str, Any]:
    truth_symbols = [symbol for symbol in truth if symbol is not None]
    target_length = len(truth_symbols)
    truth_tuple = tuple(truth_symbols)
    candidates = transcript_bank.candidates_for_length(target_length)
    coverage = any(entry.transcript == truth_tuple for entry in candidates)
    if not candidates:
        return {
            "real_downstream_supported": False,
            "real_downstream_task_name": real_downstream_config.task_name,
            "real_downstream_bank_size": 0,
            "real_downstream_bank_coverage": 0.0,
            "real_downstream_exact_match": None,
            "real_downstream_topk_recovery": None,
            "real_downstream_token_accuracy": None,
            "real_downstream_cer": None,
            "real_downstream_exact_match_if_covered": None,
            "real_downstream_topk_recovery_if_covered": None,
            "real_downstream_best_transcript": [],
        }

    ranked_sequences: list[DecodedSequence] = []
    for entry in candidates:
        visual_score = _visual_log_score(
            posterior,
            entry.transcript,
            real_downstream_config.missing_log_probability,
        )
        structural_score = _structural_score(
            method,
            entry.transcript,
            decoding_config,
            bigram_transition_model,
            trigram_transition_model,
        )
        ranked_sequences.append(
            DecodedSequence(
                symbols=list(entry.transcript),
                total_score=visual_score + structural_score,
                visual_score=visual_score,
                structural_score=structural_score,
            )
        )
    ranked_sequences.sort(key=lambda item: (item.total_score, item.visual_score), reverse=True)
    bank_result = BeamDecodingResult(
        sequences=ranked_sequences[: real_downstream_config.transcript_top_k],
        method=f"{method}_train_transcript_bank",
    )
    metrics = sequence_metric_bundle(bank_result, truth_symbols)
    return {
        "real_downstream_supported": True,
        "real_downstream_task_name": real_downstream_config.task_name,
        "real_downstream_bank_size": len(candidates),
        "real_downstream_bank_coverage": float(coverage),
        "real_downstream_exact_match": metrics["sequence_exact_match"],
        "real_downstream_topk_recovery": metrics["sequence_topk_recovery"],
        "real_downstream_token_accuracy": metrics["sequence_token_accuracy"],
        "real_downstream_cer": metrics["sequence_cer"],
        "real_downstream_exact_match_if_covered": metrics["sequence_exact_match"] if coverage else None,
        "real_downstream_topk_recovery_if_covered": metrics["sequence_topk_recovery"] if coverage else None,
        "real_downstream_best_transcript": bank_result.best.symbols if bank_result.sequences else [],
    }


def downstream_supported_ngram_path_payload(
    decoded: BeamDecodingResult,
    truth: list[str | None],
    inventory: SupportedNGramInventory,
    real_downstream_config: RealDownstreamConfig,
) -> dict[str, Any]:
    truth_symbols = tuple(symbol for symbol in truth if symbol is not None)
    truth_path, total_ngrams = _supported_ngram_path(truth_symbols, inventory)
    if total_ngrams <= 0 or len(truth_path) < real_downstream_config.min_supported_ngrams:
        return {
            "real_downstream_supported": False,
            "real_downstream_task_name": real_downstream_config.task_name,
            "real_downstream_bank_size": len(inventory.inventory),
            "real_downstream_bank_coverage": 0.0,
            "real_downstream_exact_match": None,
            "real_downstream_topk_recovery": None,
            "real_downstream_token_accuracy": None,
            "real_downstream_cer": None,
            "real_downstream_exact_match_if_covered": None,
            "real_downstream_topk_recovery_if_covered": None,
            "real_downstream_best_transcript": [],
        }

    derived_sequences = [
        DecodedSequence(
            symbols=_supported_ngram_path(tuple(sequence.symbols), inventory)[0],
            total_score=sequence.total_score,
            visual_score=sequence.visual_score,
            structural_score=sequence.structural_score,
        )
        for sequence in decoded.sequences
    ]
    derived_result = BeamDecodingResult(
        sequences=derived_sequences,
        method=f"{decoded.method}_train_supported_ngram_path",
    )
    metrics = sequence_metric_bundle(derived_result, truth_path)
    full_coverage = float(len(truth_path) == total_ngrams)
    coverage_fraction = float(len(truth_path) / total_ngrams) if total_ngrams > 0 else 0.0
    return {
        "real_downstream_supported": True,
        "real_downstream_task_name": real_downstream_config.task_name,
        "real_downstream_bank_size": len(inventory.inventory),
        "real_downstream_bank_coverage": coverage_fraction,
        "real_downstream_exact_match": metrics["sequence_exact_match"],
        "real_downstream_topk_recovery": metrics["sequence_topk_recovery"],
        "real_downstream_token_accuracy": metrics["sequence_token_accuracy"],
        "real_downstream_cer": metrics["sequence_cer"],
        "real_downstream_exact_match_if_covered": metrics["sequence_exact_match"] if full_coverage else None,
        "real_downstream_topk_recovery_if_covered": metrics["sequence_topk_recovery"] if full_coverage else None,
        "real_downstream_best_transcript": derived_result.best.symbols if derived_result.sequences else [],
    }


def downstream_payload(
    method: str,
    decoded: BeamDecodingResult,
    posterior: TranscriptionPosterior,
    truth: list[str | None],
    downstream_resource: TranscriptBank | SupportedNGramInventory,
    real_downstream_config: RealDownstreamConfig,
    decoding_config: DecodingConfig,
    bigram_transition_model: BigramTransitionModel,
    trigram_transition_model: TrigramTransitionModel | None,
) -> dict[str, Any]:
    if isinstance(downstream_resource, TranscriptBank):
        return downstream_transcript_bank_payload(
            method=method,
            posterior=posterior,
            truth=truth,
            transcript_bank=downstream_resource,
            real_downstream_config=real_downstream_config,
            decoding_config=decoding_config,
            bigram_transition_model=bigram_transition_model,
            trigram_transition_model=trigram_transition_model,
        )
    return downstream_supported_ngram_path_payload(
        decoded=decoded,
        truth=truth,
        inventory=downstream_resource,
        real_downstream_config=real_downstream_config,
    )


def downstream_supported_trigram_path_payload(
    decoded: BeamDecodingResult,
    truth: list[str | None],
    inventory: SupportedNGramInventory,
    real_downstream_config: RealDownstreamConfig,
) -> dict[str, Any]:
    return downstream_supported_ngram_path_payload(
        decoded=decoded,
        truth=truth,
        inventory=inventory,
        real_downstream_config=real_downstream_config,
    )
