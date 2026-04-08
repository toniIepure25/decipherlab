from __future__ import annotations

import math
import zlib

import networkx as nx
import numpy as np

from decipherlab.models import TriageReport, TranscriptionPosterior


def _normalize(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(value, 0.0) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(clipped), 1)
        return {key: uniform for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def _sequence_to_bytes(sequence: list[str]) -> bytes:
    return "|".join(sequence).encode("utf-8")


def _repeat_rate(sequence: list[str], ngram_sizes: list[int]) -> float:
    rates: list[float] = []
    for ngram_size in ngram_sizes:
        if len(sequence) < ngram_size:
            continue
        ngrams = [tuple(sequence[index : index + ngram_size]) for index in range(len(sequence) - ngram_size + 1)]
        counts: dict[tuple[str, ...], int] = {}
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
        repeated = sum(1 for count in counts.values() if count > 1)
        rates.append(repeated / max(len(counts), 1))
    return float(np.mean(rates)) if rates else 0.0


def _compression_ratio(sequence: list[str]) -> float:
    payload = _sequence_to_bytes(sequence)
    if not payload:
        return 1.0
    return len(zlib.compress(payload)) / max(len(payload), 1)


def _expected_unigram_probabilities(posterior: TranscriptionPosterior) -> dict[str, float]:
    counts: dict[str, float] = {}
    for distribution in posterior.iter_position_distributions():
        for symbol, probability in distribution.items():
            counts[symbol] = counts.get(symbol, 0.0) + probability
    length = max(len(posterior.candidate_ids), 1)
    return {symbol: count / length for symbol, count in counts.items()}


def _expected_bigram_probabilities(posterior: TranscriptionPosterior) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = {}
    distributions = posterior.iter_position_distributions()
    for left, right in zip(distributions[:-1], distributions[1:]):
        for left_symbol, left_probability in left.items():
            for right_symbol, right_probability in right.items():
                pair = (left_symbol, right_symbol)
                counts[pair] = counts.get(pair, 0.0) + (left_probability * right_probability)
    total = max(len(distributions) - 1, 1)
    return {pair: count / total for pair, count in counts.items()}


def sequence_metrics_from_symbols(sequence: list[str], ngram_sizes: list[int]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for symbol in sequence:
        counts[symbol] = counts.get(symbol, 0.0) + 1.0
    total = max(len(sequence), 1)
    probabilities = {symbol: count / total for symbol, count in counts.items()}
    unigram_entropy = -sum(probability * math.log2(probability) for probability in probabilities.values() if probability > 0.0)
    bigram_counts: dict[tuple[str, str], float] = {}
    for left, right in zip(sequence[:-1], sequence[1:]):
        bigram_counts[(left, right)] = bigram_counts.get((left, right), 0.0) + 1.0
    total_bigrams = max(len(sequence) - 1, 1)
    bigram_probabilities = {pair: count / total_bigrams for pair, count in bigram_counts.items()}
    conditional_entropy = 0.0
    context_totals: dict[str, float] = {}
    for (left, _), probability in bigram_probabilities.items():
        context_totals[left] = context_totals.get(left, 0.0) + probability
    for (left, _), probability in bigram_probabilities.items():
        conditional_entropy -= probability * math.log2(probability / context_totals[left])

    graph = nx.DiGraph()
    for (left, right), probability in bigram_probabilities.items():
        graph.add_edge(left, right, weight=probability)
    return {
        "alphabet_size_estimate": float(len(probabilities)),
        "unigram_entropy": unigram_entropy,
        "conditional_entropy": conditional_entropy,
        "index_of_coincidence": sum(probability**2 for probability in probabilities.values()),
        "repeat_rate": _repeat_rate(sequence, ngram_sizes),
        "compression_ratio": _compression_ratio(sequence),
        "adjacency_density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
    }


def analyze_posterior(
    family: str,
    posterior: TranscriptionPosterior,
    repeat_ngram_sizes: list[int],
    shuffled_null_trials: int,
    rng: np.random.Generator,
) -> TriageReport:
    hard_sequence = posterior.hard_sequence()
    unigram_probabilities = _expected_unigram_probabilities(posterior)
    bigram_probabilities = _expected_bigram_probabilities(posterior)
    unigram_entropy = -sum(
        probability * math.log2(probability)
        for probability in unigram_probabilities.values()
        if probability > 0.0
    )

    context_totals: dict[str, float] = {}
    for (left, _), probability in bigram_probabilities.items():
        context_totals[left] = context_totals.get(left, 0.0) + probability
    conditional_entropy = 0.0
    for (left, _), probability in bigram_probabilities.items():
        conditional_entropy -= probability * math.log2(probability / context_totals[left])

    graph = nx.DiGraph()
    for (left, right), probability in bigram_probabilities.items():
        graph.add_edge(left, right, weight=probability)

    repeat_rate = _repeat_rate(hard_sequence, repeat_ngram_sizes)
    compression_ratio = _compression_ratio(hard_sequence)
    null_repeat_rates: list[float] = []
    null_compression_ratios: list[float] = []
    for _ in range(shuffled_null_trials):
        shuffled = list(rng.permutation(hard_sequence))
        null_repeat_rates.append(_repeat_rate(shuffled, repeat_ngram_sizes))
        null_compression_ratios.append(_compression_ratio(shuffled))

    alphabet_size_estimate = sum(
        1 for probability in unigram_probabilities.values() if probability * len(hard_sequence) >= 0.25
    )
    index_of_coincidence = sum(probability**2 for probability in unigram_probabilities.values())
    null_repeat_rate = float(np.mean(null_repeat_rates)) if null_repeat_rates else 0.0
    null_compression_ratio = float(np.mean(null_compression_ratios)) if null_compression_ratios else compression_ratio

    random_ic = 1.0 / max(alphabet_size_estimate, 1)
    repeat_gain = max(repeat_rate - null_repeat_rate, 0.0)
    compression_gain = max(null_compression_ratio - compression_ratio, 0.0)
    conditional_gap = max(unigram_entropy - conditional_entropy, 0.0)
    alphabet_ratio = alphabet_size_estimate / max(len(hard_sequence), 1)
    routing_scores = _normalize(
        {
            "unknown_script": 0.2 + 0.7 * conditional_gap + 0.4 * compression_gain + 0.3 * posterior.mean_entropy(),
            "monoalphabetic": 0.2
            + 1.8 * max(index_of_coincidence - random_ic, 0.0)
            + 1.2 * repeat_gain
            + 1.0 * compression_gain,
            "homophonic": 0.2
            + 1.1 * conditional_gap
            + 1.0 * alphabet_ratio
            + 0.6 * compression_gain
            + 0.4 * max(random_ic - index_of_coincidence, 0.0),
            "transposition_heuristic": 0.2
            + 1.1 * max(index_of_coincidence - random_ic, 0.0)
            + 0.7 * max((conditional_entropy / max(unigram_entropy, 1.0e-6)) - 0.85, 0.0)
            - 0.4 * repeat_gain,
            "pseudo_text_null": 0.2
            + (1.0 / (1.0 + 8.0 * (repeat_gain + compression_gain + conditional_gap)))
            + 0.5 * max(random_ic - index_of_coincidence, 0.0),
        }
    )

    return TriageReport(
        family=family,
        sequence_length=len(hard_sequence),
        alphabet_size_estimate=alphabet_size_estimate,
        unigram_entropy=unigram_entropy,
        conditional_entropy=conditional_entropy,
        index_of_coincidence=index_of_coincidence,
        repeat_rate=repeat_rate,
        compression_ratio=compression_ratio,
        adjacency_density=nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
        mean_posterior_entropy=posterior.mean_entropy(),
        null_repeat_rate=null_repeat_rate,
        null_compression_ratio=null_compression_ratio,
        routing_scores=routing_scores,
        diagnostics={
            "repeat_gain": repeat_gain,
            "compression_gain": compression_gain,
            "conditional_gap": conditional_gap,
            "alphabet_ratio": alphabet_ratio,
        },
    )
