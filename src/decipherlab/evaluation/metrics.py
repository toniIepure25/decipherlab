from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score

from decipherlab.models import HypothesisRanking, TriageReport, TranscriptionPosterior


def canonical_family_name(family: str) -> str:
    mapping = {
        "transposition": "transposition_heuristic",
        "pseudo_text": "pseudo_text_null",
    }
    return mapping.get(family, family)


def clustering_ari(true_symbols: list[str], cluster_labels: np.ndarray) -> float:
    return float(adjusted_rand_score(true_symbols, cluster_labels.tolist()))


def top_k_success(ranking: HypothesisRanking, true_family: str, top_k: int) -> float:
    canonical = canonical_family_name(true_family)
    ranked = [evidence.family for evidence in ranking.evidences[:top_k]]
    return float(canonical in ranked)


def brier_score(ranking: HypothesisRanking, true_family: str) -> float:
    canonical = canonical_family_name(true_family)
    families = [evidence.family for evidence in ranking.evidences]
    probabilities = np.asarray([evidence.probability for evidence in ranking.evidences], dtype=float)
    targets = np.asarray([1.0 if family == canonical else 0.0 for family in families], dtype=float)
    return float(np.mean((probabilities - targets) ** 2))


def _filter_symbol_targets(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    labels: list[str] = []
    for index, label in enumerate(true_symbols):
        if label is None:
            continue
        indices.append(index)
        labels.append(label)
    return indices, labels


def symbol_top_k_accuracy(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
    top_k: int,
) -> tuple[float | None, int]:
    indices, labels = _filter_symbol_targets(posterior, true_symbols)
    if not labels:
        return None, 0
    hits = 0
    for position, label in zip(indices, labels):
        if label in posterior.candidate_ids[position][:top_k]:
            hits += 1
    return hits / len(labels), len(labels)


def symbol_negative_log_likelihood(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
) -> tuple[float | None, int]:
    indices, labels = _filter_symbol_targets(posterior, true_symbols)
    if not labels:
        return None, 0
    nll_values: list[float] = []
    for position, label in zip(indices, labels):
        distribution = posterior.iter_position_distributions()[position]
        if label in distribution:
            probability = distribution[label]
        else:
            probability = 1.0e-8
        nll_values.append(-float(np.log(probability)))
    return float(np.mean(nll_values)), len(labels)


def symbol_expected_calibration_error(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
    bins: int = 10,
) -> tuple[float | None, int]:
    indices, labels = _filter_symbol_targets(posterior, true_symbols)
    if not labels:
        return None, 0
    probabilities = posterior.probabilities()
    confidences = probabilities[indices, 0]
    correctness = np.asarray(
        [float(posterior.candidate_ids[position][0] == label) for position, label in zip(indices, labels)],
        dtype=float,
    )
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += abs(bin_accuracy - bin_confidence) * (np.sum(mask) / len(confidences))
    return float(ece), len(labels)


def symbol_entropy_by_correctness(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
) -> tuple[float | None, float | None]:
    indices, labels = _filter_symbol_targets(posterior, true_symbols)
    if not labels:
        return None, None
    entropies = posterior.entropy_per_position()
    correct: list[float] = []
    incorrect: list[float] = []
    for position, label in zip(indices, labels):
        if posterior.candidate_ids[position][0] == label:
            correct.append(float(entropies[position]))
        else:
            incorrect.append(float(entropies[position]))
    return (
        float(np.mean(correct)) if correct else None,
        float(np.mean(incorrect)) if incorrect else None,
    )


def symbol_case_breakdown(
    posterior: TranscriptionPosterior,
    true_symbols: list[str | None],
    top_k: int,
) -> dict[str, int]:
    indices, labels = _filter_symbol_targets(posterior, true_symbols)
    breakdown = {
        "labeled_count": len(labels),
        "top1_hits": 0,
        "topk_hits": 0,
        "collapse_rescued_by_topk": 0,
        "missing_from_topk": 0,
    }
    for position, label in zip(indices, labels):
        candidates = posterior.candidate_ids[position]
        if candidates and candidates[0] == label:
            breakdown["top1_hits"] += 1
        if label in candidates[:top_k]:
            breakdown["topk_hits"] += 1
            if not candidates or candidates[0] != label:
                breakdown["collapse_rescued_by_topk"] += 1
        else:
            breakdown["missing_from_topk"] += 1
    return breakdown


def expected_calibration_error(rankings: list[HypothesisRanking], true_families: list[str], bins: int = 10) -> float:
    confidences: list[float] = []
    correctness: list[float] = []
    for ranking, true_family in zip(rankings, true_families):
        canonical = canonical_family_name(true_family)
        top = ranking.best
        confidences.append(top.probability)
        correctness.append(float(top.family == canonical))
    if not confidences:
        return 0.0
    confidences_arr = np.asarray(confidences)
    correctness_arr = np.asarray(correctness)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences_arr >= lower) & (confidences_arr < upper if upper < 1.0 else confidences_arr <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correctness_arr[mask]))
        bin_confidence = float(np.mean(confidences_arr[mask]))
        ece += abs(bin_accuracy - bin_confidence) * (np.sum(mask) / len(confidences_arr))
    return float(ece)


def structural_recovery_error(report: TriageReport, reference_metrics: dict[str, float]) -> float:
    fields = [
        "unigram_entropy",
        "conditional_entropy",
        "index_of_coincidence",
        "repeat_rate",
        "compression_ratio",
    ]
    deltas = [abs(getattr(report, field) - reference_metrics[field]) for field in fields]
    return float(np.mean(deltas))


def summarize_rankings(
    rankings: list[HypothesisRanking],
    true_families: list[str | None],
    top_k: int,
) -> dict[str, float | None]:
    filtered = [
        (ranking, true_family)
        for ranking, true_family in zip(rankings, true_families)
        if true_family is not None
    ]
    if not filtered:
        return {
            "family_top1_accuracy": None,
            "family_topk_accuracy": None,
            "mean_brier_score": None,
            "expected_calibration_error": None,
        }
    top1 = [
        float(ranking.best.family == canonical_family_name(true_family))
        for ranking, true_family in filtered
    ]
    topk = [top_k_success(ranking, true_family, top_k) for ranking, true_family in filtered]
    briers = [brier_score(ranking, true_family) for ranking, true_family in filtered]
    return {
        "family_top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "family_topk_accuracy": float(np.mean(topk)) if topk else 0.0,
        "mean_brier_score": float(np.mean(briers)) if briers else 0.0,
        "expected_calibration_error": expected_calibration_error(
            [ranking for ranking, _ in filtered],
            [true_family for _, true_family in filtered],
        ),
    }


def average_probabilities_by_true_family(
    rankings: list[HypothesisRanking],
    true_families: list[str | None],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ranking, true_family in zip(rankings, true_families):
        if true_family is None:
            continue
        canonical = canonical_family_name(true_family)
        for evidence in ranking.evidences:
            grouped[canonical][evidence.family].append(evidence.probability)
    averaged: dict[str, dict[str, float]] = {}
    for family, payload in grouped.items():
        averaged[family] = {name: float(np.mean(values)) for name, values in payload.items()}
    return averaged
