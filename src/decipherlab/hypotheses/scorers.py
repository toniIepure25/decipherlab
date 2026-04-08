from __future__ import annotations

import math

import numpy as np

from decipherlab.models import HypothesisEvidence, HypothesisRanking, TriageReport


def _softmax(scores: list[float]) -> list[float]:
    values = np.asarray(scores, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities.tolist()


def _random_ic(report: TriageReport) -> float:
    return 1.0 / max(report.alphabet_size_estimate, 1)


def _structure_gain(report: TriageReport) -> float:
    return max(report.diagnostics.get("repeat_gain", 0.0), 0.0) + max(
        report.diagnostics.get("compression_gain", 0.0),
        0.0,
    )


def _conditional_gap(report: TriageReport) -> float:
    return max(report.diagnostics.get("conditional_gap", 0.0), 0.0)


def _alphabet_ratio(report: TriageReport) -> float:
    return report.diagnostics.get("alphabet_ratio", 0.0)


def _score_unknown_script(report: TriageReport) -> tuple[float, list[str], dict[str, float], list[str]]:
    structure_gain = _structure_gain(report)
    score = 0.3 + 0.8 * structure_gain + 0.9 * _conditional_gap(report) + 0.5 * report.mean_posterior_entropy
    rationale = [
        f"Moderate structure gain over nulls ({structure_gain:.3f}) keeps a script-like explanation viable.",
        f"Posterior uncertainty remains non-trivial ({report.mean_posterior_entropy:.3f}), so overcommitting would be premature.",
    ]
    caveats = ["Heuristic family score. This does not identify language or semantics."]
    return score, rationale, {"structure_gain": structure_gain}, caveats


def _score_monoalphabetic(report: TriageReport) -> tuple[float, list[str], dict[str, float], list[str]]:
    ic_excess = max(report.index_of_coincidence - _random_ic(report), 0.0)
    structure_gain = _structure_gain(report)
    score = 0.2 + 2.2 * ic_excess + 1.5 * structure_gain + 0.9 * _conditional_gap(report) - 0.4 * _alphabet_ratio(report)
    rationale = [
        f"Index of coincidence exceeds the random baseline by {ic_excess:.3f}.",
        f"Repeat and compression gains over shuffled controls total {structure_gain:.3f}.",
    ]
    caveats = ["Heuristic score based on structural regularities, not recovered keys."]
    return score, rationale, {"ic_excess": ic_excess, "structure_gain": structure_gain}, caveats


def _score_homophonic(report: TriageReport) -> tuple[float, list[str], dict[str, float], list[str]]:
    ic_excess = max(report.index_of_coincidence - _random_ic(report), 0.0)
    ic_flattening = max(_random_ic(report) - report.index_of_coincidence, 0.0)
    structure_gain = _structure_gain(report)
    alphabet_ratio = _alphabet_ratio(report)
    alphabet_excess = max(alphabet_ratio - 0.35, 0.0)
    score = (
        0.2
        + 0.7 * _conditional_gap(report)
        + 3.0 * alphabet_excess
        + 0.6 * structure_gain
        + 1.2 * ic_flattening
        - 1.8 * ic_excess
    )
    rationale = [
        f"Alphabet ratio is relatively high ({alphabet_ratio:.3f}), with excess inflation {alphabet_excess:.3f}.",
        f"Conditional structure remains above null expectations ({_conditional_gap(report):.3f}) without a strong monoalphabetic IC excess ({ic_excess:.3f}).",
    ]
    caveats = ["Heuristic score. Homophonic support is suggestive, not a proof of key structure."]
    return score, rationale, {"alphabet_ratio": alphabet_ratio, "ic_flattening": ic_flattening, "ic_excess": ic_excess}, caveats


def _score_transposition(report: TriageReport) -> tuple[float, list[str], dict[str, float], list[str]]:
    ic_excess = max(report.index_of_coincidence - _random_ic(report), 0.0)
    adjacency_loss = max(
        (report.conditional_entropy / max(report.unigram_entropy, 1.0e-6)) - 0.85,
        0.0,
    )
    repeat_gain = max(report.diagnostics.get("repeat_gain", 0.0), 0.0)
    score = 0.2 + 1.5 * ic_excess + 1.2 * adjacency_loss + 0.5 * max(report.diagnostics.get("compression_gain", 0.0), 0.0) - 0.7 * repeat_gain
    rationale = [
        f"Unigram concentration remains non-random (IC excess {ic_excess:.3f}).",
        f"Local adjacency structure is comparatively weak (adjacency loss {adjacency_loss:.3f}), which is compatible with transposition.",
    ]
    caveats = ["Transposition support is intentionally heuristic in this MVP."]
    return score, rationale, {"ic_excess": ic_excess, "adjacency_loss": adjacency_loss}, caveats


def _score_pseudo_text(report: TriageReport) -> tuple[float, list[str], dict[str, float], list[str]]:
    structure_gain = _structure_gain(report)
    anti_structure = math.exp(-6.0 * structure_gain)
    ic_excess = max(report.index_of_coincidence - _random_ic(report), 0.0)
    score = (
        0.7
        + 1.5 * anti_structure
        + 0.6 * max(report.compression_ratio - report.null_compression_ratio, 0.0)
        - 0.5 * ic_excess
    )
    rationale = [
        f"Structure gains over nulls are weak ({structure_gain:.3f}), which favors a null-style explanation.",
        f"Index of coincidence exceeds the random baseline by only {ic_excess:.3f}.",
    ]
    caveats = ["Null-style support means the system did not find strong evidence for richer structure."]
    return score, rationale, {"anti_structure": anti_structure, "ic_excess": ic_excess}, caveats


_SCORERS = {
    "unknown_script": _score_unknown_script,
    "monoalphabetic": _score_monoalphabetic,
    "homophonic": _score_homophonic,
    "transposition_heuristic": _score_transposition,
    "pseudo_text_null": _score_pseudo_text,
}


def rank_hypotheses(report: TriageReport, families: list[str]) -> HypothesisRanking:
    raw_scores: list[float] = []
    payloads: list[tuple[str, list[str], dict[str, float], list[str]]] = []
    for family in families:
        if family not in _SCORERS:
            raise ValueError(f"Unsupported hypothesis family: {family}")
        score, rationale, diagnostics, caveats = _SCORERS[family](report)
        raw_scores.append(score)
        payloads.append((family, rationale, diagnostics, caveats))

    probabilities = _softmax(raw_scores)
    evidences: list[HypothesisEvidence] = []
    for (family, rationale, diagnostics, caveats), score, probability in zip(payloads, raw_scores, probabilities):
        evidences.append(
            HypothesisEvidence(
                family=family,
                score=float(score),
                probability=float(probability),
                rationale=rationale,
                diagnostics=diagnostics,
                caveats=caveats,
                heuristic=True,
            )
        )

    evidences.sort(key=lambda evidence: (-evidence.score, evidence.family))
    return HypothesisRanking(evidences=evidences)
