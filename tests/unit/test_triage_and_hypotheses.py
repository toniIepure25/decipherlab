from __future__ import annotations

import numpy as np

from decipherlab.hypotheses.scorers import rank_hypotheses
from decipherlab.models import TranscriptionPosterior
from decipherlab.structure.triage import analyze_posterior, sequence_metrics_from_symbols


def _posterior_from_sequence(sequence: list[str]) -> TranscriptionPosterior:
    return TranscriptionPosterior(
        candidate_ids=[[symbol] for symbol in sequence],
        log_probabilities=np.zeros((len(sequence), 1), dtype=float),
    )


def test_sequence_metrics_capture_repeat_structure() -> None:
    repeated = sequence_metrics_from_symbols(list("abababab"), [2])
    random_like = sequence_metrics_from_symbols(list("abcdefgh"), [2])
    assert repeated["repeat_rate"] > random_like["repeat_rate"]
    assert repeated["compression_ratio"] < random_like["compression_ratio"]


def test_hypothesis_ranker_prefers_clear_monoalphabetic_signal(small_config) -> None:
    sequence = list("abcabcabddabcabcabddabcabcabddabcabcabdd")
    report = analyze_posterior(
        family="monoalphabetic",
        posterior=_posterior_from_sequence(sequence),
        repeat_ngram_sizes=small_config.triage.repeat_ngram_sizes,
        shuffled_null_trials=small_config.triage.shuffled_null_trials,
        rng=np.random.default_rng(small_config.experiment.seed),
    )
    ranking = rank_hypotheses(report, small_config.hypotheses.families)
    assert ranking.best.family == "monoalphabetic"


def test_hypothesis_ranker_prefers_clear_pseudo_text_signal(small_config) -> None:
    rng = np.random.default_rng(0)
    sequence = list(rng.choice([f"s{index}" for index in range(20)], size=120, replace=True))
    report = analyze_posterior(
        family="pseudo_text",
        posterior=_posterior_from_sequence(sequence),
        repeat_ngram_sizes=small_config.triage.repeat_ngram_sizes,
        shuffled_null_trials=small_config.triage.shuffled_null_trials,
        rng=np.random.default_rng(small_config.experiment.seed),
    )
    ranking = rank_hypotheses(report, small_config.hypotheses.families)
    assert ranking.best.family == "pseudo_text_null"


def test_homophonic_scorer_beats_null_distractors_on_inflated_inventory_sequence(small_config) -> None:
    rng = np.random.default_rng(1)
    underlying = "abcde" * 30
    pools = {character: [f"{character}{index}" for index in range(6)] for character in "abcde"}
    sequence = [str(rng.choice(pools[character])) for character in underlying]
    report = analyze_posterior(
        family="homophonic",
        posterior=_posterior_from_sequence(sequence),
        repeat_ngram_sizes=small_config.triage.repeat_ngram_sizes,
        shuffled_null_trials=small_config.triage.shuffled_null_trials,
        rng=np.random.default_rng(small_config.experiment.seed),
    )
    ranking = rank_hypotheses(report, small_config.hypotheses.families)
    scores = {evidence.family: evidence.score for evidence in ranking.evidences}
    assert scores["homophonic"] > scores["pseudo_text_null"]
    assert scores["homophonic"] > scores["transposition_heuristic"]
