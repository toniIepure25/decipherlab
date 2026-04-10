from __future__ import annotations

import numpy as np

from decipherlab.decoding.beam_search import (
    BigramTransitionModel,
    TrigramTransitionModel,
    beam_decode_confusion_network,
    greedy_decode_confusion_network,
    trigram_beam_decode_confusion_network,
)
from decipherlab.decoding.crf import crf_viterbi_decode_confusion_network
from decipherlab.structured_uncertainty.confusion_network import (
    ConfusionNetwork,
    ConfusionNetworkPosition,
)


def test_bigram_beam_decoder_can_recover_sequence_greedy_misses():
    network = ConfusionNetwork(
        positions=[
            ConfusionNetworkPosition(
                position=0,
                candidate_ids=["a", "b"],
                log_probabilities=np.log(np.asarray([0.55, 0.45], dtype=float)),
                source_entropy=0.68,
                retained_probability_mass=1.0,
            ),
            ConfusionNetworkPosition(
                position=1,
                candidate_ids=["b", "a"],
                log_probabilities=np.log(np.asarray([0.55, 0.45], dtype=float)),
                source_entropy=0.68,
                retained_probability_mass=1.0,
            ),
        ]
    )
    transition_model = BigramTransitionModel.fit(
        sequences=[["b", "a"], ["b", "a"], ["b", "a"], ["a", "a"]],
        smoothing=0.1,
    )

    greedy = greedy_decode_confusion_network(network)
    beam = beam_decode_confusion_network(
        network=network,
        transition_model=transition_model,
        beam_width=4,
        lm_weight=2.5,
        top_k_sequences=3,
        length_normalize=True,
    )

    assert greedy.best.symbols == ["a", "b"]
    assert beam.best.symbols == ["b", "a"]


def test_trigram_beam_decoder_can_use_higher_order_context():
    network = ConfusionNetwork(
        positions=[
            ConfusionNetworkPosition(
                position=0,
                candidate_ids=["a", "b"],
                log_probabilities=np.log(np.asarray([0.51, 0.49], dtype=float)),
                source_entropy=0.69,
                retained_probability_mass=1.0,
            ),
            ConfusionNetworkPosition(
                position=1,
                candidate_ids=["a", "b"],
                log_probabilities=np.log(np.asarray([0.51, 0.49], dtype=float)),
                source_entropy=0.69,
                retained_probability_mass=1.0,
            ),
            ConfusionNetworkPosition(
                position=2,
                candidate_ids=["b", "a"],
                log_probabilities=np.log(np.asarray([0.51, 0.49], dtype=float)),
                source_entropy=0.69,
                retained_probability_mass=1.0,
            ),
        ]
    )
    trigram_model = TrigramTransitionModel.fit(
        sequences=[["b", "a", "a"], ["b", "a", "a"], ["a", "a", "b"]],
        smoothing=0.1,
    )

    decoded = trigram_beam_decode_confusion_network(
        network=network,
        transition_model=trigram_model,
        beam_width=8,
        lm_weight=3.0,
        top_k_sequences=4,
        length_normalize=True,
    )

    assert decoded.best.symbols == ["b", "a", "a"]


def test_crf_viterbi_decoder_matches_bigram_recovery_case():
    network = ConfusionNetwork(
        positions=[
            ConfusionNetworkPosition(
                position=0,
                candidate_ids=["a", "b"],
                log_probabilities=np.log(np.asarray([0.55, 0.45], dtype=float)),
                source_entropy=0.68,
                retained_probability_mass=1.0,
            ),
            ConfusionNetworkPosition(
                position=1,
                candidate_ids=["b", "a"],
                log_probabilities=np.log(np.asarray([0.55, 0.45], dtype=float)),
                source_entropy=0.68,
                retained_probability_mass=1.0,
            ),
        ]
    )
    transition_model = BigramTransitionModel.fit(
        sequences=[["b", "a"], ["b", "a"], ["b", "a"], ["a", "a"]],
        smoothing=0.1,
    )

    decoded = crf_viterbi_decode_confusion_network(
        network=network,
        transition_model=transition_model,
        lm_weight=2.5,
    )

    assert decoded.best.symbols == ["b", "a"]
