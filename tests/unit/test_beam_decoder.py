from __future__ import annotations

import numpy as np

from decipherlab.decoding.beam_search import (
    BigramTransitionModel,
    beam_decode_confusion_network,
    greedy_decode_confusion_network,
)
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
