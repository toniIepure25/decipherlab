from __future__ import annotations

import numpy as np

from decipherlab.models import TranscriptionPosterior


def test_posterior_normalizes_and_truncates_top_k() -> None:
    posterior = TranscriptionPosterior.from_scores(
        support=["a", "b", "c"],
        scores=np.asarray([[4.0, 1.0, 0.0], [0.5, 1.0, 3.0]], dtype=float),
        top_k=2,
        floor_probability=1.0e-6,
    )
    probabilities = posterior.probabilities()
    assert probabilities.shape == (2, 2)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert posterior.hard_sequence() == ["a", "c"]
    assert posterior.candidate_ids[0] == ["a", "b"]
