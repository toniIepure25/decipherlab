from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from decipherlab.models import GlyphClusterResult, TranscriptionPosterior


def build_posterior_from_clusters(
    cluster_result: GlyphClusterResult,
    top_k: int,
    temperature: float,
    floor_probability: float,
) -> TranscriptionPosterior:
    distances = cdist(cluster_result.feature_matrix, cluster_result.centroids, metric="euclidean")
    scores = -distances / max(temperature, 1.0e-6)
    return TranscriptionPosterior.from_scores(
        support=cluster_result.inventory,
        scores=scores,
        top_k=top_k,
        floor_probability=floor_probability,
    )


def split_posterior_by_lengths(
    posterior: TranscriptionPosterior,
    lengths: list[int],
) -> list[TranscriptionPosterior]:
    sliced: list[TranscriptionPosterior] = []
    start = 0
    for length in lengths:
        end = start + length
        sliced.append(
            TranscriptionPosterior(
                candidate_ids=posterior.candidate_ids[start:end],
                log_probabilities=posterior.log_probabilities[start:end],
                boundary_probabilities=None
                if posterior.boundary_probabilities is None
                else posterior.boundary_probabilities[start:max(start, end - 1)],
            )
        )
        start = end
    return sliced
