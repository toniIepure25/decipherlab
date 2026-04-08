from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from decipherlab.models import GlyphClusterResult


def _estimate_cluster_count(
    feature_matrix: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    seed: int,
) -> tuple[int, float | None]:
    sample_count = feature_matrix.shape[0]
    upper = min(max_clusters, sample_count - 1)
    lower = max(2, min_clusters)
    if upper < lower:
        return max(1, min(sample_count, lower)), None

    best_k = lower
    best_score = float("-inf")
    for k in range(lower, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(feature_matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(feature_matrix, labels)
        if score > best_score:
            best_k = k
            best_score = float(score)

    if best_score == float("-inf"):
        return lower, None
    return best_k, best_score


def cluster_feature_matrix(
    feature_matrix: np.ndarray,
    estimate_clusters: bool,
    min_clusters: int,
    max_clusters: int,
    seed: int,
) -> GlyphClusterResult:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    if estimate_clusters:
        n_clusters, best_silhouette = _estimate_cluster_count(
            scaled_features,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            seed=seed,
        )
    else:
        n_clusters = min(max_clusters, max(min_clusters, 2))
        best_silhouette = None

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(scaled_features)
    inventory = [f"cluster_{index:03d}" for index in range(n_clusters)]
    return GlyphClusterResult(
        labels=labels,
        inventory=inventory,
        centroids=model.cluster_centers_,
        feature_matrix=scaled_features,
        estimated_cluster_count=n_clusters,
        silhouette_score=best_silhouette,
        diagnostics={
            "feature_dimension": float(feature_matrix.shape[1]),
            "scaled_feature_std": float(np.std(scaled_features)),
        },
    )
