from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from decipherlab.config import PosteriorConfig, VisionConfig
from decipherlab.models import TranscriptionPosterior


def _ensure_2d_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        return np.stack([-scores, scores], axis=1)
    return scores


def _negative_log_likelihood(scores: np.ndarray, labels: list[str], support: list[str], temperature: float) -> float:
    support_index = {label: index for index, label in enumerate(support)}
    target_indices = np.asarray([support_index[label] for label in labels], dtype=int)
    scaled = scores / max(temperature, 1.0e-6)
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return float(-np.mean(log_probs[np.arange(len(target_indices)), target_indices]))


def _tune_temperature(
    scores: np.ndarray,
    labels: list[str],
    support: list[str],
    calibration_grid: list[float],
    default_temperature: float,
) -> tuple[float, dict[str, float]]:
    if not labels:
        return default_temperature, {"calibration_nll": float("nan"), "calibration_samples": 0.0}
    best_temperature = default_temperature
    best_nll = _negative_log_likelihood(scores, labels, support, default_temperature)
    for temperature in calibration_grid:
        nll = _negative_log_likelihood(scores, labels, support, temperature)
        if nll < best_nll:
            best_temperature = temperature
            best_nll = nll
    return best_temperature, {"calibration_nll": best_nll, "calibration_samples": float(len(labels))}


@dataclass
class GlyphPosteriorModel:
    strategy: str
    support: list[str]
    scaler: StandardScaler
    pca: PCA
    temperature: float
    diagnostics: dict[str, float] = field(default_factory=dict)
    classifier: LogisticRegression | None = None
    prototypes: np.ndarray | None = None

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(feature_matrix)
        return self.pca.transform(scaled)

    def score_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        embedded = self.transform(feature_matrix)
        if self.classifier is not None:
            return _ensure_2d_scores(self.classifier.decision_function(embedded))
        if self.prototypes is None:
            raise ValueError("Unsupervised posterior model requires prototypes.")
        distances = np.linalg.norm(embedded[:, None, :] - self.prototypes[None, :, :], axis=2)
        return -distances

    def predict_posterior(
        self,
        feature_matrix: np.ndarray,
        top_k: int,
        floor_probability: float,
    ) -> TranscriptionPosterior:
        scores = self.score_matrix(feature_matrix) / max(self.temperature, 1.0e-6)
        return TranscriptionPosterior.from_scores(
            support=self.support,
            scores=scores,
            top_k=top_k,
            floor_probability=floor_probability,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "support_size": len(self.support),
            "temperature": self.temperature,
            "diagnostics": self.diagnostics,
        }


def _fit_embedding(
    train_features: np.ndarray,
    embedding_dim: int,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_features)
    n_components = max(2, min(embedding_dim, scaled.shape[0] - 1, scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=0)
    embedded = pca.fit_transform(scaled)
    return scaler, pca, embedded


def _fit_label_prototypes(
    embedded_train: np.ndarray,
    train_labels: list[str | None],
) -> tuple[list[str], np.ndarray]:
    support = sorted({label for label in train_labels if label is not None})
    if not support:
        raise ValueError("Label prototypes require at least one labeled training symbol.")
    prototypes = []
    for label in support:
        mask = np.asarray([candidate == label for candidate in train_labels], dtype=bool)
        prototypes.append(np.mean(embedded_train[mask], axis=0))
    return support, np.asarray(prototypes, dtype=float)


def fit_posterior_model(
    train_features: np.ndarray,
    train_labels: list[str | None],
    validation_features: np.ndarray,
    validation_labels: list[str | None],
    posterior_config: PosteriorConfig,
    vision_config: VisionConfig,
    seed: int,
) -> GlyphPosteriorModel:
    scaler, pca, embedded_train = _fit_embedding(train_features, posterior_config.embedding_dim)
    embedded_validation = pca.transform(scaler.transform(validation_features)) if len(validation_features) else np.empty((0, embedded_train.shape[1]))
    labeled_train_mask = np.asarray([label is not None for label in train_labels], dtype=bool)
    labeled_validation_mask = np.asarray([label is not None for label in validation_labels], dtype=bool)

    if posterior_config.strategy == "cluster_distance":
        if np.sum(labeled_train_mask) >= 2 and len({label for label in train_labels if label is not None}) >= 2:
            support, prototypes = _fit_label_prototypes(embedded_train, train_labels)
            diagnostics = {
                "embedding_dim": float(embedded_train.shape[1]),
                "prototype_count": float(len(support)),
                "supervised_train_samples": float(np.sum(labeled_train_mask)),
                "calibration": 0.0,
            }
            return GlyphPosteriorModel(
                strategy="cluster_distance",
                support=support,
                scaler=scaler,
                pca=pca,
                temperature=posterior_config.temperature,
                prototypes=prototypes,
                diagnostics=diagnostics,
            )

        cluster_count = min(
            vision_config.max_clusters,
            max(vision_config.min_clusters, int(np.sqrt(max(len(embedded_train), 2)))),
        )
        cluster_count = min(cluster_count, len(embedded_train))
        kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
        kmeans.fit(embedded_train)
        support = [f"cluster_{index:03d}" for index in range(cluster_count)]
        return GlyphPosteriorModel(
            strategy="cluster_distance",
            support=support,
            scaler=scaler,
            pca=pca,
            temperature=posterior_config.temperature,
            prototypes=kmeans.cluster_centers_,
            diagnostics={
                "embedding_dim": float(embedded_train.shape[1]),
                "cluster_count": float(cluster_count),
                "supervised_train_samples": float(np.sum(labeled_train_mask)),
                "calibration": 0.0,
            },
        )

    if (
        posterior_config.strategy == "calibrated_classifier"
        and posterior_config.use_label_supervision
        and np.sum(labeled_train_mask) >= 4
        and len({label for label in train_labels if label is not None}) >= 2
    ):
        classifier = LogisticRegression(
            max_iter=1200,
            random_state=seed,
        )
        supervised_train = embedded_train[labeled_train_mask]
        supervised_labels = np.asarray([label for label in train_labels if label is not None], dtype=object)
        classifier.fit(supervised_train, supervised_labels)
        support = classifier.classes_.tolist()
        validation_scores = _ensure_2d_scores(classifier.decision_function(embedded_validation[labeled_validation_mask])) if np.any(labeled_validation_mask) else np.empty((0, len(support)))
        validation_targets = [label for label in validation_labels if label is not None]
        temperature, calibration_diagnostics = _tune_temperature(
            validation_scores,
            validation_targets,
            support,
            posterior_config.calibration_grid,
            posterior_config.temperature,
        )
        return GlyphPosteriorModel(
            strategy="calibrated_classifier",
            support=support,
            scaler=scaler,
            pca=pca,
            temperature=temperature,
            classifier=classifier,
            diagnostics={
                "embedding_dim": float(embedded_train.shape[1]),
                "supervised_train_samples": float(supervised_train.shape[0]),
                "supervised_class_count": float(len(support)),
                **calibration_diagnostics,
            },
        )

    cluster_count = min(
        vision_config.max_clusters,
        max(vision_config.min_clusters, int(np.sqrt(max(len(embedded_train), 2)))),
    )
    cluster_count = min(cluster_count, len(embedded_train))
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(embedded_train)
    support = [f"cluster_{index:03d}" for index in range(cluster_count)]
    return GlyphPosteriorModel(
        strategy="calibrated_prototype_fallback",
        support=support,
        scaler=scaler,
        pca=pca,
        temperature=posterior_config.temperature,
        prototypes=kmeans.cluster_centers_,
        diagnostics={
            "embedding_dim": float(embedded_train.shape[1]),
            "cluster_count": float(cluster_count),
            "supervised_train_samples": float(np.sum(labeled_train_mask)),
        },
    )
