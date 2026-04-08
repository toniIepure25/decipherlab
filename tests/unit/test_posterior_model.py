from __future__ import annotations

from pathlib import Path

import numpy as np

from decipherlab.glyphs.features import extract_feature_matrix
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.transcription.model import fit_posterior_model

from tests.helpers import build_test_config, create_real_manifest_fixture


def test_calibrated_posterior_model_predicts_labeled_symbols(tmp_path: Path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    dataset = load_glyph_crop_manifest_dataset(manifest_path)
    config = build_test_config(tmp_path)
    train_examples = dataset.get_split("train")
    val_examples = dataset.get_split("val")
    test_examples = dataset.get_split("test")

    train_glyphs = [glyph for example in train_examples for glyph in example.glyphs]
    val_glyphs = [glyph for example in val_examples for glyph in example.glyphs]
    test_glyphs = [glyph for example in test_examples for glyph in example.glyphs]

    model = fit_posterior_model(
        train_features=extract_feature_matrix(train_glyphs, downsample=config.vision.feature_downsample),
        train_labels=[glyph.true_symbol for glyph in train_glyphs],
        validation_features=extract_feature_matrix(val_glyphs, downsample=config.vision.feature_downsample),
        validation_labels=[glyph.true_symbol for glyph in val_glyphs],
        posterior_config=config.posterior,
        vision_config=config.vision,
        seed=config.experiment.seed,
    )
    posterior = model.predict_posterior(
        extract_feature_matrix(test_glyphs, downsample=config.vision.feature_downsample),
        top_k=config.posterior.top_k,
        floor_probability=config.posterior.floor_probability,
    )
    hard = posterior.hard_sequence()
    truth = [glyph.true_symbol for glyph in test_glyphs]
    accuracy = np.mean([float(predicted == actual) for predicted, actual in zip(hard, truth)])
    assert model.strategy == "calibrated_classifier"
    assert accuracy >= 0.75


def test_cluster_distance_posterior_model_produces_label_support(tmp_path: Path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    dataset = load_glyph_crop_manifest_dataset(manifest_path)
    config = build_test_config(tmp_path)
    config.posterior.strategy = "cluster_distance"
    train_examples = dataset.get_split("train")
    val_examples = dataset.get_split("val")
    test_examples = dataset.get_split("test")

    train_glyphs = [glyph for example in train_examples for glyph in example.glyphs]
    val_glyphs = [glyph for example in val_examples for glyph in example.glyphs]
    test_glyphs = [glyph for example in test_examples for glyph in example.glyphs]

    model = fit_posterior_model(
        train_features=extract_feature_matrix(train_glyphs, downsample=config.vision.feature_downsample),
        train_labels=[glyph.true_symbol for glyph in train_glyphs],
        validation_features=extract_feature_matrix(val_glyphs, downsample=config.vision.feature_downsample),
        validation_labels=[glyph.true_symbol for glyph in val_glyphs],
        posterior_config=config.posterior,
        vision_config=config.vision,
        seed=config.experiment.seed,
    )
    posterior = model.predict_posterior(
        extract_feature_matrix(test_glyphs, downsample=config.vision.feature_downsample),
        top_k=config.posterior.top_k,
        floor_probability=config.posterior.floor_probability,
    )
    assert model.strategy == "cluster_distance"
    assert set(posterior.support()).issubset(set(model.support))
