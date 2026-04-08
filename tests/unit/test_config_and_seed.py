from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from decipherlab.benchmarks.synthetic import generate_synthetic_dataset
from decipherlab.config import load_config
from decipherlab.utils.randomness import set_global_seed


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad", "seed": 1, "output_root": "outputs/runs"},
                "dataset": {"source": "synthetic", "generate_if_missing": True},
                "synthetic": {
                    "families": ["monoalphabetic"],
                    "samples_per_family": 1,
                    "sequence_length": 16,
                    "alphabet_size": 8,
                    "homophonic_extra_symbols": 2,
                    "transposition_block_size": 4,
                    "noise_std": 0.1,
                    "allograph_jitter": 0.5,
                    "image_size": 18,
                    "train_fraction": 0.6,
                    "val_fraction": 0.2,
                },
                "vision": {"feature_downsample": 8, "estimate_clusters": True, "min_clusters": 2, "max_clusters": 6},
                "posterior": {"strategy": "calibrated_classifier", "top_k": 2, "temperature": 1.0, "floor_probability": 1.0e-6, "embedding_dim": 8, "use_label_supervision": True, "calibration_grid": [1.0]},
                "triage": {"repeat_ngram_sizes": [2], "shuffled_null_trials": 2},
                "hypotheses": {"families": ["unknown_script", "monoalphabetic", "pseudo_text_null"]},
                "evaluation": {"enabled": True, "ambiguity_levels": [0.0], "top_k": 2},
                "unexpected": 123,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(config_path)


def test_seed_setting_is_repeatable() -> None:
    first = set_global_seed(13).normal(size=5)
    second = set_global_seed(13).normal(size=5)
    assert np.allclose(first, second)


def test_synthetic_generation_is_deterministic(small_config) -> None:
    bundle_a = generate_synthetic_dataset(small_config.synthetic, seed=small_config.experiment.seed)
    bundle_b = generate_synthetic_dataset(small_config.synthetic, seed=small_config.experiment.seed)
    assert bundle_a.dataset.examples[0].observed_symbols == bundle_b.dataset.examples[0].observed_symbols
    assert np.allclose(bundle_a.dataset.examples[0].glyphs[0].image, bundle_b.dataset.examples[0].glyphs[0].image)
