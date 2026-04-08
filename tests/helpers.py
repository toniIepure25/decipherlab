from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from decipherlab.config import DecipherLabConfig
from decipherlab.vision.synthetic import build_symbol_prototypes, perturb_prototype

import numpy as np


def build_test_config(tmp_path: Path, noise_std: float = 0.3) -> DecipherLabConfig:
    return DecipherLabConfig.model_validate(
        {
            "experiment": {
                "name": "test_mvp",
                "seed": 11,
                "seed_sweep": [],
                "output_root": str(tmp_path / "runs"),
                "notes": "test configuration",
            },
            "dataset": {
                "source": "synthetic",
                "manifest_path": None,
                "manifest_format": "glyph_crop",
                "generate_if_missing": True,
                "train_split": "train",
                "val_split": "val",
                "evaluation_split": "test",
                "min_sequences_per_split_warning": 2,
                "min_symbol_instances_per_train_class_warning": 2,
                "min_family_instances_per_split_warning": 2,
            },
            "synthetic": {
                "families": ["monoalphabetic", "homophonic", "transposition", "pseudo_text"],
                "samples_per_family": 6,
                "sequence_length": 32,
                "alphabet_size": 12,
                "homophonic_extra_symbols": 8,
                "transposition_block_size": 4,
                "noise_std": noise_std,
                "allograph_jitter": 0.7,
                "image_size": 20,
                "train_fraction": 0.5,
                "val_fraction": 0.25,
            },
            "vision": {
                "feature_downsample": 8,
                "estimate_clusters": True,
                "min_clusters": 4,
                "max_clusters": 18,
            },
            "posterior": {
                "strategy": "calibrated_classifier",
                "top_k": 3,
                "temperature": 1.2,
                "floor_probability": 1.0e-6,
                "embedding_dim": 16,
                "use_label_supervision": True,
                "calibration_grid": [0.5, 1.0, 1.5, 2.0],
            },
            "triage": {
                "repeat_ngram_sizes": [2, 3],
                "shuffled_null_trials": 4,
            },
            "hypotheses": {
                "families": [
                    "unknown_script",
                    "monoalphabetic",
                    "homophonic",
                    "transposition_heuristic",
                    "pseudo_text_null",
                ]
            },
            "evaluation": {
                "enabled": True,
                "ambiguity_levels": [0.0, 0.35],
                "top_k": 3,
                "comparison_strategies": ["cluster_distance", "calibrated_classifier"],
                "overdiffuse_entropy_ratio": 0.8,
                "bootstrap_trials": 50,
                "bootstrap_confidence_level": 0.95,
                "bootstrap_seed": 101,
            },
        }
    )


def create_real_manifest_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "real_manifest_fixture"
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    symbol_ids = ["ga", "gb", "gc", "gd"]
    prototypes = build_symbol_prototypes(symbol_ids, image_size=22, seed=17)
    split_sequences = {
        "train": ["ga", "gb", "gc", "ga", "gd", "gc", "gb", "ga"],
        "train_b": ["gb", "gc", "ga", "gd", "ga", "gb", "gc", "gd"],
        "val": ["ga", "gc", "gd", "gb", "ga", "gd", "gc", "gb"],
        "test": ["gd", "gb", "ga", "gc", "gd", "ga", "gb", "gc"],
    }
    split_map = {"train": "train", "train_b": "train", "val": "val", "test": "test"}
    records = []
    for sequence_id, symbols in split_sequences.items():
        for position, symbol in enumerate(symbols):
            image = perturb_prototype(
                prototypes[symbol],
                rng=rng,
                noise_std=0.05,
                allograph_jitter=0.25,
            )
            image_path = image_root / f"{sequence_id}_{position:02d}.png"
            Image.fromarray((image * 255).astype("uint8"), mode="L").save(image_path)
            records.append(
                {
                    "sequence_id": sequence_id,
                    "position": position,
                    "image_path": str(image_path.relative_to(root)),
                    "split": split_map[sequence_id],
                    "example_id": f"{sequence_id}_{position:02d}",
                    "group_id": "doc_a" if sequence_id in {"train", "val"} else "doc_b",
                    "family": "monoalphabetic",
                    "transcription": symbol,
                    "metadata": {"source": "fixture"},
                }
            )

    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "fixture_real_manifest",
                "unit_type": "glyph_crop",
                "metadata": {"description": "Small real-manifest-style fixture for tests."},
                "records": records,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path
