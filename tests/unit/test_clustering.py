from __future__ import annotations

import numpy as np

from decipherlab.evaluation.metrics import clustering_ari
from decipherlab.glyphs.clustering import cluster_feature_matrix
from decipherlab.glyphs.features import extract_feature_matrix
from decipherlab.models import GlyphCrop
from decipherlab.vision.synthetic import build_symbol_prototypes, perturb_prototype


def test_glyph_clustering_recovers_controlled_inventory() -> None:
    rng = np.random.default_rng(5)
    symbol_ids = ["sym_a", "sym_b", "sym_c"]
    prototypes = build_symbol_prototypes(symbol_ids, image_size=20, seed=5)
    glyphs = []
    true_symbols = []
    position = 0
    for symbol_id in symbol_ids:
        for _ in range(6):
            glyphs.append(
                GlyphCrop(
                    position=position,
                    image=perturb_prototype(prototypes[symbol_id], rng, noise_std=0.03, allograph_jitter=0.2),
                    true_symbol=symbol_id,
                )
            )
            true_symbols.append(symbol_id)
            position += 1

    features = extract_feature_matrix(glyphs, downsample=8)
    result = cluster_feature_matrix(
        features,
        estimate_clusters=False,
        min_clusters=3,
        max_clusters=3,
        seed=5,
    )
    assert result.estimated_cluster_count == 3
    assert clustering_ari(true_symbols, result.labels) > 0.8
