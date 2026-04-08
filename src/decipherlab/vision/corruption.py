from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

from decipherlab.models import GlyphCrop, SequenceExample


def corrupt_image(
    image: np.ndarray,
    ambiguity_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if ambiguity_level <= 0.0:
        return image.astype(np.float32)
    blurred = ndimage.gaussian_filter(image, sigma=0.4 + (1.6 * ambiguity_level))
    noisy = blurred + rng.normal(loc=0.0, scale=0.08 + (0.28 * ambiguity_level), size=image.shape)
    if ambiguity_level >= 0.15:
        height, width = image.shape
        occ_h = max(1, int(height * (0.12 + 0.2 * ambiguity_level)))
        occ_w = max(1, int(width * (0.12 + 0.2 * ambiguity_level)))
        top = int(rng.integers(0, max(1, height - occ_h + 1)))
        left = int(rng.integers(0, max(1, width - occ_w + 1)))
        noisy[top : top + occ_h, left : left + occ_w] *= 1.0 - min(0.85, ambiguity_level + 0.15)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def apply_ambiguity_to_examples(
    examples: list[SequenceExample],
    ambiguity_level: float,
    seed: int,
) -> list[SequenceExample]:
    if ambiguity_level <= 0.0:
        return examples

    corrupted_examples: list[SequenceExample] = []
    for example_index, example in enumerate(examples):
        new_glyphs: list[GlyphCrop] = []
        for glyph in example.glyphs:
            rng = np.random.default_rng(seed + (example_index * 997) + glyph.position)
            new_glyphs.append(
                replace(
                    glyph,
                    image=corrupt_image(glyph.image, ambiguity_level=ambiguity_level, rng=rng),
                )
            )
        new_metadata = dict(example.metadata)
        new_metadata["ambiguity_level"] = ambiguity_level
        corrupted_examples.append(replace(example, glyphs=new_glyphs, metadata=new_metadata))
    return corrupted_examples
