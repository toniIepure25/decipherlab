from __future__ import annotations

import numpy as np
from PIL import Image

from decipherlab.models import GlyphCrop


def _resize_image(image: np.ndarray, downsample: int) -> np.ndarray:
    pil_image = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255), mode="L")
    resized = pil_image.resize((downsample, downsample), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def glyph_feature_vector(image: np.ndarray, downsample: int) -> np.ndarray:
    reduced = _resize_image(image, downsample=downsample)
    row_sums = reduced.sum(axis=1)
    column_sums = reduced.sum(axis=0)
    ink_mass = float(np.sum(reduced))
    if ink_mass > 0.0:
        yy, xx = np.indices(reduced.shape)
        center_y = float(np.sum(yy * reduced) / ink_mass) / downsample
        center_x = float(np.sum(xx * reduced) / ink_mass) / downsample
    else:
        center_y = center_x = 0.5
    summary = np.asarray(
        [
            ink_mass / (downsample * downsample),
            float(np.mean(reduced)),
            float(np.std(reduced)),
            center_y,
            center_x,
        ],
        dtype=np.float32,
    )
    return np.concatenate([reduced.ravel(), row_sums, column_sums, summary])


def extract_feature_matrix(glyphs: list[GlyphCrop], downsample: int) -> np.ndarray:
    return np.stack([glyph_feature_vector(glyph.image, downsample) for glyph in glyphs], axis=0)
