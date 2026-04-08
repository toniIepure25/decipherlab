from __future__ import annotations

from hashlib import sha256

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _symbol_seed(symbol_id: str, seed: int) -> int:
    digest = sha256(f"{symbol_id}:{seed}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def render_symbol_prototype(symbol_id: str, image_size: int, seed: int) -> np.ndarray:
    """Render a deterministic synthetic prototype for a glyph identity."""

    local_rng = np.random.default_rng(_symbol_seed(symbol_id, seed))
    canvas = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(canvas)
    stroke_count = int(local_rng.integers(2, 5))
    for _ in range(stroke_count):
        start = tuple(int(value) for value in local_rng.integers(2, image_size - 2, size=2))
        end = tuple(int(value) for value in local_rng.integers(2, image_size - 2, size=2))
        width = int(local_rng.integers(1, max(2, image_size // 7)))
        draw.line([start, end], fill=255, width=width)
        if local_rng.random() > 0.55:
            radius = int(local_rng.integers(2, max(4, image_size // 3)))
            corner = tuple(int(value) for value in local_rng.integers(1, image_size - radius - 1, size=2))
            draw.arc(
                [corner[0], corner[1], corner[0] + radius, corner[1] + radius],
                start=int(local_rng.integers(0, 180)),
                end=int(local_rng.integers(181, 360)),
                fill=255,
                width=max(1, width - 1),
            )

    image = np.asarray(canvas, dtype=np.float32) / 255.0
    return image


def build_symbol_prototypes(symbol_ids: list[str], image_size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        symbol_id: render_symbol_prototype(symbol_id, image_size=image_size, seed=seed)
        for symbol_id in symbol_ids
    }


def perturb_prototype(
    prototype: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    allograph_jitter: float,
) -> np.ndarray:
    rotated = ndimage.rotate(
        prototype,
        angle=float(rng.normal(loc=0.0, scale=allograph_jitter * 3.5)),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )
    shifted = ndimage.shift(
        rotated,
        shift=(
            float(rng.normal(loc=0.0, scale=allograph_jitter * 0.35)),
            float(rng.normal(loc=0.0, scale=allograph_jitter * 0.35)),
        ),
        order=1,
        mode="constant",
        cval=0.0,
    )
    blurred = ndimage.gaussian_filter(shifted, sigma=max(0.1, noise_std * 0.9))
    noisy = blurred + rng.normal(loc=0.0, scale=noise_std, size=prototype.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)
