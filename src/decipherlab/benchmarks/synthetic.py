from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from decipherlab.config import SyntheticConfig
from decipherlab.models import DatasetCollection, GlyphCrop, SequenceExample
from decipherlab.utils.io import ensure_directory, write_json
from decipherlab.vision.synthetic import build_symbol_prototypes, perturb_prototype

_PLAIN_ALPHABET = list("etaoinshrdlucmfwyp")
_BASE_TEXT = "".join(
    ch
    for ch in (
        "the archivists preserved the cipher ledger while hidden routes and sealed messages "
        "passed between ports and towers and the stewards repeated the patterns to train "
        "new readers in a careful sequence of signs and marginal notations"
    )
    if ch in _PLAIN_ALPHABET
)


@dataclass(frozen=True)
class SyntheticDatasetBundle:
    dataset: DatasetCollection
    prototypes: dict[str, np.ndarray]


def _sample_plaintext(length: int, rng: np.random.Generator) -> str:
    if length >= len(_BASE_TEXT):
        repetitions = (length // len(_BASE_TEXT)) + 2
        text = (_BASE_TEXT * repetitions)[: length + len(_BASE_TEXT)]
    else:
        text = _BASE_TEXT
    start = int(rng.integers(0, max(1, len(text) - length)))
    return text[start : start + length]


def _global_symbol_inventory(config: SyntheticConfig) -> list[str]:
    inventory_size = config.alphabet_size + config.homophonic_extra_symbols + 6
    return [f"sym_{index:03d}" for index in range(inventory_size)]


def _make_monoalphabetic(
    plaintext: str,
    global_symbols: list[str],
    rng: np.random.Generator,
) -> list[str]:
    unique_plain_symbols = sorted(set(plaintext))
    chosen = rng.choice(global_symbols, size=len(unique_plain_symbols), replace=False)
    mapping = {plain: observed for plain, observed in zip(unique_plain_symbols, chosen)}
    return [mapping[character] for character in plaintext]


def _make_homophonic(
    plaintext: str,
    global_symbols: list[str],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> list[str]:
    unique_plain_symbols = sorted(set(plaintext))
    available = list(rng.choice(global_symbols, size=len(unique_plain_symbols) + config.homophonic_extra_symbols, replace=False))
    frequencies = {character: plaintext.count(character) for character in unique_plain_symbols}
    allocations = {character: 1 for character in unique_plain_symbols}
    extras_remaining = config.homophonic_extra_symbols
    ranked = sorted(unique_plain_symbols, key=lambda item: frequencies[item], reverse=True)
    pointer = 0
    while extras_remaining > 0:
        allocations[ranked[pointer % len(ranked)]] += 1
        extras_remaining -= 1
        pointer += 1

    mapping: dict[str, list[str]] = {}
    cursor = 0
    for character in unique_plain_symbols:
        mapping[character] = available[cursor : cursor + allocations[character]]
        cursor += allocations[character]
    return [str(rng.choice(mapping[character])) for character in plaintext]


def _make_transposition(
    plaintext: str,
    global_symbols: list[str],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> list[str]:
    sequence = _make_monoalphabetic(plaintext, global_symbols, rng)
    block_size = max(2, config.transposition_block_size)
    transposed: list[str] = []
    for start in range(0, len(sequence), block_size):
        block = sequence[start : start + block_size]
        if len(block) < 2:
            transposed.extend(block)
            continue
        permutation = rng.permutation(len(block))
        transposed.extend([block[index] for index in permutation])
    return transposed


def _make_pseudo_text(
    length: int,
    global_symbols: list[str],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> list[str]:
    symbol_pool = list(rng.choice(global_symbols, size=config.alphabet_size, replace=False))
    return list(rng.choice(symbol_pool, size=length, replace=True))


def _render_example(
    example_id: str,
    family: str,
    observed_symbols: list[str],
    plaintext: str,
    split: str,
    prototypes: dict[str, np.ndarray],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> SequenceExample:
    glyphs: list[GlyphCrop] = []
    for position, symbol in enumerate(observed_symbols):
        image = perturb_prototype(
            prototypes[symbol],
            rng=rng,
            noise_std=config.noise_std,
            allograph_jitter=config.allograph_jitter,
        )
        glyphs.append(
            GlyphCrop(
                position=position,
                image=image,
                true_symbol=symbol,
                variant_id=f"{symbol}_var_{position:03d}",
            )
        )

    return SequenceExample(
        example_id=example_id,
        family=family,
        glyphs=glyphs,
        plaintext=plaintext,
        observed_symbols=observed_symbols,
        split=split,
        metadata={
            "generator": "synthetic_baseline",
            "noise_std": config.noise_std,
            "allograph_jitter": config.allograph_jitter,
        },
    )


def _deterministic_split(index: int, total: int, config: SyntheticConfig) -> str:
    if total <= 1:
        return "train"
    ratio = index / total
    if ratio < config.train_fraction:
        return "train"
    if ratio < config.train_fraction + config.val_fraction:
        return "val"
    return "test"


def generate_synthetic_dataset(
    config: SyntheticConfig,
    seed: int,
) -> SyntheticDatasetBundle:
    rng = np.random.default_rng(seed)
    global_symbols = _global_symbol_inventory(config)
    prototypes = build_symbol_prototypes(global_symbols, image_size=config.image_size, seed=seed)
    examples: list[SequenceExample] = []

    for family in config.families:
        for index in range(config.samples_per_family):
            plaintext = _sample_plaintext(config.sequence_length, rng)
            if family == "monoalphabetic":
                observed_symbols = _make_monoalphabetic(plaintext, global_symbols, rng)
            elif family == "homophonic":
                observed_symbols = _make_homophonic(plaintext, global_symbols, config, rng)
            elif family == "transposition":
                observed_symbols = _make_transposition(plaintext, global_symbols, config, rng)
            elif family == "pseudo_text":
                observed_symbols = _make_pseudo_text(config.sequence_length, global_symbols, config, rng)
                plaintext = "?" * config.sequence_length
            else:
                raise ValueError(f"Unsupported family: {family}")

            examples.append(
                _render_example(
                    example_id=f"{family}_{index:03d}",
                    family=family,
                    observed_symbols=observed_symbols,
                    plaintext=plaintext,
                    split=_deterministic_split(index, config.samples_per_family, config),
                    prototypes=prototypes,
                    config=config,
                    rng=rng,
                )
            )

    return SyntheticDatasetBundle(
        dataset=DatasetCollection(
            dataset_name="synthetic_benchmark",
            examples=examples,
            metadata={"format": "generated_synthetic", "seed": seed},
        ),
        prototypes=prototypes,
    )


def save_synthetic_dataset(
    bundle: SyntheticDatasetBundle,
    output_dir: str | Path,
    dataset_name: str,
    seed: int,
) -> Path:
    dataset_dir = ensure_directory(output_dir)
    artifacts_dir = ensure_directory(dataset_dir / "artifacts")
    manifest_examples: list[dict[str, object]] = []

    for example in bundle.dataset.examples:
        artifact_path = artifacts_dir / f"{example.example_id}.npz"
        np.savez_compressed(
            artifact_path,
            glyph_images=np.stack([glyph.image for glyph in example.glyphs], axis=0),
            observed_symbols=np.asarray(example.observed_symbols, dtype="U32"),
            plaintext=np.asarray(["" if example.plaintext is None else example.plaintext], dtype="U256"),
        )
        manifest_examples.append(
            {
                "example_id": example.example_id,
                "family": example.family,
                "artifact_path": str(artifact_path.relative_to(dataset_dir)),
                "sequence_length": example.sequence_length,
                "split": example.split,
                "metadata": example.metadata,
            }
        )

    manifest_path = dataset_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "examples": manifest_examples,
        },
    )
    return manifest_path
