from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
import yaml

from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.utils.io import ensure_directory, write_yaml

SKLEARN_DIGITS_SOURCE = "sklearn.datasets.load_digits"
SKLEARN_DIGITS_ORIGIN = "UCI Optical Recognition of Handwritten Digits"


def _digit_image_to_uint8(image: np.ndarray) -> np.ndarray:
    scaled = np.clip(image.astype(np.float32) / 16.0, 0.0, 1.0)
    return np.uint8(np.round(scaled * 255.0))


def build_sklearn_digits_manifest(
    output_dir: str | Path,
    manifest_path: str | Path,
    train_count_per_class: int = 100,
    val_count_per_class: int = 30,
    split_seed: int = 23,
) -> GlyphCropManifest:
    dataset = load_digits()
    images = dataset.images.astype(np.float32)
    targets = dataset.target.astype(int)

    unique_labels = sorted({int(label) for label in targets.tolist()})
    counts = Counter(targets.tolist())
    min_count = min(counts.values())
    if train_count_per_class + val_count_per_class >= min_count:
        raise ValueError(
            "train_count_per_class + val_count_per_class must be smaller than the minimum class count "
            f"({min_count}) for sklearn digits."
        )

    root = ensure_directory(output_dir)
    image_root = ensure_directory(root / "images")
    rng = np.random.default_rng(split_seed)

    records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    per_class_split_counts: dict[str, Counter[str]] = {str(label): Counter() for label in unique_labels}

    for label in unique_labels:
        class_indices = np.flatnonzero(targets == label)
        permutation = rng.permutation(class_indices)
        train_indices = permutation[:train_count_per_class]
        val_indices = permutation[train_count_per_class : train_count_per_class + val_count_per_class]
        test_indices = permutation[train_count_per_class + val_count_per_class :]
        split_map = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        class_dir = ensure_directory(image_root / f"digit_{label}")
        for split_name, split_indices in split_map.items():
            for class_rank, dataset_index in enumerate(split_indices.tolist()):
                image_path = class_dir / f"{split_name}_{class_rank:03d}_{dataset_index:04d}.png"
                if not image_path.exists():
                    Image.fromarray(_digit_image_to_uint8(images[dataset_index]), mode="L").save(image_path)
                record = {
                    "sequence_id": f"digit_{label}_{dataset_index:04d}",
                    "position": 0,
                    "image_path": str(image_path.relative_to(root)),
                    "split": split_name,
                    "example_id": f"digit_{label}_{dataset_index:04d}",
                    "group_id": None,
                    "family": None,
                    "transcription": str(label),
                    "metadata": {
                        "source_index": int(dataset_index),
                        "digit_label": int(label),
                        "source": SKLEARN_DIGITS_SOURCE,
                    },
                }
                records.append(record)
                split_counts[split_name] += 1
                per_class_split_counts[str(label)][split_name] += 1

    manifest = GlyphCropManifest.model_validate(
        {
            "dataset_name": "sklearn_digits_crops",
            "unit_type": "glyph_crop",
            "metadata": {
                "source": SKLEARN_DIGITS_SOURCE,
                "origin": SKLEARN_DIGITS_ORIGIN,
                "split_strategy": "deterministic_per_class_cap",
                "split_seed": split_seed,
                "train_count_per_class": train_count_per_class,
                "val_count_per_class": val_count_per_class,
                "test_count_per_class": "remainder",
                "class_counts": {str(label): int(counts[label]) for label in unique_labels},
                "per_class_split_counts": {
                    label: dict(sorted(counter.items())) for label, counter in sorted(per_class_split_counts.items())
                },
                "split_counts": dict(sorted(split_counts.items())),
            },
            "records": records,
        }
    )
    write_yaml(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def summarize_sklearn_digits_local_artifacts(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(dataset_root)
    manifest_file = None if manifest_path is None else Path(manifest_path)
    png_root = root / "images"
    png_size = sum(path.stat().st_size for path in png_root.rglob("*.png")) if png_root.exists() else 0
    manifest_size = manifest_file.stat().st_size if manifest_file is not None and manifest_file.exists() else 0
    total_footprint = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    manifest_summary: dict[str, Any] = {}
    if manifest_file is not None and manifest_file.exists():
        payload = GlyphCropManifest.model_validate(yaml.safe_load(manifest_file.read_text(encoding="utf-8")))
        split_counts: Counter[str] = Counter(record.split for record in payload.records)
        label_counts: Counter[str] = Counter(
            record.transcription for record in payload.records if record.transcription is not None
        )
        manifest_summary = {
            "dataset_name": payload.dataset_name,
            "record_count": len(payload.records),
            "split_counts": dict(sorted(split_counts.items())),
            "label_count": len(label_counts),
            "metadata": payload.metadata,
        }

    return {
        "dataset_root": str(root),
        "png_size": png_size,
        "manifest_size": manifest_size,
        "total_footprint": total_footprint,
        "manifest_summary": manifest_summary,
    }


def format_sklearn_digits_integration_note(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> str:
    summary = summarize_sklearn_digits_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    manifest_summary = summary["manifest_summary"]
    assert isinstance(manifest_summary, dict)
    metadata = manifest_summary.get("metadata", {})
    lines = [
        "# scikit-learn Digits Integration Note",
        "",
        "## Source",
        f"- Loader: `{SKLEARN_DIGITS_SOURCE}`",
        f"- Origin: `{SKLEARN_DIGITS_ORIGIN}`",
        "- Additional network download: `0` bytes (dataset ships with the installed scikit-learn package).",
        "",
        "## Local Storage",
        f"- Extracted PNG size: `{summary['png_size']}` bytes",
        f"- Generated manifest size: `{summary['manifest_size']}` bytes",
        f"- Total local footprint under `{Path(dataset_root)}`: `{summary['total_footprint']}` bytes",
        "",
        "## Preparation Strategy",
        "- Convert each 8x8 handwritten digit image into a single grayscale PNG crop.",
        "- Treat each crop as one single-position sequence.",
        "- Use the digit label as `transcription`.",
        "- Leave `family` and `group_id` unset because the source corpus does not provide decipherment-family or document-group structure.",
        "",
        "## Split Strategy",
        "- Deterministic per-class split using a fixed random seed.",
    ]
    if isinstance(metadata, dict):
        lines.extend(
            [
                f"- Train cap per class: `{metadata.get('train_count_per_class', 'n/a')}`",
                f"- Validation cap per class: `{metadata.get('val_count_per_class', 'n/a')}`",
                f"- Test policy: `{metadata.get('test_count_per_class', 'n/a')}`",
                f"- Split seed: `{metadata.get('split_seed', 'n/a')}`",
            ]
        )
    if manifest_summary:
        lines.extend(
            [
                "",
                "This produces:",
                f"- `{manifest_summary.get('record_count', 0)}` total labeled crops",
            ]
        )
        split_counts = manifest_summary.get("split_counts", {})
        if isinstance(split_counts, dict):
            for split_name, split_count in split_counts.items():
                lines.append(f"- `{split_count}` {split_name} examples")
        lines.append(f"- `{manifest_summary.get('label_count', 0)}` digit classes")
    lines.extend(
        [
            "",
            "## Limitations",
            "- This is a real handwritten-symbol corpus, but it is low-resolution and far less script-like than Omniglot or Kuzushiji-49.",
            "- Sequences are single-glyph crops, so downstream structural and grouped metrics remain limited.",
            "- The strongest evidence from this dataset is expected to remain symbol-level rather than decipherment-level.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_sklearn_digits_integration_note(
    dataset_root: str | Path,
    note_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    destination = Path(note_path)
    destination.write_text(
        format_sklearn_digits_integration_note(dataset_root=dataset_root, manifest_path=manifest_path),
        encoding="utf-8",
    )
    return destination
