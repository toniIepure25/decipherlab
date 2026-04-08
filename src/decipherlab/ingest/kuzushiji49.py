from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml, get_data_home
import yaml

from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.utils.io import ensure_directory, write_yaml

KUZUSHIJI49_DATASET_NAME = "Kuzushiji-49"
KUZUSHIJI49_OPENML_ID = 41991
KUZUSHIJI49_OPENML_URL = f"https://www.openml.org/d/{KUZUSHIJI49_OPENML_ID}"
KUZUSHIJI49_CACHE_FILE = (
    Path(get_data_home()) / "openml" / "openml.org" / "data" / "v1" / "download" / "21399168" / "Kuzushiji-49.arff.gz"
)


def _numeric_label_sort_key(label: str) -> tuple[int, str]:
    try:
        return int(label), label
    except ValueError:
        return 10**9, label


def _load_kuzushiji49() -> tuple[np.ndarray, np.ndarray]:
    dataset = fetch_openml(name=KUZUSHIJI49_DATASET_NAME, version=1, as_frame=False, parser="auto")
    images = np.asarray(dataset.data, dtype=np.uint8).reshape(-1, 28, 28)
    targets = np.asarray(dataset.target, dtype=str)
    del dataset
    gc.collect()
    return images, targets


def build_kuzushiji49_balanced_manifest(
    output_dir: str | Path,
    manifest_path: str | Path,
    train_count_per_class: int = 300,
    val_count_per_class: int = 75,
    test_count_per_class: int = 75,
    split_seed: int = 23,
    image_subdir: str = "images_balanced_300_75_75",
) -> GlyphCropManifest:
    images, targets = _load_kuzushiji49()
    unique_labels = sorted({label for label in targets.tolist()}, key=_numeric_label_sort_key)
    counts = Counter(targets.tolist())
    minimum_count = min(counts.values())
    requested_total = train_count_per_class + val_count_per_class + test_count_per_class
    if requested_total > minimum_count:
        raise ValueError(
            "train_count_per_class + val_count_per_class + test_count_per_class must not exceed the minimum class count "
            f"({minimum_count}) for Kuzushiji-49."
        )

    root = ensure_directory(output_dir)
    image_root = ensure_directory(root / image_subdir)
    rng = np.random.default_rng(split_seed)

    records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    per_class_split_counts: dict[str, Counter[str]] = {label: Counter() for label in unique_labels}

    for label in unique_labels:
        class_indices = np.flatnonzero(targets == label)
        permutation = rng.permutation(class_indices)
        split_map = {
            "train": permutation[:train_count_per_class],
            "val": permutation[train_count_per_class : train_count_per_class + val_count_per_class],
            "test": permutation[
                train_count_per_class + val_count_per_class : train_count_per_class + val_count_per_class + test_count_per_class
            ],
        }
        class_dir = ensure_directory(image_root / f"class_{int(label):02d}")
        for split_name, split_indices in split_map.items():
            for class_rank, dataset_index in enumerate(split_indices.tolist()):
                image_path = class_dir / f"{split_name}_{class_rank:03d}_{dataset_index:06d}.png"
                if not image_path.exists():
                    Image.fromarray(images[dataset_index], mode="L").save(image_path)
                records.append(
                    {
                        "sequence_id": f"k49_{label}_{dataset_index:06d}",
                        "position": 0,
                        "image_path": str(image_path.relative_to(root)),
                        "split": split_name,
                        "example_id": f"k49_{label}_{dataset_index:06d}",
                        "group_id": None,
                        "family": None,
                        "transcription": label,
                        "metadata": {
                            "source_index": int(dataset_index),
                            "label_id": int(label),
                            "source": KUZUSHIJI49_DATASET_NAME,
                            "openml_data_id": KUZUSHIJI49_OPENML_ID,
                        },
                    }
                )
                split_counts[split_name] += 1
                per_class_split_counts[label][split_name] += 1

    manifest = GlyphCropManifest.model_validate(
        {
            "dataset_name": "kuzushiji49_balanced_crops",
            "unit_type": "glyph_crop",
            "metadata": {
                "source": KUZUSHIJI49_DATASET_NAME,
                "openml_data_id": KUZUSHIJI49_OPENML_ID,
                "openml_url": KUZUSHIJI49_OPENML_URL,
                "cache_file": str(KUZUSHIJI49_CACHE_FILE),
                "split_strategy": "balanced_per_class_cap",
                "subset_strategy": "all_classes_preserved_balanced_cap",
                "split_seed": split_seed,
                "train_count_per_class": train_count_per_class,
                "val_count_per_class": val_count_per_class,
                "test_count_per_class": test_count_per_class,
                "requested_total_per_class": requested_total,
                "class_counts": {label: int(counts[label]) for label in unique_labels},
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


def summarize_kuzushiji49_local_artifacts(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(dataset_root)
    manifest_file = None if manifest_path is None else Path(manifest_path)
    png_size = sum(path.stat().st_size for path in root.rglob("*.png")) if root.exists() else 0
    manifest_size = manifest_file.stat().st_size if manifest_file is not None and manifest_file.exists() else 0
    total_footprint = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    cache_size = KUZUSHIJI49_CACHE_FILE.stat().st_size if KUZUSHIJI49_CACHE_FILE.exists() else 0

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
        "cache_size": cache_size,
        "png_size": png_size,
        "manifest_size": manifest_size,
        "total_footprint": total_footprint,
        "manifest_summary": manifest_summary,
    }


def format_kuzushiji49_integration_note(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> str:
    summary = summarize_kuzushiji49_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    manifest_summary = summary["manifest_summary"]
    assert isinstance(manifest_summary, dict)
    metadata = manifest_summary.get("metadata", {})

    lines = [
        "# Kuzushiji-49 Integration Note",
        "",
        "## Source",
        f"- Dataset: `{KUZUSHIJI49_DATASET_NAME}`",
        f"- OpenML ID: `{KUZUSHIJI49_OPENML_ID}`",
        f"- OpenML URL: `{KUZUSHIJI49_OPENML_URL}`",
        "",
        "## Local Storage",
        f"- Full downloaded cache size: `{summary['cache_size']}` bytes",
        f"- Generated PNG size: `{summary['png_size']}` bytes",
        f"- Generated manifest size: `{summary['manifest_size']}` bytes",
        f"- Total local footprint under `{Path(dataset_root)}`: `{summary['total_footprint']}` bytes",
        "",
        "## Preparation Strategy",
        "- Download the full Kuzushiji-49 corpus through OpenML and keep the cached archive intact.",
        "- Preserve all 49 classes, but build a balanced crop manifest with deterministic per-class caps so the frozen multi-seed paper pack remains tractable.",
        "- Treat each image as one single-position glyph sequence.",
        "- Use the OpenML class label as `transcription`.",
        "- Leave `family` and `group_id` unset because the source corpus does not provide decipherment-family or document-group structure compatible with the current downstream metrics.",
        "",
        "## Subset Strategy",
        "- Download mode: **full dataset download through OpenML cache**.",
        "- Evaluation mode: **balanced capped subset with all classes preserved**.",
    ]

    if isinstance(metadata, dict):
        lines.extend(
            [
                f"- Train cap per class: `{metadata.get('train_count_per_class', 'n/a')}`",
                f"- Validation cap per class: `{metadata.get('val_count_per_class', 'n/a')}`",
                f"- Test cap per class: `{metadata.get('test_count_per_class', 'n/a')}`",
                f"- Split seed: `{metadata.get('split_seed', 'n/a')}`",
            ]
        )

    if manifest_summary:
        split_counts = manifest_summary.get("split_counts", {})
        lines.extend(
            [
                "",
                "This produces:",
                f"- `{manifest_summary.get('record_count', 0)}` total labeled crops",
            ]
        )
        if isinstance(split_counts, dict):
            for split_name, split_count in split_counts.items():
                lines.append(f"- `{split_count}` {split_name} examples")
        lines.append(f"- `{manifest_summary.get('label_count', 0)}` class labels")

    lines.extend(
        [
            "",
            "## Why This Strengthens The Paper",
            "- Kuzushiji-49 is historically grounded and visually closer to manuscript conditions than scikit-learn digits.",
            "- It complements Omniglot by adding a real historical character corpus with heavier class imbalance and different stroke statistics.",
            "- It tests whether the symbol-level uncertainty-rescue effect survives on a corpus that is more manuscript-like without changing the core protocol.",
            "",
            "## Limitations",
            "- The current manifest is a balanced evaluation subset rather than a full-manifest sweep, chosen to keep the frozen multi-seed paper pack tractable while preserving all classes.",
            "- Sequences are single-glyph crops, so grouped and downstream structural metrics remain limited.",
            "- The strongest evidence from this dataset is still expected to remain symbol-level rather than semantic or family-level.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_kuzushiji49_integration_note(
    dataset_root: str | Path,
    note_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    destination = Path(note_path)
    destination.write_text(
        format_kuzushiji49_integration_note(dataset_root=dataset_root, manifest_path=manifest_path),
        encoding="utf-8",
    )
    return destination
