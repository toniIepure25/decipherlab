from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import yaml

from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.utils.io import ensure_directory, write_yaml

OMNIGLOT_REPOSITORY_URL = "https://github.com/brendenlake/omniglot"
OMNIGLOT_ARCHIVES = {
    "images_background.zip": f"{OMNIGLOT_REPOSITORY_URL}/raw/master/python/images_background.zip",
    "images_evaluation.zip": f"{OMNIGLOT_REPOSITORY_URL}/raw/master/python/images_evaluation.zip",
}


def extract_omniglot_archives(
    archives: list[str | Path],
    output_dir: str | Path,
) -> list[Path]:
    destination = ensure_directory(output_dir)
    extracted_roots: list[Path] = []
    for archive in archives:
        archive_path = Path(archive)
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(destination)
            top_levels = sorted({Path(name).parts[0] for name in handle.namelist() if name and not name.endswith("/")})
            for top_level in top_levels:
                extracted_roots.append(destination / top_level)
    unique_roots: list[Path] = []
    for root in extracted_roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def build_omniglot_manifest(
    image_roots: list[str | Path],
    output_path: str | Path,
    train_count_per_class: int = 12,
    val_count_per_class: int = 4,
    test_count_per_class: int = 4,
    split_seed: int = 23,
) -> GlyphCropManifest:
    total_expected = train_count_per_class + val_count_per_class + test_count_per_class
    records: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[tuple[Path, str]]] = defaultdict(list)
    output_file = Path(output_path)

    for root in image_roots:
        root_path = Path(root)
        source_split = root_path.name
        for alphabet_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            for character_dir in sorted(path for path in alphabet_dir.iterdir() if path.is_dir()):
                label = f"{alphabet_dir.name}__{character_dir.name}"
                for image_path in sorted(character_dir.glob("*.png")):
                    grouped[(alphabet_dir.name, character_dir.name)].append((image_path, source_split))

    rng = np.random.default_rng(split_seed)
    for (alphabet, character), entries in sorted(grouped.items()):
        if len(entries) != total_expected:
            raise ValueError(
                f"Omniglot class {alphabet}/{character} has {len(entries)} instances; expected {total_expected}."
            )
        permutation = rng.permutation(len(entries))
        split_map = {}
        for index, permuted_index in enumerate(permutation.tolist()):
            if index < train_count_per_class:
                split = "train"
            elif index < train_count_per_class + val_count_per_class:
                split = "val"
            else:
                split = "test"
            split_map[permuted_index] = split
        for instance_index, (image_path, source_split) in enumerate(entries):
            split = split_map[instance_index]
            relative_image_path = image_path.resolve()
            try:
                relative_image_path = image_path.resolve().relative_to(output_file.parent.resolve())
            except ValueError:
                relative_image_path = image_path.resolve()
            record = {
                "sequence_id": f"{alphabet}__{character}__{image_path.stem}",
                "position": 0,
                "image_path": str(relative_image_path),
                "split": split,
                "example_id": f"{alphabet}__{character}__{image_path.stem}",
                "group_id": alphabet,
                "family": None,
                "transcription": f"{alphabet}__{character}",
                "metadata": {
                    "alphabet": alphabet,
                    "character": character,
                    "source_split": source_split,
                    "original_filename": image_path.name,
                },
            }
            records.append(record)

    manifest = GlyphCropManifest.model_validate(
        {
            "dataset_name": "omniglot_character_crops",
            "unit_type": "glyph_crop",
            "metadata": {
                "source": "Omniglot",
                "split_strategy": "within_character_deterministic",
                "split_seed": split_seed,
                "train_count_per_class": train_count_per_class,
                "val_count_per_class": val_count_per_class,
                "test_count_per_class": test_count_per_class,
            },
            "records": records,
        }
    )
    write_yaml(output_file, manifest.model_dump(mode="json"))
    return manifest


def summarize_omniglot_local_artifacts(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(dataset_root)
    manifest_file = None if manifest_path is None else Path(manifest_path)
    archive_sizes = {
        archive_name: (root / archive_name).stat().st_size
        for archive_name in OMNIGLOT_ARCHIVES
        if (root / archive_name).exists()
    }
    extracted_root = root / "extracted"
    extracted_png_size = sum(path.stat().st_size for path in extracted_root.rglob("*.png")) if extracted_root.exists() else 0
    manifest_size = manifest_file.stat().st_size if manifest_file is not None and manifest_file.exists() else 0
    total_footprint = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    manifest_summary: dict[str, object] = {}
    if manifest_file is not None and manifest_file.exists():
        payload = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
        manifest = GlyphCropManifest.model_validate(payload)
        split_counts: dict[str, int] = defaultdict(int)
        label_counts: dict[str, int] = defaultdict(int)
        group_counts: dict[str, int] = defaultdict(int)
        for record in manifest.records:
            split_counts[record.split] += 1
            if record.transcription is not None:
                label_counts[record.transcription] += 1
            if record.group_id is not None:
                group_counts[record.group_id] += 1
        manifest_summary = {
            "dataset_name": manifest.dataset_name,
            "record_count": len(manifest.records),
            "split_counts": dict(sorted(split_counts.items())),
            "label_count": len(label_counts),
            "group_count": len(group_counts),
            "metadata": manifest.metadata,
        }

    return {
        "dataset_root": str(root),
        "archive_sizes": archive_sizes,
        "extracted_png_size": extracted_png_size,
        "manifest_size": manifest_size,
        "total_footprint": total_footprint,
        "manifest_summary": manifest_summary,
    }


def format_omniglot_integration_note(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> str:
    summary = summarize_omniglot_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    manifest_summary = summary["manifest_summary"]
    assert isinstance(manifest_summary, dict)
    lines = [
        "# Omniglot Integration Note",
        "",
        "## Source",
        f"- Repository: `{OMNIGLOT_REPOSITORY_URL}`",
    ]
    for archive_name, archive_url in OMNIGLOT_ARCHIVES.items():
        lines.append(f"- Official archive: `{archive_name}` -> `{archive_url}`")
    lines.extend(
        [
            "",
            "## Local Storage",
            "- Full download strategy: **full dataset**",
        ]
    )
    archive_sizes = summary["archive_sizes"]
    assert isinstance(archive_sizes, dict)
    if archive_sizes:
        lines.append("- Downloaded archive size:")
        for archive_name, size_bytes in archive_sizes.items():
            lines.append(f"  - `{archive_name}`: `{size_bytes}` bytes")
    lines.extend(
        [
            f"- Extracted PNG size: `{summary['extracted_png_size']}` bytes",
            f"- Generated manifest size: `{summary['manifest_size']}` bytes",
            f"- Total local footprint under `{Path(dataset_root)}`: `{summary['total_footprint']}` bytes",
            "",
            "This is well below the project’s `40 GB` maximum dataset budget.",
            "",
            "## Preparation Strategy",
            "- Extract both official archives under `data/raw/omniglot/extracted/`.",
            "- Treat each image as one labeled glyph crop.",
            "- Use `alphabet__character` as the transcription label.",
            "- Use the alphabet name as `group_id`.",
            "- Preserve the original Omniglot archive source (`images_background` vs `images_evaluation`) in per-record metadata.",
            "",
            "## Split Strategy",
            "",
            "Omniglot’s original background/evaluation split holds out whole character classes and therefore does not support same-label symbol recovery. To make the current fixed-vs-uncertainty protocol meaningful, we use a deterministic within-character split:",
            "",
            "- train: `12` samples per character",
            "- val: `4` samples per character",
            "- test: `4` samples per character",
        ]
    )
    if manifest_summary:
        split_counts = manifest_summary.get("split_counts", {})
        metadata = manifest_summary.get("metadata", {})
        lines.extend(
            [
                "",
                "This produces:",
                "",
                f"- `{manifest_summary.get('record_count', 0)}` total labeled crops",
            ]
        )
        if isinstance(split_counts, dict):
            for split_name, split_count in split_counts.items():
                lines.append(f"- `{split_count}` {split_name} examples")
        lines.extend(
            [
                f"- `{manifest_summary.get('label_count', 0)}` character classes",
                f"- `{manifest_summary.get('group_count', 0)}` alphabet groups",
            ]
        )
        if isinstance(metadata, dict) and metadata.get("split_seed") is not None:
            lines.append(f"- split seed: `{metadata['split_seed']}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Sequences are single-glyph crops, so sequence-level structural metrics are limited.",
            "- Family labels matching the current decipherment hypothesis families are not available.",
            "- The strongest evidence from this dataset is expected to remain symbol-level rather than semantic or family-level.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_omniglot_integration_note(
    dataset_root: str | Path,
    note_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    destination = Path(note_path)
    destination.write_text(
        format_omniglot_integration_note(dataset_root=dataset_root, manifest_path=manifest_path),
        encoding="utf-8",
    )
    return destination
