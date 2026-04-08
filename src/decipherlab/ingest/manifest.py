from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from decipherlab.config import DatasetConfig
from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.ingest.validation import summarize_glyph_crop_manifest
from decipherlab.models import DatasetCollection, GlyphCrop, SequenceExample


def _load_manifest_payload(manifest_path: Path) -> dict[str, object]:
    if manifest_path.suffix.lower() in {".yaml", ".yml"}:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    else:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    return payload


def _load_grayscale_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_synthetic_manifest_dataset(manifest_path: str | Path) -> DatasetCollection:
    manifest_file = Path(manifest_path)
    manifest = _load_manifest_payload(manifest_file)

    examples: list[SequenceExample] = []
    for record in manifest["examples"]:
        artifact_path = manifest_file.parent / record["artifact_path"]
        payload = np.load(artifact_path, allow_pickle=False)
        glyph_images = payload["glyph_images"]
        observed_symbols = payload["observed_symbols"].tolist()
        plaintext = str(payload["plaintext"][0]) if "plaintext" in payload else None
        glyphs = [
            GlyphCrop(position=index, image=image.astype(np.float32), true_symbol=observed_symbols[index])
            for index, image in enumerate(glyph_images)
        ]
        examples.append(
            SequenceExample(
                example_id=record["example_id"],
                family=record["family"],
                glyphs=glyphs,
                plaintext=plaintext,
                observed_symbols=observed_symbols,
                split=record.get("split", "test"),
                metadata=record.get("metadata", {}),
            )
        )
    return DatasetCollection(
        dataset_name=str(manifest.get("dataset_name", "synthetic_manifest")),
        examples=examples,
        manifest_path=str(manifest_file),
        metadata={"format": "synthetic_npz", "seed": manifest.get("seed")},
    )


def load_glyph_crop_manifest_dataset(
    manifest_path: str | Path,
    dataset_config: DatasetConfig | None = None,
) -> DatasetCollection:
    manifest_file = Path(manifest_path)
    payload = _load_manifest_payload(manifest_file)
    manifest = GlyphCropManifest.model_validate(payload)
    validation_report = summarize_glyph_crop_manifest(
        manifest,
        manifest_path=manifest_file,
        dataset_config=dataset_config,
    )

    grouped_records: dict[str, list] = {}
    for record in manifest.records:
        grouped_records.setdefault(record.sequence_id, []).append(record)

    examples: list[SequenceExample] = []
    for sequence_id, records in grouped_records.items():
        ordered = sorted(records, key=lambda record: record.position)
        glyphs: list[GlyphCrop] = []
        labels: list[str | None] = []
        families = {record.family for record in ordered if record.family is not None}
        group_ids = {record.group_id for record in ordered if record.group_id is not None}
        for record in ordered:
            image_path = (manifest_file.parent / record.image_path).resolve()
            glyphs.append(
                GlyphCrop(
                    position=record.position,
                    image=_load_grayscale_image(image_path),
                    true_symbol=record.transcription,
                    variant_id=record.example_id,
                    source_path=str(image_path),
                )
            )
            labels.append(record.transcription)

        examples.append(
            SequenceExample(
                example_id=sequence_id,
                family=next(iter(families)) if len(families) == 1 else None,
                glyphs=glyphs,
                plaintext=None,
                observed_symbols=labels,
                split=ordered[0].split,
                metadata={
                    "unit_type": manifest.unit_type,
                    "labeled_positions": sum(label is not None for label in labels),
                    "group_id": next(iter(group_ids)) if len(group_ids) == 1 else None,
                    "manifest_records": [record.model_dump(mode="python") for record in ordered],
                },
            )
        )

    split_counts = {split: len([example for example in examples if example.split == split]) for split in sorted({example.split for example in examples})}
    labeled_positions = sum(example.labeled_symbol_count for example in examples)
    total_positions = sum(example.sequence_length for example in examples)
    return DatasetCollection(
        dataset_name=manifest.dataset_name,
        examples=sorted(examples, key=lambda example: (example.split, example.example_id)),
        manifest_path=str(manifest_file),
        metadata=manifest.metadata
        | {
            "format": "glyph_crop",
            "split_counts": split_counts,
            "sequence_count": len(examples),
            "label_coverage": (labeled_positions / total_positions) if total_positions else 0.0,
            "family_label_coverage": (
                sum(example.family is not None for example in examples) / len(examples)
                if examples
                else 0.0
            ),
            "validation_report": validation_report,
        },
    )
