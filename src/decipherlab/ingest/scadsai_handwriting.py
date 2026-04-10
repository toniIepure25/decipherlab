from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re
import tarfile

import yaml

from decipherlab.config import DatasetConfig
from decipherlab.ingest.schema import GlyphCropManifest, GlyphCropManifestRecord
from decipherlab.ingest.validation import format_manifest_summary_markdown, summarize_glyph_crop_manifest
from decipherlab.utils.io import ensure_directory, write_json, write_text, write_yaml


@dataclass(frozen=True)
class ScadsWordToken:
    page_id: str
    line_id: str
    word_id: str
    word_file: str
    word_idx: int
    raw_text: str
    normalized_text: str
    xml_file: str
    line_file: str


def normalize_scads_word(value: str) -> str:
    lowered = value.strip().lower()
    lowered = (
        lowered.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    lowered = re.sub(r"(^[^a-z]+|[^a-z]+$)", "", lowered)
    if not lowered or len(lowered) < 2 or len(lowered) > 16:
        return ""
    if not re.fullmatch(r"[a-z]+", lowered):
        return ""
    return lowered


def _page_split_map(page_ids: list[str], seed: int, train_pages: int, val_pages: int) -> dict[str, str]:
    if train_pages + val_pages >= len(page_ids):
        raise ValueError("train_pages + val_pages must be smaller than the number of available pages.")
    shuffled = list(page_ids)
    random.Random(seed).shuffle(shuffled)
    train = set(shuffled[:train_pages])
    val = set(shuffled[train_pages : train_pages + val_pages])
    return {
        page_id: ("train" if page_id in train else "val" if page_id in val else "test")
        for page_id in page_ids
    }


def _load_scads_rows(archive_path: Path) -> list[dict[str, str]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        word_member = archive.getmember("./ground_truth/csv/word_annotations.csv")
        word_extracted = archive.extractfile(word_member)
        if word_extracted is None:
            raise ValueError("Could not extract ScaDS word annotation CSV from the archive.")
        line_member = archive.getmember("./ground_truth/csv/line_annotations.csv")
        line_extracted = archive.extractfile(line_member)
        if line_extracted is None:
            raise ValueError("Could not extract ScaDS line annotation CSV from the archive.")

        word_rows = [dict(row) for row in csv.DictReader(io.TextIOWrapper(word_extracted, encoding="utf-8"))]
        line_rows = [dict(row) for row in csv.DictReader(io.TextIOWrapper(line_extracted, encoding="utf-8"))]
        line_file_map = {row["line_id"]: row["line_file"] for row in line_rows}
        for row in word_rows:
            row["line_file"] = line_file_map.get(row["line_id"], f"{row['line_id']}.png")
        return word_rows


def prepare_scadsai_grouped_manifest(
    archive_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    note_path: str | Path,
    split_seed: int = 23,
    train_pages: int = 220,
    val_pages: int = 80,
    selected_vocabulary_size: int = 8,
    min_token_count_per_split: int = 8,
    min_sequence_length: int = 4,
) -> dict[str, object]:
    archive = Path(archive_path)
    output_root = ensure_directory(output_dir)
    rows = _load_scads_rows(archive)
    page_ids = sorted({row["page_id"] for row in rows})
    split_map = _page_split_map(page_ids, seed=split_seed, train_pages=train_pages, val_pages=val_pages)

    per_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    line_tokens: dict[tuple[str, str], list[ScadsWordToken]] = defaultdict(list)
    for row in rows:
        normalized_text = normalize_scads_word(row["text"])
        if not normalized_text:
            continue
        page_id = row["page_id"]
        split = split_map[page_id]
        token = ScadsWordToken(
            page_id=page_id,
            line_id=row["line_id"],
            word_id=row["word_id"],
            word_file=row["word_file"],
            word_idx=int(row["word_idx"]),
            raw_text=row["text"],
            normalized_text=normalized_text,
            xml_file=row["xml_file"],
            line_file=row["line_file"],
        )
        line_tokens[(page_id, row["line_id"])].append(token)
        per_split_counts[split][normalized_text] += 1

    common_tokens = sorted(
        set(per_split_counts["train"]) & set(per_split_counts["val"]) & set(per_split_counts["test"])
    )
    scored_tokens: list[tuple[int, str, list[int]]] = []
    for token in common_tokens:
        counts = [per_split_counts[split][token] for split in ("train", "val", "test")]
        if min(counts) >= min_token_count_per_split:
            scored_tokens.append((sum(counts), token, counts))
    scored_tokens.sort(key=lambda item: (-item[0], item[1]))
    selected_tokens = [token for _, token, _ in scored_tokens[:selected_vocabulary_size]]
    if len(selected_tokens) < selected_vocabulary_size:
        raise ValueError(
            "Not enough repeated token labels for the requested ScaDS grouped benchmark vocabulary: "
            f"required {selected_vocabulary_size}, found {len(selected_tokens)}."
        )

    selected_set = set(selected_tokens)
    crop_root = ensure_directory(output_root / "images")
    manifest_records: list[GlyphCropManifestRecord] = []
    sequence_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    selected_word_files: set[str] = set()
    for (page_id, line_id), tokens in sorted(line_tokens.items()):
        split = split_map[page_id]
        filtered = [token for token in sorted(tokens, key=lambda item: item.word_idx) if token.normalized_text in selected_set]
        if len(filtered) < min_sequence_length:
            continue
        sequence_counts[split] += 1
        sequence_lengths.append(len(filtered))
        for position, token in enumerate(filtered):
            selected_word_files.add(token.word_file)
            crop_relative_path = Path("images") / split / token.word_file
            manifest_records.append(
                GlyphCropManifestRecord(
                    sequence_id=line_id,
                    position=position,
                    image_path=str(crop_relative_path),
                    split=split,
                    example_id=f"{line_id}_tok_{position:02d}",
                    group_id=page_id,
                    transcription=token.normalized_text,
                    metadata={
                        "source_page_id": page_id,
                        "source_line_id": line_id,
                        "source_word_id": token.word_id,
                        "source_word_file": token.word_file,
                        "source_line_file": token.line_file,
                        "source_xml_file": token.xml_file,
                        "raw_text": token.raw_text,
                        "word_index_in_line": token.word_idx,
                        "label_source": "ground_truth_word_annotation",
                        "grouped_sequence_source": "real_handwritten_line",
                    },
                )
            )

    with tarfile.open(archive, "r:gz") as tar:
        for word_file in sorted(selected_word_files):
            member_name = f"./images/words/{word_file}"
            try:
                member = tar.getmember(member_name)
            except KeyError as exc:
                raise ValueError(f"Missing word image {member_name} in ScaDS archive.") from exc
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not extract word image {member_name} from ScaDS archive.")
            member_bytes = extracted.read()
            for split in ("train", "val", "test"):
                candidate_path = crop_root / split / word_file
                if candidate_path.exists():
                    continue
            output_path = None
            for record in manifest_records:
                if Path(record.image_path).name == word_file:
                    output_path = output_root / record.image_path
                    break
            if output_path is None:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(member_bytes)

    manifest = GlyphCropManifest(
        dataset_name="scadsai_grouped_words",
        metadata={
            "source_corpus": "ScaDS.AI German line- and word-level handwriting dataset (Zenodo 18301532)",
            "task_scope": "real_grouped_token_sequence_public_integration",
            "label_source": "ground_truth_word_annotations",
            "split_seed": split_seed,
            "selected_tokens": selected_tokens,
            "min_sequence_length": min_sequence_length,
            "token_granularity": "word",
            "group_structure": "handwritten_line",
            "normalization": "lowercase_ascii_german_transliteration",
        },
        records=manifest_records,
    )
    write_yaml(manifest_path, manifest.model_dump(mode="json"))

    validation_dataset_config = DatasetConfig(
        source="manifest",
        manifest_path=Path(manifest_path),
        manifest_format="glyph_crop",
        min_sequences_per_split_warning=2,
        min_symbol_instances_per_train_class_warning=2,
        min_family_instances_per_split_warning=1,
    )
    validation_summary = summarize_glyph_crop_manifest(
        manifest,
        manifest_path=manifest_path,
        dataset_config=validation_dataset_config,
    )
    write_json(Path(manifest_path).with_suffix(".summary.json"), validation_summary)
    write_text(Path(manifest_path).with_suffix(".summary.md"), format_manifest_summary_markdown(validation_summary))

    note_lines = [
        "# ScaDS.AI Grouped Token Manifest",
        "",
        "- Source: `ScaDS.AI German line- and word-level handwriting dataset` (Zenodo record 18301532).",
        "- Token alignment source: `ground_truth/csv/word_annotations.csv` plus `ground_truth/xml/*.xml`.",
        "- Images source: pre-extracted word crops under `images/words/` in the public archive.",
        "- Labels are human-provided word annotations normalized to lowercase ASCII transliterations for grouped-sequence evaluation.",
        f"- Selected vocabulary: `{selected_tokens}`.",
        f"- Sequence granularity: real handwritten lines filtered to selected vocabulary with minimum length `{min_sequence_length}`.",
        f"- Train/val/test grouped sequences: `{dict(sequence_counts)}`.",
        f"- Mean retained sequence length: `{(sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0:.3f}`.",
        "",
        "## Caveats",
        "",
        "- This is a real grouped/token-aligned corpus with public word-level annotations, not an OCR-derived benchmark.",
        "- The current grouped task still evaluates grouped recovery rather than semantic or family-level manuscript reasoning.",
    ]
    write_text(note_path, "\n".join(note_lines) + "\n")

    return {
        "manifest_path": str(Path(manifest_path)),
        "note_path": str(Path(note_path)),
        "sequence_counts": dict(sequence_counts),
        "selected_tokens": selected_tokens,
        "page_count": len(page_ids),
    }
