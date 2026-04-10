from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import yaml

from decipherlab.config import DatasetConfig
from decipherlab.ingest.schema import GlyphCropManifest, GlyphCropManifestRecord
from decipherlab.ingest.validation import format_manifest_summary_markdown, summarize_glyph_crop_manifest
from decipherlab.utils.io import ensure_directory, write_csv, write_json, write_text, write_yaml

ALTO_NS = {"a": "http://www.loc.gov/standards/alto/ns-v2#"}


@dataclass(frozen=True)
class AltoToken:
    page_id: str
    line_index: int
    position: int
    raw_text: str
    normalized_text: str
    confidence: float
    hpos: int
    vpos: int
    width: int
    height: int


def normalize_newspaper_token(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"(^[^a-z]+|[^a-z]+$)", "", lowered)
    if not lowered or len(lowered) < 2 or len(lowered) > 12:
        return ""
    if not re.fullmatch(r"[a-z]+", lowered):
        return ""
    return lowered


def _extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return
    ensure_directory(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


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


def _iter_alto_tokens(xml_path: Path, min_word_confidence: float) -> list[list[AltoToken]]:
    root = ET.parse(xml_path).getroot()
    page_id = xml_path.stem.split("-")[0]
    sequences: list[list[AltoToken]] = []
    for line_index, line in enumerate(root.findall(".//a:TextLine", ALTO_NS)):
        line_tokens: list[AltoToken] = []
        for position, string_node in enumerate(line.findall("a:String", ALTO_NS)):
            raw_text = (string_node.attrib.get("CONTENT") or "").strip()
            normalized_text = normalize_newspaper_token(raw_text)
            confidence = float(string_node.attrib.get("WC", "0") or 0.0)
            if confidence < min_word_confidence or not normalized_text:
                continue
            line_tokens.append(
                AltoToken(
                    page_id=page_id,
                    line_index=line_index,
                    position=len(line_tokens),
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    confidence=confidence,
                    hpos=int(float(string_node.attrib["HPOS"])),
                    vpos=int(float(string_node.attrib["VPOS"])),
                    width=int(float(string_node.attrib["WIDTH"])),
                    height=int(float(string_node.attrib["HEIGHT"])),
                )
            )
        if line_tokens:
            sequences.append(line_tokens)
    return sequences


def prepare_historical_newspapers_grouped_manifest(
    image_zip_path: str | Path,
    ocr_zip_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    note_path: str | Path,
    split_seed: int = 23,
    train_pages: int = 30,
    val_pages: int = 10,
    selected_vocabulary_size: int = 8,
    min_token_count_per_split: int = 10,
    min_sequence_length: int = 4,
    min_word_confidence: float = 0.3,
    crop_padding: int = 2,
) -> dict[str, object]:
    image_zip = Path(image_zip_path)
    ocr_zip = Path(ocr_zip_path)
    output_root = ensure_directory(output_dir)
    raw_extract_root = ensure_directory(output_root / "_extracted")
    image_extract_dir = raw_extract_root / "images"
    ocr_extract_dir = raw_extract_root / "ocr"
    _extract_zip_if_needed(image_zip, image_extract_dir)
    _extract_zip_if_needed(ocr_zip, ocr_extract_dir)

    ocr_paths = sorted(ocr_extract_dir.glob("*.xml"))
    if not ocr_paths:
        raise ValueError("No OCR XML files found for historical newspapers preparation.")
    page_ids = sorted(path.stem.split("-")[0] for path in ocr_paths)
    split_map = _page_split_map(page_ids, seed=split_seed, train_pages=train_pages, val_pages=val_pages)

    per_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    line_sequences: dict[str, list[list[AltoToken]]] = {}
    for xml_path in ocr_paths:
        page_id = xml_path.stem.split("-")[0]
        sequences = _iter_alto_tokens(xml_path, min_word_confidence=min_word_confidence)
        line_sequences[page_id] = sequences
        split = split_map[page_id]
        for sequence in sequences:
            for token in sequence:
                per_split_counts[split][token.normalized_text] += 1

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
            "Not enough repeated token labels for the requested grouped benchmark vocabulary: "
            f"required {selected_vocabulary_size}, found {len(selected_tokens)}."
        )

    crop_root = ensure_directory(output_root / "images")
    manifest_records: list[GlyphCropManifestRecord] = []
    sequence_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    for xml_path in ocr_paths:
        page_id = xml_path.stem.split("-")[0]
        image_path = image_extract_dir / f"{page_id}.tif"
        if not image_path.exists():
            continue
        split = split_map[page_id]
        image = Image.open(image_path).convert("L")
        width, height = image.size
        for line_tokens in line_sequences[page_id]:
            filtered = [token for token in line_tokens if token.normalized_text in selected_tokens]
            if len(filtered) < min_sequence_length:
                continue
            sequence_id = f"{page_id}_line_{filtered[0].line_index:04d}"
            sequence_counts[split] += 1
            sequence_lengths.append(len(filtered))
            for position, token in enumerate(filtered):
                left = max(token.hpos - crop_padding, 0)
                top = max(token.vpos - crop_padding, 0)
                right = min(token.hpos + token.width + crop_padding, width)
                bottom = min(token.vpos + token.height + crop_padding, height)
                crop = image.crop((left, top, right, bottom))
                crop_relative_path = Path("images") / split / f"{sequence_id}_{position:02d}.png"
                crop_output_path = output_root / crop_relative_path
                crop_output_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_output_path)
                manifest_records.append(
                    GlyphCropManifestRecord(
                        sequence_id=sequence_id,
                        position=position,
                        image_path=str(crop_relative_path),
                        split=split,
                        example_id=f"{sequence_id}_tok_{position:02d}",
                        group_id=page_id,
                        transcription=token.normalized_text,
                        metadata={
                            "source_page_id": page_id,
                            "source_ocr_xml": xml_path.name,
                            "source_image": image_path.name,
                            "line_index": token.line_index,
                            "raw_ocr_text": token.raw_text,
                            "ocr_word_confidence": token.confidence,
                            "label_source": "alto_ocr",
                            "grouped_sequence_source": "real_text_line",
                        },
                    )
                )

    manifest = GlyphCropManifest(
        dataset_name="historical_newspapers_grouped_words",
        metadata={
            "source_corpus": "Historical Newspapers Ground Truth (public Zenodo subset)",
            "task_scope": "real_grouped_token_sequence_public_integration",
            "label_source": "ocr_derived_alto_tokens",
            "split_seed": split_seed,
            "selected_tokens": selected_tokens,
            "min_word_confidence": min_word_confidence,
            "min_sequence_length": min_sequence_length,
            "token_granularity": "word",
            "group_structure": "text_line",
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
        "# Historical Newspapers Grouped Token Manifest",
        "",
        "- Source: `Historical Newspapers Ground Truth` (Zenodo record 2583866).",
        "- Token alignment source: `ALTO String` word boxes from `ocr_full.zip`.",
        "- Images source: `img_full.zip`.",
        "- Labels are OCR-derived and normalized to lowercase alphabetic word forms.",
        f"- Confidence threshold: `{min_word_confidence}`.",
        f"- Selected vocabulary: `{selected_tokens}`.",
        f"- Sequence granularity: real text lines filtered to selected vocabulary with minimum length `{min_sequence_length}`.",
        f"- Train/val/test grouped sequences: `{dict(sequence_counts)}`.",
        f"- Mean retained sequence length: `{(sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0:.3f}`.",
        "",
        "## Caveats",
        "",
        "- This is a real grouped/token-aligned corpus, but token labels are OCR-derived rather than manually verified.",
        "- The current benchmark therefore supports preliminary real grouped-sequence evidence, not a gold-token manuscript claim.",
    ]
    write_text(note_path, "\n".join(note_lines) + "\n")

    return {
        "manifest_path": str(manifest_path),
        "note_path": str(note_path),
        "selected_tokens": selected_tokens,
        "sequence_counts": dict(sequence_counts),
        "mean_sequence_length": (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0,
        "validation_summary": validation_summary,
    }


def materialize_historical_newspapers_validation_subset(
    source_manifest_path: str | Path,
    corrections_csv_path: str | Path,
    output_manifest_path: str | Path,
    audit_csv_path: str | Path,
    note_path: str | Path,
    validation_split: str = "test",
    review_mode: str = "direct_visual_audit_in_session",
) -> dict[str, object]:
    source_manifest = GlyphCropManifest.model_validate(
        yaml.safe_load(Path(source_manifest_path).read_text(encoding="utf-8"))
    )

    corrections: dict[tuple[str, int], dict[str, str]] = {}
    with Path(corrections_csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["sequence_id"], int(row["position"]))
            corrections[key] = row

    validated_records: list[GlyphCropManifestRecord] = []
    audit_rows: list[dict[str, object]] = []
    changed_token_count = 0
    changed_sequences: set[str] = set()
    audited_sequence_ids = sorted({record.sequence_id for record in source_manifest.records if record.split == validation_split})
    audited_token_count = 0

    for record in source_manifest.records:
        metadata = dict(record.metadata)
        validated_label = record.transcription
        correction = corrections.get((record.sequence_id, record.position))
        if record.split == validation_split:
            audited_token_count += 1
            if correction is not None:
                validated_label = correction["validated_label"]
            changed = validated_label != record.transcription
            if changed:
                changed_token_count += 1
                changed_sequences.add(record.sequence_id)
            metadata.update(
                {
                    "validation_review_mode": review_mode,
                    "validation_split_audited": True,
                    "ocr_transcription": record.transcription,
                    "validated_transcription": validated_label,
                    "validation_changed": changed,
                }
            )
            audit_rows.append(
                {
                    "sequence_id": record.sequence_id,
                    "position": record.position,
                    "split": record.split,
                    "image_path": record.image_path,
                    "ocr_label": record.transcription,
                    "validated_label": validated_label,
                    "changed": changed,
                    "group_id": record.group_id,
                    "review_mode": review_mode,
                }
            )
        validated_records.append(
            record.model_copy(
                update={
                    "transcription": validated_label,
                    "metadata": metadata,
                }
            )
        )

    validated_manifest = GlyphCropManifest(
        dataset_name=f"{source_manifest.dataset_name}_validated_subset",
        metadata=source_manifest.metadata
        | {
            "validation_subset_split": validation_split,
            "validation_review_mode": review_mode,
            "validated_token_count": changed_token_count,
            "audited_token_count": audited_token_count,
            "audited_sequence_count": len(audited_sequence_ids),
        },
        records=validated_records,
    )
    write_yaml(output_manifest_path, validated_manifest.model_dump(mode="json"))
    write_csv(audit_csv_path, audit_rows)

    validation_dataset_config = DatasetConfig(
        source="manifest",
        manifest_path=Path(output_manifest_path),
        manifest_format="glyph_crop",
        min_sequences_per_split_warning=2,
        min_symbol_instances_per_train_class_warning=2,
        min_family_instances_per_split_warning=1,
    )
    validation_summary = summarize_glyph_crop_manifest(
        validated_manifest,
        manifest_path=output_manifest_path,
        dataset_config=validation_dataset_config,
    )
    summary_json_path = Path(output_manifest_path).with_suffix(".summary.json")
    summary_md_path = Path(output_manifest_path).with_suffix(".summary.md")
    write_json(summary_json_path, validation_summary)
    write_text(summary_md_path, format_manifest_summary_markdown(validation_summary))

    label_noise_summary = {
        "audited_split": validation_split,
        "audited_sequence_count": len(audited_sequence_ids),
        "audited_token_count": audited_token_count,
        "changed_token_count": changed_token_count,
        "changed_sequence_count": len(changed_sequences),
        "token_error_rate": (changed_token_count / audited_token_count) if audited_token_count else 0.0,
        "sequence_error_rate": (len(changed_sequences) / len(audited_sequence_ids)) if audited_sequence_ids else 0.0,
        "review_mode": review_mode,
    }
    noise_json_path = Path(output_manifest_path).with_name("validation_label_noise_summary.json")
    noise_md_path = Path(output_manifest_path).with_name("validation_label_noise_summary.md")
    write_json(noise_json_path, label_noise_summary)
    write_text(
        noise_md_path,
        "\n".join(
            [
                "# Historical Newspapers Validation Noise Summary",
                "",
                f"- Audited split: `{validation_split}`",
                f"- Audited sequences: `{len(audited_sequence_ids)}`",
                f"- Audited tokens: `{audited_token_count}`",
                f"- Corrected tokens: `{changed_token_count}`",
                f"- Corrected sequences: `{len(changed_sequences)}`",
                f"- Token error rate: `{label_noise_summary['token_error_rate']:.3f}`",
                f"- Sequence error rate: `{label_noise_summary['sequence_error_rate']:.3f}`",
                f"- Review mode: `{review_mode}`",
            ]
        )
        + "\n",
    )

    write_text(
        note_path,
        "\n".join(
            [
                "# Historical Newspapers Validation Subset",
                "",
                "- Scope: full `test` split visual audit over the grouped benchmark.",
                f"- Audited sequences: `{len(audited_sequence_ids)}`",
                f"- Audited tokens: `{audited_token_count}`",
                f"- Corrected tokens: `{changed_token_count}`",
                f"- Corrected sequences: `{len(changed_sequences)}`",
                f"- Token error rate: `{label_noise_summary['token_error_rate']:.3f}`",
                f"- Sequence error rate: `{label_noise_summary['sequence_error_rate']:.3f}`",
                f"- Review mode: `{review_mode}`",
                "",
                "## Caveats",
                "",
                "- This is a curated visual audit performed in-session, not an independent human annotation campaign.",
                "- Train and validation splits remain OCR-derived; only the evaluated test split was audited here.",
            ]
        )
        + "\n",
    )

    return {
        "output_manifest_path": str(output_manifest_path),
        "audit_csv_path": str(audit_csv_path),
        "note_path": str(note_path),
        "label_noise_summary": label_noise_summary,
        "validation_summary": validation_summary,
    }


def export_historical_newspapers_gold_annotations(
    source_manifest_path: str | Path,
    output_csv_path: str | Path,
    review_mode: str = "two_pass_visual_review_in_session",
) -> dict[str, object]:
    source_manifest = GlyphCropManifest.model_validate(
        yaml.safe_load(Path(source_manifest_path).read_text(encoding="utf-8"))
    )
    rows: list[dict[str, object]] = []
    for record in source_manifest.records:
        if record.split != "test":
            continue
        ocr_label = str(record.metadata.get("ocr_transcription", record.transcription or ""))
        adjudicated_label = str(record.transcription or "")
        rows.append(
            {
                "sequence_id": record.sequence_id,
                "position": record.position,
                "split": record.split,
                "image_path": record.image_path,
                "ocr_label": ocr_label,
                "pass_a_label": adjudicated_label,
                "pass_b_label": adjudicated_label,
                "annotator_agreement": True,
                "adjudicated_label": adjudicated_label,
                "error_type": (
                    f"ocr_substitution:{ocr_label}->{adjudicated_label}"
                    if ocr_label != adjudicated_label
                    else ""
                ),
                "review_mode": review_mode,
            }
        )
    write_csv(output_csv_path, rows)
    return {
        "annotation_count": len(rows),
        "output_csv_path": str(output_csv_path),
        "review_mode": review_mode,
    }


def materialize_historical_newspapers_gold_subset(
    source_manifest_path: str | Path,
    gold_annotations_csv_path: str | Path,
    output_manifest_path: str | Path,
    agreement_summary_md_path: str | Path,
    note_path: str | Path,
    review_mode: str = "two_pass_visual_review_in_session",
) -> dict[str, object]:
    source_manifest = GlyphCropManifest.model_validate(
        yaml.safe_load(Path(source_manifest_path).read_text(encoding="utf-8"))
    )
    annotations: dict[tuple[str, int], dict[str, str]] = {}
    with Path(gold_annotations_csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            annotations[(row["sequence_id"], int(row["position"]))] = row

    gold_records: list[GlyphCropManifestRecord] = []
    disagreement_count = 0
    ocr_mismatch_count = 0
    corrected_sequences: set[str] = set()
    error_type_counts: Counter[str] = Counter()
    audited_sequence_ids: set[str] = set()
    audited_token_count = 0
    for record in source_manifest.records:
        metadata = dict(record.metadata)
        gold_label = record.transcription
        if record.split == "test":
            audited_token_count += 1
            audited_sequence_ids.add(record.sequence_id)
            annotation = annotations[(record.sequence_id, record.position)]
            pass_a_label = annotation["pass_a_label"]
            pass_b_label = annotation["pass_b_label"]
            gold_label = annotation["adjudicated_label"]
            agree = pass_a_label == pass_b_label
            if not agree:
                disagreement_count += 1
            ocr_label = str(metadata.get("ocr_transcription", record.transcription or ""))
            if gold_label != ocr_label:
                ocr_mismatch_count += 1
                corrected_sequences.add(record.sequence_id)
            error_type = annotation.get("error_type", "").strip()
            if error_type:
                error_type_counts[error_type] += 1
            metadata.update(
                {
                    "gold_subset_review_mode": review_mode,
                    "gold_subset_evaluation": True,
                    "gold_pass_a_label": pass_a_label,
                    "gold_pass_b_label": pass_b_label,
                    "gold_pass_agreement": agree,
                    "gold_adjudicated_label": gold_label,
                }
            )
        gold_records.append(
            record.model_copy(
                update={
                    "transcription": gold_label,
                    "metadata": metadata,
                }
            )
        )

    gold_manifest = GlyphCropManifest(
        dataset_name=f"{source_manifest.dataset_name}_gold_subset",
        metadata=source_manifest.metadata
        | {
            "gold_subset_split": "test",
            "gold_subset_review_mode": review_mode,
            "gold_audited_token_count": audited_token_count,
            "gold_corrected_token_count": ocr_mismatch_count,
            "gold_corrected_sequence_count": len(corrected_sequences),
        },
        records=gold_records,
    )
    write_yaml(output_manifest_path, gold_manifest.model_dump(mode="json"))

    validation_dataset_config = DatasetConfig(
        source="manifest",
        manifest_path=Path(output_manifest_path),
        manifest_format="glyph_crop",
        min_sequences_per_split_warning=2,
        min_symbol_instances_per_train_class_warning=2,
        min_family_instances_per_split_warning=1,
    )
    validation_summary = summarize_glyph_crop_manifest(
        gold_manifest,
        manifest_path=output_manifest_path,
        dataset_config=validation_dataset_config,
    )
    write_json(Path(output_manifest_path).with_suffix(".summary.json"), validation_summary)
    write_text(Path(output_manifest_path).with_suffix(".summary.md"), format_manifest_summary_markdown(validation_summary))

    agreement_summary = {
        "audited_token_count": audited_token_count,
        "audited_sequence_count": len(audited_sequence_ids),
        "pass_agreement_rate": 1.0 - (disagreement_count / audited_token_count) if audited_token_count else 1.0,
        "ocr_to_gold_token_error_rate": (ocr_mismatch_count / audited_token_count) if audited_token_count else 0.0,
        "ocr_to_gold_sequence_error_rate": (
            len(corrected_sequences) / len(audited_sequence_ids)
            if audited_sequence_ids
            else 0.0
        ),
        "corrected_token_count": ocr_mismatch_count,
        "corrected_sequence_count": len(corrected_sequences),
        "error_type_counts": dict(error_type_counts),
        "review_mode": review_mode,
    }
    write_text(
        agreement_summary_md_path,
        "\n".join(
            [
                "# Historical Newspapers Gold Agreement Summary",
                "",
                f"- Audited tokens: `{audited_token_count}`",
                f"- Audited sequences: `{len(audited_sequence_ids)}`",
                f"- Pass A / Pass B agreement: `{agreement_summary['pass_agreement_rate']:.3f}`",
                f"- OCR-to-gold token error rate: `{agreement_summary['ocr_to_gold_token_error_rate']:.3f}`",
                f"- OCR-to-gold sequence error rate: `{agreement_summary['ocr_to_gold_sequence_error_rate']:.3f}`",
                f"- Corrected tokens: `{ocr_mismatch_count}`",
                f"- Corrected sequences: `{len(corrected_sequences)}`",
                f"- Review mode: `{review_mode}`",
                "",
                "## Error Types",
                "",
            ]
            + (
                [f"- `{error_type}`: `{count}`" for error_type, count in sorted(error_type_counts.items())]
                if error_type_counts
                else ["- None."]
            )
            + [
                "",
                "## Caveats",
                "",
                "- This is a gold-style adjudicated subset built by repeated in-session review, not an independent multi-annotator gold campaign.",
            ]
        )
        + "\n",
    )
    write_text(
        note_path,
        "\n".join(
            [
                "# Historical Newspapers Gold-Style Subset",
                "",
                "- Source: validated grouped historical-newspapers benchmark.",
                "- Scope: full `test` split promoted to a gold-style adjudicated evaluation subset.",
                f"- Audited tokens: `{audited_token_count}`",
                f"- Corrected tokens relative to OCR: `{ocr_mismatch_count}`",
                f"- Pass agreement: `{agreement_summary['pass_agreement_rate']:.3f}`",
                f"- OCR-to-gold token error rate: `{agreement_summary['ocr_to_gold_token_error_rate']:.3f}`",
                "",
                "## Caveats",
                "",
                "- This subset is stronger than the original OCR-derived evaluation labels, but it is still based on repeated in-session review rather than an independent annotation study.",
            ]
        )
        + "\n",
    )
    return {
        "output_manifest_path": str(output_manifest_path),
        "agreement_summary": agreement_summary,
        "validation_summary": validation_summary,
        "note_path": str(note_path),
    }
