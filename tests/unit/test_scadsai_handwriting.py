from __future__ import annotations

import csv
from pathlib import Path
import tarfile

from PIL import Image
import yaml

from decipherlab.ingest.scadsai_handwriting import normalize_scads_word, prepare_scadsai_grouped_manifest


def _create_scads_fixture(archive_path: Path) -> None:
    source_root = archive_path.parent / "scads_fixture"
    (source_root / "ground_truth" / "csv").mkdir(parents=True, exist_ok=True)
    (source_root / "images" / "words").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    line_rows: list[dict[str, str]] = []
    page_plan = {
        "0001": ("train", [["der", "die", "und", "in"], ["der", "die", "und", "in"]]),
        "0002": ("train", [["der", "die", "und", "in"], ["der", "die", "und", "in"]]),
        "0003": ("val", [["der", "die", "und", "in"], ["der", "die", "und", "in"]]),
        "0004": ("test", [["der", "die", "und", "in"], ["der", "die", "und", "in"]]),
    }
    for page_id, (_, lines) in page_plan.items():
        for line_offset, tokens in enumerate(lines):
            line_id = f"{page_id}-{line_offset:03d}"
            line_rows.append(
                {
                    "page_id": page_id,
                    "line_id": line_id,
                    "xml_file": f"{page_id}.xml",
                    "line_file": f"{line_id}.png",
                    "line_idx": str(line_offset),
                    "x": "0",
                    "y": str(line_offset * 10),
                    "width": "40",
                    "height": "12",
                    "text": " ".join(tokens),
                }
            )
            for word_idx, token in enumerate(tokens):
                word_id = f"{line_id}-{word_idx:02d}"
                word_file = f"{word_id}.png"
                rows.append(
                    {
                        "page_id": page_id,
                        "line_id": line_id,
                        "word_id": word_id,
                        "xml_file": f"{page_id}.xml",
                        "word_file": word_file,
                        "word_idx": str(word_idx),
                        "text": token,
                        "x_rel": str(word_idx * 10),
                        "y_rel": "0",
                        "width": "10",
                        "height": "10",
                        "x_abs": str(word_idx * 10),
                        "y_abs": "0",
                        "line_file": f"{line_id}.png",
                    }
                )
                image = Image.new("L", (12, 12), color=200)
                image.save(source_root / "images" / "words" / word_file)
    csv_path = source_root / "ground_truth" / "csv" / "word_annotations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page_id",
                "line_id",
                "word_id",
                "xml_file",
                "word_file",
                "word_idx",
                "text",
                "x_rel",
                "y_rel",
                "width",
                "height",
                "x_abs",
                "y_abs",
                "line_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    line_csv_path = source_root / "ground_truth" / "csv" / "line_annotations.csv"
    with line_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page_id",
                "line_id",
                "xml_file",
                "line_file",
                "line_idx",
                "x",
                "y",
                "width",
                "height",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(line_rows)

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source_root.rglob("*")):
            archive.add(path, arcname=f"./{path.relative_to(source_root)}")


def test_normalize_scads_word_transliterates_german_tokens():
    assert normalize_scads_word("Märchen") == "maerchen"
    assert normalize_scads_word("größere") == "groessere"
    assert normalize_scads_word("1812") == ""


def test_prepare_scadsai_grouped_manifest_builds_grouped_lines(tmp_path):
    archive_path = tmp_path / "scads_fixture.tar.gz"
    _create_scads_fixture(archive_path)
    output_dir = tmp_path / "processed"
    manifest_path = output_dir / "manifest.yaml"
    note_path = output_dir / "README.md"

    result = prepare_scadsai_grouped_manifest(
        archive_path=archive_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        note_path=note_path,
        split_seed=23,
        train_pages=2,
        val_pages=1,
        selected_vocabulary_size=4,
        min_token_count_per_split=1,
        min_sequence_length=4,
    )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "scadsai_grouped_words"
    assert len(payload["records"]) == 32
    assert Path(result["note_path"]).exists()
    assert (output_dir / "images" / "train").exists()
    assert any(record["metadata"]["label_source"] == "ground_truth_word_annotation" for record in payload["records"])
