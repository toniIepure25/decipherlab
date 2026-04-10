from __future__ import annotations

from pathlib import Path

import yaml

from decipherlab.ingest.historical_newspapers import (
    export_historical_newspapers_gold_annotations,
    materialize_historical_newspapers_gold_subset,
    materialize_historical_newspapers_validation_subset,
    normalize_newspaper_token,
)
from tests.helpers import create_real_manifest_fixture


def test_normalize_newspaper_token_keeps_simple_alpha_tokens():
    assert normalize_newspaper_token("Der,") == "der"
    assert normalize_newspaper_token("von") == "von"


def test_normalize_newspaper_token_rejects_short_numeric_and_mixed_tokens():
    assert normalize_newspaper_token("a") == ""
    assert normalize_newspaper_token("1897") == ""
    assert normalize_newspaper_token("ab3") == "ab"


def test_materialize_historical_newspapers_validation_subset_applies_corrections(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path)
    corrections_path = tmp_path / "corrections.csv"
    corrections_path.write_text(
        "sequence_id,position,ocr_label,validated_label,reason\n"
        "test,0,gd,ga,unit_test\n",
        encoding="utf-8",
    )
    output_manifest = tmp_path / "validated_manifest.yaml"
    audit_csv = tmp_path / "audit.csv"
    note_path = tmp_path / "note.md"

    result = materialize_historical_newspapers_validation_subset(
        source_manifest_path=manifest_path,
        corrections_csv_path=corrections_path,
        output_manifest_path=output_manifest,
        audit_csv_path=audit_csv,
        note_path=note_path,
    )

    payload = yaml.safe_load(output_manifest.read_text(encoding="utf-8"))
    corrected_record = next(
        record
        for record in payload["records"]
        if record["sequence_id"] == "test" and record["position"] == 0
    )

    assert corrected_record["transcription"] == "ga"
    assert corrected_record["metadata"]["validation_changed"] is True
    assert result["label_noise_summary"]["changed_token_count"] == 1
    assert Path(result["audit_csv_path"]).exists()


def test_materialize_historical_newspapers_gold_subset_tracks_agreement(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path)
    corrections_path = tmp_path / "corrections.csv"
    corrections_path.write_text(
        "sequence_id,position,ocr_label,validated_label,reason\n"
        "test,0,gd,ga,unit_test\n",
        encoding="utf-8",
    )
    validated_manifest = tmp_path / "validated_manifest.yaml"
    materialize_historical_newspapers_validation_subset(
        source_manifest_path=manifest_path,
        corrections_csv_path=corrections_path,
        output_manifest_path=validated_manifest,
        audit_csv_path=tmp_path / "audit.csv",
        note_path=tmp_path / "note.md",
    )
    gold_annotations = tmp_path / "gold_annotations.csv"
    export_historical_newspapers_gold_annotations(
        source_manifest_path=validated_manifest,
        output_csv_path=gold_annotations,
    )
    gold_manifest = tmp_path / "gold_manifest.yaml"
    result = materialize_historical_newspapers_gold_subset(
        source_manifest_path=validated_manifest,
        gold_annotations_csv_path=gold_annotations,
        output_manifest_path=gold_manifest,
        agreement_summary_md_path=tmp_path / "agreement.md",
        note_path=tmp_path / "gold_note.md",
    )

    payload = yaml.safe_load(gold_manifest.read_text(encoding="utf-8"))
    corrected_record = next(
        record
        for record in payload["records"]
        if record["sequence_id"] == "test" and record["position"] == 0
    )

    assert corrected_record["transcription"] == "ga"
    assert corrected_record["metadata"]["gold_pass_agreement"] is True
    assert result["agreement_summary"]["pass_agreement_rate"] == 1.0
