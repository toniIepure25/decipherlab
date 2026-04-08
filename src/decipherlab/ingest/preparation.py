from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.utils.io import write_yaml


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _normalize_metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        return json.loads(value)
    raise ValueError("metadata must be a JSON object, dict, or blank value.")


def load_record_table(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return rows
    if suffix == ".json":
        return json.loads(input_path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            return payload["records"]
        return payload
    raise ValueError(f"Unsupported record-table format: {input_path.suffix}")


def _normalize_record(
    raw_record: dict[str, Any],
    output_path: Path,
    image_root: Path | None = None,
) -> dict[str, Any]:
    record = {key: _blank_to_none(value) for key, value in raw_record.items()}
    if "position" in record and record["position"] is not None:
        record["position"] = int(record["position"])
    image_path_value = record.get("image_path")
    if image_path_value is None:
        raise ValueError("Each record must contain image_path.")
    image_path = Path(str(image_path_value))
    if image_root is not None and not image_path.is_absolute():
        image_path = image_root / image_path
    if image_path.is_absolute():
        try:
            record["image_path"] = str(image_path.relative_to(output_path.parent))
        except ValueError:
            record["image_path"] = str(image_path)
    else:
        record["image_path"] = str(image_path)
    record["metadata"] = _normalize_metadata(record.get("metadata"))
    return record


def build_glyph_crop_manifest_from_table(
    records_path: str | Path,
    output_path: str | Path,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
    image_root: str | Path | None = None,
) -> GlyphCropManifest:
    output_file = Path(output_path)
    raw_records = load_record_table(records_path)
    normalized_records = [
        _normalize_record(
            raw_record,
            output_path=output_file,
            image_root=None if image_root is None else Path(image_root),
        )
        for raw_record in raw_records
    ]
    manifest = GlyphCropManifest.model_validate(
        {
            "dataset_name": dataset_name,
            "unit_type": "glyph_crop",
            "metadata": metadata or {},
            "records": normalized_records,
        }
    )
    write_yaml(output_file, manifest.model_dump(mode="json"))
    return manifest
