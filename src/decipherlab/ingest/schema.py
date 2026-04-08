from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GlyphCropManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence_id: str
    position: int = Field(ge=0)
    image_path: str
    split: Literal["train", "val", "test"]
    example_id: str | None = None
    group_id: str | None = None
    family: str | None = None
    transcription: str | None = None
    plaintext_symbol: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "sequence_id",
        "image_path",
        "example_id",
        "group_id",
        "family",
        "transcription",
        "plaintext_symbol",
    )
    @classmethod
    def validate_non_blank_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("String fields must not be blank or whitespace-only.")
        return value


class GlyphCropManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    unit_type: Literal["glyph_crop"] = "glyph_crop"
    metadata: dict[str, Any] = Field(default_factory=dict)
    records: list[GlyphCropManifestRecord]

    @model_validator(mode="after")
    def validate_records(self) -> "GlyphCropManifest":
        seen_positions: set[tuple[str, int]] = set()
        sequence_splits: dict[str, str] = {}
        sequence_positions: dict[str, list[int]] = {}
        sequence_groups: dict[str, str | None] = {}
        for record in self.records:
            key = (record.sequence_id, record.position)
            if key in seen_positions:
                raise ValueError(f"Duplicate glyph position detected for {record.sequence_id}@{record.position}.")
            seen_positions.add(key)
            prior_split = sequence_splits.get(record.sequence_id)
            if prior_split is not None and prior_split != record.split:
                raise ValueError(f"Sequence {record.sequence_id} appears in multiple splits.")
            sequence_splits[record.sequence_id] = record.split
            sequence_positions.setdefault(record.sequence_id, []).append(record.position)
            prior_group = sequence_groups.get(record.sequence_id)
            if prior_group is not None and record.group_id is not None and prior_group != record.group_id:
                raise ValueError(f"Sequence {record.sequence_id} appears in multiple groups.")
            sequence_groups[record.sequence_id] = record.group_id if record.group_id is not None else prior_group
        available_splits = set(sequence_splits.values())
        if "train" not in available_splits:
            raise ValueError("Glyph-crop manifests must include a train split.")
        if len(available_splits) < 2:
            raise ValueError("Glyph-crop manifests must include at least two splits.")
        for sequence_id, positions in sequence_positions.items():
            expected = list(range(len(positions)))
            observed = sorted(positions)
            if observed != expected:
                raise ValueError(
                    f"Sequence {sequence_id} positions must be contiguous from 0..n-1; got {observed}."
                )
        return self
