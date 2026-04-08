from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.utils.io import write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an alphabet-balanced Omniglot subset manifest for faster external validation runs.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/raw/omniglot/manifest.yaml"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/raw/omniglot/manifest_balanced_50x10.yaml"),
    )
    parser.add_argument(
        "--classes-per-alphabet",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=23,
    )
    args = parser.parse_args()

    payload = yaml.safe_load(args.source_manifest.read_text(encoding="utf-8"))
    manifest = GlyphCropManifest.model_validate(payload)
    by_alphabet: dict[str, set[str]] = defaultdict(set)
    records_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)

    for record in manifest.records:
        label = record.transcription
        if label is None or record.group_id is None:
            continue
        by_alphabet[record.group_id].add(label)
        records_by_label[label].append(record.model_dump(mode="json"))

    rng = random.Random(args.selection_seed)
    selected_labels: set[str] = set()
    for alphabet, labels in sorted(by_alphabet.items()):
        label_list = sorted(labels)
        if len(label_list) < args.classes_per_alphabet:
            raise ValueError(
                f"Alphabet {alphabet} has only {len(label_list)} classes; requested {args.classes_per_alphabet}."
            )
        chosen = rng.sample(label_list, args.classes_per_alphabet)
        selected_labels.update(chosen)

    subset_records = []
    for label in sorted(selected_labels):
        subset_records.extend(records_by_label[label])

    subset_manifest = {
        "dataset_name": f"{manifest.dataset_name}_balanced_50x{args.classes_per_alphabet}",
        "unit_type": manifest.unit_type,
        "metadata": manifest.metadata
        | {
            "subset_strategy": "alphabet_balanced_class_cap",
            "classes_per_alphabet": args.classes_per_alphabet,
            "selection_seed": args.selection_seed,
            "source_manifest": str(args.source_manifest),
        },
        "records": subset_records,
    }
    validated = GlyphCropManifest.model_validate(subset_manifest)
    write_yaml(args.output_manifest, validated.model_dump(mode="json"))
    print(f"Subset manifest written to {args.output_manifest}")
    print(f"Records: {len(validated.records)}")
    print(f"Dataset: {validated.dataset_name}")


if __name__ == "__main__":
    main()
