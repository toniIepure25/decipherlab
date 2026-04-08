from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.ingest.sklearn_digits import (
    build_sklearn_digits_manifest,
    summarize_sklearn_digits_local_artifacts,
    write_sklearn_digits_integration_note,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a manifest-backed scikit-learn digits crop dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/sklearn_digits"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/raw/sklearn_digits/manifest.yaml"),
    )
    parser.add_argument(
        "--note-path",
        type=Path,
        default=Path("data/raw/sklearn_digits/README.md"),
    )
    parser.add_argument("--train-count-per-class", type=int, default=100)
    parser.add_argument("--val-count-per-class", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=23)
    args = parser.parse_args()

    manifest = build_sklearn_digits_manifest(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        train_count_per_class=args.train_count_per_class,
        val_count_per_class=args.val_count_per_class,
        split_seed=args.split_seed,
    )
    write_sklearn_digits_integration_note(
        dataset_root=args.output_dir,
        note_path=args.note_path,
        manifest_path=args.manifest_path,
    )
    dataset = load_glyph_crop_manifest_dataset(args.manifest_path)
    storage_summary = summarize_sklearn_digits_local_artifacts(
        dataset_root=args.output_dir,
        manifest_path=args.manifest_path,
    )
    print(f"Manifest written to {args.manifest_path}")
    print(f"Integration note: {args.note_path}")
    print(f"Dataset: {manifest.dataset_name}")
    print(f"Examples: {dataset.count_examples()}")
    print(f"Splits: {dataset.metadata.get('split_counts')}")
    print(f"Label coverage: {dataset.metadata.get('label_coverage')}")
    print(f"Local footprint: {storage_summary['total_footprint']} bytes")


if __name__ == "__main__":
    main()
