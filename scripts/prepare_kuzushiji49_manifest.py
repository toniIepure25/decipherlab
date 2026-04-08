from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.kuzushiji49 import (
    build_kuzushiji49_balanced_manifest,
    summarize_kuzushiji49_local_artifacts,
    write_kuzushiji49_integration_note,
)
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a manifest-backed Kuzushiji-49 crop dataset with a balanced evaluation subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/kuzushiji49"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/raw/kuzushiji49/manifest_balanced_49x300_75_75.yaml"),
    )
    parser.add_argument(
        "--note-path",
        type=Path,
        default=Path("data/raw/kuzushiji49/README.md"),
    )
    parser.add_argument("--train-count-per-class", type=int, default=300)
    parser.add_argument("--val-count-per-class", type=int, default=75)
    parser.add_argument("--test-count-per-class", type=int, default=75)
    parser.add_argument("--split-seed", type=int, default=23)
    args = parser.parse_args()

    manifest = build_kuzushiji49_balanced_manifest(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        train_count_per_class=args.train_count_per_class,
        val_count_per_class=args.val_count_per_class,
        test_count_per_class=args.test_count_per_class,
        split_seed=args.split_seed,
    )
    write_kuzushiji49_integration_note(
        dataset_root=args.output_dir,
        note_path=args.note_path,
        manifest_path=args.manifest_path,
    )
    dataset = load_glyph_crop_manifest_dataset(args.manifest_path)
    storage_summary = summarize_kuzushiji49_local_artifacts(
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
    print(f"OpenML cache size: {storage_summary['cache_size']} bytes")


if __name__ == "__main__":
    main()
