from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.omniglot import build_omniglot_manifest, extract_omniglot_archives
from decipherlab.ingest.omniglot import summarize_omniglot_local_artifacts, write_omniglot_integration_note
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a manifest-backed Omniglot crop dataset.")
    parser.add_argument(
        "--background-zip",
        type=Path,
        default=Path("data/raw/omniglot/images_background.zip"),
    )
    parser.add_argument(
        "--evaluation-zip",
        type=Path,
        default=Path("data/raw/omniglot/images_evaluation.zip"),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/raw/omniglot/extracted"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/raw/omniglot/manifest.yaml"),
    )
    parser.add_argument(
        "--note-path",
        type=Path,
        default=Path("data/raw/omniglot/README.md"),
    )
    parser.add_argument("--split-seed", type=int, default=23)
    args = parser.parse_args()

    roots = extract_omniglot_archives(
        archives=[args.background_zip, args.evaluation_zip],
        output_dir=args.extract_dir,
    )
    manifest = build_omniglot_manifest(
        image_roots=roots,
        output_path=args.manifest_path,
        split_seed=args.split_seed,
    )
    write_omniglot_integration_note(
        dataset_root=args.manifest_path.parent,
        note_path=args.note_path,
        manifest_path=args.manifest_path,
    )
    dataset = load_glyph_crop_manifest_dataset(args.manifest_path)
    storage_summary = summarize_omniglot_local_artifacts(
        dataset_root=args.manifest_path.parent,
        manifest_path=args.manifest_path,
    )
    print(f"Manifest written to {args.manifest_path}")
    print(f"Integration note: {args.note_path}")
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Examples: {dataset.count_examples()}")
    print(f"Splits: {dataset.metadata.get('split_counts')}")
    print(f"Label coverage: {dataset.metadata.get('label_coverage')}")
    print(f"Local footprint: {storage_summary['total_footprint']} bytes")


if __name__ == "__main__":
    main()
