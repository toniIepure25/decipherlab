from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.scadsai_handwriting import prepare_scadsai_grouped_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a grouped word-sequence manifest from the ScaDS.AI handwriting dataset.")
    parser.add_argument(
        "--archive-path",
        default="data/raw/scadsai_handwriting/scadsai_german_handwriting_line_word_level_v01.tar.gz",
        help="Path to the downloaded ScaDS.AI dataset archive.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/scadsai_grouped_words",
        help="Directory for the processed grouped manifest and copied word crops.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/processed/scadsai_grouped_words/manifest.yaml",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--note-path",
        default="data/processed/scadsai_grouped_words/README.md",
        help="Dataset integration note path.",
    )
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--train-pages", type=int, default=220)
    parser.add_argument("--val-pages", type=int, default=80)
    parser.add_argument("--selected-vocabulary-size", type=int, default=8)
    parser.add_argument("--min-token-count-per-split", type=int, default=8)
    parser.add_argument("--min-sequence-length", type=int, default=4)
    args = parser.parse_args()

    result = prepare_scadsai_grouped_manifest(
        archive_path=args.archive_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        note_path=args.note_path,
        split_seed=args.split_seed,
        train_pages=args.train_pages,
        val_pages=args.val_pages,
        selected_vocabulary_size=args.selected_vocabulary_size,
        min_token_count_per_split=args.min_token_count_per_split,
        min_sequence_length=args.min_sequence_length,
    )
    print(result)


if __name__ == "__main__":
    main()
