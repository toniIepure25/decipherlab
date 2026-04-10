from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.historical_newspapers import prepare_historical_newspapers_grouped_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a real grouped-token manifest from the Historical Newspapers Ground Truth corpus."
    )
    parser.add_argument("--image-zip", required=True)
    parser.add_argument("--ocr-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--note-path", required=True)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--train-pages", type=int, default=30)
    parser.add_argument("--val-pages", type=int, default=10)
    parser.add_argument("--selected-vocabulary-size", type=int, default=8)
    parser.add_argument("--min-token-count-per-split", type=int, default=10)
    parser.add_argument("--min-sequence-length", type=int, default=4)
    parser.add_argument("--min-word-confidence", type=float, default=0.3)
    parser.add_argument("--crop-padding", type=int, default=2)
    args = parser.parse_args()

    result = prepare_historical_newspapers_grouped_manifest(
        image_zip_path=args.image_zip,
        ocr_zip_path=args.ocr_zip,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        note_path=args.note_path,
        split_seed=args.split_seed,
        train_pages=args.train_pages,
        val_pages=args.val_pages,
        selected_vocabulary_size=args.selected_vocabulary_size,
        min_token_count_per_split=args.min_token_count_per_split,
        min_sequence_length=args.min_sequence_length,
        min_word_confidence=args.min_word_confidence,
        crop_padding=args.crop_padding,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
