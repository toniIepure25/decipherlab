from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.historical_newspapers import (
    export_historical_newspapers_gold_annotations,
    materialize_historical_newspapers_gold_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gold-style historical-newspapers grouped subset.")
    parser.add_argument(
        "--source-manifest",
        default="data/processed/historical_newspapers_grouped_words/validation_subset_manifest.yaml",
    )
    parser.add_argument(
        "--gold-annotations",
        default="data/processed/historical_newspapers_grouped_words/gold_annotations.csv",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/processed/historical_newspapers_grouped_words/gold_subset_manifest.yaml",
    )
    parser.add_argument(
        "--agreement-summary",
        default="data/processed/historical_newspapers_grouped_words/gold_agreement_summary.md",
    )
    parser.add_argument(
        "--note-path",
        default="data/processed/historical_newspapers_grouped_words/gold_subset_README.md",
    )
    args = parser.parse_args()

    export_historical_newspapers_gold_annotations(
        source_manifest_path=args.source_manifest,
        output_csv_path=args.gold_annotations,
    )
    result = materialize_historical_newspapers_gold_subset(
        source_manifest_path=args.source_manifest,
        gold_annotations_csv_path=args.gold_annotations,
        output_manifest_path=args.output_manifest,
        agreement_summary_md_path=args.agreement_summary,
        note_path=args.note_path,
    )
    print(result)


if __name__ == "__main__":
    main()
