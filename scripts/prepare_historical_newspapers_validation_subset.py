from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.ingest.historical_newspapers import materialize_historical_newspapers_validation_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validated historical-newspapers grouped manifest.")
    parser.add_argument(
        "--source-manifest",
        default="data/processed/historical_newspapers_grouped_words/manifest.yaml",
    )
    parser.add_argument(
        "--corrections-csv",
        default="data/processed/historical_newspapers_grouped_words/validation_corrections.csv",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/processed/historical_newspapers_grouped_words/validation_subset_manifest.yaml",
    )
    parser.add_argument(
        "--audit-csv",
        default="data/processed/historical_newspapers_grouped_words/validation_subset_annotations.csv",
    )
    parser.add_argument(
        "--note-path",
        default="data/processed/historical_newspapers_grouped_words/validation_subset_README.md",
    )
    parser.add_argument(
        "--validation-split",
        default="test",
    )
    args = parser.parse_args()

    result = materialize_historical_newspapers_validation_subset(
        source_manifest_path=args.source_manifest,
        corrections_csv_path=args.corrections_csv,
        output_manifest_path=args.output_manifest,
        audit_csv_path=args.audit_csv,
        note_path=args.note_path,
        validation_split=args.validation_split,
    )
    print(result)


if __name__ == "__main__":
    main()
