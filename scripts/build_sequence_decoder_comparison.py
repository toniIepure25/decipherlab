from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.sequence.cross_dataset import build_sequence_decoder_comparison_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build decoder-comparison outputs from completed sequence task packs.")
    parser.add_argument("--markov-effect-rows", default="outputs/sequence_cross_dataset_effect_rows.csv")
    parser.add_argument("--markov-failure-summary", default="outputs/sequence_cross_dataset_failure_summary.csv")
    parser.add_argument("--process-effect-rows", default="outputs/sequence_process_family_cross_dataset_effect_rows.csv")
    parser.add_argument("--process-failure-summary", default="outputs/sequence_process_family_cross_dataset_failure_summary.csv")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    outputs = build_sequence_decoder_comparison_outputs(
        [
            {
                "task_label": "real_glyph_markov_sequences",
                "effect_rows_csv": Path(args.markov_effect_rows),
                "failure_csv": Path(args.markov_failure_summary),
            },
            {
                "task_label": "real_glyph_process_family_sequences",
                "effect_rows_csv": Path(args.process_effect_rows),
                "failure_csv": Path(args.process_failure_summary),
            },
        ],
        output_root=args.output_root,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
