from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.sequence.cross_dataset import build_sequence_cross_dataset_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-dataset sequence summary outputs from completed run bundles.")
    parser.add_argument("--omniglot-cluster", required=True)
    parser.add_argument("--omniglot-calibrated", required=True)
    parser.add_argument("--digits-cluster", required=True)
    parser.add_argument("--digits-calibrated", required=True)
    parser.add_argument("--kuzushiji-cluster", required=True)
    parser.add_argument("--kuzushiji-calibrated", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    outputs = build_sequence_cross_dataset_outputs(
        [
            {
                "dataset_label": "omniglot_sequence_markov",
                "task_label": "real_glyph_markov_sequences",
                "cluster_distance_run": args.omniglot_cluster,
                "calibrated_classifier_run": args.omniglot_calibrated,
            },
            {
                "dataset_label": "sklearn_digits_sequence_markov",
                "task_label": "real_glyph_markov_sequences",
                "cluster_distance_run": args.digits_cluster,
                "calibrated_classifier_run": args.digits_calibrated,
            },
            {
                "dataset_label": "kuzushiji49_sequence_markov",
                "task_label": "real_glyph_markov_sequences",
                "cluster_distance_run": args.kuzushiji_cluster,
                "calibrated_classifier_run": args.kuzushiji_calibrated,
            },
        ],
        output_root=args.output_root,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
