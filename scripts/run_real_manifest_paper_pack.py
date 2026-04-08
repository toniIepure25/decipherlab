from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.workflows import run_real_manifest_paper_pack, summarize_real_manifest_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DecipherLab paper pack on a manifest-backed real dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/real_manifest_large_paper_pack.yaml"),
        help="Path to a manifest-backed experiment config.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Directory where manuscript drafts should be refreshed.",
    )
    args = parser.parse_args()

    summary = summarize_real_manifest_dataset(args.config)
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Sequences: {summary['sequence_count']}")
    print(f"Split counts: {summary['split_sequence_counts']}")
    print(f"Label coverage: {summary['split_label_coverage']}")
    print(f"Group counts: {summary['split_group_counts']}")
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")

    result = run_real_manifest_paper_pack(args.config, paper_dir=args.paper_dir)
    print(f"Paper pack complete: {result['run_dir']}")
    print(f"Dataset summary: {Path(result['run_dir']) / 'dataset_summary.md'}")
    if result["paper_outputs"] is not None:
        print(f"Paper drafts refreshed under: {args.paper_dir}")


if __name__ == "__main__":
    main()
