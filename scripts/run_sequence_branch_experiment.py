from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.sequence.runner import run_sequence_branch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequence-branch structured uncertainty experiment.")
    parser.add_argument("--config", required=True, help="Path to a validated DecipherLab YAML config.")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["cluster_distance", "calibrated_classifier"],
        help="Optional posterior strategy override for the structured decoder branch.",
    )
    args = parser.parse_args()
    result = run_sequence_branch_experiment(args.config, strategy_override=args.strategy)
    print(json.dumps({"run_dir": str(result["run_dir"]), "summary_count": len(result["summary_rows"])}, indent=2))


if __name__ == "__main__":
    main()
