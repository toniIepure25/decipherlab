from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.sequence.cross_dataset import build_sequence_cross_dataset_outputs
from decipherlab.sequence.runner import run_sequence_branch_experiment


CONFIG_PATH = Path("configs/experiments/sequence_historical_newspapers_real_grouped.yaml")


def main() -> None:
    strategy_runs: dict[str, str] = {}
    for strategy in ["cluster_distance", "calibrated_classifier"]:
        result = run_sequence_branch_experiment(CONFIG_PATH, strategy_override=strategy)
        strategy_runs[strategy] = str(result["run_dir"])
    outputs = build_sequence_cross_dataset_outputs(
        [
            {
                "dataset_label": "historical_newspapers_real_grouped",
                "task_label": "real_grouped_manifest_sequences",
                "cluster_distance_run": strategy_runs["cluster_distance"],
                "calibrated_classifier_run": strategy_runs["calibrated_classifier"],
            }
        ],
        output_root="outputs/real_grouped_historical_newspapers",
    )
    print(json.dumps({"runs": strategy_runs, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
