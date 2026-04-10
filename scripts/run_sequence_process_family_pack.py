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


DATASET_CONFIGS = [
    ("omniglot_process_family", Path("configs/experiments/sequence_omniglot_process_family.yaml")),
    ("sklearn_digits_process_family", Path("configs/experiments/sequence_sklearn_digits_process_family.yaml")),
    ("kuzushiji49_process_family", Path("configs/experiments/sequence_kuzushiji_process_family.yaml")),
]


def main() -> None:
    dataset_runs: list[dict[str, str | Path]] = []
    for dataset_label, config_path in DATASET_CONFIGS:
        strategy_runs: dict[str, str] = {}
        for strategy in ["cluster_distance", "calibrated_classifier"]:
            result = run_sequence_branch_experiment(config_path, strategy_override=strategy)
            strategy_runs[strategy] = str(result["run_dir"])
        dataset_runs.append(
            {
                "dataset_label": dataset_label,
                "task_label": "real_glyph_process_family_sequences",
                "cluster_distance_run": strategy_runs["cluster_distance"],
                "calibrated_classifier_run": strategy_runs["calibrated_classifier"],
            }
        )
    outputs = build_sequence_cross_dataset_outputs(
        dataset_runs,
        output_root="outputs",
        output_prefix="sequence_process_family_cross_dataset",
        markdown_title="Sequence Process-Family Cross-Dataset Summary",
        alias_stem="sequence_process_family_cross_dataset",
    )
    print(json.dumps({"runs": dataset_runs, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
