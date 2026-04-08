from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.config import load_config
from decipherlab.evaluation.runner import run_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the focused DecipherLab uncertainty experiment on a manifest-backed dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/real_manifest_uncertainty.yaml"),
        help="Path to an experiment config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.dataset.source != "manifest":
        raise SystemExit("The core-claim experiment expects a manifest-backed dataset config.")
    if config.dataset.manifest_path is None or not config.dataset.manifest_path.exists():
        raise SystemExit(
            "The manifest-backed dataset path does not exist. Update dataset.manifest_path in the config first."
        )

    result = run_ablation_suite(config)
    run_dir = Path(result["run_dir"])
    print(f"Experiment completed: {run_dir}")
    print(f"Paired summary: {run_dir / 'pairwise_summary.json'}")
    print(f"Paper-style report: {run_dir / 'paper_experiment_summary.md'}")


if __name__ == "__main__":
    main()
