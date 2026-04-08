from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.evaluation.cross_dataset import build_cross_dataset_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-dataset summary artifacts from completed DecipherLab runs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional repeated dataset specification in the form label=run_dir. When provided, overrides the legacy flags.",
    )
    parser.add_argument(
        "--omniglot-run",
        type=Path,
        default=Path("outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation"),
    )
    parser.add_argument(
        "--secondary-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--secondary-label",
        type=str,
        default="sklearn_digits_crops",
    )
    parser.add_argument(
        "--tertiary-run",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tertiary-label",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    args = parser.parse_args()

    if args.dataset:
        datasets: list[dict[str, Path | str]] = []
        for item in args.dataset:
            if "=" not in item:
                raise ValueError("--dataset entries must use label=run_dir format.")
            label, run_dir = item.split("=", 1)
            datasets.append({"dataset_label": label, "run_dir": Path(run_dir)})
    else:
        datasets = [
            {"dataset_label": "omniglot_character_crops", "run_dir": args.omniglot_run},
            {"dataset_label": args.secondary_label, "run_dir": args.secondary_run},
        ]
        if args.tertiary_run is not None:
            datasets.append(
                {
                    "dataset_label": args.tertiary_label or args.tertiary_run.name,
                    "run_dir": args.tertiary_run,
                }
            )

    outputs = build_cross_dataset_outputs(
        datasets=datasets,
        output_root=args.output_root,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
