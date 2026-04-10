from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.sequence.sensitivity import (
    build_process_family_sensitivity_outputs,
    discover_latest_process_family_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build process-family sensitivity outputs from completed sequence-branch runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Completed process-family run directory. Can be supplied multiple times.",
    )
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dir] if args.run_dir else discover_latest_process_family_runs()
    outputs = build_process_family_sensitivity_outputs(run_dirs, output_root=args.output_root)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
