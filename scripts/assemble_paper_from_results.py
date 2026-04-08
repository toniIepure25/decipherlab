from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.manuscript import assemble_paper_sections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble manuscript-ready drafts from a DecipherLab results pack.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Results-pack directory produced by the evaluation runner.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Directory where manuscript drafts should be written.",
    )
    args = parser.parse_args()

    outputs = assemble_paper_sections(run_dir=args.run_dir, paper_dir=args.paper_dir)
    print(f"Experiments draft: {outputs['experiments']}")
    print(f"Results draft: {outputs['results']}")
    print(f"Limitations draft: {outputs['limitations']}")


if __name__ == "__main__":
    main()
