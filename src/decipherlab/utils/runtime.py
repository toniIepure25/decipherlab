from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from decipherlab.config import DecipherLabConfig
from decipherlab.utils.io import ensure_directory, stable_hash, write_json


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    config_hash: str
    timestamp: str
    code_version: str


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not_a_git_repository"
    return result.stdout.strip()


def prepare_run_context(
    config: DecipherLabConfig,
    suffix: str | None = None,
) -> RunContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = config.experiment.name if suffix is None else f"{config.experiment.name}_{suffix}"
    run_dir = ensure_directory(config.experiment.output_root / f"{timestamp}_{name}")
    config_hash = stable_hash(config.model_dump(mode="json"))
    context = RunContext(
        run_dir=run_dir,
        config_hash=config_hash,
        timestamp=timestamp,
        code_version=_git_revision(),
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "experiment_name": config.experiment.name,
            "timestamp": timestamp,
            "seed": config.experiment.seed,
            "config_hash": config_hash,
            "output_dir": str(run_dir),
            "code_version": context.code_version,
            "notes": config.experiment.notes,
        },
    )
    return context
