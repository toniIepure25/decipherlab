from __future__ import annotations

import logging
from pathlib import Path

from decipherlab.utils.io import ensure_directory


def configure_logging(run_dir: str | Path) -> logging.Logger:
    directory = ensure_directory(run_dir)
    logger = logging.getLogger("decipherlab")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
