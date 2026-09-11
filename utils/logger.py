"""Application logging. Technical errors go to a local file, not the UI."""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root = logging.getLogger("martial_arts_tracker")
    root.setLevel(numeric)
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    from config.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_path, settings.log_level)
    return logging.getLogger(f"martial_arts_tracker.{name}")
