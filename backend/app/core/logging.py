"""Logging configuration."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)

    # Third-party libraries are noisy at INFO level.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
