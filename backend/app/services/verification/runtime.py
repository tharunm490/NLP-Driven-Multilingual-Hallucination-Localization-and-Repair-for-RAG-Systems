"""Shared lazy torch device selection for verification models."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_torch_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception as exc:
        logger.debug("Torch device probing failed: %s", exc)
    return "cpu"
