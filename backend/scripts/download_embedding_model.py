"""Pre-download and load the configured embedding model.

Useful before starting the backend so the first upload is not delayed by the
model download.

Usage:
    python scripts/download_embedding_model.py                 # uses EMBEDDING_MODEL_NAME
    python scripts/download_embedding_model.py BAAI/bge-m3     # explicit model
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
    )
    device = os.environ.get("EMBEDDING_DEVICE", "cpu")
    print(f"Loading embedding model '{model_name}' on '{device}' ...", flush=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    dimension = getter() if callable(getter) else "unknown"
    print(f"MODEL_READY model={model_name} dimension={dimension}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
