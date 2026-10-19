"""Lazy, reusable multilingual embedding generation.

One model is loaded per process. Model-specific details (E5 prefixes,
normalisation) are handled here so callers only deal with plain text.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.core.config import Settings
from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate passage and query embeddings with a lazily loaded model."""

    def __init__(
        self,
        settings: Settings,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None
        self._dimension: int | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    # -- model lifecycle ---------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def device(self) -> str | None:
        return self._device

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load_model()

    def _resolve_device(self) -> str:
        requested = self._settings.embedding_device
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001 - device detection must not crash
            pass
        return "cpu"

    def _load_model(self) -> None:
        device = self._resolve_device()

        if self._model_factory is not None:
            try:
                model = self._model_factory()
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"Could not initialise the embedding model: {exc}") from exc
            self._apply_model(model, device)
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install -r backend/requirements.txt"
            ) from exc

        logger.info(
            "Loading embedding model %s on %s (first load may download weights)",
            self.model_name,
            device,
        )
        try:
            model = SentenceTransformer(
                self.model_name,
                device=device,
                trust_remote_code=self._settings.embedding_trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            if device != "cpu":
                logger.warning(
                    "Could not load %s on %s (%s); retrying on CPU",
                    self.model_name,
                    device,
                    exc,
                )
                try:
                    model = SentenceTransformer(
                        self.model_name,
                        device="cpu",
                        trust_remote_code=self._settings.embedding_trust_remote_code,
                    )
                    device = "cpu"
                except Exception as cpu_exc:  # noqa: BLE001
                    logger.exception("Embedding model failed to load on CPU")
                    raise EmbeddingError(
                        f"Could not load embedding model '{self.model_name}'. "
                        "Check the model name, your internet connection for the first "
                        "download, and available memory."
                    ) from cpu_exc
            else:
                logger.exception("Embedding model failed to load")
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Check the model name, your internet connection for the first "
                    "download, and available memory."
                ) from exc

        self._apply_model(model, device)

    def _apply_model(self, model: Any, device: str) -> None:
        try:
            model.max_seq_length = self._settings.embedding_max_seq_length
        except Exception:  # noqa: BLE001 - not all backends expose this
            logger.debug("Embedding model does not accept max_seq_length")

        dimension = None
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if callable(getter):
            try:
                dimension = int(getter())
            except Exception:  # noqa: BLE001
                dimension = None

        if not dimension:
            raise EmbeddingError(
                "The embedding model did not report a usable vector dimension."
            )

        self._model = model
        self._dimension = dimension
        self._device = device
        logger.info(
            "Embedding model ready: %s (dimension=%s, device=%s)",
            self.model_name,
            dimension,
            device,
        )

    # -- inference ---------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise EmbeddingError("Cannot embed empty text.")
        prefix = self._settings.effective_passage_prefix
        inputs = [f"{prefix}{text}" for text in cleaned]
        return self._encode(inputs)

    def embed_query(self, query: str) -> list[float]:
        query = query.strip()
        if not query:
            raise EmbeddingError("Cannot embed an empty query.")
        prefix = self._settings.effective_query_prefix
        return self._encode([f"{prefix}{query}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        assert self._model is not None  # for type checkers

        batch_size = self._settings.embedding_batch_size
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                encoded = self._model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=self._settings.embedding_normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Embedding inference failed")
                raise EmbeddingError(
                    "Embedding generation failed. The model may be out of memory; "
                    "try a smaller EMBEDDING_BATCH_SIZE or EMBEDDING_DEVICE=cpu."
                ) from exc

            batch_vectors = self._to_python_vectors(encoded)
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    "Embedding model returned a different number of vectors than inputs."
                )
            for vector in batch_vectors:
                self._validate_vector(vector)
            vectors.extend(batch_vectors)

        return vectors

    def _to_python_vectors(self, encoded: Any) -> list[list[float]]:
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, list):
            raise EmbeddingError("Embedding model returned an unexpected output type.")
        if encoded and not isinstance(encoded[0], list):
            encoded = [encoded]
        try:
            return [[float(value) for value in vector] for vector in encoded]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding model returned non-numeric vectors.") from exc

    def _validate_vector(self, vector: list[float]) -> None:
        if self._dimension is not None and len(vector) != self._dimension:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"received {len(vector)}."
            )
        if not vector:
            raise EmbeddingError("Embedding model returned an empty vector.")
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding model returned non-finite values.")
        if all(value == 0.0 for value in vector):
            raise EmbeddingError("Embedding model returned an all-zero vector.")
