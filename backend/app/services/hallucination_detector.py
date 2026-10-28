"""Local binary groundedness detector used by Hallucination Checks.

The checkpoint is a text-classification model that receives retrieved evidence
first and a claim/response second. It is loaded lazily and reused across
requests. The implementation deliberately keeps NLI verification separate.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from app.core.config import Settings
from app.core.errors import HallucinationCheckError, HallucinationModelError
from app.services.verification.runtime import resolve_torch_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HallucinationPrediction:
    raw_label: str
    support_probability: float


class HallucinationDetector(Protocol):
    model_name: str

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[HallucinationPrediction]: ...


class TransformerHallucinationDetector:
    def __init__(
        self,
        settings: Settings,
        *,
        tokenizer_factory: Callable[[str], Any] | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = settings.hallucination_model_name
        self._settings = settings
        self._device = resolve_torch_device(settings.hallucination_model_device)
        self._batch_size = max(1, settings.hallucination_model_batch_size)
        self._max_tokens = max(32, settings.hallucination_model_max_tokens)
        self._local_files_only = settings.hallucination_model_local_files_only
        self._configured_support_index = (
            settings.hallucination_supported_label_index
        )
        self._tokenizer_factory = tokenizer_factory
        self._model_factory = model_factory
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._support_index: int | None = None
        self._support_label = "SUPPORTED"
        self._unsupported_label = "UNSUPPORTED"
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        self._ensure_loaded()

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[HallucinationPrediction]:
        if not pairs:
            return []
        tokenizer, model = self._ensure_loaded()
        unique_pairs = list(dict.fromkeys(pairs))
        predictions: list[HallucinationPrediction] = []
        for start in range(0, len(unique_pairs), self._batch_size):
            if cancel_event is not None and cancel_event.is_set():
                raise HallucinationCheckError("The hallucination check was cancelled.")
            if deadline is not None and time.monotonic() >= deadline:
                raise HallucinationCheckError(
                    "The hallucination check exceeded its time limit."
                )
            batch = unique_pairs[start : start + self._batch_size]
            predictions.extend(
                self._predict_batch(tokenizer, model, batch)
            )
        by_pair = dict(zip(unique_pairs, predictions))
        return [by_pair[pair] for pair in pairs]

    def _predict_batch(
        self,
        tokenizer: Any,
        model: Any,
        batch: Sequence[tuple[str, str]],
    ) -> list[HallucinationPrediction]:
        sources = [pair[0] for pair in batch]
        claims = [pair[1] for pair in batch]
        try:
            import torch

            encoded = tokenizer(
                sources,
                claims,
                padding=True,
                truncation="only_first",
                max_length=self._max_tokens,
                return_tensors="pt",
            )
            if hasattr(encoded, "to"):
                encoded = encoded.to(self._device)
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        except Exception as exc:  # noqa: BLE001 - convert model failures
            raise HallucinationModelError(
                "The local hallucination model could not score the extracted claims."
            ) from exc

        support_index = self._support_index
        if support_index is None:
            raise HallucinationModelError(
                "The hallucination model's supported label is not configured."
            )
        rows: list[HallucinationPrediction] = []
        for row in probabilities:
            if int(row.numel()) <= support_index:
                raise HallucinationModelError(
                    "The hallucination model returned an unexpected output shape."
                )
            probability = float(row[support_index].item())
            rows.append(
                HallucinationPrediction(
                    raw_label=(
                        self._support_label
                        if probability >= 0.5
                        else self._unsupported_label
                    ),
                    support_probability=max(0.0, min(1.0, probability)),
                )
            )
        return rows

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._load_lock:
            if self._tokenizer is None or self._model is None:
                self._load_models()
        assert self._tokenizer is not None and self._model is not None
        return self._tokenizer, self._model

    def _load_models(self) -> None:
        try:
            if self._tokenizer_factory is not None:
                tokenizer = self._tokenizer_factory(self.model_name)
            else:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=self._local_files_only,
                )
            if self._model_factory is not None:
                model = self._model_factory(self.model_name)
            else:
                from transformers import AutoModelForSequenceClassification

                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    local_files_only=self._local_files_only,
                )
            if hasattr(model, "to"):
                model = model.to(self._device)
            if hasattr(model, "eval"):
                model.eval()

            self._support_index, self._support_label, self._unsupported_label = (
                self._resolve_labels(getattr(model, "config", None))
            )
            tokenizer_limit = getattr(tokenizer, "model_max_length", None)
            if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
                self._max_tokens = min(self._max_tokens, tokenizer_limit)
            self._tokenizer = tokenizer
            self._model = model
        except HallucinationModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - include actionable context
            raise HallucinationModelError(
                "Could not load the local hallucination model "
                f"'{self.model_name}'. Pre-download it or adjust "
                "HALLUCINATION_MODEL_NAME."
            ) from exc

    def _resolve_labels(self, config: Any) -> tuple[int, str, str]:
        id2label = getattr(config, "id2label", None) or {}
        labels = {
            int(index): str(label)
            for index, label in dict(id2label).items()
        }
        if self._configured_support_index is not None:
            support_index = self._configured_support_index
            support_label = labels.get(support_index, "SUPPORTED")
        else:
            support_index = next(
                (
                    index
                    for index, label in sorted(labels.items())
                    if "SUPPORT" in label.upper()
                    and "UNSUPPORT" not in label.upper()
                ),
                None,
            )
            if support_index is None:
                raise HallucinationModelError(
                    "The hallucination model does not expose a SUPPORTED label. "
                    "Set HALLUCINATION_SUPPORTED_LABEL_INDEX explicitly."
                )
            support_label = labels[support_index]

        unsupported_index = next(
            (index for index in sorted(labels) if index != support_index),
            None,
        )
        unsupported_label = (
            labels[unsupported_index]
            if unsupported_index is not None
            else "UNSUPPORTED"
        )
        return support_index, support_label, unsupported_label


def build_hallucination_detector(settings: Settings) -> HallucinationDetector:
    return TransformerHallucinationDetector(settings)
