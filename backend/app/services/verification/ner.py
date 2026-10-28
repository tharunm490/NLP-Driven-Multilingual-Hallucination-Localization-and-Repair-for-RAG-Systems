"""Lazy multilingual named-entity recognition for extracted claims."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from app.core.config import Settings
from app.core.errors import NerError
from app.schemas.verification import NamedEntity, TextSpan
from app.services.verification.runtime import resolve_torch_device
from app.services.verification.text import split_sentences, word_spans

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NerResult:
    entities: tuple[NamedEntity, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class NerTagger(Protocol):
    model_name: str | None

    @property
    def is_loaded(self) -> bool: ...

    def tag(self, text: str) -> NerResult: ...


class NullNerTagger:
    model_name = None

    @property
    def is_loaded(self) -> bool:
        return True

    def tag(self, text: str) -> NerResult:
        return NerResult()


class TransformerNerTagger:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self._max_tokens = max(32, max_tokens)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._id2label: dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def tag(self, text: str) -> NerResult:
        value = text.strip()
        if not value:
            return NerResult()
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        spans = split_sentences(value)
        chunks = self._chunks(value, spans)
        entities: list[NamedEntity] = []
        warnings: list[str] = []
        for chunk in chunks:
            try:
                entities.extend(self._tag_chunk(value, chunk))
            except Exception as exc:
                raise NerError(f"NER inference failed: {exc}") from exc
        return NerResult(entities=tuple(entities), warnings=tuple(warnings))

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise NerError(
                "transformers/torch are required for named-entity recognition."
            ) from exc

        device = resolve_torch_device(self._requested_device)
        logger.info(
            "Loading NER model %s on %s (first load may download weights)",
            self.model_name,
            device,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            model.to(device)
            model.eval()
        except Exception as exc:
            if device != "cpu":
                logger.warning(
                    "Could not load NER model %s on %s (%s); retrying on CPU",
                    self.model_name,
                    device,
                    exc,
                )
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModelForTokenClassification.from_pretrained(
                        self.model_name
                    )
                    device = "cpu"
                    model.to(device)
                    model.eval()
                except Exception as cpu_exc:
                    raise NerError(
                        f"Could not load NER model '{self.model_name}'."
                    ) from cpu_exc
            else:
                raise NerError(
                    f"Could not load NER model '{self.model_name}'."
                ) from exc

        labels = getattr(getattr(model, "config", None), "id2label", {}) or {}
        self._id2label = {int(index): str(label) for index, label in labels.items()}
        if not self._id2label:
            raise NerError("The NER model did not provide an id-to-label mapping.")
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        model_max = _model_max_tokens(getattr(model, "config", None), tokenizer)
        self._max_tokens = min(self._max_tokens, model_max)
        logger.info(
            "NER model ready: %s (device=%s, max_tokens=%s)",
            self.model_name,
            device,
            self._max_tokens,
        )

    def _chunks(self, text: str, spans: Sequence[TextSpan]) -> list[TextSpan]:
        assert self._tokenizer is not None
        if not spans:
            spans = [TextSpan(start=0, end=len(text), text=text)]
        chunks: list[TextSpan] = []
        for span in spans:
            if self._token_count(span.text) <= self._max_tokens:
                chunks.append(span)
                continue
            chunks.extend(self._split_long_span(span))
        return chunks

    def _split_long_span(self, span: TextSpan) -> list[TextSpan]:
        words = word_spans(span.text)
        if not words:
            return []
        result: list[TextSpan] = []
        group_start: int | None = None
        group_end: int | None = None
        for word in words:
            candidate_start = group_start if group_start is not None else word.start
            candidate_text = span.text[candidate_start : word.end]
            if group_start is not None and self._token_count(candidate_text) > self._max_tokens:
                assert group_end is not None
                result.append(
                    TextSpan(
                        start=span.start + group_start,
                        end=span.start + group_end,
                        text=span.text[group_start:group_end],
                    )
                )
                group_start = word.start
            elif group_start is None:
                group_start = word.start
            group_end = word.end
        if group_start is not None and group_end is not None:
            result.append(
                TextSpan(
                    start=span.start + group_start,
                    end=span.start + group_end,
                    text=span.text[group_start:group_end],
                )
            )
        return result

    def _token_count(self, text: str) -> int:
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def _tag_chunk(self, full_text: str, chunk: TextSpan) -> list[NamedEntity]:
        assert self._tokenizer is not None
        assert self._model is not None
        import torch

        encoded = self._tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        token_count = len(encoded["input_ids"][0])
        if token_count > self._max_tokens:
            raise NerError(
                f"NER chunk has {token_count} tokens, above the {self._max_tokens}-token limit."
            )
        encoded = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).cpu()

        entities: list[NamedEntity] = []
        current_label: str | None = None
        current_start: int | None = None
        current_end: int | None = None
        current_scores: list[float] = []

        def flush() -> None:
            nonlocal current_label, current_start, current_end, current_scores
            if current_label is None or current_start is None or current_end is None:
                return
            local_start = current_start
            local_end = current_end
            while local_start < local_end and chunk.text[local_start].isspace():
                local_start += 1
            while local_end > local_start and chunk.text[local_end - 1].isspace():
                local_end -= 1
            if local_end > local_start:
                absolute_start = chunk.start + local_start
                absolute_end = chunk.start + local_end
                entities.append(
                    NamedEntity(
                        text=full_text[absolute_start:absolute_end],
                        label=current_label,
                        start=absolute_start,
                        end=absolute_end,
                        score=sum(current_scores) / len(current_scores),
                    )
                )
            current_label = None
            current_start = None
            current_end = None
            current_scores = []

        for index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            label_id = int(torch.argmax(logits[index]).item())
            score = float(probabilities[index, label_id].item())
            raw_label = self._id2label.get(label_id, f"LABEL_{label_id}")
            prefix, label = _split_entity_label(raw_label)
            if prefix == "O" or not label:
                flush()
                continue
            if prefix == "B" or current_label != label:
                flush()
                current_label = label
                current_start = int(start)
                current_end = int(end)
                current_scores = [score]
                continue
            current_end = int(end)
            current_scores.append(score)
        flush()
        return entities


def _split_entity_label(label: str) -> tuple[str, str | None]:
    value = label.strip()
    if value.upper() == "O":
        return "O", None
    if len(value) > 2 and value[1] == "-" and value[0].upper() in {"B", "I"}:
        return value[0].upper(), value[2:].upper()
    if value.upper().startswith("LABEL_"):
        return "O", None
    return "B", value.upper()


def _model_max_tokens(config: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    for value in (
        getattr(config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < number < 100_000:
            candidates.append(number)
    return min(candidates) if candidates else 512


def build_ner_tagger(settings: Settings) -> NerTagger:
    if not settings.verification_ner_enabled:
        return NullNerTagger()
    return TransformerNerTagger(
        model_name=settings.verification_ner_model_name,
        device=settings.verification_ner_device,
        max_tokens=settings.verification_ner_max_tokens,
    )
