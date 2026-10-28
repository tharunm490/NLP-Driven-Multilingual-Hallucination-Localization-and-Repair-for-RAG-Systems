"""Multilingual natural-language-inference inference and evidence windowing."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import httpx

from app.core.config import Settings
from app.core.errors import NliError, NliInputTooLong
from app.schemas.verification import NliLabel, TextSpan
from app.services.verification.runtime import resolve_torch_device
from app.services.verification.text import split_sentences, word_spans

logger = logging.getLogger(__name__)

_LABEL_ALIASES: dict[str, NliLabel] = {
    "entail": "entailment",
    "entailed": "entailment",
    "entailment": "entailment",
    "support": "entailment",
    "supported": "entailment",
    "contradict": "contradiction",
    "contradicted": "contradiction",
    "contradiction": "contradiction",
    "neutral": "neutral",
    "neither": "neutral",
    "unknown": "neutral",
}


@dataclass(frozen=True)
class NliPrediction:
    label: NliLabel
    confidence: float
    probabilities: dict[str, float]
    logits: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PremiseWindow:
    span: TextSpan
    token_count: int


@runtime_checkable
class NliModel(Protocol):
    model_name: str
    max_pair_tokens: int

    def count_pair_tokens(self, premise: str, hypothesis: str) -> int: ...

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]: ...


class TransformerNliModel:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_tokens: int = 512,
        max_batch_tokens: int = 8192,
        cache_size: int = 256,
        num_threads: int = 0,
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self._batch_size = max(1, batch_size)
        self._max_batch_tokens = max(1, max_batch_tokens)
        self._num_threads = max(0, num_threads)
        self._configured_max_tokens = max(64, max_tokens)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._id2label: dict[int, NliLabel] = {}
        self._max_pair_tokens = self._configured_max_tokens
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._cache_size = max(0, cache_size)
        self._prediction_cache: OrderedDict[
            tuple[str, str], NliPrediction
        ] = OrderedDict()

    @property
    def max_pair_tokens(self) -> int:
        return self._max_pair_tokens

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load_model()

    def _load_model(self) -> None:
        load_started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise NliError("transformers/torch are required for NLI verification.") from exc

        device = resolve_torch_device(self._requested_device)
        if self._num_threads > 0:
            torch.set_num_threads(self._num_threads)
        logger.info(
            "Loading NLI model %s on %s (threads=%d, first load may download weights)",
            self.model_name,
            device,
            torch.get_num_threads(),
        )
        load_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, dtype=load_dtype
            )
            model.to(device)
            model.eval()
        except Exception as exc:
            if device != "cpu":
                logger.warning(
                    "Could not load NLI model %s on %s (%s); retrying on CPU",
                    self.model_name,
                    device,
                    exc,
                )
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name, dtype=torch.float32
                    )
                    device = "cpu"
                    model.to(device)
                    model.eval()
                except Exception as cpu_exc:
                    raise NliError(
                        f"Could not load NLI model '{self.model_name}'."
                    ) from cpu_exc
            else:
                raise NliError(
                    f"Could not load NLI model '{self.model_name}'."
                ) from exc

        raw_labels = getattr(getattr(model, "config", None), "id2label", {}) or {}
        label_map: dict[int, NliLabel] = {}
        for index, raw_label in raw_labels.items():
            canonical = _canonical_label(str(raw_label))
            if canonical is None:
                raise NliError(
                    f"The NLI model exposes an unusable label '{raw_label}'. "
                    "Expected entailment, neutral, and contradiction labels."
                )
            label_map[int(index)] = canonical
        if set(label_map.values()) != {"entailment", "neutral", "contradiction"}:
            raise NliError(
                "The NLI model does not expose entailment, neutral, and "
                "contradiction labels."
            )

        model_max = _model_max_tokens(getattr(model, "config", None), tokenizer)
        self._id2label = label_map
        self._tokenizer = tokenizer
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._model = model
        self._device = device
        self._max_pair_tokens = min(self._configured_max_tokens, model_max)
        logger.info(
            "NLI model ready: %s (device=%s, dtype=%s, labels=%s, "
            "max_pair_tokens=%s, load_ms=%.1f)",
            self.model_name,
            device,
            getattr(model, "dtype", "unknown"),
            {index: label for index, label in sorted(label_map.items())},
            self._max_pair_tokens,
            (time.perf_counter() - load_started) * 1000,
        )

    def count_pair_tokens(self, premise: str, hypothesis: str) -> int:
        self.load()
        assert self._tokenizer is not None
        return _count_pair_tokens(self._tokenizer, premise, hypothesis)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        if not pairs:
            return []
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        with self._inference_lock:
            started = time.perf_counter()
            unique_pairs = list(dict.fromkeys(pairs))
            predictions_by_pair: dict[tuple[str, str], NliPrediction] = {}
            pending: list[tuple[str, str]] = []
            cache_hits = 0
            for pair in unique_pairs:
                cached = self._cache_get(pair)
                if cached is None:
                    pending.append(pair)
                else:
                    predictions_by_pair[pair] = cached
                    cache_hits += 1

            batches = 0
            if pending:
                lengths: list[tuple[int, tuple[str, str]]] = []
                for pair in pending:
                    token_count = self.count_pair_tokens(*pair)
                    if token_count > self._max_pair_tokens:
                        raise NliInputTooLong(
                            f"A claim-evidence pair has {token_count} tokens, "
                            f"above the {self._max_pair_tokens}-token limit."
                        )
                    lengths.append((token_count, pair))
                ordered_batches = _length_aware_batches(
                    lengths,
                    self._batch_size,
                    self._max_batch_tokens,
                )
                batches = len(ordered_batches)
                for batch_index, batch in enumerate(ordered_batches, start=1):
                    batch_started = time.perf_counter()
                    batch_predictions = self._predict_batch(batch)
                    if len(batch_predictions) != len(batch):
                        raise NliError(
                            "The NLI model returned a different number of "
                            "predictions than claim-evidence pairs."
                        )
                    for pair, prediction in zip(batch, batch_predictions):
                        predictions_by_pair[pair] = prediction
                        self._cache_put(pair, prediction)
                    logger.info(
                        "NLI batch %d/%d: pairs=%d elapsed_ms=%.1f",
                        batch_index,
                        batches,
                        len(batch),
                        (time.perf_counter() - batch_started) * 1000,
                    )

            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "NLI inference: model=%s device=%s pairs=%d unique_pairs=%d "
                "batches=%d cache_hits=%d elapsed_ms=%.1f peak_cuda_mb=%.1f",
                self.model_name,
                self._device,
                len(pairs),
                len(unique_pairs),
                batches,
                cache_hits,
                elapsed_ms,
                self._peak_cuda_mb(),
            )
            return [predictions_by_pair[pair] for pair in pairs]

    def _predict_batch(
        self, batch: Sequence[tuple[str, str]]
    ) -> list[NliPrediction]:
        assert self._tokenizer is not None
        assert self._model is not None
        import torch

        premises = [premise for premise, _ in batch]
        hypotheses = [hypothesis for _, hypothesis in batch]
        encoded = self._tokenizer(
            premises,
            hypotheses,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        longest = int(max(lengths)) if lengths else 0
        if longest > self._max_pair_tokens:
            raise NliInputTooLong(
                f"A claim-evidence pair has {longest} tokens, above the "
                f"{self._max_pair_tokens}-token limit."
            )
        encoded = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        try:
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu()
                raw_logits = logits.detach().cpu().tolist()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise NliError(
                "NLI inference ran out of GPU memory. Reduce "
                "VERIFICATION_NLI_BATCH_SIZE or use VERIFICATION_NLI_DEVICE=cpu."
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).casefold():
                raise NliError(
                    "NLI inference ran out of memory. Reduce "
                    "VERIFICATION_NLI_BATCH_SIZE or "
                    "VERIFICATION_NLI_MAX_BATCH_TOKENS."
                ) from exc
            raise NliError("NLI inference failed.") from exc
        except Exception as exc:
            raise NliError("NLI inference failed.") from exc
        if (
            getattr(probabilities, "ndim", 0) != 2
            or int(probabilities.shape[0]) != len(batch)
        ):
            raise NliError(
                "The NLI model returned a malformed probability matrix."
            )
        return [
            _prediction_from_probabilities(
                row,
                self._id2label,
                logits=raw_logits[index],
            )
            for index, row in enumerate(probabilities.tolist())
        ]

    def _cache_get(
        self, pair: tuple[str, str]
    ) -> NliPrediction | None:
        if self._cache_size == 0:
            return None
        prediction = self._prediction_cache.get(pair)
        if prediction is None:
            return None
        self._prediction_cache.move_to_end(pair)
        return _copy_prediction(prediction)

    def _cache_put(
        self, pair: tuple[str, str], prediction: NliPrediction
    ) -> None:
        if self._cache_size == 0:
            return
        self._prediction_cache[pair] = _copy_prediction(prediction)
        self._prediction_cache.move_to_end(pair)
        while len(self._prediction_cache) > self._cache_size:
            self._prediction_cache.popitem(last=False)

    def _peak_cuda_mb(self) -> float:
        try:
            import torch

            if self._device and self._device.startswith("cuda"):
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:  # noqa: BLE001 - metrics must not break inference
            pass
        return 0.0


def _length_aware_batches(
    lengths: Sequence[tuple[int, tuple[str, str]]],
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[tuple[str, str]]]:
    """Group similar-length pairs without exceeding the size or token budget."""

    ordered = sorted(lengths, key=lambda item: (item[0], item[1]))
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[int, tuple[str, str]]] = []
    current_max = 0
    for token_count, pair in ordered:
        candidate_max = max(current_max, token_count)
        exceeds_size = len(current) >= batch_size
        exceeds_tokens = bool(current) and (
            candidate_max * (len(current) + 1) > max_batch_tokens
        )
        if exceeds_size or exceeds_tokens:
            batches.append([item[1] for item in current])
            current = []
            current_max = 0
        current.append((token_count, pair))
        current_max = max(current_max, token_count)
    if current:
        batches.append([item[1] for item in current])
    return batches


def _copy_prediction(prediction: NliPrediction) -> NliPrediction:
    return NliPrediction(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=dict(prediction.probabilities),
        logits=(
            tuple(prediction.logits)
            if prediction.logits is not None
            else None
        ),
    )


class HostedNliModel:
    """NLI client for a Hugging Face Inference Endpoint.

    The endpoint must use the Text Classification task so that it accepts
    ``{"text": premise, "text_pair": hypothesis}`` and returns the model's
    entailment/neutral/contradiction scores. The serverless zero-shot API is
    intentionally not used here: it treats candidate labels as hypotheses,
    which cannot preserve the claim-evidence direction required by this app.
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        api_key: str,
        batch_size: int = 1,
        max_concurrency: int = 2,
        max_tokens: int = 512,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        *,
        tokenizer_factory: Callable[[str], Any] | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model_name = f"hf-endpoint:{model_name}"
        self._tokenizer_model_name = model_name
        self._endpoint_url = endpoint_url.strip()
        self._api_key = api_key.strip()
        self._batch_size = max(1, batch_size)
        self._max_concurrency = max(1, max_concurrency)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._timeout = httpx.Timeout(timeout_seconds)
        self._configured_max_tokens = max(64, max_tokens)
        self._max_pair_tokens = self._configured_max_tokens
        self._tokenizer_factory = tokenizer_factory
        self._tokenizer: Any | None = None
        self._client = client
        self._owns_client = client is None
        self._client_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._sleep = sleep or time.sleep

    @property
    def max_pair_tokens(self) -> int:
        return self._max_pair_tokens

    @property
    def is_loaded(self) -> bool:
        return self._client is not None or self._tokenizer is not None

    def load(self) -> None:
        """Load only the tokenizer needed for local window sizing.

        The hosted model itself is never downloaded or instantiated.
        """

        self._ensure_tokenizer()

    def count_pair_tokens(self, premise: str, hypothesis: str) -> int:
        tokenizer = self._ensure_tokenizer()
        return _count_pair_tokens(tokenizer, premise, hypothesis)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        if not pairs:
            return []
        unique_pairs = list(dict.fromkeys(pairs))
        pair_index = {pair: index for index, pair in enumerate(unique_pairs)}
        client = self._ensure_client()
        batches = [
            unique_pairs[start : start + self._batch_size]
            for start in range(0, len(unique_pairs), self._batch_size)
        ]
        if self._max_concurrency <= 1 or len(batches) <= 1:
            unique_predictions = [
                prediction
                for batch in batches
                for prediction in self._predict_batch(client, batch)
            ]
        else:
            workers = min(self._max_concurrency, len(batches))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch_predictions = list(
                    executor.map(
                        lambda batch: self._predict_batch(client, batch),
                        batches,
                    )
                )
            unique_predictions = [
                prediction
                for batch in batch_predictions
                for prediction in batch
            ]
        return [unique_predictions[pair_index[pair]] for pair in pairs]

    def _predict_batch(
        self, client: httpx.Client, batch: Sequence[tuple[str, str]]
    ) -> list[NliPrediction]:
        payload = _remote_payload(batch)
        data = self._post_with_retries(client, payload)
        rows = _remote_prediction_rows(data, expected=len(batch))
        return [_prediction_from_remote_row(row) for row in rows]

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None and self._owns_client:
            client.close()

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._tokenizer_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            try:
                if self._tokenizer_factory is not None:
                    tokenizer = self._tokenizer_factory(
                        self._tokenizer_model_name
                    )
                else:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        self._tokenizer_model_name
                    )
            except Exception as exc:
                raise NliError(
                    "Could not load the NLI tokenizer for hosted inference."
                ) from exc
            self._tokenizer = tokenizer
            self._max_pair_tokens = min(
                self._configured_max_tokens,
                _tokenizer_max_tokens(tokenizer),
            )
            return tokenizer

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=self._timeout,
                    follow_redirects=True,
                )
            return self._client

    def _post_with_retries(
        self, client: httpx.Client, payload: dict[str, Any]
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                response = client.post(
                    self._endpoint_url,
                    headers=headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 >= attempts:
                    raise NliError(
                        "The hosted NLI endpoint could not be reached."
                    ) from exc
                self._sleep(self._retry_delay(attempt, None))
                continue

            status = response.status_code
            if 200 <= status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise NliError(
                        "The hosted NLI endpoint returned invalid JSON."
                    ) from exc

            if status == 429:
                if attempt + 1 >= attempts:
                    raise NliError(
                        "The hosted NLI endpoint is rate limited "
                        "(HTTP 429). Reduce concurrency or retry later."
                    )
                self._sleep(
                    self._retry_delay(
                        attempt, response.headers.get("Retry-After")
                    )
                )
                continue

            if status in {408, 425, 500, 502, 503, 504}:
                if attempt + 1 >= attempts:
                    raise NliError(
                        f"The hosted NLI endpoint is unavailable "
                        f"(HTTP {status})."
                    )
                self._sleep(
                    self._retry_delay(
                        attempt, response.headers.get("Retry-After")
                    )
                )
                continue

            if status == 402:
                raise NliError(
                    "Hosted NLI quota or credits are exhausted (HTTP 402). "
                    "Check the Hugging Face account billing settings."
                )
            if status in {401, 403}:
                raise NliError(
                    "Hosted NLI authentication failed. Check "
                    "VERIFICATION_NLI_API_KEY and endpoint permissions."
                )
            if status == 404:
                raise NliError(
                    "The hosted NLI endpoint or model was not found "
                    "(HTTP 404). Check VERIFICATION_NLI_ENDPOINT_URL."
                )
            if status in {400, 422}:
                raise NliError(
                    f"The hosted NLI endpoint rejected the request "
                    f"(HTTP {status}). Verify that it uses the Text "
                    "Classification task and the expected text-pair schema."
                )
            raise NliError(
                f"The hosted NLI endpoint rejected the request (HTTP {status})."
            )

        raise NliError("The hosted NLI request did not complete.")

    def _retry_delay(
        self, attempt: int, retry_after_header: str | None
    ) -> float:
        delay = min(
            self._retry_backoff_seconds * (2**attempt),
            8.0,
        )
        retry_after = _parse_retry_after(retry_after_header)
        if retry_after is not None:
            delay = max(delay, retry_after)
        delay = min(delay, 10.0)
        if delay <= 0.0:
            return 0.0
        return delay + random.uniform(0.0, min(0.25, delay * 0.5))


def build_nli_model(settings: Settings) -> NliModel:
    if settings.verification_nli_backend == "onnx":
        from app.services.verification.onnx import OnnxNliModel

        logger.info(
            "Using ONNX NLI model %s from %s",
            settings.verification_nli_model_name,
            settings.resolved_verification_nli_onnx_model_path,
        )
        return OnnxNliModel(
            model_name=settings.verification_nli_model_name,
            model_path=settings.resolved_verification_nli_onnx_model_path,
            provider=settings.verification_nli_onnx_provider,
            batch_size=settings.verification_nli_batch_size,
            max_tokens=settings.verification_nli_max_tokens,
            max_batch_tokens=settings.verification_nli_max_batch_tokens,
            cache_size=settings.verification_nli_cache_size,
            intra_op_threads=settings.verification_nli_onnx_intra_threads,
            inter_op_threads=settings.verification_nli_onnx_inter_threads,
        )
    if settings.verification_nli_backend == "hf_endpoint":
        logger.info(
            "Using hosted NLI endpoint for model %s",
            settings.verification_nli_model_name,
        )
        return HostedNliModel(
            model_name=settings.verification_nli_model_name,
            endpoint_url=settings.verification_nli_endpoint_url,
            api_key=settings.resolved_verification_nli_api_key,
            batch_size=settings.verification_nli_hosted_batch_size,
            max_concurrency=settings.verification_nli_max_concurrency,
            max_tokens=settings.verification_nli_max_tokens,
            timeout_seconds=settings.verification_nli_timeout_seconds,
            max_retries=settings.verification_nli_max_retries,
            retry_backoff_seconds=(
                settings.verification_nli_retry_backoff_seconds
            ),
        )
    return TransformerNliModel(
        model_name=settings.verification_nli_model_name,
        device=settings.verification_nli_device,
        batch_size=settings.verification_nli_batch_size,
        max_tokens=settings.verification_nli_max_tokens,
        max_batch_tokens=settings.verification_nli_max_batch_tokens,
        cache_size=settings.verification_nli_cache_size,
        num_threads=settings.verification_nli_num_threads,
    )


def build_premise_windows(
    text: str,
    hypothesis: str,
    model: NliModel,
    max_windows: int = 8,
) -> tuple[list[PremiseWindow], list[str]]:
    base = text.strip()
    if not base:
        return [], ["Evidence text was empty."]
    warnings: list[str] = []
    fitted = _fitted_sentences(text, hypothesis, model, warnings)

    windows: list[PremiseWindow] = []
    group_start: int | None = None
    group_end: int | None = None
    for span in fitted:
        candidate_start = group_start if group_start is not None else span.start
        candidate = text[candidate_start : span.end]
        if (
            group_start is not None
            and model.count_pair_tokens(candidate, hypothesis) > model.max_pair_tokens
        ):
            assert group_end is not None
            windows.append(
                _make_window(text, group_start, group_end, hypothesis, model)
            )
            group_start = span.start
        elif group_start is None:
            group_start = span.start
        group_end = span.end
    if group_start is not None and group_end is not None:
        windows.append(_make_window(text, group_start, group_end, hypothesis, model))

    if max_windows > 0 and len(windows) > max_windows:
        warnings.append(
            f"Evidence produced {len(windows)} NLI windows; only the first "
            f"{max_windows} were evaluated."
        )
        windows = windows[:max_windows]
    return windows, warnings


def build_context_windows(
    text: str,
    focus_start: int,
    focus_end: int,
    hypothesis: str,
    model: NliModel,
    max_tokens: int | None = None,
) -> tuple[list[PremiseWindow], list[str]]:
    """Build one NLI window around a matched sentence, adding adjacent context.

    The matched sentence alone often lacks the subject or qualifier needed for
    NLI. This keeps the focused sentence in place while expanding to complete,
    token-bounded neighbouring sentences. ``max_tokens`` is a deliberate budget
    for the first, focused pass; it is never used to cut a sentence mid-word.
    """

    if not text.strip():
        return [], ["Evidence text was empty."]
    warnings: list[str] = []
    token_budget = min(
        model.max_pair_tokens,
        max(32, max_tokens if max_tokens is not None else model.max_pair_tokens),
    )
    fitted = _fitted_sentences(
        text, hypothesis, model, warnings, max_tokens=token_budget
    )
    if not fitted:
        return [], warnings

    focus_index = next(
        (
            index
            for index, span in enumerate(fitted)
            if span.start < max(focus_end, focus_start + 1)
            and span.end > focus_start
        ),
        None,
    )
    if focus_index is None:
        focus_index = min(
            range(len(fitted)),
            key=lambda index: abs(fitted[index].start - focus_start),
        )

    left = right = focus_index
    while True:
        expanded = False
        if right + 1 < len(fitted):
            candidate = text[fitted[left].start : fitted[right + 1].end]
            if model.count_pair_tokens(candidate, hypothesis) <= token_budget:
                right += 1
                expanded = True
        if left > 0:
            candidate = text[fitted[left - 1].start : fitted[right].end]
            if model.count_pair_tokens(candidate, hypothesis) <= token_budget:
                left -= 1
                expanded = True
        if not expanded:
            break

    return [
        _make_window(
            text, fitted[left].start, fitted[right].end, hypothesis, model
        )
    ], warnings


def _fitted_sentences(
    text: str,
    hypothesis: str,
    model: NliModel,
    warnings: list[str],
    *,
    max_tokens: int | None = None,
) -> list[TextSpan]:
    token_budget = min(
        model.max_pair_tokens,
        max(32, max_tokens if max_tokens is not None else model.max_pair_tokens),
    )
    sentences = split_sentences(text, join_wrapped_lines=True) or [
        TextSpan(0, len(text), text)
    ]
    fitted: list[TextSpan] = []
    for sentence in sentences:
        if model.count_pair_tokens(sentence.text, hypothesis) <= token_budget:
            fitted.append(sentence)
            continue
        split = _split_long_span(
            sentence, hypothesis, model, max_tokens=token_budget
        )
        if not split:
            warnings.append(
                "A sentence could not be split into an NLI-sized evidence window."
            )
        fitted.extend(split)
    return fitted


def _remote_payload(
    batch: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    rows = [
        {"text": premise, "text_pair": hypothesis}
        for premise, hypothesis in batch
    ]
    inputs: Any = rows[0] if len(rows) == 1 else rows
    return {
        "inputs": inputs,
        "parameters": {
            "function_to_apply": "softmax",
            "top_k": 3,
        },
    }


def _remote_prediction_rows(data: Any, expected: int) -> list[list[Any]]:
    if isinstance(data, dict):
        raise NliError("The hosted NLI endpoint returned an error response.")
    if not isinstance(data, list) or not data:
        raise NliError("The hosted NLI endpoint returned a malformed response.")

    if expected == 1 and all(isinstance(item, dict) for item in data):
        rows: list[Any] = [data]
    elif expected == 1 and len(data) == 1 and isinstance(data[0], list):
        rows = [data[0]]
    else:
        rows = list(data)

    if len(rows) != expected:
        raise NliError(
            "The hosted NLI endpoint returned a different number of "
            "predictions than requested pairs."
        )
    if not all(
        isinstance(row, list) and all(isinstance(item, dict) for item in row)
        for row in rows
    ):
        raise NliError("The hosted NLI endpoint returned malformed labels.")
    return rows


def _prediction_from_remote_row(row: Sequence[Any]) -> NliPrediction:
    scores: dict[str, float] = {}
    for item in row:
        canonical = _canonical_label(str(item.get("label", "")))
        if canonical is None:
            raise NliError(
                "The hosted NLI endpoint returned an unusable label. "
                "Expected entailment, neutral, and contradiction."
            )
        if canonical in scores:
            raise NliError(
                "The hosted NLI endpoint returned duplicate labels."
            )
        try:
            value = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise NliError(
                "The hosted NLI endpoint returned malformed scores."
            ) from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise NliError(
                "The hosted NLI endpoint returned a score outside the "
                "valid range."
            )
        scores[canonical] = value

    if set(scores) != {"entailment", "neutral", "contradiction"}:
        raise NliError(
            "The hosted NLI endpoint did not return entailment, neutral, "
            "and contradiction scores."
        )
    total = sum(scores.values())
    if abs(total - 1.0) > 1e-3:
        raise NliError(
            "The hosted NLI endpoint returned scores that do not sum to one."
        )
    label = max(scores, key=scores.get)
    return NliPrediction(
        label=label,  # type: ignore[arg-type]
        confidence=scores[label],
        probabilities=scores,
    )


def _count_pair_tokens(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
) -> int:
    encoded = tokenizer(
        premise,
        hypothesis,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def _tokenizer_max_tokens(tokenizer: Any) -> int:
    try:
        value = int(getattr(tokenizer, "model_max_length", 0))
    except (TypeError, ValueError):
        return 512
    return value if 0 < value < 100_000 else 512


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else None


def _prediction_from_probabilities(
    values: Sequence[float],
    id2label: dict[int, NliLabel],
    *,
    logits: Sequence[float] | None = None,
) -> NliPrediction:
    if len(values) != len(id2label):
        raise NliError(
            "The NLI model returned a probability row with an unexpected size."
        )
    mapped: dict[str, float] = {}
    for index, label in id2label.items():
        try:
            value = float(values[index])
        except (IndexError, TypeError, ValueError) as exc:
            raise NliError(
                "The NLI model returned malformed probability values."
            ) from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise NliError(
                "The NLI model returned a probability outside the valid range."
            )
        mapped[label] = value
    if set(mapped) != {"entailment", "neutral", "contradiction"}:
        raise NliError(
            "The NLI model did not return entailment, neutral, and "
            "contradiction probabilities."
        )
    total = sum(mapped.values())
    if abs(total - 1.0) > 1e-3:
        raise NliError(
            "The NLI model returned probabilities that do not sum to one."
        )
    label = max(mapped, key=mapped.get)
    return NliPrediction(
        label=label,  # type: ignore[arg-type]
        confidence=mapped[label],
        probabilities=mapped,
        logits=tuple(float(value) for value in logits) if logits is not None else None,
    )


def _split_long_span(
    span: TextSpan,
    hypothesis: str,
    model: NliModel,
    *,
    max_tokens: int | None = None,
) -> list[TextSpan]:
    token_budget = min(
        model.max_pair_tokens,
        max(32, max_tokens if max_tokens is not None else model.max_pair_tokens),
    )
    words = word_spans(span.text)
    if not words:
        return []
    spans: list[TextSpan] = []
    group_start: int | None = None
    group_end: int | None = None
    for word in words:
        candidate_start = group_start if group_start is not None else word.start
        candidate = span.text[candidate_start : word.end]
        token_count = model.count_pair_tokens(candidate, hypothesis)
        if group_start is not None and token_count > token_budget:
            assert group_end is not None
            spans.append(
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
        segment = span.text[group_start:group_end]
        if model.count_pair_tokens(segment, hypothesis) <= token_budget:
            spans.append(
                TextSpan(
                    start=span.start + group_start,
                    end=span.start + group_end,
                    text=segment,
                )
            )
    return spans


def _make_window(
    text: str, start: int, end: int, hypothesis: str, model: NliModel
) -> PremiseWindow:
    value = text[start:end].strip()
    token_count = model.count_pair_tokens(value, hypothesis)
    return PremiseWindow(
        span=TextSpan(start=start, end=end, text=value),
        token_count=token_count,
    )


def _canonical_label(label: str) -> NliLabel | None:
    return _LABEL_ALIASES.get(label.strip().casefold())


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

