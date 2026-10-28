"""ONNX Runtime inference for the local multilingual NLI model.

The ONNX path is an optional, local-only alternative to the PyTorch path. It
uses the same tokenizer, label mapping, batching logic, cache, and input-length
checks so that switching backends does not silently change the verification
contract.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

from app.core.errors import NliError, NliInputTooLong
from app.schemas.verification import NliLabel
from app.services.verification.nli import (
    NliPrediction,
    _canonical_label,
    _copy_prediction,
    _length_aware_batches,
    _prediction_from_probabilities,
)

logger = logging.getLogger(__name__)

_SUPPORTED_INPUTS = frozenset(
    {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
)


class OnnxNliModel:
    """Local ONNX NLI model with the same public contract as the PyTorch model."""

    def __init__(
        self,
        model_name: str,
        model_path: str | Path,
        provider: str = "CPUExecutionProvider",
        batch_size: int = 8,
        max_tokens: int = 512,
        max_batch_tokens: int = 8192,
        cache_size: int = 256,
        intra_op_threads: int = 0,
        inter_op_threads: int = 1,
        *,
        tokenizer_factory: Callable[[str], Any] | None = None,
        session_factory: Callable[[str], Any] | None = None,
        label_map: dict[int, NliLabel] | None = None,
        configured_max_tokens: int | None = None,
    ) -> None:
        self.model_name = f"onnx:{model_name}"
        self._tokenizer_name = model_name
        self._model_path = Path(model_path)
        self._provider = provider
        self._batch_size = max(1, batch_size)
        self._max_batch_tokens = max(1, max_batch_tokens)
        self._cache_size = max(0, cache_size)
        self._intra_op_threads = max(0, intra_op_threads)
        self._inter_op_threads = max(0, inter_op_threads)
        self._tokenizer_factory = tokenizer_factory
        self._session_factory = session_factory
        self._configured_label_map = dict(label_map or {})
        self._configured_max_tokens = max(
            8, configured_max_tokens or max_tokens
        )
        self._max_pair_tokens = self._configured_max_tokens
        self._tokenizer: Any | None = None
        self._session: Any | None = None
        self._input_names: frozenset[str] = frozenset()
        self._output_name = "logits"
        self._id2label: dict[int, NliLabel] = {}
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._prediction_cache: OrderedDict[
            tuple[str, str], NliPrediction
        ] = OrderedDict()

    @property
    def max_pair_tokens(self) -> int:
        return self._max_pair_tokens

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            self._load()

    def _load(self) -> None:
        started = time.perf_counter()
        if not self._model_path.exists():
            raise NliError(
                f"ONNX NLI model not found at '{self._model_path}'. "
                "Run scripts/export_nli_onnx.py first or set "
                "VERIFICATION_NLI_ONNX_MODEL_PATH."
            )

        tokenizer = self._load_tokenizer()
        manifest = self._load_manifest()
        id2label = self._resolve_label_map(manifest)
        session = self._create_session()
        input_names = frozenset(
            item.name for item in session.get_inputs() if item.name in _SUPPORTED_INPUTS
        )
        if "input_ids" not in input_names or "attention_mask" not in input_names:
            raise NliError(
                "The ONNX NLI model must expose input_ids and attention_mask."
            )
        outputs = session.get_outputs()
        if not outputs:
            raise NliError("The ONNX NLI model has no outputs.")
        output_name = next(
            (item.name for item in outputs if item.name == "logits"),
            outputs[0].name,
        )
        max_tokens = self._resolve_max_tokens(manifest, tokenizer)
        self._tokenizer = tokenizer
        self._session = session
        self._input_names = input_names
        self._output_name = output_name
        self._id2label = id2label
        self._max_pair_tokens = max_tokens
        logger.info(
            "ONNX NLI model ready: %s provider=%s max_pair_tokens=%s "
            "inputs=%s load_ms=%.1f",
            self._model_path,
            self._provider,
            self._max_pair_tokens,
            sorted(input_names),
            (time.perf_counter() - started) * 1000,
        )

    def _load_tokenizer(self) -> Any:
        if self._tokenizer_factory is not None:
            tokenizer = self._tokenizer_factory(self._tokenizer_name)
        else:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            except Exception as exc:
                raise NliError(
                    "Could not load the tokenizer for the ONNX NLI model."
                ) from exc
        if tokenizer is None:
            raise NliError("The ONNX NLI tokenizer is unavailable.")
        return tokenizer

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self._model_path.with_name("manifest.json")
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NliError(
                f"Could not read the ONNX NLI manifest at '{manifest_path}'."
            ) from exc
        if not isinstance(data, dict):
            raise NliError("The ONNX NLI manifest is malformed.")
        return data

    def _resolve_label_map(self, manifest: dict[str, Any]) -> dict[int, NliLabel]:
        raw_labels = self._configured_label_map
        if not raw_labels:
            manifest_labels = manifest.get("id2label")
            if isinstance(manifest_labels, dict):
                raw_labels = {
                    int(index): label
                    for index, label in manifest_labels.items()
                }
        if not raw_labels:
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self._tokenizer_name)
                raw_labels = {
                    int(index): label
                    for index, label in (config.id2label or {}).items()
                }
            except Exception as exc:
                raise NliError(
                    "Could not determine the ONNX NLI label mapping."
                ) from exc

        label_map: dict[int, NliLabel] = {}
        for index, raw_label in raw_labels.items():
            canonical = _canonical_label(str(raw_label))
            if canonical is None:
                raise NliError(
                    f"The ONNX NLI model exposes an unusable label '{raw_label}'."
                )
            label_map[int(index)] = canonical
        if set(label_map.values()) != {
            "entailment",
            "neutral",
            "contradiction",
        }:
            raise NliError(
                "The ONNX NLI model does not expose entailment, neutral, and "
                "contradiction labels."
            )
        return label_map

    def _create_session(self) -> Any:
        if self._session_factory is not None:
            session = self._session_factory(str(self._model_path))
            if session is None:
                raise NliError("The injected ONNX session is unavailable.")
            return session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise NliError(
                "onnxruntime is required when VERIFICATION_NLI_BACKEND=onnx."
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._intra_op_threads > 0:
            options.intra_op_num_threads = self._intra_op_threads
        if self._inter_op_threads > 0:
            options.inter_op_num_threads = self._inter_op_threads
        providers = [self._provider]
        try:
            return ort.InferenceSession(
                str(self._model_path),
                sess_options=options,
                providers=providers,
            )
        except Exception as exc:
            if self._provider != "CPUExecutionProvider":
                logger.warning(
                    "ONNX provider %s failed (%s); falling back to CPU.",
                    self._provider,
                    exc,
                )
                try:
                    return ort.InferenceSession(
                        str(self._model_path),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as cpu_exc:
                    raise NliError(
                        "Could not load the ONNX NLI model."
                    ) from cpu_exc
            raise NliError("Could not load the ONNX NLI model.") from exc

    def _resolve_max_tokens(
        self, manifest: dict[str, Any], tokenizer: Any
    ) -> int:
        candidates = [self._configured_max_tokens]
        for value in (
            manifest.get("max_position_embeddings"),
            getattr(tokenizer, "model_max_length", None),
        ):
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 0 < number < 100_000:
                candidates.append(number)
        return max(8, min(candidates))

    def count_pair_tokens(self, premise: str, hypothesis: str) -> int:
        self.load()
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            truncation=True,
            max_length=self._max_pair_tokens + 1,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        if not pairs:
            return []
        self.load()
        assert self._session is not None
        assert self._tokenizer is not None

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

            lengths: list[tuple[int, tuple[str, str]]] = []
            for pair in pending:
                token_count = self.count_pair_tokens(*pair)
                if token_count > self._max_pair_tokens:
                    raise NliInputTooLong(
                        f"A claim-evidence pair has {token_count} tokens, "
                        f"above the {self._max_pair_tokens}-token limit."
                    )
                lengths.append((token_count, pair))
            batches = _length_aware_batches(
                lengths,
                self._batch_size,
                self._max_batch_tokens,
            )
            for batch_index, batch in enumerate(batches, start=1):
                batch_started = time.perf_counter()
                predictions = self._predict_batch(batch)
                if len(predictions) != len(batch):
                    raise NliError(
                        "The ONNX NLI model returned a different number of "
                        "predictions than claim-evidence pairs."
                    )
                for pair, prediction in zip(batch, predictions):
                    predictions_by_pair[pair] = prediction
                    self._cache_put(pair, prediction)
                logger.info(
                    "ONNX NLI batch %d/%d: pairs=%d elapsed_ms=%.1f",
                    batch_index,
                    len(batches),
                    len(batch),
                    (time.perf_counter() - batch_started) * 1000,
                )

            logger.info(
                "ONNX NLI inference: pairs=%d unique_pairs=%d batches=%d "
                "cache_hits=%d elapsed_ms=%.1f",
                len(pairs),
                len(unique_pairs),
                len(batches),
                cache_hits,
                (time.perf_counter() - started) * 1000,
            )
            return [predictions_by_pair[pair] for pair in pairs]

    def _predict_batch(
        self, batch: Sequence[tuple[str, str]]
    ) -> list[NliPrediction]:
        assert self._session is not None
        assert self._tokenizer is not None
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - numpy ships with torch
            raise NliError("numpy is required for ONNX NLI inference.") from exc

        encoded = self._tokenizer(
            [premise for premise, _ in batch],
            [hypothesis for _, hypothesis in batch],
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="np",
        )
        feeds: dict[str, Any] = {}
        for name in self._input_names:
            value = encoded.get(name)
            if value is None and name in {"token_type_ids", "position_ids"}:
                value = np.zeros_like(encoded["input_ids"])
            if value is None:
                raise NliError(
                    f"The ONNX NLI model requires an unsupported input '{name}'."
                )
            feeds[name] = value

        lengths = feeds["attention_mask"].sum(axis=1).tolist()
        longest = int(max(lengths)) if lengths else 0
        if longest > self._max_pair_tokens:
            raise NliInputTooLong(
                f"A claim-evidence pair has {longest} tokens, above the "
                f"{self._max_pair_tokens}-token limit."
            )

        try:
            logits = self._session.run([self._output_name], feeds)[0]
        except Exception as exc:
            raise NliError("ONNX NLI inference failed.") from exc
        try:
            values = np.asarray(logits, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != len(batch):
                raise NliError(
                    "The ONNX NLI model returned a malformed logits matrix."
                )
            shifted = values - values.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        except NliError:
            raise
        except Exception as exc:
            raise NliError(
                "The ONNX NLI model returned malformed logits."
            ) from exc
        if not all(
            math.isfinite(float(value))
            for row in probabilities.tolist()
            for value in row
        ):
            raise NliError("The ONNX NLI model returned non-finite scores.")
        return [
            _prediction_from_probabilities(row, self._id2label)
            for row in probabilities.tolist()
        ]

    def _cache_get(self, pair: tuple[str, str]) -> NliPrediction | None:
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
