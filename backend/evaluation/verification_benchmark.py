r"""Benchmark NLI batching, evidence windows, ONNX, threads, and model reuse.

The benchmark calls the local verification engine directly with a controlled,
labelled ISO/IEC 27001 fixture. Retrieval, embeddings, and the generator are
excluded so that the measurements isolate claim verification. Each job runs in
its own process so model memory does not leak between configurations.

Example (from ``backend``):
    .\.venv\Scripts\python.exe -m evaluation.verification_benchmark
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import queue as queue_module
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import psutil

from app.core.config import Settings
from app.schemas.verification import VerificationEvidenceInput, VerificationResult
from app.services.query_language_detection import QueryLanguageDetector
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import ClaimVerificationEngine
from app.services.verification.matching import LexicalEvidenceMatcher
from app.services.verification.ner import build_ner_tagger
from app.services.verification.nli import (
    TransformerNliModel,
    build_context_windows,
    build_nli_model,
)
from app.services.verification.onnx import OnnxNliModel
from evaluation.iso27001_dataset import CHUNKS_BY_ID
from evaluation.verification_cases import (
    EngineScenario,
    NliCase,
    engine_scenarios,
    nli_cases,
)

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_ONNX_MODEL = "data/onnx/mdeberta-v3-base-mnli-xnli/model.onnx"
DEFAULT_INT8_MODEL = "data/onnx/mdeberta-v3-base-mnli-xnli/model_int8.onnx"
DEFAULT_RESULTS_DIR = "evaluation/results"
_LABELS = ("entailment", "neutral", "contradiction")
_ENGINE_LABELS = (
    "supported",
    "contradicted",
    "neutral",
    "unevaluable",
    "verification_error",
)


@dataclass(frozen=True)
class NliBenchmarkConfig:
    config_id: str
    backend: str
    batch_size: int
    max_tokens: int = 512
    window_tokens: int = 0
    num_threads: int = 0
    intra_op_threads: int = 0
    quantized: bool = False


@dataclass(frozen=True)
class EngineBenchmarkConfig:
    config_id: str
    max_claims: int
    max_extracted_claims: int
    max_evidence_per_claim: int
    max_nli_pairs: int
    primary_window_tokens: int
    nli_max_tokens: int
    nli_batch_size: int
    nli_max_batch_tokens: int
    min_evidence_relevance: float = 0.5
    nli_backend: str = "local"
    contradiction_threshold: float = 0.5
    deadline_seconds: float = 600.0
    warm_repeat: bool = True


class _ResourceSampler:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._cpu_samples: list[float] = []
        self._warning: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        average_cpu = (
            sum(self._cpu_samples) / len(self._cpu_samples)
            if self._cpu_samples
            else None
        )
        return {
            "peak_rss_mb": round(self._peak_rss / (1024 * 1024), 1)
            if self._peak_rss
            else None,
            "average_cpu_percent": round(average_cpu, 1)
            if average_cpu is not None
            else None,
            "cpu_samples": len(self._cpu_samples),
            "warning": self._warning,
        }

    def _run(self) -> None:
        try:
            process = psutil.Process(os.getpid())
            process.cpu_percent(None)
            while not self._event.wait(0.05):
                rss = process.memory_info().rss
                self._peak_rss = max(self._peak_rss, rss)
                self._cpu_samples.append(process.cpu_percent(None))
        except Exception as exc:  # pragma: no cover - platform dependent
            self._warning = f"Resource sampling failed: {exc}"


def nli_benchmark_configs(include_int8: bool, include_onnx: bool) -> list[NliBenchmarkConfig]:
    configs = [
        NliBenchmarkConfig(
            config_id="torch-sequential-full",
            backend="torch",
            batch_size=1,
        ),
        NliBenchmarkConfig(
            config_id="torch-batched-full",
            backend="torch",
            batch_size=8,
        ),
        NliBenchmarkConfig(
            config_id="torch-batched-window-192",
            backend="torch",
            batch_size=8,
            window_tokens=192,
        ),
        NliBenchmarkConfig(
            config_id="torch-batched-window-256",
            backend="torch",
            batch_size=8,
            window_tokens=256,
        ),
        NliBenchmarkConfig(
            config_id="torch-batched-window-192-t8",
            backend="torch",
            batch_size=8,
            window_tokens=192,
            num_threads=8,
        ),
        NliBenchmarkConfig(
            config_id="torch-batched-window-192-t4",
            backend="torch",
            batch_size=8,
            window_tokens=192,
            num_threads=4,
        ),
    ]
    if include_onnx:
        configs.extend(
            [
                NliBenchmarkConfig(
                    config_id="onnx-fp32-window-192",
                    backend="onnx",
                    batch_size=8,
                    window_tokens=192,
                ),
                NliBenchmarkConfig(
                    config_id="onnx-fp32-window-192-t6",
                    backend="onnx",
                    batch_size=8,
                    window_tokens=192,
                    intra_op_threads=6,
                ),
                NliBenchmarkConfig(
                    config_id="onnx-fp32-full",
                    backend="onnx",
                    batch_size=8,
                ),
            ]
        )
        if include_int8:
            configs.append(
                NliBenchmarkConfig(
                    config_id="onnx-int8-window-192",
                    backend="onnx",
                    batch_size=8,
                    window_tokens=192,
                    quantized=True,
                )
            )
    return configs


def engine_benchmark_configs(
    *,
    include_onnx: bool,
) -> list[EngineBenchmarkConfig]:
    configs = [
        EngineBenchmarkConfig(
            config_id="unbatched-exhaustive",
            max_claims=20,
            max_extracted_claims=64,
            max_evidence_per_claim=50,
            max_nli_pairs=250,
            primary_window_tokens=192,
            nli_max_tokens=384,
            nli_batch_size=1,
            nli_max_batch_tokens=4096,
            contradiction_threshold=0.6,
            deadline_seconds=600.0,
            warm_repeat=True,
        ),
        EngineBenchmarkConfig(
            config_id="batched-exhaustive",
            max_claims=20,
            max_extracted_claims=64,
            max_evidence_per_claim=50,
            max_nli_pairs=250,
            primary_window_tokens=192,
            nli_max_tokens=256,
            nli_batch_size=8,
            nli_max_batch_tokens=8192,
            contradiction_threshold=0.6,
            deadline_seconds=120.0,
            warm_repeat=True,
        ),
        EngineBenchmarkConfig(
            config_id="legacy-8x2-12pair",
            max_claims=8,
            max_extracted_claims=64,
            max_evidence_per_claim=2,
            max_nli_pairs=12,
            primary_window_tokens=192,
            nli_max_tokens=256,
            nli_batch_size=8,
            nli_max_batch_tokens=8192,
            contradiction_threshold=0.6,
            deadline_seconds=120.0,
            warm_repeat=False,
        ),
    ]
    if include_onnx:
        configs.extend(
            [
                EngineBenchmarkConfig(
                    config_id="onnx-batched-exhaustive",
                    max_claims=20,
                    max_extracted_claims=64,
                    max_evidence_per_claim=50,
                    max_nli_pairs=250,
                    primary_window_tokens=192,
                    nli_max_tokens=256,
                    nli_batch_size=8,
                    nli_max_batch_tokens=8192,
                    nli_backend="onnx",
                    contradiction_threshold=0.6,
                    deadline_seconds=120.0,
                    warm_repeat=False,
                ),
                EngineBenchmarkConfig(
                    config_id="onnx-legacy-8x2-12pair",
                    max_claims=8,
                    max_extracted_claims=64,
                    max_evidence_per_claim=2,
                    max_nli_pairs=12,
                    primary_window_tokens=192,
                    nli_max_tokens=256,
                    nli_batch_size=8,
                    nli_max_batch_tokens=8192,
                    nli_backend="onnx",
                    contradiction_threshold=0.6,
                    deadline_seconds=120.0,
                    warm_repeat=False,
                ),
            ]
        )
    return configs


def _run_in_subprocess(
    kind: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue: Any = context.Queue()
    process = context.Process(
        target=_worker_entry,
        args=(kind, payload, result_queue),
    )
    process.start()
    try:
        status, result = result_queue.get(timeout=timeout_seconds)
    except queue_module.Empty as exc:
        process.terminate()
        process.join(timeout=5.0)
        raise RuntimeError(
            f"{kind} benchmark worker exceeded {timeout_seconds:.0f}s."
        ) from exc
    finally:
        if process.is_alive():
            process.join(timeout=5.0)
    if status == "error":
        raise RuntimeError(f"{kind} benchmark worker failed:\n{result}")
    return result


def _worker_entry(kind: str, payload: dict[str, Any], sink: Any) -> None:
    try:
        if kind == "nli":
            result = _run_nli_job(**payload)
        elif kind == "engine":
            result = _run_engine_job(**payload)
        else:
            raise ValueError(f"Unknown benchmark job kind: {kind}")
        sink.put(("ok", result))
    except BaseException:
        sink.put(("error", traceback.format_exc()))


def _run_nli_job(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    onnx_model_path: str,
    int8_model_path: str,
) -> dict[str, Any]:
    cfg = NliBenchmarkConfig(**config)
    sampler = _ResourceSampler()
    sampler.start()
    model = _build_nli_model(cfg, onnx_model_path, int8_model_path)
    load_started = time.perf_counter()
    model.load()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    selected: list[tuple[NliCase, str]] = []
    window_build_ms = 0.0
    window_warnings: list[str] = []
    for raw_case in cases:
        case = NliCase(**raw_case)
        if cfg.window_tokens > 0:
            focus_start = case.evidence.find(case.focus)
            if focus_start < 0:
                raise ValueError(
                    f"Focus sentence was not found for {case.case_id}."
                )
            started = time.perf_counter()
            windows, warnings = build_context_windows(
                case.evidence,
                focus_start,
                focus_start + len(case.focus),
                case.claim,
                model,
                max_tokens=cfg.window_tokens,
            )
            window_build_ms += (time.perf_counter() - started) * 1000.0
            window_warnings.extend(warnings)
            if not windows:
                raise ValueError(f"No window was built for {case.case_id}.")
            selected.append((case, windows[0].span.text))
        else:
            selected.append((case, case.evidence))

    pairs = [(text, case.claim) for case, text in selected]
    tokenization_started = time.perf_counter()
    token_counts = [
        model.count_pair_tokens(premise, hypothesis)
        for premise, hypothesis in pairs
    ]
    tokenization_ms = (time.perf_counter() - tokenization_started) * 1000.0
    for case, token_count in zip([item[0] for item in selected], token_counts):
        if token_count > model.max_pair_tokens:
            raise ValueError(
                f"{case.case_id} needs {token_count} tokens, above "
                f"{model.max_pair_tokens}."
            )

    batch_times: list[tuple[int, float]] = []
    original_predict_batch = model._predict_batch

    def timed_predict_batch(batch: Sequence[tuple[str, str]]) -> Any:
        started = time.perf_counter()
        predictions = original_predict_batch(batch)
        batch_times.append(
            (len(batch), (time.perf_counter() - started) * 1000.0)
        )
        return predictions

    model._predict_batch = timed_predict_batch  # type: ignore[method-assign]
    inference_started = time.perf_counter()
    try:
        if cfg.batch_size == 1:
            predictions = []
            for pair in pairs:
                predictions.extend(model.predict([pair]))
        else:
            predictions = model.predict(pairs)
    finally:
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        model._predict_batch = original_predict_batch  # type: ignore[method-assign]
        resources = sampler.stop()

    if len(predictions) != len(pairs):
        raise ValueError(
            f"{cfg.config_id} returned {len(predictions)} predictions for "
            f"{len(pairs)} pairs."
        )

    records = []
    for case, prediction in zip([item[0] for item in selected], predictions):
        records.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "category": case.category,
                "gold": case.gold_label,
                "predicted": prediction.label,
                "confidence": round(prediction.confidence, 6),
                "probabilities": {
                    key: round(value, 6)
                    for key, value in prediction.probabilities.items()
                },
            }
        )

    accuracy = _nli_accuracy(records)
    state = _resource_state(model)
    return {
        "config": asdict(cfg),
        "model_state": state,
        "pairs": len(pairs),
        "mean_pair_tokens": round(
            sum(token_counts) / len(token_counts), 1
        ),
        "max_pair_tokens": max(token_counts),
        "window_build_ms": round(window_build_ms, 1),
        "window_warnings": sorted(set(window_warnings)),
        "load_ms": round(load_ms, 1),
        "tokenization_ms": round(tokenization_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "per_pair_ms": round(inference_ms / len(pairs), 1),
        "batches": len(batch_times),
        "batch_size_mean": round(
            sum(size for size, _ in batch_times) / len(batch_times), 2
        )
        if batch_times
        else 0,
        "batch_ms_mean": round(
            sum(ms for _, ms in batch_times) / len(batch_times), 1
        )
        if batch_times
        else 0,
        "batch_ms_p95": round(_p95([ms for _, ms in batch_times]), 1),
        "accuracy": accuracy,
        "records": records,
        "resources": resources,
    }


def _build_nli_model(
    config: NliBenchmarkConfig,
    onnx_model_path: str,
    int8_model_path: str,
) -> Any:
    if config.backend == "torch":
        return TransformerNliModel(
            model_name=DEFAULT_NLI_MODEL,
            device="cpu",
            batch_size=config.batch_size,
            max_tokens=config.max_tokens,
            max_batch_tokens=8192,
            num_threads=config.num_threads,
        )
    path = Path(int8_model_path if config.quantized else onnx_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at '{path}'. Run "
            "scripts/export_nli_onnx.py first."
        )
    return OnnxNliModel(
        model_name=DEFAULT_NLI_MODEL,
        model_path=path,
        provider="CPUExecutionProvider",
        batch_size=config.batch_size,
        max_tokens=config.max_tokens,
        max_batch_tokens=8192,
        intra_op_threads=config.intra_op_threads,
        inter_op_threads=1,
    )


def _resource_state(model: Any) -> dict[str, Any]:
    return {
        "max_pair_tokens": model.max_pair_tokens,
        "model_name": getattr(model, "model_name", None),
    }


def _nli_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("short", "long", "all"):
        selected = [
            record
            for record in records
            if split == "all" or record["split"] == split
        ]
        gold = [record["gold"] for record in selected]
        predicted = [record["predicted"] for record in selected]
        result[split] = _classification_metrics(gold, predicted)
    for category in ("supported", "contradicted", "neutral"):
        selected = [
            record for record in records if record["category"] == category
        ]
        if not selected:
            continue
        gold = [record["gold"] for record in selected]
        predicted = [record["predicted"] for record in selected]
        result[category] = {
            "count": len(selected),
            "accuracy": round(
                sum(a == b for a, b in zip(gold, predicted)) / len(selected),
                4,
            ),
        }
    return result


def _classification_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, Any]:
    if not gold:
        return {"count": 0, "accuracy": None}
    confusion = {
        expected: {actual: 0 for actual in _LABELS} for expected in _LABELS
    }
    for expected, actual in zip(gold, predicted):
        confusion.setdefault(expected, {label: 0 for label in _LABELS})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    per_label = {}
    for label in _LABELS:
        true_positive = sum(
            1 for expected, actual in zip(gold, predicted)
            if expected == label and actual == label
        )
        false_positive = sum(
            1 for expected, actual in zip(gold, predicted)
            if expected != label and actual == label
        )
        false_negative = sum(
            1 for expected, actual in zip(gold, predicted)
            if expected == label and actual != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": sum(1 for expected in gold if expected == label),
        }
    return {
        "count": len(gold),
        "accuracy": round(
            sum(a == b for a, b in zip(gold, predicted)) / len(gold), 4
        ),
        "false_support": sum(
            1 for expected, actual in zip(gold, predicted)
            if expected != "entailment" and actual == "entailment"
        ),
        "false_contradiction": sum(
            1 for expected, actual in zip(gold, predicted)
            if expected != "contradiction" and actual == "contradiction"
        ),
        "per_label": per_label,
        "confusion": confusion,
    }


def _run_engine_job(
    config: dict[str, Any],
    scenarios: list[dict[str, Any]],
    onnx_model_path: str,
) -> dict[str, Any]:
    cfg = EngineBenchmarkConfig(**config)
    settings = Settings(
        _env_file=None,
        verification_max_extracted_claims=cfg.max_extracted_claims,
        verification_max_claims=cfg.max_claims,
        verification_max_evidence_per_claim=cfg.max_evidence_per_claim,
        verification_max_nli_pairs=cfg.max_nli_pairs,
        verification_primary_window_tokens=cfg.primary_window_tokens,
        verification_min_evidence_relevance=cfg.min_evidence_relevance,
        verification_nli_max_tokens=cfg.nli_max_tokens,
        verification_nli_batch_size=cfg.nli_batch_size,
        verification_nli_max_batch_tokens=cfg.nli_max_batch_tokens,
        verification_nli_backend=cfg.nli_backend,
        verification_nli_onnx_model_path=onnx_model_path
        if cfg.nli_backend == "onnx"
        else "",
        verification_nli_device="cpu",
        verification_nli_contradiction_threshold=cfg.contradiction_threshold,
        verification_ner_device="cpu",
        verification_deadline_seconds=cfg.deadline_seconds,
    )
    detector = QueryLanguageDetector(
        enabled=settings.language_detection_enabled,
        min_chars=settings.query_language_min_chars,
        min_confidence=settings.query_language_min_confidence,
        min_margin=settings.query_language_min_margin,
        fallback_language=settings.query_language_fallback,
    )
    nli_model = build_nli_model(settings)
    ner = build_ner_tagger(settings)
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=detector,
        claim_extractor=HeuristicClaimExtractor(cfg.max_extracted_claims),
        ner_tagger=ner,
        matcher=LexicalEvidenceMatcher(),
        nli_model=nli_model,
    )

    sampler = _ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    nli_model.load()
    nli_load_ms = (time.perf_counter() - started) * 1000.0
    ner_load_ms = 0.0
    if not ner.is_loaded:
        started = time.perf_counter()
        ner.load()
        ner_load_ms = (time.perf_counter() - started) * 1000.0
    load_ms = nli_load_ms + ner_load_ms

    scenario_objects = [EngineScenario(**item) for item in scenarios]
    runs: list[dict[str, Any]] = []

    def execute(scenario: EngineScenario, run_id: str) -> None:
        run_evidence = [
            VerificationEvidenceInput(
                text=CHUNKS_BY_ID[chunk_id].text,
                evidence_id=chunk_id,
                document_id=CHUNKS_BY_ID[chunk_id].document_id,
                filename=CHUNKS_BY_ID[chunk_id].filename,
                chunk_index=CHUNKS_BY_ID[chunk_id].chunk_index,
                page_number=CHUNKS_BY_ID[chunk_id].page_number,
            )
            for chunk_id in scenario.evidence_ids
        ]
        wall_started = time.perf_counter()
        result = engine.verify(scenario.answer, run_evidence)
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        runs.append(
            {
                "run_id": run_id,
                "scenario_id": scenario.scenario_id,
                "wall_ms": round(wall_ms, 1),
                "verification_ms": round(result.verification_ms, 1),
                "timings": result.timings.model_dump(),
                "status": result.status,
                "summary": result.summary.model_dump(),
                "warnings": result.warnings,
                "error": result.error,
                "metrics": _engine_metrics(result, scenario),
            }
        )

    long_scenario = next(
        item for item in scenario_objects if item.scenario_id == "long-request"
    )
    short_scenario = next(
        item for item in scenario_objects if item.scenario_id == "short-request"
    )
    execute(long_scenario, "long-first")
    if cfg.warm_repeat:
        execute(long_scenario, "long-repeat")
    execute(short_scenario, "short-warm")

    resources = sampler.stop()
    return {
        "config": asdict(cfg),
        "model_state": {
            "nli_model": getattr(nli_model, "model_name", None),
            "nli_max_pair_tokens": nli_model.max_pair_tokens,
            "ner_model": getattr(ner, "model_name", None),
        },
        "load_ms": round(load_ms, 1),
        "nli_load_ms": round(nli_load_ms, 1),
        "ner_load_ms": round(ner_load_ms, 1),
        "runs": runs,
        "resources": resources,
    }


def _engine_metrics(
    result: VerificationResult, scenario: EngineScenario
) -> dict[str, Any]:
    actual_claims = list(result.claims)
    assigned: set[int] = set()
    records: list[dict[str, Any]] = []
    for expected_claim, expected_status in zip(
        scenario.claims, scenario.expected_statuses
    ):
        best_index = None
        best_score = 0.0
        for index, actual in enumerate(actual_claims):
            if index in assigned:
                continue
            score = SequenceMatcher(
                None,
                _normalize_claim(expected_claim),
                _normalize_claim(actual.claim.claim),
            ).ratio()
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None or best_score < 0.55:
            records.append(
                {
                    "claim": expected_claim,
                    "expected": expected_status,
                    "actual": "missing",
                    "matched_claim": None,
                    "similarity": round(best_score, 4),
                    "label": None,
                    "confidence": None,
                }
            )
            continue
        assigned.add(best_index)
        actual = actual_claims[best_index]
        records.append(
            {
                "claim": expected_claim,
                "expected": expected_status,
                "actual": actual.status,
                "matched_claim": actual.claim.claim,
                "similarity": round(best_score, 4),
                "label": actual.label,
                "confidence": actual.confidence,
            }
        )

    extras = [
        {
            "claim": actual.claim.claim,
            "status": actual.status,
            "label": actual.label,
            "confidence": actual.confidence,
        }
        for index, actual in enumerate(actual_claims)
        if index not in assigned
    ]
    correct = sum(
        1 for record in records if record["expected"] == record["actual"]
    )
    checked_records = [
        record
        for record in records
        if record["actual"] not in {"missing", "verification_error"}
    ]
    checked_correct = sum(
        1
        for record in checked_records
        if record["expected"] == record["actual"]
    )
    windows = sum(
        len(evidence.windows)
        for claim in actual_claims
        for evidence in claim.matched_evidence
    )
    summary = result.summary
    return {
        "expected_claims": len(scenario.claims),
        "actual_claims": len(actual_claims),
        "claims_complete": summary.claims_complete,
        "claims_incomplete": summary.claims_incomplete,
        "checked": len(checked_records),
        "coverage": round(len(checked_records) / len(scenario.claims), 4),
        "pairs_expected": summary.pairs_expected,
        "pairs_evaluated": summary.pairs_evaluated,
        "pairs_missing": summary.pairs_missing,
        "pair_coverage": round(
            summary.pairs_evaluated / summary.pairs_expected, 4
        )
        if summary.pairs_expected
        else None,
        "correct": correct,
        "accuracy": round(correct / len(scenario.claims), 4),
        "checked_accuracy": round(
            checked_correct / len(checked_records), 4
        )
        if checked_records
        else None,
        "false_support": sum(
            1
            for record in records
            if record["expected"] != "supported"
            and record["actual"] == "supported"
        ),
        "false_contradiction": sum(
            1
            for record in records
            if record["expected"] != "contradicted"
            and record["actual"] == "contradicted"
        ),
        "nli_windows": windows,
        "status_counts": {
            label: sum(
                1 for record in records if record["actual"] == label
            )
            for label in _ENGINE_LABELS
        },
        "records": records,
        "extra_claims": extras,
    }


def _normalize_claim(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _environment() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_gb": round(
            psutil.virtual_memory().total / (1024**3), 1
        ),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "onnx": version("onnx"),
        "onnxruntime": version("onnxruntime"),
        "psutil": version("psutil"),
    }


def run_benchmark(
    *,
    onnx_model_path: str = DEFAULT_ONNX_MODEL,
    int8_model_path: str = DEFAULT_INT8_MODEL,
    include_int8: bool = False,
    nli_only: bool = False,
    engine_only: bool = False,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    onnx_available = Path(onnx_model_path).exists()
    include_onnx = onnx_available
    if include_int8 and not Path(int8_model_path).exists():
        raise FileNotFoundError(
            f"INT8 model not found at '{int8_model_path}'."
        )

    cases = [asdict(case) for case in nli_cases()]
    scenarios = [asdict(scenario) for scenario in engine_scenarios()]
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "dataset": {
            "nli_cases": len(cases),
            "short_cases": sum(1 for case in cases if case["split"] == "short"),
            "long_cases": sum(1 for case in cases if case["split"] == "long"),
            "engine_scenarios": [
                {
                    "scenario_id": scenario["scenario_id"],
                    "claims": len(scenario["claims"]),
                    "evidence_chunks": len(scenario["evidence_ids"]),
                }
                for scenario in scenarios
            ],
            "note": (
                "Small controlled ISO/IEC 27001 fixture; not a substitute for "
                "a public NLI benchmark."
            ),
        },
        "onnx_available": onnx_available,
        "nli_results": [],
        "engine_results": [],
    }

    if not engine_only:
        for config in nli_benchmark_configs(include_int8, include_onnx):
            print(f"[benchmark] NLI {config.config_id}", flush=True)
            report["nli_results"].append(
                _run_in_subprocess(
                    "nli",
                    {
                        "config": asdict(config),
                        "cases": cases,
                        "onnx_model_path": onnx_model_path,
                        "int8_model_path": int8_model_path,
                    },
                    timeout_seconds=timeout_seconds,
                )
            )

    if not nli_only:
        for config in engine_benchmark_configs(include_onnx=include_onnx):
            print(f"[benchmark] engine {config.config_id}", flush=True)
            report["engine_results"].append(
                _run_in_subprocess(
                    "engine",
                    {
                        "config": asdict(config),
                        "scenarios": scenarios,
                        "onnx_model_path": onnx_model_path,
                    },
                    timeout_seconds=timeout_seconds,
                )
            )

    report["conclusions"] = _conclusions(report)
    return report


def _conclusions(report: dict[str, Any]) -> list[str]:
    conclusions: list[str] = []
    nli_results = report.get("nli_results", [])
    windows = [
        item for item in nli_results if item["config"]["window_tokens"] > 0
    ]
    if windows:
        fastest = min(windows, key=lambda item: item["per_pair_ms"])
        most_accurate = max(
            windows,
            key=lambda item: (item["accuracy"]["all"]["accuracy"], -item["per_pair_ms"]),
        )
        conclusions.append(
            f"Fastest windowed NLI configuration: {fastest['config']['config_id']} "
            f"({fastest['per_pair_ms']:.1f} ms/pair, "
            f"{fastest['accuracy']['all']['accuracy']:.1%} accuracy)."
        )
        conclusions.append(
            f"Highest-accuracy windowed NLI configuration: "
            f"{most_accurate['config']['config_id']} "
            f"({most_accurate['accuracy']['all']['accuracy']:.1%})."
        )
    full = [item for item in nli_results if item["config"]["window_tokens"] == 0]
    if full:
        baseline = max(full, key=lambda item: item["per_pair_ms"])
        conclusions.append(
            f"Full-evidence baseline max latency: "
            f"{baseline['config']['config_id']} at "
            f"{baseline['per_pair_ms']:.1f} ms/pair."
        )
    if report.get("onnx_available"):
        torch_window = next(
            (
                item
                for item in windows
                if item["config"]["backend"] == "torch"
                and item["config"]["num_threads"] == 0
            ),
            None,
        )
        onnx_window = next(
            (
                item
                for item in windows
                if item["config"]["backend"] == "onnx"
                and not item["config"]["quantized"]
            ),
            None,
        )
        if torch_window and onnx_window:
            speedup = torch_window["per_pair_ms"] / onnx_window["per_pair_ms"]
            conclusions.append(
                f"ONNX FP32 windowed NLI was {speedup:.2f}x the PyTorch "
                "per-pair speed on this machine."
            )
    int8 = [
        item
        for item in nli_results
        if item["config"].get("quantized")
    ]
    if int8:
        best_int8 = max(
            int8, key=lambda item: item["accuracy"]["all"]["accuracy"]
        )
        conclusions.append(
            f"INT8 accuracy was {best_int8['accuracy']['all']['accuracy']:.1%}; "
            "it should not replace FP32 unless a larger labelled set shows "
            "acceptable accuracy."
        )
    for result in report.get("engine_results", []):
        for run in result["runs"]:
            if run["run_id"] == "long-first":
                conclusions.append(
                    f"{result['config']['config_id']} long request: "
                    f"{run['wall_ms'] / 1000.0:.1f}s, "
                    f"{run['metrics']['checked']}/"
                    f"{run['metrics']['expected_claims']} checked, "
                    f"{run['metrics']['pairs_evaluated']}/"
                    f"{run['metrics']['pairs_expected']} pairs evaluated, "
                    f"{run['metrics']['coverage']:.1%} claim coverage, "
                    f"{run['metrics']['accuracy']:.1%} accuracy, "
                    f"{run['metrics']['false_support']} false support."
                )
    return conclusions


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    environment = report["environment"]
    lines.extend(
        [
            "# Verification performance benchmark",
            "",
            f"Generated: `{report['generated_at']}`",
            "",
            "## Environment",
            "",
            f"- Platform: {environment['platform']}",
            f"- Python: {environment['python']}",
            f"- CPU: {environment['cpu_count']} logical / "
            f"{environment['physical_cpu_count']} physical",
            f"- Memory: {environment['total_memory_gb']} GB",
            f"- torch: {environment['torch'] or 'not installed'}",
            f"- transformers: {environment['transformers'] or 'not installed'}",
            f"- onnxruntime: {environment['onnxruntime'] or 'not installed'}",
            "",
            "## Dataset",
            "",
            f"- NLI cases: {report['dataset']['nli_cases']} "
            f"({report['dataset']['short_cases']} short, "
            f"{report['dataset']['long_cases']} long)",
        ]
    )
    for scenario in report["dataset"]["engine_scenarios"]:
        lines.append(
            f"- Engine scenario `{scenario['scenario_id']}`: "
            f"{scenario['claims']} claims, "
            f"{scenario['evidence_chunks']} evidence chunks"
        )
    lines.extend(["", report["dataset"]["note"]])
    nli_results = report.get("nli_results", [])
    if nli_results:
        lines.extend(
            [
                "",
                "## NLI-only results",
                "",
                "| Config | Split | Pairs | Mean tokens | Load s | Inference s | "
                "Per pair ms | Batches | Batch mean ms | Accuracy | "
                "False support | False contradiction | Peak MB |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: |",
            ]
        )
        for result in nli_results:
            config_id = result["config"]["config_id"]
            for split in ("short", "long", "all"):
                metrics = result["accuracy"][split]
                lines.append(
                    f"| {config_id} | {split} | {result['pairs']} | "
                    f"{result['mean_pair_tokens']} | "
                    f"{result['load_ms'] / 1000.0:.2f} | "
                    f"{result['inference_ms'] / 1000.0:.2f} | "
                    f"{result['per_pair_ms']:.1f} | {result['batches']} | "
                    f"{result['batch_ms_mean']:.1f} | "
                    f"{_format_metric(metrics, 'accuracy')} | "
                    f"{_format_metric(metrics, 'false_support')} | "
                    f"{_format_metric(metrics, 'false_contradiction')} | "
                    f"{_format_optional(result['resources']['peak_rss_mb'])} |"
                )

    lines.extend(
        [
            "",
            "## Engine results",
            "",
            "| Config | Run | Wall s | Verify s | Extract ms | Select ms | "
            "NLI ms | Checked | Coverage | Pairs | Pair coverage | "
            "Incomplete | Accuracy | Checked accuracy | False support | "
            "False contradiction | Peak MB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report.get("engine_results", []):
        config_id = result["config"]["config_id"]
        for run in result["runs"]:
            metrics = run["metrics"]
            timings = run["timings"]
            lines.append(
                f"| {config_id} | {run['run_id']} | "
                f"{run['wall_ms'] / 1000.0:.1f} | "
                f"{run['verification_ms'] / 1000.0:.1f} | "
                f"{timings['claim_extraction_ms']:.1f} | "
                f"{timings['evidence_selection_ms']:.1f} | "
                f"{timings['nli_ms']:.1f} | "
                f"{metrics['checked']}/{metrics['expected_claims']} | "
                f"{metrics['coverage']:.1%} | "
                f"{metrics['pairs_evaluated']}/"
                f"{metrics['pairs_expected']} | "
                f"{_format_rate(metrics['pair_coverage'])} | "
                f"{metrics['claims_incomplete']} | "
                f"{metrics['accuracy']:.1%} | "
                f"{_format_rate(metrics['checked_accuracy'])} | "
                f"{metrics['false_support']} | "
                f"{metrics['false_contradiction']} | "
                f"{_format_optional(result['resources']['peak_rss_mb'])} |"
            )

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
        ]
    )
    for conclusion in report.get("conclusions", []):
        lines.append(f"- {conclusion}")
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "- Engine runs use the labelled ISO/IEC 27001 fixture with a lexical "
            "matcher and supplied evidence. RAG retrieval, embedding generation, "
            "and the LLM are excluded so the measurements isolate verification.",
            "- The engine loads NLI and NER before timing request runs, so "
            "`load_ms` is reported separately and request timings are warm.",
            "- `long-first` and `long-repeat` use the same engine instance to "
            "measure model reuse.",
            "- Times are single-process CPU measurements. Accuracy is on a small "
            "controlled fixture and is not a claim of general NLI accuracy.",
        ]
    )
    if nli_results:
        lines.append(
            "- Full-evidence NLI rows are baselines and may exceed the engine's "
            "window limits; they are not production defaults."
        )
    if any(item["config"].get("quantized") for item in nli_results):
        lines.append(
            "- INT8 is evaluated but is not a default because dynamic "
            "quantization changed labels on several probe pairs."
        )
    return "\n".join(lines) + "\n"


def _format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    if key == "accuracy":
        return f"{value:.1%}"
    return str(value)


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "verification_performance.json"
    markdown_path = target / "verification_performance.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--int8-model", default=DEFAULT_INT8_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--nli-only", action="store_true")
    parser.add_argument("--engine-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.nli_only and args.engine_only:
        parser.error("--nli-only and --engine-only cannot be combined")

    report = run_benchmark(
        onnx_model_path=args.onnx_model,
        int8_model_path=args.int8_model,
        include_int8=args.include_int8,
        nli_only=args.nli_only,
        engine_only=args.engine_only,
        timeout_seconds=args.timeout_seconds,
    )
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)
    for conclusion in report["conclusions"]:
        print(f"- {conclusion}", flush=True)


if __name__ == "__main__":
    main()
