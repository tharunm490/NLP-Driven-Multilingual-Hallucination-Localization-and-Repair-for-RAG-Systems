"""Tests for NLI windowing and prediction handling."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import NliError
from app.services.verification.nli import (
    TransformerNliModel,
    _prediction_from_probabilities,
    build_context_windows,
    build_nli_model,
    build_premise_windows,
)
from tests.verification_support import FakeNliModel


def test_windows_never_exceed_model_limit_and_preserve_offsets():
    evidence = (
        "The policy requires annual reviews. "
        "The review must be documented. "
        "The record is retained for three years."
    )
    model = FakeNliModel()
    model.max_pair_tokens = 16

    windows, warnings = build_premise_windows(
        evidence,
        "The policy requires annual reviews.",
        model,
        max_windows=8,
    )

    assert windows
    assert not warnings
    for window in windows:
        assert model.count_pair_tokens(window.span.text, "The policy requires annual reviews.") <= 16
        assert evidence[window.span.start : window.span.end] == window.span.text


def test_windows_are_capped_with_an_explicit_warning():
    evidence = "One fact. Two facts. Three facts. Four facts."
    model = FakeNliModel()
    model.max_pair_tokens = 8

    windows, warnings = build_premise_windows(
        evidence, "A claim.", model, max_windows=2
    )

    assert len(windows) <= 2
    assert any("only the first 2" in warning for warning in warnings)


def test_context_window_adds_neighbouring_sentences_around_focus():
    evidence = (
        "The policy applies to all staff. "
        "The policy requires annual reviews. "
        "Reviews must be documented."
    )
    model = FakeNliModel()
    model.max_pair_tokens = 48
    focus_start = evidence.index("The policy requires annual reviews.")
    focus_end = focus_start + len("The policy requires annual reviews.")

    windows, warnings = build_context_windows(
        evidence,
        focus_start,
        focus_end,
        "The policy requires annual reviews.",
        model,
    )

    assert not warnings
    assert len(windows) == 1
    premise = windows[0].span.text
    assert "applies to all staff" in premise
    assert "requires annual reviews" in premise
    assert "must be documented" in premise
    assert evidence[windows[0].span.start : windows[0].span.end] == premise


def test_prediction_mapping_uses_model_label_order():
    prediction = _prediction_from_probabilities(
        [0.05, 0.1, 0.85],
        {0: "contradiction", 1: "neutral", 2: "entailment"},
    )

    assert prediction.label == "entailment"
    assert prediction.confidence == pytest.approx(0.85)
    assert prediction.probabilities["contradiction"] == pytest.approx(0.05)


def test_malformed_probability_row_is_rejected():
    with pytest.raises(NliError, match="unexpected size"):
        _prediction_from_probabilities(
            [0.5, 0.5],
            {0: "entailment", 1: "neutral", 2: "contradiction"},
        )
import pytest

from app.services.verification.nli import (
    NliPrediction,
    TransformerNliModel,
    _length_aware_batches,
)


class _CountingTokenizer:
    model_max_length = 512

    def __call__(self, premise, hypothesis=None, **kwargs):
        if isinstance(premise, list):
            texts = [f"{item} {hypothesis}" for item in premise]
        else:
            texts = [f"{premise} {hypothesis}"]
        return {"input_ids": [list(range(len(text.split()) + 3)) for text in texts]}


def _prediction(label: str, value: float) -> NliPrediction:
    return NliPrediction(
        label=label,  # type: ignore[arg-type]
        confidence=value,
        probabilities={"entailment": value, "neutral": 0.0, "contradiction": 0.0},
    )


def test_length_aware_batches_sort_pairs_and_respect_token_budget():
    pairs = [
        (100, ("long evidence", "claim")),
        (10, ("short evidence", "claim")),
        (20, ("medium evidence", "claim")),
        (30, ("another medium", "claim")),
    ]

    batches = _length_aware_batches(pairs, batch_size=2, max_batch_tokens=50)

    assert batches == [
        [("short evidence", "claim"), ("medium evidence", "claim")],
        [("another medium", "claim")],
        [("long evidence", "claim")],
    ]


def test_local_predict_deduplicates_and_caches_exact_pairs():
    model = TransformerNliModel("test/model", batch_size=8)
    model._tokenizer = _CountingTokenizer()
    model._model = object()
    model._device = "cpu"
    model._max_pair_tokens = 512
    model._id2label = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }
    calls: list[list[tuple[str, str]]] = []

    def fake_predict(batch):
        calls.append(list(batch))
        return [
            _prediction("entailment", 0.9)
            if premise == "same evidence"
            else _prediction("contradiction", 0.8)
            for premise, _ in batch
        ]

    model._predict_batch = fake_predict  # type: ignore[method-assign]
    pairs = [
        ("same evidence", "same claim"),
        ("same evidence", "same claim"),
        ("other evidence", "other claim"),
    ]

    first = model.predict(pairs)
    second = model.predict(pairs)

    assert len(calls) == 1
    assert set(calls[0]) == {
        ("same evidence", "same claim"),
        ("other evidence", "other claim"),
    }
    assert [item.label for item in first] == [
        "entailment",
        "entailment",
        "contradiction",
    ]
    assert [item.label for item in second] == [
        "entailment",
        "entailment",
        "contradiction",
    ]


def test_build_nli_model_defaults_to_local_transformer_inference(monkeypatch):
    monkeypatch.delenv("VERIFICATION_NLI_BACKEND", raising=False)
    settings = Settings(_env_file=None)

    model = build_nli_model(settings)

    assert settings.verification_nli_backend == "local"
    assert isinstance(model, TransformerNliModel)
    assert model.model_name == settings.verification_nli_model_name

def test_context_windows_expand_forward_before_consuming_the_token_budget():
    text = (
        "Earlier context is unrelated. "
        "The organization monitors controls. "
        "It evaluates effectiveness."
    )
    hypothesis = (
        "The organization monitors controls and evaluates effectiveness."
    )
    model = FakeNliModel()
    model.max_pair_tokens = 20

    windows, _ = build_context_windows(
        text,
        text.index("The organization monitors"),
        text.index("It evaluates"),
        hypothesis,
        model,
        max_tokens=20,
    )

    assert len(windows) == 1
    assert "Earlier context" not in windows[0].span.text
    assert "effectiveness" in windows[0].span.text
    assert text[windows[0].span.start : windows[0].span.end] == (
        windows[0].span.text
    )
