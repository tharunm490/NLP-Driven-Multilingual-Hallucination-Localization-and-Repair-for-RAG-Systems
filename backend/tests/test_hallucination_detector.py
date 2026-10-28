"""Tests for local hallucination model input order and label mapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.core.config import Settings
from app.core.errors import HallucinationModelError
from app.services.hallucination_detector import TransformerHallucinationDetector


class FakeEncoded(dict):
    def to(self, device: str):
        return self


class FakeTokenizer:
    model_max_length = 512

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.calls: list[dict] = []

    def __call__(
        self,
        sources,
        claims,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.calls.append(
            {
                "sources": list(sources),
                "claims": list(claims),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        return FakeEncoded(
            input_ids=torch.zeros((len(sources), 4), dtype=torch.long)
        )


class FakeModel:
    def __init__(self, logits: torch.Tensor, id2label: dict[int, str]) -> None:
        self._logits = logits
        self.config = SimpleNamespace(id2label=id2label)

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self._logits)


def build_detector(
    settings: Settings,
    id2label: dict[int, str],
    logits: list[list[float]],
    *,
    update: dict | None = None,
) -> tuple[TransformerHallucinationDetector, FakeTokenizer]:
    effective = settings.model_copy(update=update or {})
    tokenizer = FakeTokenizer(torch.tensor(logits))
    model = FakeModel(torch.tensor(logits), id2label)
    detector = TransformerHallucinationDetector(
        effective,
        tokenizer_factory=lambda _: tokenizer,
        model_factory=lambda _: model,
    )
    return detector, tokenizer


def test_detector_uses_source_first_and_supported_probability(settings: Settings):
    detector, tokenizer = build_detector(
        settings,
        {0: "UNSUPPORTED", 1: "SUPPORTED"},
        [[2.0, 4.0], [4.0, 2.0]],
    )

    predictions = detector.predict(
        [("Acme was founded in 2015.", "Acme was founded in 2015."),
         ("Beta was founded in 2020.", "Acme was founded in 2015.")]
    )

    assert tokenizer.calls[0]["sources"][0] == "Acme was founded in 2015."
    assert tokenizer.calls[0]["claims"][0] == "Acme was founded in 2015."
    assert tokenizer.calls[0]["truncation"] == "only_first"
    assert predictions[0].raw_label == "SUPPORTED"
    assert predictions[0].support_probability == pytest.approx(0.88, abs=0.01)
    assert predictions[1].raw_label == "UNSUPPORTED"
    assert predictions[1].support_probability == pytest.approx(0.12, abs=0.01)


def test_detector_requires_resolvable_support_label(settings: Settings):
    detector, _ = build_detector(
        settings,
        {0: "LABEL_0", 1: "LABEL_1"},
        [[0.0, 1.0]],
    )

    with pytest.raises(HallucinationModelError, match="SUPPORTED"):
        detector.predict([("source", "claim")])


def test_detector_accepts_explicit_label_override(settings: Settings):
    detector, _ = build_detector(
        settings,
        {0: "LABEL_0", 1: "LABEL_1"},
        [[0.0, 2.0]],
        update={"hallucination_supported_label_index": 1},
    )

    prediction = detector.predict([("source", "claim")])[0]
    assert prediction.raw_label == "LABEL_1"
    assert prediction.support_probability == pytest.approx(0.88, abs=0.01)
