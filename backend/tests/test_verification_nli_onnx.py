"""Tests for the optional ONNX Runtime NLI backend (no real model needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.config import Settings
from app.core.errors import NliError, NliInputTooLong
from app.services.verification.nli import build_nli_model
from app.services.verification.onnx import OnnxNliModel


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shape = ["batch", "sequence"]


class _FakeSession:
    def __init__(self, *, require_token_types: bool = False) -> None:
        self._require_token_types = require_token_types
        self.calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_FakeInput]:
        names = ["input_ids", "attention_mask"]
        if self._require_token_types:
            names.append("token_type_ids")
        return [_FakeInput(name) for name in names]

    def get_outputs(self) -> list[_FakeInput]:
        return [_FakeInput("logits")]

    def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
        self.calls.append(dict(feeds))
        batch = np.asarray(feeds["input_ids"]).shape[0]
        return [np.asarray([[0.05, 0.05, 0.9]] * batch, dtype=np.float32)]


class _FakeTokenizer:
    model_max_length = 512

    def __call__(
        self,
        premises: str | list[str],
        hypotheses: str | list[str] | None = None,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
    ) -> dict[str, Any]:
        premise_list = [premises] if isinstance(premises, str) else list(premises)
        hypothesis_list = (
            [hypotheses] if isinstance(hypotheses, str) else list(hypotheses or [])
        )
        rows = [
            [1] + [11 + (index % 7) for index, _ in enumerate(text.split())] + [2]
            for text in premise_list
        ]
        if hypothesis_list:
            for row, hypothesis in zip(rows, hypothesis_list):
                row.extend([21 + (index % 5) for index, _ in enumerate(hypothesis.split())])
                row.append(2)
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        if not padding and return_tensors is None:
            return {"input_ids": rows[0]}
        width = max(len(row) for row in rows)
        input_ids = np.asarray(
            [row + [0] * (width - len(row)) for row in rows],
            dtype=np.int64,
        )
        attention = np.asarray(
            [[1] * len(row) + [0] * (width - len(row)) for row in rows],
            dtype=np.int64,
        )
        token_types = np.zeros_like(input_ids)
        result: dict[str, Any] = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention
        if return_token_type_ids:
            result["token_type_ids"] = token_types
        return result


def _model(
    tmp_path: Path,
    *,
    session: _FakeSession | None = None,
    max_tokens: int = 256,
) -> OnnxNliModel:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    fake_session = session or _FakeSession()
    return OnnxNliModel(
        model_name="test/tokenizer",
        model_path=model_path,
        max_tokens=max_tokens,
        tokenizer_factory=lambda _: _FakeTokenizer(),
        session_factory=lambda _: fake_session,
        label_map={0: "entailment", 1: "neutral", 2: "contradiction"},
    )


def test_onnx_model_maps_logits_and_uses_only_available_inputs(
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    model = _model(tmp_path, session=session)

    predictions = model.predict(
        [("The policy requires annual reviews.", "Reviews are annual.")]
    )

    assert predictions[0].label == "contradiction"
    assert predictions[0].confidence == pytest.approx(0.539133, abs=1e-5)
    assert set(session.calls[0]) == {"input_ids", "attention_mask"}


def test_onnx_model_supplies_token_type_ids_when_required(
    tmp_path: Path,
) -> None:
    session = _FakeSession(require_token_types=True)
    model = _model(tmp_path, session=session)

    model.predict([("Premise text.", "Hypothesis.")])

    assert set(session.calls[0]) == {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }


def test_onnx_model_rejects_pairs_above_the_limit(tmp_path: Path) -> None:
    model = _model(tmp_path, max_tokens=8)

    with pytest.raises(NliInputTooLong):
        model.predict(
            [
                (
                    "This premise has far too many words for the configured limit.",
                    "The claim is also too long for this test.",
                )
            ]
        )


def test_onnx_model_requires_a_model_path() -> None:
    with pytest.raises(ValueError, match="ONNX_MODEL_PATH"):
        Settings(
            _env_file=None,
            verification_nli_backend="onnx",
        )


def test_build_nli_model_selects_onnx_backend(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    settings = Settings(
        _env_file=None,
        verification_nli_backend="onnx",
        verification_nli_onnx_model_path=str(model_path),
    )

    model = build_nli_model(settings)

    assert isinstance(model, OnnxNliModel)
    assert model.model_name.startswith("onnx:")


def test_onnx_model_reports_a_missing_file(tmp_path: Path) -> None:
    model = OnnxNliModel(
        model_name="test/tokenizer",
        model_path=tmp_path / "missing.onnx",
    )

    with pytest.raises(NliError, match="not found"):
        model.load()
