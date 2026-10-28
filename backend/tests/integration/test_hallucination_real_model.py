"""Opt-in real inference check for the hallucination detector."""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.services.hallucination_detector import TransformerHallucinationDetector

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_HALLUCINATION_INTEGRATION") != "1",
    reason="set RUN_HALLUCINATION_INTEGRATION=1 to load the real hallucination model",
)
def test_real_model_supports_and_rejects_a_claim() -> None:
    model_name = os.getenv(
        "HALLUCINATION_REAL_MODEL",
        "Horizon-Labs/hallucination-guard-base",
    )
    settings = Settings(
        _env_file=None,
        hallucination_model_name=model_name,
        hallucination_model_local_files_only=(
            os.getenv("HALLUCINATION_REAL_LOCAL_ONLY") == "1"
        ),
        hallucination_model_device="cpu",
        hallucination_model_max_tokens=2048,
    )
    detector = TransformerHallucinationDetector(settings)
    source = (
        "The Eiffel Tower is 330 metres tall and was completed in 1889 "
        "for the World's Fair."
    )

    supported, unsupported = detector.predict(
        [
            (source, "The tower was finished in 1889."),
            (source, "The tower was finished in 1899."),
        ]
    )

    assert supported.raw_label == "SUPPORTED"
    assert supported.support_probability > 0.5
    assert unsupported.raw_label == "UNSUPPORTED"
    assert unsupported.support_probability < 0.5
    assert supported.support_probability > unsupported.support_probability
