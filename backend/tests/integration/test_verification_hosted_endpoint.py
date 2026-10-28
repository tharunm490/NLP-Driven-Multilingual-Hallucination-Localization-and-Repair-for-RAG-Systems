r"""Opt-in smoke test for a real hosted NLI endpoint.

    $env:RUN_HOSTED_NLI_INTEGRATION="1"
    .\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_hosted_endpoint.py -m integration -s

Configure ``VERIFICATION_NLI_BACKEND=hf_endpoint`` plus the endpoint URL and
token in ``backend/.env`` first. This test makes real API calls and may incur
provider charges.
"""

from __future__ import annotations

import os
import time

import pytest

from app.core.config import Settings
from app.services.verification.nli import HostedNliModel, build_nli_model

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_HOSTED_NLI_INTEGRATION") != "1",
        reason="set RUN_HOSTED_NLI_INTEGRATION=1 to call a real hosted endpoint",
    ),
]


def test_hosted_endpoint_returns_three_class_scores():
    settings = Settings()
    if settings.verification_nli_backend != "hf_endpoint":
        pytest.fail(
            "Set VERIFICATION_NLI_BACKEND=hf_endpoint in backend/.env before "
            "running this test."
        )

    model = build_nli_model(settings)
    assert isinstance(model, HostedNliModel)

    pairs = [
        (
            "The policy is reviewed annually.",
            "The policy is reviewed every year.",
        ),
        (
            "The policy is reviewed annually.",
            "The policy is reviewed daily.",
        ),
        (
            "The policy is reviewed annually.",
            "The policy covers access control.",
        ),
    ]

    started = time.perf_counter()
    try:
        predictions = model.predict(pairs)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        model.close()

    assert len(predictions) == len(pairs)
    for prediction in predictions:
        assert prediction.label in {
            "entailment",
            "neutral",
            "contradiction",
        }
        assert set(prediction.probabilities) == {
            "entailment",
            "neutral",
            "contradiction",
        }
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
        assert 0.0 <= prediction.confidence <= 1.0

    print(f"hosted_nli_pairs={len(pairs)} elapsed_ms={elapsed_ms:.3f}")
    print(
        "hosted_nli_labels="
        + ",".join(
            f"{prediction.label}:{prediction.confidence:.4f}"
            for prediction in predictions
        )
    )
