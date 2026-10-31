"""Tests for the authoritative two-class hallucination verdict policy."""

from __future__ import annotations

import pytest

from app.services.hallucination_policy import HallucinationVerdictPolicy


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("SUPPORTED", "supported"),
        ("supported", "supported"),
        ("UNSUPPORTED", "potentially_hallucinated"),
        ("unsupported", "potentially_hallucinated"),
    ],
)
def test_policy_maps_binary_model_labels(
    raw_label: str,
    expected: str,
) -> None:
    policy = HallucinationVerdictPolicy()

    assert policy.classify(raw_label) == expected


def test_policy_never_produces_a_review_verdict() -> None:
    policy = HallucinationVerdictPolicy()
    verdicts = {
        policy.classify(label)
        for label in ("SUPPORTED", "UNSUPPORTED", "UNKNOWN", "")
    }

    assert verdicts == {"supported", "potentially_hallucinated"}
