"""Authoritative two-class verdict policy for Hallucination Checks."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.hallucination import HallucinationVerdict


@dataclass(frozen=True)
class HallucinationVerdictPolicy:
    supported_label: str = "SUPPORTED"

    def classify(self, raw_model_label: str) -> HallucinationVerdict:
        if raw_model_label.strip().upper() == self.supported_label.upper():
            return "supported"
        return "potentially_hallucinated"
