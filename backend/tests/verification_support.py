"""Deterministic test doubles for verification tests."""

from __future__ import annotations

from typing import Callable, Sequence

from app.core.errors import NliError
from app.schemas.verification import (
    ClaimVerificationResult,
    DetectedLanguage,
    EvidenceReference,
    EvidenceVerification,
    ExtractedClaim,
    NamedEntity,
    NliWindowPrediction,
    TextSpan,
    VerificationEvidenceInput,
    VerificationResult,
    VerificationSummary,
)
from app.services.verification.matching import EvidenceMatch, MatchingResult
from app.services.verification.ner import NerResult
from app.services.verification.nli import NliPrediction


class FakeNliModel:
    model_name = "test/fake-nli"
    max_pair_tokens = 48

    def __init__(
        self,
        default: NliPrediction | None = None,
        by_hypothesis: dict[str, NliPrediction] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self._default = default or prediction("entailment", 0.95)
        self._by_hypothesis = dict(by_hypothesis or {})
        self._fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def count_pair_tokens(self, premise: str, hypothesis: str) -> int:
        return len((premise + " " + hypothesis).split()) + 3

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        if self._fail:
            raise NliError("simulated NLI failure")
        self.calls.append(list(pairs))
        return [
            self._by_hypothesis.get(hypothesis, self._default)
            for _, hypothesis in pairs
        ]


class FakeNerTagger:
    model_name = "test/fake-ner"

    def __init__(
        self,
        entities: Sequence[NamedEntity] = (),
        *,
        fail: bool = False,
    ) -> None:
        self._entities = tuple(entities)
        self._fail = fail
        self.is_loaded = True

    def tag(self, text: str) -> NerResult:
        if self._fail:
            raise RuntimeError("simulated NER failure")
        return NerResult(entities=self._entities)


class FixedMatcher:
    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self._scores = list(scores or [])
        self.calls: list[tuple[list[str], list[str]]] = []

    def match_many(
        self, claims: Sequence[str], evidence: Sequence[str]
    ) -> MatchingResult:
        self.calls.append((list(claims), list(evidence)))
        matches: list[tuple[EvidenceMatch, ...]] = []
        for index, _ in enumerate(claims):
            scores = self._scores or [1.0 - evidence_index * 0.1 for evidence_index in range(len(evidence))]
            matches.append(
                tuple(
                    sorted(
                        (
                            EvidenceMatch(evidence_index=evidence_index, similarity=score)
                            for evidence_index, score in enumerate(scores[: len(evidence)])
                        ),
                        key=lambda item: (-item.similarity, item.evidence_index),
                    )
                )
            )
        return MatchingResult(matches=tuple(matches), method="fixed")


class StubVerificationEngine:
    enabled = True

    def __init__(self, result: VerificationResult | None = None) -> None:
        self.result = result or supported_result()
        self.calls: list[tuple[str, list[VerificationEvidenceInput]]] = []

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: object | None = None,
    ) -> VerificationResult:
        self.calls.append((answer, list(evidence)))
        return self.result


def prediction(label: str, confidence: float) -> NliPrediction:
    remainder = max(0.0, 1.0 - confidence)
    probabilities = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    probabilities[label] = confidence
    others = [item for item in probabilities if item != label]
    probabilities[others[0]] = remainder
    return NliPrediction(label=label, confidence=confidence, probabilities=probabilities)  # type: ignore[arg-type]


def claim(
    text: str,
    *,
    offset: int = 0,
    claim_id: str = "c1",
    method: str = "heuristic",
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=claim_id,
        sentence_index=1,
        sentence=text,
        sentence_start=offset,
        sentence_end=offset + len(text),
        claim=text,
        claim_start=offset,
        claim_end=offset + len(text),
        language=DetectedLanguage(code="en", confidence=1.0, reliable=True, method="test"),
        extraction_method=method,
    )


def supported_result(
    *,
    claims_total: int = 1,
    supported: int = 1,
    contradicted: int = 0,
    neutral: int = 0,
    claim_text: str = "Acme Corporation was founded in 2015.",
    evidence_passed: int = 1,
) -> VerificationResult:
    claim_result = ClaimVerificationResult(
        claim=claim(claim_text),
        status="supported" if supported else ("contradicted" if contradicted else "neutral"),
        label="entailment" if supported else ("contradiction" if contradicted else "neutral"),
        confidence=0.95 if supported else 0.6,
        probabilities={"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}
        if supported
        else {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7},
        pairs_expected=1,
        pairs_evaluated=1,
        matched_evidence=[
            EvidenceVerification(
                evidence=EvidenceReference(
                    evidence_id="e1",
                    filename="acme.txt",
                    page_number=1,
                    text_preview="Acme Corporation was founded in 2015.",
                ),
                rank=1,
                similarity=0.9,
                label="entailment" if supported else "neutral",
                confidence=0.95 if supported else 0.6,
                probabilities={"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}
                if supported
                else {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7},
                relevance="relevant",
                relevance_score=0.9,
                claim_term_coverage=0.9,
                counts_for_verdict=bool(supported),
                premise_text=claim_text,
                premise_strategy="full_evidence",
                evidence_char_length=len(claim_text),
                premise_char_length=len(claim_text),
                windows=[
                    NliWindowPrediction(
                        window_index=0,
                        start=0,
                        end=40,
                        label="entailment" if supported else "neutral",
                        confidence=0.95 if supported else 0.6,
                        probabilities={"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}
                        if supported
                        else {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7},
                    )
                ],
            )
        ],
    )
    return VerificationResult(
        status="completed",
        enabled=True,
        claim_extraction_method="heuristic",
        nli_model="test/fake-nli",
        ner_model="test/fake-ner",
        claims=[claim_result],
        summary=VerificationSummary(
            claims_total=claims_total,
            claims_checked=claims_total,
            claims_complete=claims_total,
            supported=supported,
            contradicted=contradicted,
            neutral=neutral,
            evidence_passed=evidence_passed,
            evidence_total=evidence_passed,
            pairs_expected=claims_total,
            pairs_evaluated=claims_total,
        ),
        verification_ms=1.0,
    )


class FailingVerificationEngine:
    enabled = True

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: object | None = None,
    ) -> VerificationResult:
        raise RuntimeError("simulated verification outage")


def fake_entity(text: str, label: str = "ORG", start: int = 0) -> NamedEntity:
    return NamedEntity(text=text, label=label, start=start, end=start + len(text), score=0.99)


def text_span(text: str, start: int = 0) -> TextSpan:
    return TextSpan(start=start, end=start + len(text), text=text)
