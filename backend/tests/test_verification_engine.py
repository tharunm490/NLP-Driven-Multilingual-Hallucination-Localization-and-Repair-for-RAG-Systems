"""Tests for the claim-verification orchestration layer."""

from __future__ import annotations

import threading
import time
from typing import Sequence

import pytest

from app.core.config import Settings
from app.schemas.verification import VerificationEvidenceInput
from app.services.query_language_detection import QueryLanguageDetector
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import (
    ClaimVerificationEngine,
    DisabledVerificationEngine,
    _best_candidates_by_evidence,
    _candidate_windows,
    _claim_term_coverage,
)
from app.services.verification.matching import (
    EvidenceMatch,
    LexicalEvidenceMatcher,
    MatchingResult,
)
from app.services.verification.nli import NliPrediction
from tests.verification_support import (
    FakeNerTagger,
    FakeNliModel,
    FixedMatcher,
    fake_entity,
    prediction,
)


class _PremiseAwareNliModel(FakeNliModel):
    def __init__(
        self,
        required_terms: Sequence[str],
        neutral_confidence: float = 0.9,
    ) -> None:
        super().__init__(default=prediction("neutral", neutral_confidence))
        self._required_terms = tuple(term.casefold() for term in required_terms)
        self._neutral_confidence = neutral_confidence

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        self.calls.append(list(pairs))
        return [
            prediction("entailment", 0.95)
            if all(term in premise.casefold() for term in self._required_terms)
            else prediction("neutral", self._neutral_confidence)
            for premise, _ in pairs
        ]


class _ComplementaryNliModel(FakeNliModel):
    def __init__(self) -> None:
        super().__init__(default=prediction("neutral", 0.9))

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        self.calls.append(list(pairs))
        return [
            prediction("entailment", 0.94)
            if "annual reviews" in premise.casefold()
            and "daily audits" in premise.casefold()
            else prediction("neutral", 0.9)
            for premise, _ in pairs
        ]


class _ConflictingNliModel(FakeNliModel):
    def __init__(self) -> None:
        super().__init__(default=prediction("neutral", 0.9))

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        self.calls.append(list(pairs))
        results: list[NliPrediction] = []
        for premise, _ in pairs:
            if "1920" in premise:
                results.append(prediction("contradiction", 0.93))
            elif "2015" in premise:
                results.append(prediction("entailment", 0.95))
            else:
                results.append(prediction("neutral", 0.9))
        return results


class _ShortPredictionNliModel(FakeNliModel):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        self.calls.append(list(pairs))
        return [prediction("entailment", 0.95) for _ in pairs[:-1]]


class _SlowNerTagger(FakeNerTagger):
    def tag(self, text: str):
        time.sleep(0.08)
        return super().tag(text)


class _PremiseContradictionNliModel(FakeNliModel):
    def __init__(
        self,
        contradiction_term: str,
        neutral_confidence: float = 0.99,
    ) -> None:
        super().__init__(default=prediction("neutral", neutral_confidence))
        self._contradiction_term = contradiction_term.casefold()
        self._neutral_confidence = neutral_confidence

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliPrediction]:
        self.calls.append(list(pairs))
        return [
            prediction("contradiction", 0.95)
            if self._contradiction_term in premise.casefold()
            else prediction("neutral", self._neutral_confidence)
            for premise, _ in pairs
        ]


def _engine(
    settings: Settings,
    detector: QueryLanguageDetector,
    *,
    nli_model: FakeNliModel | None = None,
    ner_tagger: FakeNerTagger | None = None,
    matcher: FixedMatcher | None = None,
) -> ClaimVerificationEngine:
    return ClaimVerificationEngine(
        settings=settings,
        detector=detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=ner_tagger or FakeNerTagger(),
        matcher=matcher or FixedMatcher(),
        nli_model=nli_model or FakeNliModel(),
    )


def _evidence(text: str) -> list[VerificationEvidenceInput]:
    return [
        VerificationEvidenceInput(
            text=text,
            evidence_id="e1",
            filename="acme.txt",
            page_number=1,
            chunk_id="chunk-1",
        )
    ]


def test_supported_claim_produces_supported_summary_and_entity_offsets(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    engine = _engine(
        settings,
        query_language_detector,
        ner_tagger=FakeNerTagger([fake_entity("Acme Corporation")]),
    )

    result = engine.verify(answer, _evidence("Acme Corporation was founded in 2015."))

    assert result.status == "completed"
    assert result.summary.claims_total == 1
    assert result.summary.supported == 1
    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.label == "entailment"
    assert claim.claim.language.code == "en"
    assert claim.claim.entities[0].start == 0
    assert claim.claim.entities[0].text == "Acme Corporation"
    assert claim.matched_evidence[0].windows


def test_contradicted_claim_is_reported_as_contradicted(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 1920."
    nli = FakeNliModel(
        by_hypothesis={
            answer: prediction("contradiction", 0.93),
        }
    )
    engine = _engine(settings, query_language_detector, nli_model=nli)

    result = engine.verify(answer, _evidence("Acme Corporation was founded in 2015."))

    assert result.claims[0].status == "contradicted"
    assert result.summary.contradicted == 1


def test_no_evidence_yields_insufficient_claim_status(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = _engine(settings, query_language_detector)

    result = engine.verify("Acme Corporation was founded in 2015.", [])

    assert result.status == "completed"
    assert result.summary.claims_total == 1
    assert result.claims[0].status == "neutral"
    assert result.summary.evidence_passed == 0


def test_ner_failure_is_isolated_and_verification_continues(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = _engine(
        settings, query_language_detector, ner_tagger=FakeNerTagger(fail=True)
    )

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation was founded in 2015."),
    )

    assert result.status == "completed"
    assert result.claims[0].status == "supported"
    assert any("Named-entity recognition failed" in warning for warning in result.claims[0].warnings)


def test_all_claim_evidence_pairs_are_batched_into_one_nli_call(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020."
    )
    nli = FakeNliModel()
    engine = _engine(settings, query_language_detector, nli_model=nli)

    result = engine.verify(
        answer, _evidence("Acme and Beta are corporations founded in the 2010s.")
    )

    assert result.summary.claims_total == 2
    assert len(nli.calls) == 1
    assert len(nli.calls[0]) == 2
    assert {hypothesis for _, hypothesis in nli.calls[0]} == {
        "Acme Corporation was founded in 2015.",
        "Beta Corporation was founded in 2020.",
    }


def test_every_claim_is_compared_with_every_retrieved_chunk(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020."
    )
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Beta Corporation was founded in 2020.", evidence_id="e2"
        ),
        VerificationEvidenceInput(
            text="Acme publishes software.", evidence_id="e3"
        ),
        VerificationEvidenceInput(
            text="Acme publishes software.", evidence_id="e4"
        ),
    ]
    nli = FakeNliModel()
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9, 0.8, 0.7, 0.6]),
    )

    result = engine.verify(answer, evidence)

    assert result.summary.pairs_expected == 8
    assert result.summary.pairs_evaluated == 8
    assert result.summary.pairs_missing == 0
    assert result.summary.claims_complete == 2
    assert len(nli.calls) == 1
    assert len(nli.calls[0]) == 8
    assert {hypothesis for _, hypothesis in nli.calls[0]} == {
        "Acme Corporation was founded in 2015.",
        "Beta Corporation was founded in 2020.",
    }
    for claim in result.claims:
        assert len(claim.matched_evidence) == 4
        assert [item.evidence.evidence_id for item in claim.matched_evidence] == [
            "e1",
            "e2",
            "e3",
            "e4",
        ]
        assert all(item.evaluated for item in claim.matched_evidence)
        assert claim.complete is True


def test_retrieval_similarity_is_preserved_separately_from_matcher_similarity(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.",
            evidence_id="e1",
            retrieval_similarity=0.73,
            retrieval_score=0.27,
        )
    ]
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=FakeNliModel(),
        matcher=FixedMatcher([0.42]),
    )

    result = engine.verify("Acme Corporation was founded in 2015.", evidence)

    row = result.claims[0].matched_evidence[0]
    assert row.similarity == pytest.approx(0.42)
    assert row.retrieval_similarity == pytest.approx(0.73)
    assert row.retrieval_score == pytest.approx(0.27)


def test_fifteen_claims_by_five_chunks_evaluates_all_seventy_five_pairs(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = " ".join(
        f"Acme Corporation office {index} was founded in {1990 + index}."
        for index in range(1, 16)
    )
    evidence = [
        VerificationEvidenceInput(
            text=(
                f"Document {index} records general governance information "
                "for the organisation."
            ),
            evidence_id=f"e{index}",
        )
        for index in range(1, 6)
    ]
    nli = FakeNliModel(default=prediction("neutral", 0.9))
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher(),
    )

    result = engine.verify(answer, evidence)

    assert result.summary.claims_total == 15
    assert result.summary.claims_complete == 15
    assert result.summary.pairs_expected == 75
    assert result.summary.pairs_evaluated == 75
    assert result.summary.pairs_missing == 0
    assert sum(len(call) for call in nli.calls) == 75
    assert len(result.claims) == 15
    for claim in result.claims:
        assert claim.status == "neutral"
        assert len(claim.matched_evidence) == 5
        assert [
            item.evidence.evidence_id for item in claim.matched_evidence
        ] == ["e1", "e2", "e3", "e4", "e5"]
        assert all(item.evaluated for item in claim.matched_evidence)


def test_duplicate_retrieved_chunks_are_each_reported_as_separate_pairs(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e2"
        ),
    ]
    nli = FakeNliModel()
    engine = _engine(settings, query_language_detector, nli_model=nli)

    result = engine.verify(
        "Acme Corporation was founded in 2015.", evidence
    )

    claim = result.claims[0]
    assert claim.pairs_expected == 2
    assert claim.pairs_evaluated == 2
    assert [item.evidence.evidence_id for item in claim.matched_evidence] == [
        "e1",
        "e2",
    ]
    assert len(nli.calls[0]) == 2


def test_neutral_comparison_does_not_override_supporting_comparison(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Acme Corporation publishes software.", evidence_id="e2"
        ),
    ]
    nli = _PremiseAwareNliModel(["founded in 2015"])
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.95, 0.9]),
    )

    result = engine.verify(answer, evidence)

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.pairs_evaluated == claim.pairs_expected == 2
    assert [item.label for item in claim.matched_evidence] == [
        "entailment",
        "neutral",
    ]
    assert [item.counts_for_verdict for item in claim.matched_evidence] == [
        True,
        False,
    ]


def test_raw_contradiction_from_irrelevant_passage_is_not_counted(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="The Eiffel Tower is in Paris and was completed in 1920.",
            evidence_id="e2",
        ),
    ]
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=_ConflictingNliModel(),
        matcher=FixedMatcher([0.95, 0.01]),
    )

    result = engine.verify(answer, evidence)

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.conflict is False
    assert claim.support_confidence == pytest.approx(0.95)
    assert claim.contradiction_confidence is None
    irrelevant = claim.matched_evidence[1]
    assert irrelevant.label == "contradiction"
    assert irrelevant.relevance == "irrelevant"
    assert irrelevant.counts_for_verdict is False
    assert "not counted" in irrelevant.explanation
    assert any("raw contradiction" in warning for warning in claim.warnings)


def test_table_of_contents_evidence_cannot_support_a_claim(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "The standard specifies that an ISMS must establish a management system."
    toc = (
        "Contents\n"
        "4 Context of the organization ........ 9\n"
        "5 Leadership and commitment ........ 11\n"
        "6 Planning the ISMS ........ 14\n"
        "7 Support ........ 17\n"
        "9 Performance evaluation ........ 24"
    )
    nli = FakeNliModel(default=prediction("entailment", 0.99))
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.99]),
    )

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=toc, evidence_id="toc", page_number=3)],
    )

    claim = result.claims[0]
    pair = claim.matched_evidence[0]
    assert claim.status == "neutral"
    assert pair.label == "entailment"
    assert pair.evidence_quality == "table_of_contents"
    assert pair.relevance == "non_substantive"
    assert pair.counts_for_verdict is False
    assert "not counted" in pair.explanation


def test_high_similarity_without_claim_term_coverage_is_not_counted(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = "The Eiffel Tower is in Paris and was completed in 1889."
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=FakeNliModel(default=prediction("entailment", 0.99)),
        matcher=FixedMatcher([0.99]),
    )

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=evidence, evidence_id="e1")],
    )

    claim = result.claims[0]
    pair = claim.matched_evidence[0]
    assert claim.status == "neutral"
    assert pair.relevance == "irrelevant"
    assert pair.claim_term_coverage == 0.0
    assert pair.counts_for_verdict is False


def test_weaker_secondary_support_does_not_count_for_the_verdict(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "1. Monitor, measure, and evaluate performance "
        f"{chr(8212)} The organization determines what to monitor, the "
        "methods to use, and when results must be evaluated."
    )
    weak = "The organization monitors performance and evaluates results."
    strong = (
        "The organization determines what needs to be monitored and "
        "measured, the methods to use, when the activities occur, who "
        "performs them, and when results are analysed and evaluated. It "
        "retains documented evidence and evaluates information security "
        "performance and the effectiveness of the ISMS."
    )
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=FakeNliModel(default=prediction("entailment", 0.99)),
        matcher=FixedMatcher([0.80, 0.70]),
    )

    result = engine.verify(
        answer,
        [
            VerificationEvidenceInput(text=weak, evidence_id="weak"),
            VerificationEvidenceInput(text=strong, evidence_id="strong"),
        ],
    )

    claim = result.claims[0]
    assert claim.status == "supported"
    counted = [
        item.evidence.evidence_id
        for item in claim.matched_evidence
        if item.counts_for_verdict
    ]
    assert counted == ["strong"]
    weak_pair = claim.matched_evidence[0]
    assert weak_pair.relevance == "relevant"
    assert weak_pair.counts_for_verdict is False
    assert "materially weaker" in weak_pair.explanation
    assert any("materially weaker" in warning for warning in claim.warnings)


def test_single_moderate_overlap_support_still_counts(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=FakeNliModel(default=prediction("entailment", 0.99)),
        matcher=FixedMatcher([0.50]),
    )

    result = engine.verify(
        "The organization monitors performance and evaluates results.",
        [
            VerificationEvidenceInput(
                text="The organization monitors performance at planned intervals.",
                evidence_id="e1",
            )
        ],
    )

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.matched_evidence[0].relevance == "relevant"
    assert claim.matched_evidence[0].counts_for_verdict is True


def test_anaphoric_claim_is_evaluated_with_the_raw_claim_fallback(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    nli = FakeNliModel(default=prediction("entailment", 0.99))
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.99]),
    )

    result = engine.verify(
        "It does this by applying a risk management process.",
        [VerificationEvidenceInput(text="The ISMS applies a risk process.", evidence_id="e1")],
    )

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.diagnostic_status is None
    assert claim.claim.evaluable is False
    assert claim.pairs_expected == 1
    assert claim.pairs_evaluated == 1
    assert len(claim.matched_evidence) == 1
    assert len(nli.calls) == 1
    assert result.summary.claims_total == 1
    assert result.summary.claims_evaluable == 1
    assert result.summary.claims_unevaluable == 0
    assert result.summary.unevaluable == 0
    assert result.summary.pairs_expected == 1
    assert result.status == "completed"


def test_pair_exposes_full_nli_inputs_scores_and_token_accounting(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence_text = "Acme Corporation was founded in 2015. It publishes software."
    engine = _engine(settings, query_language_detector)

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=evidence_text, evidence_id="e1")],
    )

    pair = result.claims[0].matched_evidence[0]
    assert pair.claim_id == "c1"
    assert pair.hypothesis_text == answer
    assert pair.evidence_text == evidence_text
    assert pair.premise_text == evidence_text
    assert pair.nli_input_order == "premise_evidence,hypothesis_claim"
    assert pair.evidence_quality == "substantive"
    assert pair.full_pair_token_count is not None
    assert pair.input_pair_token_count is not None
    assert pair.full_pair_token_count == pair.input_pair_token_count
    assert pair.max_pair_tokens == 48
    assert pair.probabilities["entailment"] == pytest.approx(0.95)


def test_list_item_hypothesis_includes_its_lead_in_context(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "The standard requires that an ISMS must:\n- Establish the ISMS."
    evidence = "The standard requires that an ISMS must establish the ISMS."
    nli = FakeNliModel(
        by_hypothesis={
            "An ISMS must establish the ISMS.": prediction(
                "entailment", 0.95
            )
        },
        default=prediction("neutral", 0.9),
    )
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9]),
    )

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=evidence, evidence_id="e1")],
    )

    claim = result.claims[0]
    assert claim.claim.claim == "Establish the ISMS."
    assert claim.claim.context is not None
    assert claim.claim.hypothesis == "An ISMS must establish the ISMS."
    assert claim.status == "supported"
    assert nli.calls[0][0][1] == claim.claim.hypothesis


def test_long_evidence_uses_and_exposes_a_focused_window(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    long_evidence = (
        "Acme Corporation publishes software for many customers. "
        + "The company sells products in many countries around the world. " * 8
        + "Acme Corporation was founded in 2015."
    )
    nli = _PremiseAwareNliModel(["founded in 2015"])
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger(),
        matcher=LexicalEvidenceMatcher(),
        nli_model=nli,
    )

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=long_evidence, evidence_id="e1")],
    )

    claim = result.claims[0]
    evidence = claim.matched_evidence[0]
    assert claim.status == "supported"
    assert evidence.premise_strategy == "focused_window"
    assert evidence.truncated is True
    assert evidence.premise_text in long_evidence
    assert "founded in 2015" in evidence.premise_text
    assert evidence.evidence_char_length == len(long_evidence)
    assert evidence.premise_char_length == len(evidence.premise_text)
    assert evidence.window_start is not None
    assert evidence.window_end is not None
    assert nli.calls[0][0][0] == evidence.premise_text


def test_full_evidence_is_used_when_it_fits_the_nli_limit(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence_text = (
        "Acme Corporation was founded in 2015. It publishes software."
    )
    engine = _engine(settings, query_language_detector)

    result = engine.verify(
        answer,
        [VerificationEvidenceInput(text=evidence_text, evidence_id="e1")],
    )

    evidence = result.claims[0].matched_evidence[0]
    assert evidence.premise_strategy == "full_evidence"
    assert evidence.premise_text == evidence_text
    assert evidence.truncated is False
    assert evidence.window_start == 0
    assert evidence.window_end == len(evidence_text)


def test_evidence_limit_is_reported_as_missing_pairs_not_silently_dropped(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Acme Corporation sells software.", evidence_id="e2"
        ),
    ]
    engine = _engine(
        settings.model_copy(
            update={"verification_max_evidence_per_claim": 1}
        ),
        query_language_detector,
    )

    result = engine.verify(
        "Acme Corporation was founded in 2015.", evidence
    )

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.complete is False
    assert claim.pairs_expected == 2
    assert claim.pairs_evaluated == 1
    assert len(claim.matched_evidence) == 2
    assert claim.matched_evidence[1].evaluated is False
    assert "evidence limit" in claim.matched_evidence[1].explanation
    assert result.summary.pairs_missing == 1
    assert result.summary.claims_complete == 0
    assert result.summary.claims_incomplete == 1


def test_later_relevant_sentence_is_selected_over_earlier_text(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = (
        "Acme Corporation publishes software. "
        "The company was incorporated in 2004. "
        "Acme Corporation was founded in 2015 in Springfield."
    )
    nli = _PremiseAwareNliModel(["founded in 2015"])
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger(),
        matcher=LexicalEvidenceMatcher(),
        nli_model=nli,
    )

    result = engine.verify(answer, _evidence(evidence))

    assert result.claims[0].status == "supported"
    assert any(
        "founded in 2015" in premise for premise, _ in nli.calls[0]
    )


def test_second_relevant_chunk_is_evaluated_when_ranker_prefers_another(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "The ISMS preserves confidentiality, integrity, and availability."
    evidence = [
        VerificationEvidenceInput(
            text="The ISMS preserves safeguards for physical assets.",
            evidence_id="e1",
            filename="iso27001.pdf",
            page_number=1,
        ),
        VerificationEvidenceInput(
            text=(
                "The information security management system preserves the "
                "confidentiality, integrity and availability of information."
            ),
            evidence_id="e2",
            filename="iso27001.pdf",
            page_number=5,
        ),
    ]
    nli = _PremiseAwareNliModel(["confidentiality", "availability"])
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9, 0.2]),
    )

    result = engine.verify(answer, evidence)

    assert result.claims[0].status == "supported"
    assert {item.evidence.evidence_id for item in result.claims[0].matched_evidence} == {
        "e1",
        "e2",
    }
    assert any(
        "confidentiality" in premise
        for call in nli.calls
        for premise, _ in call
    )


def test_neutral_label_is_reported_as_neutral_not_contradicted(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    nli = FakeNliModel(default=prediction("neutral", 0.9))
    engine = _engine(settings, query_language_detector, nli_model=nli)

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation develops software."),
    )

    claim = result.claims[0]
    assert claim.status == "neutral"
    assert claim.label == "neutral"
    assert result.summary.neutral == 1
    assert result.summary.contradicted == 0


def test_neutral_window_does_not_override_a_supporting_window(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = (
        "Acme Corporation was founded in 2015. "
        "Acme Corporation publishes software. "
        "Acme Corporation sells hardware."
    )
    nli = _PremiseAwareNliModel(["founded in 2015"], neutral_confidence=0.99)
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([1.0, 1.0, 1.0]),
    )

    result = engine.verify(answer, _evidence(evidence))

    claim = result.claims[0]
    assert claim.status == "supported"
    assert claim.label == "entailment"
    assert claim.confidence == pytest.approx(0.95)
    assert claim.probabilities["neutral"] < 0.9


def test_contradiction_window_is_not_overridden_by_neutral_window(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = (
        "Acme Corporation was founded in 1920. "
        "Acme Corporation publishes software. "
        "Acme Corporation sells hardware."
    )
    nli = _PremiseContradictionNliModel("founded in 1920")
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([1.0, 1.0, 1.0]),
    )

    result = engine.verify(answer, _evidence(evidence))

    claim = result.claims[0]
    assert claim.status == "contradicted"
    assert claim.label == "contradiction"
    assert claim.confidence == pytest.approx(0.95)
    assert result.summary.contradicted == 1
    assert result.summary.supported == 0


def test_relevant_third_chunk_is_evaluated_when_similarity_is_highest(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "The ISMS preserves confidentiality, integrity, and availability."
    evidence = [
        VerificationEvidenceInput(
            text="The ISMS preserves safeguards for physical assets.",
            evidence_id="e1",
            filename="iso27001.pdf",
            page_number=1,
        ),
        VerificationEvidenceInput(
            text="The ISMS protects paper records in storage.",
            evidence_id="e2",
            filename="iso27001.pdf",
            page_number=2,
        ),
        VerificationEvidenceInput(
            text=(
                "The information security management system preserves the "
                "confidentiality, integrity and availability of information."
            ),
            evidence_id="e3",
            filename="iso27001.pdf",
            page_number=5,
        ),
    ]
    nli = _PremiseAwareNliModel(["confidentiality", "availability"])
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9, 0.2, 0.95]),
    )

    result = engine.verify(answer, evidence)

    assert result.claims[0].status == "supported"
    assert "e3" in {
        item.evidence.evidence_id for item in result.claims[0].matched_evidence
    }
    assert any(
        "confidentiality" in premise
        for call in nli.calls
        for premise, _ in call
    )


def test_multilingual_supported_claim_is_verified(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "सूचना सुरक्षा प्रबंधन प्रणाली गोपनीयता, अखंडता और उपलब्धता को बनाए रखती है।"
    engine = _engine(settings, query_language_detector)

    result = engine.verify(answer, _evidence(answer))

    assert result.summary.claims_total == 1
    assert result.claims[0].status == "supported"
    assert result.claims[0].claim.language.code == "hi"


def test_context_around_matched_sentence_is_sent_to_nli(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = (
        "Acme Corporation publishes software. "
        "The company was incorporated in Springfield. "
        "Acme Corporation was founded in 2015 in Springfield."
    )
    nli = _PremiseAwareNliModel(["founded in 2015"])
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger(),
        matcher=LexicalEvidenceMatcher(),
        nli_model=nli,
    )

    result = engine.verify(answer, _evidence(evidence))

    assert result.claims[0].status == "supported"
    premise = next(
        item
        for item in nli.calls[0]
        if "founded in 2015" in item[0]
    )[0]
    assert "incorporated in Springfield" in premise


def test_complementary_evidence_is_not_combined_into_a_single_verdict(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "The policy requires annual reviews and daily audits."
    evidence = [
        VerificationEvidenceInput(
            text="The policy requires annual reviews.",
            evidence_id="e1",
            filename="policy.txt",
            page_number=1,
        ),
        VerificationEvidenceInput(
            text="The policy requires daily audits.",
            evidence_id="e2",
            filename="policy.txt",
            page_number=2,
        ),
    ]
    nli = _ComplementaryNliModel()
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9, 0.8]),
    )

    result = engine.verify(answer, evidence)

    claim = result.claims[0]
    assert claim.status == "neutral"
    assert claim.aggregation == "no_thresholded_label"
    assert claim.composite_premise is None
    assert claim.pairs_expected == 2
    assert claim.pairs_evaluated == 2
    assert len(claim.matched_evidence) == 2
    assert all(
        item.counts_for_verdict is False
        for item in claim.matched_evidence
    )


def test_conflicting_support_and_contradiction_is_preserved(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.",
            evidence_id="e1",
            filename="history.txt",
            page_number=1,
        ),
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 1920.",
            evidence_id="e2",
            filename="history.txt",
            page_number=2,
        ),
    ]
    nli = _ConflictingNliModel()
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.95, 0.93]),
    )

    result = engine.verify(answer, evidence)

    claim = result.claims[0]
    assert claim.status == "contradicted"
    assert claim.conflict is True
    assert claim.support_confidence == pytest.approx(0.95)
    assert claim.contradiction_confidence == pytest.approx(0.93)
    assert claim.aggregation == "conflict_contradiction_precedence"
    assert result.summary.contradicted == 1
    assert result.summary.neutral == 0
    assert (
        result.summary.supported
        + result.summary.contradicted
        + result.summary.neutral
        + result.summary.unevaluable
        + result.summary.verification_error
        == result.summary.claims_total
    )


def test_lower_ranked_third_chunk_is_still_evaluated(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Acme Corporation was founded in 2015."
    evidence = [
        VerificationEvidenceInput(text="Acme publishes software.", evidence_id="e1"),
        VerificationEvidenceInput(text="Acme sells hardware.", evidence_id="e2"),
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e3"
        ),
    ]
    nli = _PremiseAwareNliModel(["founded in 2015"])
    engine = _engine(
        settings.model_copy(
            update={
                "verification_max_evidence_per_claim": 3,
            }
        ),
        query_language_detector,
        nli_model=nli,
        matcher=FixedMatcher([0.9, 0.8, 0.7]),
    )

    result = engine.verify(answer, evidence)

    assert result.claims[0].status == "supported"
    assert "e3" in {
        item.evidence.evidence_id
        for item in result.claims[0].matched_evidence
    }


def test_prediction_count_mismatch_returns_error_result(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020."
    )
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Beta Corporation was founded in 2020.", evidence_id="e2"
        ),
    ]
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=_ShortPredictionNliModel(),
    )

    result = engine.verify(answer, evidence)

    assert result.status == "error"
    assert "different number of predictions" in (result.error or "")
    assert result.timings.nli_ms > 0.0


def test_nli_failure_returns_an_error_result_without_raising(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = _engine(
        settings,
        query_language_detector,
        nli_model=FakeNliModel(fail=True),
    )

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation was founded in 2015."),
    )

    assert result.status == "error"
    assert result.error


def test_disabled_engine_returns_disabled_result(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    result = DisabledVerificationEngine().verify("A factual answer.", [])

    assert result.status == "disabled"
    assert result.enabled is False


def test_model_loading_time_does_not_consume_the_nli_deadline(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = _engine(
        settings.model_copy(update={"verification_deadline_seconds": 0.05}),
        query_language_detector,
        ner_tagger=_SlowNerTagger(),
    )

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation was founded in 2015."),
    )

    assert result.status == "completed"
    assert result.claims[0].status == "supported"
    assert result.summary.pairs_evaluated == 1


def test_expired_request_deadline_does_not_skip_claim_evidence_pairs(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    nli = FakeNliModel(default=prediction("entailment", 0.95))
    engine = _engine(settings, query_language_detector, nli_model=nli)

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation was founded in 2015."),
        deadline=time.monotonic() - 1.0,
    )

    assert result.status == "completed"
    assert result.claims[0].status == "supported"
    assert result.summary.claims_checked == 1
    assert result.summary.verification_error == 0
    assert result.summary.pairs_evaluated == result.summary.pairs_expected == 1
    assert len(nli.calls) == 1


def test_cancelled_request_marks_claims_verification_error_and_returns_partial(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    cancel_event = threading.Event()
    cancel_event.set()
    engine = _engine(settings, query_language_detector)

    result = engine.verify(
        "Acme Corporation was founded in 2015.",
        _evidence("Acme Corporation was founded in 2015."),
        cancel_event=cancel_event,
    )

    assert result.status == "partial"
    assert result.claims[0].status == "neutral"
    assert result.claims[0].diagnostic_status == "verification_error"
    assert result.summary.verification_error == 1
    assert "disconnected" in (result.claims[0].explanation or "").lower()


def test_claim_limit_prioritizes_numbers_and_reports_unchecked_claims(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "Acme Corporation has offices in Paris. "
        "The policy requires 30 days notice. "
        "Beta Corporation publishes a newsletter."
    )
    evidence = [
        VerificationEvidenceInput(
            text=answer,
            evidence_id="e1",
            filename="acme.txt",
        )
    ]
    engine = _engine(
        settings.model_copy(
            update={
                "verification_max_claims": 2,
                "verification_max_extracted_claims": 10,
                "verification_max_nli_pairs": 10,
            }
        ),
        query_language_detector,
    )

    result = engine.verify(answer, evidence)

    assert result.summary.claims_total == 3
    assert result.summary.claims_checked == 2
    assert result.summary.verification_error == 1
    assert result.status == "partial"
    checked_texts = [
        item.claim.claim
        for item in result.claims
        if item.aggregation != "not_checked"
    ]
    skipped_texts = [
        item.claim.claim
        for item in result.claims
        if item.aggregation == "not_checked"
    ]
    assert any("30 days" in text for text in checked_texts)
    assert len(skipped_texts) == 1
    assert "30 days" not in skipped_texts[0]
    assert any("left unchecked" in warning for warning in result.warnings)


def test_pair_budget_leaves_remaining_claim_verification_error(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020."
    )
    evidence = [
        VerificationEvidenceInput(
            text="Acme Corporation was founded in 2015.", evidence_id="e1"
        ),
        VerificationEvidenceInput(
            text="Beta Corporation was founded in 2020.", evidence_id="e2"
        ),
    ]
    engine = _engine(
        settings.model_copy(
            update={"verification_max_nli_pairs": 1}
        ),
        query_language_detector,
    )

    result = engine.verify(answer, evidence)

    assert result.status == "partial"
    statuses = [item.status for item in result.claims]
    assert statuses == ["supported", "neutral"]
    assert result.claims[1].diagnostic_status == "verification_error"
    assert result.summary.claims_checked == 1
    assert result.summary.verification_error == 1
    assert result.summary.pairs_expected == 4
    assert result.summary.pairs_evaluated == 1
    assert result.summary.pairs_missing == 3
    assert result.summary.claims_complete == 0
    assert result.summary.claims_incomplete == 2


def test_candidate_selection_prefers_claim_coverage_over_similarity():
    evidence = [
        VerificationEvidenceInput(
            text=(
                "An unrelated sentence has a high semantic score. "
                "The organization monitors and evaluates information "
                "security performance."
            ),
            evidence_id="e1",
        )
    ]
    candidates = _candidate_windows(evidence)
    matching = MatchingResult(
        matches=(
            (
                EvidenceMatch(evidence_index=0, similarity=0.99),
                EvidenceMatch(evidence_index=1, similarity=0.10),
            ),
        ),
        method="test",
    )

    best = _best_candidates_by_evidence(
        matching,
        candidates,
        ["The organization monitors and evaluates information security performance."],
        1,
    )

    chosen = best[0][0][0]
    assert chosen.span.text == (
        "The organization monitors and evaluates information security performance."
    )


def test_claim_term_coverage_handles_common_morphological_variants():
    claim = (
        "Assess risks, monitor controls, evaluate performance, and retain "
        "documentation."
    )
    premise = (
        "The organization shall assess risks, monitor controls, evaluate "
        "performance, and retain documentation."
    )

    assert _claim_term_coverage(claim, premise) == 1.0
    assert _claim_term_coverage(
        "Risk assessments and treatment plans",
        "The risk assessment and treatment planning process",
    ) >= 0.5

