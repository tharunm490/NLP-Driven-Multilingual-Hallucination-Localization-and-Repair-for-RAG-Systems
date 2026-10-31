"""Exhaustive claim-by-evidence verification orchestration."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

from app.core.config import Settings
from app.core.errors import NliError, VerificationError
from app.schemas.verification import (
    ClaimVerificationResult,
    ComparisonRelevance,
    DetectedLanguage,
    EvidenceQuality,
    EvidenceReference,
    EvidenceVerification,
    ExtractedClaim,
    NliLabel,
    NliWindowPrediction,
    PremiseStrategy,
    TextSpan,
    VerificationEvidenceInput,
    VerificationResult,
    VerificationSummary,
    VerificationTimings,
)
from app.services.generator import Generator
from app.services.query_language_detection import (
    LanguageDetection,
    QueryLanguageDetector,
)
from app.services.verification.claims import ClaimExtractor, build_claim_extractor
from app.services.verification.matching import (
    Embedder,
    EvidenceMatcher,
    MatchingResult,
    build_evidence_matcher,
)
from app.services.verification.ner import NerTagger, build_ner_tagger
from app.services.verification.nli import (
    NliModel,
    NliPrediction,
    PremiseWindow,
    build_context_windows,
    build_nli_model,
)
from app.services.verification.text import content_terms, split_sentences

logger = logging.getLogger(__name__)

_IMPORTANT_CLAIM_PATTERNS = (
    re.compile(r"\b\d{1,4}(?:[.,]\d+)?%?\b"),
    re.compile(
        r"\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not|no|never|none|neither|nor|without|cannot|can't|"
        r"must\s+not|shall\s+not|does\s+not|do\s+not|is\s+not|are\s+not)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all|any|each|every|only|at\s+least|at\s+most|more\s+than|"
        r"fewer\s+than|less\s+than|exactly|approximately|about|up\s+to)\b",
        re.IGNORECASE,
    ),
)

_DOT_LEADER = re.compile(r"\.{4,}")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)

_RELEVANCE_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her his how if in into is it its may might must not of on or our should
    that the their them there these they this those to was we were what when
    where which who will with would you your
    का की के को और में से है हैं
    """.split()
)

_RELATIVE_SUPPORT_COVERAGE = 0.75


@runtime_checkable
class VerificationEngine(Protocol):
    enabled: bool

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VerificationResult: ...


class DisabledVerificationEngine:
    enabled = False

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            status="disabled",
            enabled=False,
            warnings=["Claim-level verification is disabled by configuration."],
        )


@dataclass(frozen=True)
class _CandidateWindow:
    evidence_index: int
    span: TextSpan


@dataclass(frozen=True)
class _PairPlan:
    evidence_index: int
    rank: int
    similarity: float
    hypothesis_text: str
    premise_text: str
    premise_strategy: PremiseStrategy
    evidence_char_length: int
    premise_char_length: int
    truncated: bool
    window_start: int | None
    window_end: int | None
    claim_term_coverage: float
    relevance: ComparisonRelevance
    evidence_quality: EvidenceQuality | None
    evidence_quality_reason: str | None
    full_pair_token_count: int | None
    input_pair_token_count: int | None
    max_pair_tokens: int | None
    unavailable_reason: str | None = None


@dataclass
class _PairOutcome:
    plan: _PairPlan
    prediction: NliPrediction | None = None
    error: str | None = None
    skip_reason: str | None = None
    verdict_exclusion_reason: str | None = None
    counts_for_verdict: bool = False
    attempts: int = 0


@dataclass
class _ClaimState:
    claim: ExtractedClaim
    pairs: list[_PairOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unchecked_reason: str | None = None
    error: str | None = None


class ClaimVerificationEngine:
    enabled = True

    def __init__(
        self,
        settings: Settings,
        detector: QueryLanguageDetector,
        claim_extractor: ClaimExtractor,
        ner_tagger: NerTagger,
        matcher: EvidenceMatcher,
        nli_model: NliModel,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._claim_extractor = claim_extractor
        self._ner = ner_tagger
        self._matcher = matcher
        self._nli = nli_model

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VerificationResult:
        started = time.perf_counter()
        timings = {
            "claim_extraction_ms": 0.0,
            "evidence_selection_ms": 0.0,
            "nli_ms": 0.0,
            "aggregation_ms": 0.0,
        }
        try:
            result = self._verify(
                answer,
                evidence,
                timings,
                deadline=deadline,
                cancel_event=cancel_event,
            )
        except VerificationError as exc:
            logger.warning("Claim verification failed: %s", exc)
            result = self._error_result(str(exc), timings)
        except Exception as exc:
            logger.exception("Unexpected claim verification failure")
            result = self._error_result(
                f"Claim verification failed: {exc}", timings
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result.model_copy(update={"verification_ms": round(elapsed_ms, 3)})

    def _verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        timings: dict[str, float],
        *,
        deadline: float | None,
        cancel_event: threading.Event | None,
    ) -> VerificationResult:
        warnings: list[str] = []
        extraction_started = time.perf_counter()
        answer_language = self._detect(answer, fallback=None)
        extraction = self._claim_extractor.extract(answer)
        warnings.extend(extraction.warnings)

        claims = [
            self._enrich_claim(claim, answer_language, warnings)
            for claim in extraction.claims
        ]
        timings["claim_extraction_ms"] = (
            time.perf_counter() - extraction_started
        ) * 1000
        evidence_total = len(evidence)
        if not claims:
            warnings.append("No factual claims were extracted from the answer.")
            return VerificationResult(
                status="completed",
                enabled=True,
                answer_language=answer_language,
                claim_extraction_method=extraction.method,
                nli_model=self._nli.model_name,
                ner_model=self._ner.model_name,
                summary=VerificationSummary(
                    valid_claims_extracted=0,
                    invalid_fragments_skipped=len(extraction.skipped),
                    claims_location_unresolved=len(extraction.unresolved),
                    valid_claims_evaluated=0,
                    evidence_passed=evidence_total,
                    evidence_total=evidence_total,
                ),
                timings=VerificationTimings(**timings),
                warnings=_dedupe(warnings),
            )

        evaluable_claims = [claim for claim in claims if claim.evaluable]
        unevaluable_claims = [claim for claim in claims if not claim.evaluable]
        max_claims = max(1, self._settings.verification_max_claims)
        selected_claims, skipped_claims = _prioritize_claims(
            evaluable_claims, max_claims
        )
        states = [
            _ClaimState(claim=claim, warnings=list(claim.warnings))
            for claim in selected_claims
        ]
        if skipped_claims:
            warnings.append(
                f"Verification was capped at {max_claims} claims; "
                f"{len(skipped_claims)} claim(s) were left unchecked. Claims "
                "with numbers, dates, negation, or quantifiers were prioritized."
            )

        selection_started = time.perf_counter()
        candidates = _candidate_windows(evidence)
        if candidates and selected_claims:
            matching = self._matcher.match_many(
                [claim.hypothesis or claim.claim for claim in selected_claims],
                [candidate.span.text for candidate in candidates],
            )
        else:
            matching = MatchingResult(
                matches=tuple(() for _ in selected_claims),
                method="no_windows",
            )
        warnings.extend(matching.warnings)
        best_candidates = _best_candidates_by_evidence(
            matching,
            candidates,
            [claim.claim for claim in selected_claims],
            evidence_total,
        )
        for state, best in zip(states, best_candidates, strict=True):
            state.pairs, pair_warnings = self._build_pairs(
                state,
                best,
                evidence,
            )
            state.warnings.extend(pair_warnings)
        timings["evidence_selection_ms"] = (
            time.perf_counter() - selection_started
        ) * 1000

        nli_started = time.perf_counter()
        fatal_error = self._evaluate_pairs(
            states,
            cancel_event,
        )
        timings["nli_ms"] = (time.perf_counter() - nli_started) * 1000
        if fatal_error:
            warnings.append(fatal_error)

        aggregation_started = time.perf_counter()
        results = self._finalize_results(states, evidence)
        results.extend(
            self._unevaluable_result(claim) for claim in unevaluable_claims
        )
        results.extend(
            self._unchecked_result(
                claim,
                "The claim extraction limit was reached before this claim "
                "could be verified.",
                evidence,
            )
            for claim in skipped_claims
        )
        result_by_id = {
            item.claim.claim_id: item for item in results
        }
        results = [
            result_by_id[claim.claim_id]
            for claim in claims
            if claim.claim_id in result_by_id
        ]
        if unevaluable_claims:
            warnings.append(
                f"{len(unevaluable_claims)} extracted fragment(s) were skipped "
                "because they were not independently evaluable factual claims."
            )
        summary = _summarize(
            results,
            evidence_total=evidence_total,
            invalid_fragments_skipped=len(extraction.skipped),
            claims_location_unresolved=len(extraction.unresolved),
        )
        timings["aggregation_ms"] = (
            time.perf_counter() - aggregation_started
        ) * 1000
        status = _run_status(results, summary)
        run_error = next(
            (
                claim.error
                for claim in results
                if claim.diagnostic_status == "verification_error"
                and claim.error
            ),
            None,
        )
        if summary.pairs_missing > 0:
            warnings.append(
                f"{summary.pairs_missing} of {summary.pairs_expected} expected "
                "claim-evidence comparison(s) were not evaluated. Verdicts for "
                "affected claims use only the pairs with usable NLI results."
            )
        return VerificationResult(
            status=status,
            enabled=True,
            answer_language=answer_language,
            claim_extraction_method=extraction.method,
            nli_model=self._nli.model_name,
            ner_model=self._ner.model_name,
            claims=results,
            summary=summary,
            timings=VerificationTimings(
                **{key: round(value, 3) for key, value in timings.items()}
            ),
            warnings=_dedupe(warnings),
            error=run_error,
        )

    def _enrich_claim(
        self,
        claim: ExtractedClaim,
        answer_language: DetectedLanguage,
        warnings: list[str],
    ) -> ExtractedClaim:
        language = self._detect(claim.claim, fallback=answer_language)
        claim_warnings = list(claim.warnings)
        if claim.location_status != "verified":
            claim_warnings.append(
                "Named-entity recognition was skipped because the claim's "
                "location in the answer could not be verified."
            )
            return claim.model_copy(
                update={
                    "language": language,
                    "entities": [],
                    "warnings": claim_warnings,
                }
            )
        entities = []
        try:
            ner_result = self._ner.tag(claim.claim)
            entities = [
                entity.model_copy(
                    update={
                        "start": claim.claim_start + entity.start,
                        "end": claim.claim_start + entity.end,
                    }
                )
                for entity in ner_result.entities
                if 0 <= entity.start < entity.end <= len(claim.claim)
            ]
            claim_warnings.extend(ner_result.warnings)
        except Exception as exc:
            logger.warning("NER failed for claim %s: %s", claim.claim_id, exc)
            claim_warnings.append(f"Named-entity recognition failed: {exc}")
        return claim.model_copy(
            update={
                "language": language,
                "entities": entities,
                "warnings": claim_warnings,
            }
        )

    def _build_pairs(
        self,
        state: _ClaimState,
        best_candidates: dict[int, tuple[_CandidateWindow, float]],
        evidence: Sequence[VerificationEvidenceInput],
    ) -> tuple[list[_PairOutcome], list[str]]:
        warnings: list[str] = []
        max_evidence = max(
            1, self._settings.verification_max_evidence_per_claim
        )
        outcomes: list[_PairOutcome] = []
        for evidence_index, source in enumerate(evidence):
            choice = best_candidates.get(evidence_index)
            best = choice[0] if choice is not None else None
            similarity = choice[1] if choice is not None else 0.0
            hypothesis = state.claim.hypothesis or state.claim.claim
            if evidence_index >= max_evidence:
                outcomes.append(
                    _PairOutcome(
                        plan=_unavailable_plan(
                            evidence_index,
                            source,
                            similarity,
                            (
                                "The configured per-claim evidence limit "
                                f"({max_evidence}) was reached."
                            ),
                            hypothesis=hypothesis,
                        )
                    )
                )
                continue

            (
                premise,
                strategy,
                start,
                end,
                truncated,
                full_pair_tokens,
                premise_warnings,
            ) = self._premise_for_pair(source, best, hypothesis)
            warnings.extend(premise_warnings)
            if not premise:
                outcomes.append(
                    _PairOutcome(
                        plan=_unavailable_plan(
                            evidence_index,
                            source,
                            similarity,
                            "No NLI-sized evidence premise could be built.",
                            hypothesis=hypothesis,
                        )
                    )
                )
                continue

            quality, quality_reason = _evidence_quality(premise)
            coverage = _claim_term_coverage(state.claim.claim, premise)
            relevance = _relevance_label(
                coverage,
                quality,
                self._settings.verification_min_evidence_relevance,
            )
            input_pair_tokens = self._nli.count_pair_tokens(premise, hypothesis)
            outcomes.append(
                _PairOutcome(
                    plan=_PairPlan(
                        evidence_index=evidence_index,
                        rank=evidence_index + 1,
                        similarity=similarity,
                        hypothesis_text=hypothesis,
                        premise_text=premise,
                        premise_strategy=strategy,
                        evidence_char_length=len(source.text),
                        premise_char_length=len(premise),
                        truncated=truncated,
                        window_start=start,
                        window_end=end,
                        claim_term_coverage=coverage,
                        relevance=relevance,
                        evidence_quality=quality,
                        evidence_quality_reason=quality_reason,
                        full_pair_token_count=full_pair_tokens,
                        input_pair_token_count=input_pair_tokens,
                        max_pair_tokens=self._nli.max_pair_tokens,
                    )
                )
            )
        return outcomes, _dedupe(warnings)

    def _premise_for_pair(
        self,
        source: VerificationEvidenceInput,
        best: _CandidateWindow | None,
        hypothesis: str,
    ) -> tuple[
        str,
        PremiseStrategy,
        int | None,
        int | None,
        bool,
        int,
        list[str],
    ]:
        full_tokens = self._nli.count_pair_tokens(source.text, hypothesis)
        if full_tokens <= self._nli.max_pair_tokens:
            return (
                source.text,
                "full_evidence",
                0,
                len(source.text),
                False,
                full_tokens,
                [],
            )
        if best is None:
            return "", "unavailable", None, None, False, full_tokens, []
        windows, warnings = build_context_windows(
            source.text,
            best.span.start,
            best.span.end,
            hypothesis,
            self._nli,
            max_tokens=self._settings.verification_primary_window_tokens,
        )
        if not windows:
            return "", "unavailable", None, None, False, full_tokens, warnings
        window = windows[0]
        return (
            window.span.text,
            "focused_window",
            window.span.start,
            window.span.end,
            True,
            full_tokens,
            warnings,
        )

    def _evaluate_pairs(
        self,
        states: Sequence[_ClaimState],
        cancel_event: threading.Event | None,
    ) -> str | None:
        pending: list[tuple[int, int]] = []
        for state_index, state in enumerate(states):
            for pair_index, outcome in enumerate(state.pairs):
                if outcome.plan.unavailable_reason:
                    outcome.skip_reason = outcome.plan.unavailable_reason
                else:
                    pending.append((state_index, pair_index))
        if not pending:
            return None

        deadline_reason = _cancel_reason(cancel_event)
        if deadline_reason is not None:
            self._mark_unprocessed(states, pending, deadline_reason)
            return deadline_reason

        pair_budget = max(1, self._settings.verification_max_nli_pairs)
        if len(pending) > pair_budget:
            overflow = pending[pair_budget:]
            reason = (
                "The configured NLI pair budget was exhausted before this "
                "comparison was evaluated."
            )
            self._mark_unprocessed(states, overflow, reason)
            pending = pending[:pair_budget]

        pairs = [
            (
                states[state_index].pairs[pair_index].plan.premise_text,
                states[state_index].pairs[pair_index].plan.hypothesis_text,
            )
            for state_index, pair_index in pending
        ]
        predictions, attempts, message = self._predict_with_retries(pairs)
        if predictions is None:
            logger.warning(
                "Batch NLI inference failed after %d attempt(s); retrying "
                "claim-evidence pairs individually: %s",
                attempts,
                message,
            )
            return self._evaluate_pairs_individually(
                states,
                pending,
                cancel_event,
                initial_message=message,
            )

        for (state_index, pair_index), prediction in zip(pending, predictions):
            outcome = states[state_index].pairs[pair_index]
            outcome.prediction = prediction
            outcome.attempts = attempts
        return None

    def _evaluate_pairs_individually(
        self,
        states: Sequence[_ClaimState],
        pending: Sequence[tuple[int, int]],
        cancel_event: threading.Event | None,
        *,
        initial_message: str,
    ) -> str | None:
        last_message = initial_message
        for offset, (state_index, pair_index) in enumerate(pending):
            stop_reason = _cancel_reason(cancel_event)
            if stop_reason is not None:
                self._mark_unprocessed(
                    states, pending[offset:], stop_reason
                )
                return stop_reason
            outcome = states[state_index].pairs[pair_index]
            predictions, attempts, message = self._predict_with_retries(
                [(outcome.plan.premise_text, outcome.plan.hypothesis_text)]
            )
            outcome.attempts = attempts
            if predictions:
                outcome.prediction = predictions[0]
                continue
            last_message = message or "NLI inference failed."
            self._mark_errors(states, [(state_index, pair_index)], last_message)
        return last_message

    def _predict_with_retries(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[list[NliPrediction] | None, int, str | None]:
        attempts = max(1, self._settings.verification_nli_engine_retries + 1)
        last_message: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                predictions = self._nli.predict(pairs)
            except (NliError, VerificationError) as exc:
                last_message = f"NLI inference failed: {exc}"
            except Exception as exc:
                last_message = f"Unexpected NLI failure: {exc}"
            else:
                if len(predictions) == len(pairs):
                    return predictions, attempt, None
                last_message = (
                    "The NLI model returned a different number of "
                    "predictions than claim-evidence pairs."
                )
            logger.warning(
                "NLI attempt %d/%d failed for %d pair(s): %s",
                attempt,
                attempts,
                len(pairs),
                last_message,
            )
        return None, attempts, last_message

    def _mark_errors(
        self,
        states: Sequence[_ClaimState],
        chunk: Sequence[tuple[int, int]],
        message: str,
    ) -> None:
        for state_index, pair_index in chunk:
            state = states[state_index]
            outcome = state.pairs[pair_index]
            outcome.error = message
            state.error = message
            state.warnings.append(message)

    def _mark_unprocessed(
        self,
        states: Sequence[_ClaimState],
        pending: Sequence[tuple[int, int]],
        reason: str,
    ) -> None:
        for state_index, pair_index in pending:
            outcome = states[state_index].pairs[pair_index]
            outcome.skip_reason = reason

    def _finalize_results(
        self,
        states: Sequence[_ClaimState],
        evidence: Sequence[VerificationEvidenceInput],
    ) -> list[ClaimVerificationResult]:
        results: list[ClaimVerificationResult] = []
        for state in states:
            if state.unchecked_reason:
                results.append(
                    self._unchecked_result(
                        state.claim, state.unchecked_reason, evidence
                    )
                )
                continue
            results.append(self._claim_result(state, evidence))
        return results

    def _claim_result(
        self,
        state: _ClaimState,
        evidence: Sequence[VerificationEvidenceInput],
    ) -> ClaimVerificationResult:
        entailment_threshold = (
            self._settings.verification_nli_entailment_threshold
        )
        contradiction_threshold = (
            self._settings.verification_nli_contradiction_threshold
        )
        rows: list[EvidenceVerification] = []
        supporting: list[_PairOutcome] = []
        contradicting: list[_PairOutcome] = []
        evaluated: list[_PairOutcome] = []
        ignored_contradictions = 0
        ignored_weak_supports = 0
        for outcome in state.pairs:
            prediction = outcome.prediction
            if prediction is not None:
                evaluated.append(outcome)
                if prediction.label == "contradiction":
                    if (
                        outcome.plan.relevance != "relevant"
                        or prediction.confidence < contradiction_threshold
                    ):
                        ignored_contradictions += 1
                outcome.counts_for_verdict = _counts_for_verdict(
                    outcome,
                    entailment_threshold,
                    contradiction_threshold,
                )

        support_candidates = [
            outcome
            for outcome in evaluated
            if outcome.counts_for_verdict
            and outcome.prediction is not None
            and outcome.prediction.label == "entailment"
        ]
        if support_candidates:
            best_support_coverage = max(
                outcome.plan.claim_term_coverage
                for outcome in support_candidates
            )
            support_coverage_floor = max(
                self._settings.verification_min_evidence_relevance,
                best_support_coverage * _RELATIVE_SUPPORT_COVERAGE,
            )
            for outcome in support_candidates:
                if outcome.plan.claim_term_coverage < support_coverage_floor:
                    outcome.counts_for_verdict = False
                    outcome.verdict_exclusion_reason = (
                        "its claim-term coverage is materially weaker than "
                        "another supporting passage for the same claim "
                        f"({outcome.plan.claim_term_coverage:.3f} < "
                        f"{support_coverage_floor:.3f})"
                    )
                    ignored_weak_supports += 1

        for outcome in evaluated:
            prediction = outcome.prediction
            if prediction is None:
                continue
            if outcome.counts_for_verdict:
                if prediction.label == "entailment":
                    supporting.append(outcome)
                elif prediction.label == "contradiction":
                    contradicting.append(outcome)
        for outcome in state.pairs:
            rows.append(
                _evidence_result(
                    outcome,
                    evidence[outcome.plan.evidence_index],
                    state.claim.claim_id,
                    entailment_threshold,
                    contradiction_threshold,
                )
            )

        expected = len(evidence)
        evaluated_count = len(evaluated)
        complete = evaluated_count == expected
        support_confidence = (
            max(item.prediction.confidence for item in supporting)
            if supporting
            else None
        )
        contradiction_confidence = (
            max(item.prediction.confidence for item in contradicting)
            if contradicting
            else None
        )
        conflict = bool(supporting and contradicting)
        chosen: _PairOutcome | None
        if contradicting:
            chosen = max(
                contradicting,
                key=lambda item: item.prediction.confidence,  # type: ignore[union-attr]
            )
            status = "contradicted"
            aggregation = (
                "conflict_contradiction_precedence"
                if conflict
                else "thresholded_contradiction"
            )
        elif supporting:
            chosen = max(
                supporting,
                key=lambda item: item.prediction.confidence,  # type: ignore[union-attr]
            )
            status = "supported"
            aggregation = "thresholded_entailment"
        else:
            chosen = (
                max(
                    evaluated,
                    key=lambda item: item.prediction.confidence,  # type: ignore[union-attr]
                )
                if evaluated
                else None
            )
            status = "neutral"
            aggregation = "no_thresholded_label"

        diagnostic_status: str | None = None
        incomplete_reason: str | None = None
        if expected > 0 and evaluated_count == 0:
            status = "neutral"
            diagnostic_status = "verification_error"
            aggregation = "no_usable_pair_result"
            incomplete_reason = (
                state.error
                or _incomplete_reason(state, evaluated_count, expected)
            )
        elif not complete:
            incomplete_reason = _incomplete_reason(
                state, evaluated_count, expected
            )

        claim_warnings = list(state.warnings)
        if ignored_contradictions > 0:
            claim_warnings.append(
                f"{ignored_contradictions} raw contradiction prediction(s) "
                "were not counted because the evidence was not relevant or "
                "substantive enough, or did not meet the contradiction "
                "threshold."
            )
        if ignored_weak_supports > 0:
            claim_warnings.append(
                f"{ignored_weak_supports} supporting comparison(s) were not "
                "counted because their claim-term coverage was materially "
                "weaker than the strongest supporting passage for this claim."
            )
        if conflict:
            claim_warnings.append(
                "Relevant evaluated evidence contains both support and "
                "contradiction; the claim is marked contradicted, and the "
                "conflict is retained in this result."
            )
        if status == "neutral" and not state.pairs:
            claim_warnings.append("No evidence chunks were provided.")
        if status == "neutral" and state.pairs:
            claim_warnings.append(
                "Relevant evidence was evaluated, but no comparison met the "
                "configured support or contradiction threshold. This is a "
                "neutral outcome, not a factual refutation."
            )
        if not complete and diagnostic_status is None:
            claim_warnings.append(
                incomplete_reason
                or "Some expected claim-evidence comparisons were not evaluated."
            )

        prediction = chosen.prediction if chosen is not None else None
        reference = (
            _to_reference(
                evidence[chosen.plan.evidence_index],
                chosen.plan.evidence_index,
            )
            if chosen is not None
            else None
        )
        return ClaimVerificationResult(
            claim=state.claim,
            status=status,
            diagnostic_status=diagnostic_status,
            label=prediction.label if prediction is not None else None,
            confidence=(
                prediction.confidence if prediction is not None else None
            ),
            probabilities=(
                dict(prediction.probabilities)
                if prediction is not None
                else {}
            ),
            logits=(
                list(prediction.logits)
                if prediction is not None and prediction.logits is not None
                else None
            ),
            support_confidence=support_confidence,
            contradiction_confidence=contradiction_confidence,
            conflict=conflict,
            composite_premise=None,
            explanation=_claim_explanation(
                status=status or diagnostic_status or "neutral",
                reference=reference,
                prediction=prediction,
                support_confidence=support_confidence,
                contradiction_confidence=contradiction_confidence,
                evaluated_count=evaluated_count,
                expected=expected,
                incomplete_reason=incomplete_reason,
                has_evidence=bool(evidence),
                conflict=conflict,
            ),
            pairs_expected=expected,
            pairs_evaluated=evaluated_count,
            pairs_retried=sum(
                outcome.attempts > 1 for outcome in state.pairs
            ),
            pairs_failed=sum(
                outcome.error is not None for outcome in state.pairs
            ),
            complete=complete,
            incomplete_reason=incomplete_reason,
            matched_evidence=rows,
            aggregation=aggregation,
            error=state.error,
            warnings=_dedupe(claim_warnings),
        )

    def _unevaluable_result(
        self,
        claim: ExtractedClaim,
    ) -> ClaimVerificationResult:
        reason = (
            claim.incomplete_reason
            or "The fragment was not an independently evaluable proposition."
        )
        return ClaimVerificationResult(
            claim=claim,
            status=None,
            diagnostic_status="unevaluable",
            explanation=(
                "The extracted fragment was preserved for audit but was not "
                f"sent to evidence matching or NLI: {reason}"
            ),
            pairs_expected=0,
            pairs_evaluated=0,
            complete=False,
            incomplete_reason=reason,
            matched_evidence=[],
            aggregation="not_evaluable",
            warnings=[reason],
        )

    def _unchecked_result(
        self,
        claim: ExtractedClaim,
        reason: str,
        evidence: Sequence[VerificationEvidenceInput],
    ) -> ClaimVerificationResult:
        rows = [
            EvidenceVerification(
                claim_id=claim.claim_id,
                hypothesis_text=claim.hypothesis or claim.claim,
                evidence=_to_reference(item, index),
                evidence_text=item.text,
                rank=index + 1,
                similarity=0.0,
                retrieval_similarity=item.retrieval_similarity,
                retrieval_score=item.retrieval_score,
                evaluated=False,
                relevance="unknown",
                premise_strategy="unavailable",
                evidence_char_length=len(item.text),
                max_pair_tokens=self._nli.max_pair_tokens,
                explanation=f"Not compared: {reason}",
                error=None,
                warnings=[reason],
            )
            for index, item in enumerate(evidence)
        ]
        return ClaimVerificationResult(
            claim=claim,
            status="neutral",
            diagnostic_status="verification_error",
            explanation=(
                "No comparison was completed for this claim, so the neutral "
                f"outcome reflects an absence of evaluated evidence: {reason}"
            ),
            pairs_expected=len(evidence),
            pairs_evaluated=0,
            complete=False,
            incomplete_reason=reason,
            matched_evidence=rows,
            aggregation="not_checked",
            warnings=[reason],
        )

    def _detect(
        self, text: str, fallback: DetectedLanguage | None
    ) -> DetectedLanguage:
        detection = self._detector.detect(text)
        detected = _to_detected_language(detection)
        if detected.code is None and fallback is not None and fallback.code is not None:
            return fallback.model_copy(
                update={"method": f"{fallback.method}+claim_fallback"}
            )
        return detected

    def _error_result(
        self, message: str, timings: dict[str, float]
    ) -> VerificationResult:
        return VerificationResult(
            status="error",
            enabled=True,
            nli_model=self._nli.model_name,
            ner_model=self._ner.model_name,
            timings=VerificationTimings(
                **{key: round(value, 3) for key, value in timings.items()}
            ),
            error=message[:1000],
            warnings=["Verification failed; the generated answer was preserved."],
        )


def build_verification_engine(
    settings: Settings,
    embedder: Embedder,
    generator: Generator,
    detector: QueryLanguageDetector,
) -> VerificationEngine:
    if not settings.verification_enabled:
        return DisabledVerificationEngine()
    return ClaimVerificationEngine(
        settings=settings,
        detector=detector,
        claim_extractor=build_claim_extractor(settings, generator, detector),
        ner_tagger=build_ner_tagger(settings),
        matcher=build_evidence_matcher(embedder),
        nli_model=build_nli_model(settings),
    )


def _prioritize_claims(
    claims: Sequence[ExtractedClaim], limit: int
) -> tuple[list[ExtractedClaim], list[ExtractedClaim]]:
    """Select claims deterministically, keeping important facts when capped.

    Numbers, dates, negation, and quantifiers are the most common sources of
    high-impact hallucination, so they receive priority. The original sentence
    order is preserved in the selected list for readable results.
    """

    if len(claims) <= limit:
        return list(claims), []

    def score(index: int) -> tuple[int, int]:
        text = claims[index].claim
        matches = sum(
            1 for pattern in _IMPORTANT_CLAIM_PATTERNS if pattern.search(text)
        )
        return (-matches, index)

    selected_indexes = sorted(range(len(claims)), key=score)[:limit]
    chosen = set(selected_indexes)
    selected = [claim for index, claim in enumerate(claims) if index in chosen]
    skipped = [
        claim for index, claim in enumerate(claims) if index not in chosen
    ]
    return selected, skipped


def _cancel_reason(cancel_event: threading.Event | None) -> str | None:
    if cancel_event is not None and cancel_event.is_set():
        return (
            "Verification was stopped because the client disconnected before "
            "all evidence could be checked."
        )
    return None


def _candidate_windows(
    evidence: Sequence[VerificationEvidenceInput],
) -> list[_CandidateWindow]:
    candidates: list[_CandidateWindow] = []
    for evidence_index, item in enumerate(evidence):
        spans = split_sentences(item.text, join_wrapped_lines=True)
        if not spans and item.text.strip():
            spans = [TextSpan(0, len(item.text), item.text)]
        for span in spans:
            if span.text.strip():
                candidates.append(
                    _CandidateWindow(evidence_index=evidence_index, span=span)
                )
    return candidates


def _best_candidates_by_evidence(
    matching: MatchingResult,
    candidates: Sequence[_CandidateWindow],
    claim_texts: Sequence[str],
    evidence_count: int,
) -> list[dict[int, tuple[_CandidateWindow, float]]]:
    first_by_evidence: dict[int, _CandidateWindow] = {}
    for candidate in candidates:
        first_by_evidence.setdefault(candidate.evidence_index, candidate)

    output: list[dict[int, tuple[_CandidateWindow, float]]] = []
    for claim_index, claim_text in enumerate(claim_texts):
        best: dict[int, tuple[_CandidateWindow, float]] = {
            evidence_index: (candidate, 0.0)
            for evidence_index, candidate in first_by_evidence.items()
        }
        best_scores = {
            evidence_index: (0.0, 0.0)
            for evidence_index in best
        }
        if claim_index < len(matching.matches):
            for match in matching.matches[claim_index]:
                if not 0 <= match.evidence_index < len(candidates):
                    continue
                candidate = candidates[match.evidence_index]
                coverage = _claim_term_coverage(
                    claim_text, candidate.span.text
                )
                score = (coverage, match.similarity)
                current = best_scores.get(candidate.evidence_index)
                if current is None or score > current:
                    best[candidate.evidence_index] = (
                        candidate,
                        match.similarity,
                    )
                    best_scores[candidate.evidence_index] = score
        output.append(
            {
                index: choice
                for index, choice in best.items()
                if index < evidence_count
            }
        )
    return output


def _unavailable_plan(
    evidence_index: int,
    source: VerificationEvidenceInput,
    similarity: float,
    reason: str,
    *,
    hypothesis: str = "",
) -> _PairPlan:
    return _PairPlan(
        evidence_index=evidence_index,
        rank=evidence_index + 1,
        similarity=similarity,
        hypothesis_text=hypothesis,
        premise_text="",
        premise_strategy="unavailable",
        evidence_char_length=len(source.text),
        premise_char_length=0,
        truncated=False,
        window_start=None,
        window_end=None,
        claim_term_coverage=0.0,
        relevance="unknown",
        evidence_quality=None,
        evidence_quality_reason=None,
        full_pair_token_count=None,
        input_pair_token_count=None,
        max_pair_tokens=None,
        unavailable_reason=reason,
    )


def _relevance_terms(text: str) -> set[str]:
    return {
        _relevance_key(term)
        for term in content_terms(text)
        if term not in _RELEVANCE_STOPWORDS
        and (len(term) > 1 or term.isdigit())
    }


def _relevance_key(term: str) -> str:
    value = term.casefold()
    value = value.replace("analyse", "analyze").replace(
        "analysis", "analyze"
    )
    if len(value) > 5 and value.endswith("ments"):
        value = value[:-5]
    elif len(value) > 6 and value.endswith("ations"):
        value = value[:-6]
    elif len(value) > 4 and value.endswith("ies"):
        value = f"{value[:-3]}y"
    elif len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ied"):
        value = f"{value[:-3]}y"
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 6 and value.endswith("ment"):
        value = value[:-4]
    elif len(value) > 6 and value.endswith("ness"):
        value = value[:-4]
    elif len(value) > 6 and value.endswith("ation"):
        value = value[:-5]
    elif len(value) > 5 and value.endswith("ion"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("es"):
        value = value[:-2]
    elif len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    if len(value) > 5 and value.endswith("e") and value[-2] in "tzgv":
        value = value[:-1]
    return value


def _claim_term_coverage(claim: str, premise: str) -> float:
    claim_terms = _relevance_terms(claim)
    if not claim_terms:
        return 1.0
    premise_terms = _relevance_terms(premise)
    return len(claim_terms & premise_terms) / len(claim_terms)


def _relevance_label(
    coverage: float, quality: EvidenceQuality, threshold: float
) -> ComparisonRelevance:
    if quality != "substantive":
        return "non_substantive"
    if coverage >= threshold:
        return "relevant"
    if coverage > 0.0:
        return "weakly_relevant"
    return "irrelevant"


def _evidence_quality(text: str) -> tuple[EvidenceQuality, str | None]:
    stripped = text.strip()
    if not stripped:
        return "empty", "The evaluated evidence premise is empty."

    dot_leaders = _DOT_LEADER.findall(stripped)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    dot_lines = [line for line in lines if _DOT_LEADER.search(line)]
    if len(dot_lines) >= 2 or len(dot_leaders) >= 5:
        return (
            "table_of_contents",
            (
                "The evidence contains table-of-contents-style dot leaders or "
                "page-number index lines rather than substantive prose."
            ),
        )

    lower = stripped.casefold()
    if "all rights reserved" in lower and len(_WORD.findall(stripped)) <= 12:
        return (
            "boilerplate",
            "The evidence is publication boilerplate rather than substantive content.",
        )

    prose = [
        span.text.strip()
        for span in split_sentences(stripped)
        if _is_prose_sentence(span.text.strip())
    ]
    if prose:
        return "substantive", None

    if len(lines) == 1 and len(_WORD.findall(stripped)) <= 8:
        return (
            "heading_or_index",
            "The evidence is a short heading or index line without a substantive sentence.",
        )
    return (
        "short",
        "The evidence does not contain a complete substantive prose sentence.",
    )


def _is_prose_sentence(text: str) -> bool:
    if not text or _DOT_LEADER.search(text):
        return False
    if not text.rstrip().endswith((".", "!", "?", ";", "。", "！", "？", "।", "॥")):
        return False
    return len(_WORD.findall(text)) >= 3


def _counts_for_verdict(
    outcome: _PairOutcome,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> bool:
    prediction = outcome.prediction
    if prediction is None or outcome.plan.relevance != "relevant":
        return False
    if outcome.plan.evidence_quality != "substantive":
        return False
    if prediction.label == "entailment":
        return prediction.confidence >= entailment_threshold
    if prediction.label == "contradiction":
        return prediction.confidence >= contradiction_threshold
    return False


def _evidence_result(
    outcome: _PairOutcome,
    source: VerificationEvidenceInput,
    claim_id: str,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> EvidenceVerification:
    plan = outcome.plan
    prediction = outcome.prediction
    windows: list[NliWindowPrediction] = []
    if prediction is not None:
        windows.append(
            NliWindowPrediction(
                window_index=0,
                start=plan.window_start or 0,
                end=plan.window_end or 0,
                text=plan.premise_text,
                label=prediction.label,
                confidence=prediction.confidence,
                probabilities=dict(prediction.probabilities),
                logits=(
                    list(prediction.logits)
                    if prediction.logits is not None
                    else None
                ),
            )
        )
    return EvidenceVerification(
        claim_id=claim_id,
        hypothesis_text=plan.hypothesis_text,
        evidence=_to_reference(source, plan.evidence_index),
        evidence_text=source.text,
        rank=plan.rank,
        similarity=plan.similarity,
        retrieval_similarity=source.retrieval_similarity,
        retrieval_score=source.retrieval_score,
        evaluated=prediction is not None,
        label=prediction.label if prediction is not None else None,
        confidence=prediction.confidence if prediction is not None else None,
        probabilities=(
            dict(prediction.probabilities) if prediction is not None else {}
        ),
        logits=(
            list(prediction.logits)
            if prediction is not None and prediction.logits is not None
            else None
        ),
        relevance=plan.relevance,
        relevance_score=plan.claim_term_coverage,
        claim_term_coverage=plan.claim_term_coverage,
        counts_for_verdict=outcome.counts_for_verdict,
        premise_text=plan.premise_text,
        premise_strategy=plan.premise_strategy,
        evidence_char_length=plan.evidence_char_length,
        premise_char_length=plan.premise_char_length,
        truncated=plan.truncated,
        window_start=plan.window_start,
        window_end=plan.window_end,
        evidence_quality=plan.evidence_quality,
        evidence_quality_reason=plan.evidence_quality_reason,
        full_pair_token_count=plan.full_pair_token_count,
        input_pair_token_count=plan.input_pair_token_count,
        max_pair_tokens=plan.max_pair_tokens,
        explanation=_pair_explanation(
            outcome,
            source,
            entailment_threshold,
            contradiction_threshold,
        ),
        windows=windows,
        error=outcome.error,
        warnings=(
            [outcome.skip_reason]
            if outcome.skip_reason and not outcome.error
            else [outcome.error]
            if outcome.error
            else []
        ),
    )


def _pair_explanation(
    outcome: _PairOutcome,
    source: VerificationEvidenceInput,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> str:
    where = _reference_place(_to_reference(source, outcome.plan.evidence_index))
    if outcome.error:
        return f"NLI inference failed for this comparison from {where}: {outcome.error}"
    if outcome.skip_reason:
        return f"This comparison from {where} was not evaluated: {outcome.skip_reason}"
    prediction = outcome.prediction
    if prediction is None:
        return f"No NLI result is available for this comparison from {where}."
    if prediction.label == "neutral":
        return (
            f"The passage from {where} was classified as neutral "
            f"({prediction.confidence:.3f}); it neither supports nor refutes "
            "the claim."
        )
    if outcome.verdict_exclusion_reason:
        return (
            f"The raw NLI label was {prediction.label} "
            f"({prediction.confidence:.3f}), but it was not counted because "
            f"{outcome.verdict_exclusion_reason}."
        )
    if outcome.plan.relevance != "relevant":
        quality = (
            f" Evidence quality: {outcome.plan.evidence_quality.replace('_', ' ')}."
            if outcome.plan.evidence_quality
            else ""
        )
        reason = (
            f" {outcome.plan.evidence_quality_reason}"
            if outcome.plan.evidence_quality_reason
            else ""
        )
        return (
            f"The raw NLI label was {prediction.label} "
            f"({prediction.confidence:.3f}), but it was not counted because "
            f"the passage relevance is {outcome.plan.relevance.replace('_', ' ')}."
            f"{quality}{reason}"
        )
    threshold = (
        entailment_threshold
        if prediction.label == "entailment"
        else contradiction_threshold
    )
    if prediction.confidence < threshold:
        return (
            f"The passage from {where} was classified as {prediction.label} "
            f"({prediction.confidence:.3f}), below the {threshold:.2f} "
            "threshold, so it did not determine the verdict."
        )
    if prediction.label == "entailment":
        result = (
            f"The passage from {where} supports the claim "
            f"(entailment confidence {prediction.confidence:.3f})."
        )
    else:
        result = (
            f"The passage from {where} contradicts the claim "
            f"(contradiction confidence {prediction.confidence:.3f})."
        )
    if outcome.plan.truncated:
        result = (
            f"{result} A focused evidence window was evaluated because the "
            "full chunk exceeded the NLI token limit."
        )
    return result


def _claim_explanation(
    *,
    status: str,
    reference: EvidenceReference | None,
    prediction: NliPrediction | None,
    support_confidence: float | None,
    contradiction_confidence: float | None,
    evaluated_count: int,
    expected: int,
    incomplete_reason: str | None,
    has_evidence: bool,
    conflict: bool = False,
) -> str:
    where = _reference_place(reference) if reference is not None else "the evidence"
    coverage = ""
    if expected > 0 and evaluated_count < expected:
        coverage = (
            f" Only {evaluated_count} of {expected} expected "
            "claim-evidence comparison(s) were evaluated."
        )
        if incomplete_reason:
            coverage = f"{coverage} {incomplete_reason}"
    if status == "verification_error":
        return (
            "No evidence comparison produced a usable NLI result, so this "
            f"claim could not be verified.{coverage}"
        )
    if status == "supported":
        confidence = (
            prediction.confidence if prediction is not None else support_confidence
        )
        return (
            f"At least one relevant evidence comparison from {where} supported "
            f"the claim (entailment confidence {confidence:.3f}), and no counted "
            f"relevant contradiction met the configured threshold.{coverage}"
        )
    if status == "contradicted" and conflict:
        support = support_confidence or 0.0
        contradiction = contradiction_confidence or 0.0
        return (
            f"Relevant evidence contains both support and contradiction "
            f"(support confidence {support:.3f}, contradiction confidence "
            f"{contradiction:.3f}); contradiction takes precedence for safety, "
            f"and the conflict is retained for inspection.{coverage}"
        )
    if status == "contradicted":
        confidence = (
            prediction.confidence
            if prediction is not None
            else contradiction_confidence
        )
        return (
            f"Relevant evidence from {where} contradicted the claim "
            f"(contradiction confidence {confidence:.3f}).{coverage}"
        )
    if status == "neutral":
        if not has_evidence:
            return "No evidence chunks were available for this claim."
        return (
            "No evaluated relevant evidence comparison met the configured "
            "support or contradiction threshold. This is a neutral result: the "
            "retrieved evidence neither establishes nor refutes the claim."
            f"{coverage}"
        )
    if status == "unevaluable":
        return "The claim was preserved for transparency but was not evaluated."
    return (
        "No claim verdict could be determined from the available comparison "
        f"results.{coverage}"
    )


def _incomplete_reason(
    state: _ClaimState, evaluated_count: int, expected: int
) -> str:
    reasons = [
        outcome.error or outcome.skip_reason
        for outcome in state.pairs
        if outcome.prediction is None
        and (outcome.error or outcome.skip_reason)
    ]
    reason = _dedupe([item for item in reasons if item])
    suffix = f" {reason[0]}" if reason else ""
    return (
        f"Only {evaluated_count} of {expected} evidence comparison(s) were "
        f"evaluated.{suffix}"
    )


def _run_status(
    results: Sequence[ClaimVerificationResult],
    summary: VerificationSummary,
) -> str:
    if summary.claims_total == 0:
        return "completed"
    if (
        summary.verification_error == summary.claims_evaluable
        and summary.claims_evaluable > 0
        and summary.pairs_evaluated == 0
        and any(item.error for item in results)
    ):
        return "error"
    if summary.claims_incomplete > 0 or summary.verification_error > 0:
        return "partial"
    return "completed"


def _summarize(
    results: Sequence[ClaimVerificationResult],
    evidence_total: int,
    invalid_fragments_skipped: int = 0,
    claims_location_unresolved: int = 0,
) -> VerificationSummary:
    evaluable = [item for item in results if item.claim.evaluable]
    pairs_expected = sum(item.pairs_expected for item in evaluable)
    pairs_evaluated = sum(item.pairs_evaluated for item in evaluable)
    pairs_retried = sum(item.pairs_retried for item in evaluable)
    pairs_failed = sum(item.pairs_failed for item in evaluable)
    claims_complete = sum(item.complete for item in evaluable)
    return VerificationSummary(
        valid_claims_extracted=len(evaluable),
        invalid_fragments_skipped=invalid_fragments_skipped,
        claims_location_unresolved=claims_location_unresolved,
        valid_claims_evaluated=sum(item.pairs_evaluated > 0 for item in evaluable),
        claims_supported_by_nli=sum(
            item.label == "entailment" for item in evaluable
        ),
        claims_contradicted_by_nli=sum(
            item.label == "contradiction" for item in evaluable
        ),
        claims_neutral_nli=sum(item.label == "neutral" for item in evaluable),
        claims_total=len(results),
        claims_evaluable=len(evaluable),
        claims_unevaluable=len(results) - len(evaluable),
        claims_checked=sum(item.pairs_evaluated > 0 for item in evaluable),
        claims_complete=claims_complete,
        claims_incomplete=len(evaluable) - claims_complete,
        supported=sum(item.status == "supported" for item in evaluable),
        contradicted=sum(
            item.status == "contradicted" for item in evaluable
        ),
        neutral=sum(item.status == "neutral" for item in evaluable),
        unevaluable=len(results) - len(evaluable),
        verification_error=sum(
            item.diagnostic_status == "verification_error" for item in evaluable
        ),
        evidence_passed=evidence_total,
        evidence_total=evidence_total,
        pairs_expected=pairs_expected,
        pairs_evaluated=pairs_evaluated,
        pairs_missing=max(0, pairs_expected - pairs_evaluated),
        pairs_retried=pairs_retried,
        pairs_failed=pairs_failed,
    )


def _to_detected_language(detection: LanguageDetection) -> DetectedLanguage:
    return DetectedLanguage(
        code=detection.code,
        confidence=detection.confidence,
        reliable=detection.reliable,
        mixed=detection.mixed,
        method=detection.method,
        reason=detection.reason,
    )


def _to_reference(
    item: VerificationEvidenceInput, index: int
) -> EvidenceReference:
    preview = " ".join(item.text.split())
    if len(preview) > 240:
        preview = f"{preview[:237]}..."
    return EvidenceReference(
        evidence_id=item.evidence_id or f"e{index + 1}",
        document_id=item.document_id,
        filename=item.filename,
        chunk_id=item.chunk_id,
        chunk_index=item.chunk_index,
        page_number=item.page_number,
        text_preview=preview,
    )


def _reference_place(reference: EvidenceReference) -> str:
    name = reference.filename or reference.evidence_id
    if reference.page_number is not None:
        return f"{name}, page {reference.page_number}"
    return name


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        cleaned = value.strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)
