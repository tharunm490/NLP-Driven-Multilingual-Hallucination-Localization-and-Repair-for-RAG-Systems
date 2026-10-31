"""Tests for language-specific claim extraction and offset-safe location."""

from __future__ import annotations

from app.services.verification.claim_common import MAX_CLAIMS
from app.services.verification.claim_location import (
    locate_claim,
    normalize_for_matching,
    trim_citation_suffix,
)
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import _summarize


def test_heuristic_hindi_extractor_extracts_complete_claims_only():
    answer = (
        "# जोखिम मूल्यांकन\n\n"
        "जोखिम मानदंड\n"
        "संगठन जोखिम मानदंड स्थापित करता है।\n\n"
        "- जोखिम स्वामी\n"
        "संगठन को जोखिम स्वामियों की पहचान करनी चाहिए।\n"
        "- जोखिम विश्लेषण: संगठन जोखिमों का विश्लेषण करता है।\n"
        "- संगठन जोखिम मानदंड स्थापित करता है और संगठन जोखिम "
        "स्वामियों की पहचान करता है।\n"
        "- संगठन जोखिम मानदंड स्थापित करता है।\n"
        "- क्या संगठन को जोखिम का मूल्यांकन करना चाहिए?\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "संगठन जोखिम मानदंड स्थापित करता है।",
        "संगठन को जोखिम स्वामियों की पहचान करनी चाहिए।",
        "संगठन जोखिमों का विश्लेषण करता है।",
        "संगठन जोखिम स्वामियों की पहचान करता है।",
    ]
    assert all(
        answer[claim.claim_start : claim.claim_end] == claim.claim
        for claim in result.claims
    )
    assert all(
        claim.location_status == "verified" for claim in result.claims
    )
    skipped_texts = [fragment.text for fragment in result.skipped]
    assert "जोखिम मूल्यांकन" in skipped_texts
    assert "जोखिम स्वामी" in skipped_texts
    assert any(
        fragment.reason == "A question is not a factual claim."
        for fragment in result.skipped
    )


def test_heuristic_hindi_extractor_enforces_the_hard_claim_limit():
    answer = "".join(
        f"संगठन जोखिम संख्या {index} का मूल्यांकन करता है।\n"
        for index in range(1, 36)
    )

    result = HeuristicClaimExtractor(max_claims=100).extract(answer)

    assert len(result.claims) == MAX_CLAIMS
    assert any(
        "more than the hard limit" in warning for warning in result.warnings
    )


def test_summary_reports_location_unresolved_count():
    summary = _summarize(
        [],
        evidence_total=4,
        invalid_fragments_skipped=2,
        claims_location_unresolved=3,
    )

    assert summary.claims_location_unresolved == 3
    assert summary.invalid_fragments_skipped == 2
