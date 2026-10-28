"""Tests for local factual claim extraction."""

from __future__ import annotations

from app.core.config import Settings
from app.services.verification.claims import (
    HeuristicClaimExtractor,
    build_claim_extractor,
)

ANSWER = (
    "# Findings\n"
    "- The policy requires annual reviews [Source 1].\n"
    "2. The policy requires daily reviews [Source 2].\n"
    "Sources: [Source 1]\n"
    "What is the review frequency?\n"
)

HINDI_ANSWER = (
    "सूचना सुरक्षा प्रबंधन प्रणाली संगठन की जानकारी की गोपनीयता, "
    "अखंडता और उपलब्धता की रक्षा करती है।"
)


class ExplodingGenerator:
    provider = "groq"

    def generate(self, messages):
        raise AssertionError("claim extraction must not call an LLM")


def test_heuristic_extractor_removes_markdown_and_citations():
    result = HeuristicClaimExtractor().extract(ANSWER)

    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews",
        "The policy requires daily reviews",
    ]
    assert all(claim.extraction_method == "heuristic" for claim in result.claims)
    for claim in result.claims:
        assert ANSWER[claim.claim_start : claim.claim_end] == claim.claim
        assert claim.language.code is None


def test_heuristic_extractor_removes_adjacent_citations():
    answer = "The standard was published in 2024 [Source 3][Source 4]."

    result = HeuristicClaimExtractor().extract(answer)

    assert result.claims[0].claim == "The standard was published in 2024"
    assert result.claims[0].claim_end == len("The standard was published in 2024")


def test_heuristic_extractor_skips_headings_and_introductory_text():
    answer = (
        "# ISO 27001 Overview\n"
        "**Main Objectives**\n"
        "1. Scope\n"
        "This section provides an overview of the ISMS.\n"
        "Here are the key benefits:\n"
        "The following points summarize the standard:\n"
        "An ISMS is a systematic approach to managing information security.\n"
        "What is an ISMS?\n"
        "Sources: [Source 1]\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "An ISMS is a systematic approach to managing information security."
    ]
    for claim in result.claims:
        assert answer[claim.claim_start : claim.claim_end] == claim.claim


def test_heuristic_extractor_returns_no_claims_for_heading_only_answer():
    answer = "# Scope\n**Main Objectives**\nThis section explains the ISMS.\n"

    result = HeuristicClaimExtractor().extract(answer)

    assert result.claims == ()
    assert result.warnings == ()


def test_extractor_separates_plain_headings_from_following_paragraphs():
    answer = (
        "ISO/IEC 27001:2022 specifies ISMS requirements.\n\n"
        "Purpose of ISO/IEC 27001:2022\n"
        "The standard provides a common framework for managing risks.\n\n"
        "How an organization uses an ISMS\n"
        "1. The ISMS identifies information-security risks.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "ISO/IEC 27001:2022 specifies ISMS requirements.",
        "The standard provides a common framework for managing risks.",
        "The ISMS identifies information-security risks.",
    ]
    assert "Purpose of ISO/IEC 27001:2022" not in result.claims[0].claim
    assert "How an organization uses an ISMS" not in result.claims[2].claim


def test_heuristic_extractor_splits_compound_claims():
    answer = (
        "ISO develops standards, and IEC publishes them. "
        "The ISMS protects data; it also ensures compliance. "
        "The policy applies to staff and contractors, and it requires annual training."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "ISO develops standards",
        "IEC publishes them.",
        "The ISMS protects data",
        "it also ensures compliance.",
        "The policy applies to staff and contractors",
        "it requires annual training.",
    ]
    for claim in result.claims:
        assert answer[claim.claim_start : claim.claim_end] == claim.claim


def test_heuristic_extractor_does_not_split_shared_predicate_lists():
    answer = "The policy requires annual reviews and daily audits."

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [answer]


def test_heuristic_extractor_does_not_split_a_list_noun_phrase():
    answer = "The policy applies to staff, contractors, and auditors."

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [answer]


def test_heuristic_extractor_preserves_multilingual_claim():
    result = HeuristicClaimExtractor().extract(HINDI_ANSWER)

    assert [claim.claim for claim in result.claims] == [HINDI_ANSWER]


def test_heuristic_extractor_reports_claim_limit():
    answer = (
        "Alpha Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020. "
        "Gamma Corporation was founded in 2025."
    )

    result = HeuristicClaimExtractor(max_claims=2).extract(answer)

    assert len(result.claims) == 2
    assert any("limit of 2 claims" in warning for warning in result.warnings)


def test_heuristic_extractor_deduplicates_identical_claims():
    answer = (
        "The policy requires annual reviews. "
        "The policy requires annual reviews! "
        "The policy requires daily audits."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews.",
        "The policy requires daily audits.",
    ]


def test_extractor_keeps_complete_propositions_and_marks_unevaluable_fragments():
    answer = (
        "An ISMS is a systematic approach to managing information security. "
        "It does this by applying a risk management process and by giving "
        "interested parties confidence that risks are adequately managed.\n\n"
        "The standard ISO/IEC 27001:2022 specifies that an ISMS must:\n"
        "- **Establish, implement, maintain and continually improve the ISMS.**\n"
        "- **Integrate the ISMS into the organization's processes.**\n\n"
        "**Risk assessment**: The organization shall define and apply an "
        "information security risk assessment process.\n\n"
        "In summary, an ISMS helps organizations manage information security "
        "risks and protect confidentiality, integrity and availability."
    )

    result = HeuristicClaimExtractor().extract(answer)
    claims = [claim.claim for claim in result.claims]

    assert not any(claim.startswith("by giving") for claim in claims)
    assert not any("**" in claim for claim in claims)
    assert not any(claim.startswith("continuously monitors") for claim in claims)
    assert not any(claim.startswith("and protect confidentiality") for claim in claims)

    merged = result.claims[0]
    assert merged.claim.startswith(
        "An ISMS is a systematic approach to managing information security. "
        "It does this"
    )
    assert merged.evaluable is True
    assert merged.complete is True
    assert merged.context is None
    assert merged.hypothesis == merged.claim

    establish = next(
        claim for claim in result.claims if claim.claim.startswith("Establish")
    )
    assert establish.evaluable is True
    assert establish.complete is True
    assert establish.claim == (
        "Establish, implement, maintain and continually improve the ISMS."
    )
    assert establish.context is not None
    assert "must" in establish.context
    assert establish.hypothesis.startswith("An ISMS must establish")

    risk = next(
        claim
        for claim in result.claims
        if claim.claim.startswith("The organization shall define")
    )
    assert risk.evaluable is True
    assert risk.claim.startswith("The organization shall define")

    summary = next(
        claim
        for claim in result.claims
        if claim.claim.startswith("An ISMS helps organizations")
    )
    assert summary.evaluable is True
    assert summary.claim.endswith(
        "protect confidentiality, integrity and availability."
    )


def test_extractor_records_anaphoric_claim_as_not_evaluable():
    answer = "It does this by applying a risk management process."

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 1
    assert result.claims[0].claim == answer
    assert result.claims[0].evaluable is False
    assert "pronoun" in (result.claims[0].incomplete_reason or "").lower()


def test_extractor_resolves_pronoun_in_same_list_item_continuation():
    dash = chr(8212)
    answer = (
        f"1. **Risk assessment and treatment** {dash} At planned intervals "
        "or when significant changes occur, the organization performs "
        "information-security risk assessments and retains documented "
        "results. It then formulates and implements a risk-treatment plan, "
        "retaining documentation of the treatment outcomes."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 2
    continuation = result.claims[1]
    assert continuation.claim.startswith("It then formulates")
    assert continuation.hypothesis == (
        "The organization then formulates and implements a risk-treatment "
        "plan, retaining documentation of the treatment outcomes."
    )
    assert continuation.evaluable is True
    assert continuation.incomplete_reason is None


def test_extractor_does_not_split_an_infinitive_verb_list_after_malformed_bold():
    answer = (
        "ISO/IEC 27001:2022 specifies the *requirements* that an organization "
        "must meet to **establish, implement, maintain and continually improve* "
        "an information-security management system (ISMS)** [Source 1]. It also "
        "requires the organization to assess and treat information-security "
        "risks in a way that is tailored to its own needs [Source 1]."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.claim.startswith("ISO/IEC 27001:2022 specifies")
    assert "to establish, implement, maintain and continually improve" in claim.claim
    assert "*" not in claim.claim
    assert claim.evaluable is True


def test_extractor_evaluates_parenthetical_body_and_resolves_followup_pronoun():
    dash = chr(8211)
    answer = (
        f"4. **Perform risk assessment and treatment** {dash} At planned "
        "intervals (or when significant changes occur) the organization "
        "conducts risk assessments, documents the results, and implements a "
        "risk-treatment plan [Source 5]. This ensures that identified risks "
        "to confidentiality, integrity or availability are addressed."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 2
    assessment, followup = result.claims
    assert assessment.evaluable is True
    assert assessment.hypothesis.startswith("At planned intervals")
    assert followup.evaluable is True
    assert followup.hypothesis == (
        "At planned intervals the organization ensures that identified risks "
        "to confidentiality, integrity or availability are addressed."
    )


def test_extractor_keeps_dash_parenthetical_summary_sentence_intact():
    dash = chr(8212)
    answer = (
        f"Through these activities{dash}risk assessment and treatment, "
        "objective setting, controlled change management, performance "
        f"monitoring, and internal auditing{dash}the ISMS systematically "
        "addresses threats to information, thereby maintaining its "
        "confidentiality, integrity, and availability as required by "
        "ISO/IEC 27001:2022."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.hypothesis == answer
    assert claim.evaluable is True


def test_extractor_reconstructs_numbered_items_from_shared_modal_context():
    answer = (
        "An ISMS is a systematic approach to managing information security.\n\n"
        "The standard lists several objectives that an ISMS must achieve, "
        "including:\n"
        "1. Define and maintain an information security policy.\n"
        "2. Establish, implement, maintain and continually improve the ISMS.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims[1:]] == [
        "Define and maintain an information security policy.",
        "Establish, implement, maintain and continually improve the ISMS.",
    ]
    assert [claim.hypothesis for claim in result.claims[1:]] == [
        "An ISMS must define and maintain an information security policy.",
        "An ISMS must establish, implement, maintain and continually improve the ISMS.",
    ]
    assert all(claim.evaluable for claim in result.claims)
    assert all(claim.context for claim in result.claims[1:])


def test_extractor_reconstructs_noun_led_items_from_heading_context():
    answer = (
        "# How an ISMS helps manage information-security risks\n"
        "- Risk assessment.\n"
        "- Risk treatment.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "Risk assessment.",
        "Risk treatment.",
    ]
    assert [claim.hypothesis for claim in result.claims] == [
        "An ISMS helps manage information-security risks through risk assessment.",
        "An ISMS helps manage information-security risks through risk treatment.",
    ]
    assert all(claim.evaluable for claim in result.claims)


def test_extractor_uses_nested_item_context_and_resolves_pronouns():
    answer = (
        "# ISMS components\n"
        "- Governance structure\n"
        "  - Roles and responsibilities\n"
        "  - It identifies threats to the ISMS.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "Governance structure",
        "Roles and responsibilities",
        "It identifies threats to the ISMS.",
    ]
    assert result.claims[1].hypothesis == (
        "Governance structure includes roles and responsibilities."
    )
    assert result.claims[2].hypothesis == (
        "Governance structure identifies threats to the ISMS."
    )
    assert all(claim.evaluable for claim in result.claims)


def test_extractor_reconstructs_objective_items_without_a_modal():
    answer = (
        "Main objectives of an ISMS:\n"
        "- Protect information.\n"
        "- Manage information-security risks.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.hypothesis for claim in result.claims] == [
        "An objective of an ISMS is to protect information.",
        "An objective of an ISMS is to manage information-security risks.",
    ]
    assert all(claim.evaluable for claim in result.claims)


def test_extractor_reconstructs_label_body_list_items_as_propositions():
    answer = (
        "# How an organization uses an ISMS\n"
        "1. Adopt a risk-management approach – The ISMS applies a systematic "
        "process to identify and treat information-security risks.\n"
        "2. Set clear objectives – The organization establishes measurable "
        "information-security objectives.\n"
        "3. Assign responsibilities – The organization assigns ISMS roles and "
        "responsibilities.\n"
        "4. Monitor performance – The organization monitors and reviews ISMS "
        "performance.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "Adopt a risk-management approach – The ISMS applies a systematic "
        "process to identify and treat information-security risks.",
        "Set clear objectives – The organization establishes measurable "
        "information-security objectives.",
        "Assign responsibilities – The organization assigns ISMS roles and "
        "responsibilities.",
        "Monitor performance – The organization monitors and reviews ISMS "
        "performance.",
    ]
    assert [claim.hypothesis for claim in result.claims] == [
        "The ISMS applies a systematic process to identify and treat "
        "information-security risks.",
        "The organization establishes measurable information-security "
        "objectives.",
        "The organization assigns ISMS roles and responsibilities.",
        "The organization monitors and reviews ISMS performance.",
    ]
    assert [claim.context for claim in result.claims] == [
        "Adopt a risk-management approach",
        "Set clear objectives",
        "Assign responsibilities",
        "Monitor performance",
    ]
    assert all(claim.evaluable for claim in result.claims)
    for claim in result.claims:
        assert answer[claim.claim_start : claim.claim_end] == claim.claim


def test_extractor_does_not_split_dash_inside_an_incomplete_list_item():
    answer = (
        "# Scope\n"
        "- The policy applies to permanent - temporary staff.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "The policy applies to permanent - temporary staff."
    ]
    assert result.claims[0].evaluable is True


def test_build_claim_extractor_is_local_even_with_a_real_provider(settings: Settings):
    updated = settings.model_copy(update={"llm_provider": "groq"})

    extractor = build_claim_extractor(updated, ExplodingGenerator())

    assert isinstance(extractor, HeuristicClaimExtractor)
    result = extractor.extract("The policy requires annual reviews.")
    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews."
    ]
