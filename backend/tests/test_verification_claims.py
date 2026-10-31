"""Tests for factual claim extraction."""

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

ISO_RISK_ANSWER = (
    "# Risk assessment\n\n"
    "Risk criteria\n"
    "The organization must establish and maintain risk criteria that include:\n\n"
    "- Risk-acceptance criteria\n"
    "The level of risk the organization is willing to tolerate.\n"
    "- Risk acceptance criteria: The level of risk the organization is "
    "willing to tolerate.\n"
    "- Risk owners\n"
    "The organization must identify the risk owners.\n"
    "- Risk analysis\n"
    "The organization must analyze the risks.\n"
    "- Risk evaluation: The organization must compare risk analysis results "
    "with the risk criteria.\n"
    "- If a risk exceeds the 5% tolerance, the organization must treat the "
    "risk.\n"
    "- The organization must not accept a risk without documented approval.\n"
    "- The organization must identify the risk owners, and the organization "
    "must document the risk criteria.\n"
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


def test_extractor_accepts_assertions_and_skips_malformed_fragments():
    cases = {
        "Risk criteria": False,
        "Risk analysis": False,
        "For each identified risk the organization:": False,
        "The organization must risk owners.": False,
        (
            "Risk-acceptance criteria define the level of risk an organization "
            "is willing to tolerate."
        ): True,
        "The organization must identify the risk owners.": True,
        "The organization assesses potential consequences and likelihood.": True,
        "The organization does not accept risks above its defined criteria.": True,
    }

    for answer, expected_claim in cases.items():
        result = HeuristicClaimExtractor().extract(answer)
        assert bool(result.claims) is expected_claim
        if not expected_claim:
            assert result.skipped


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

    assert len(result.claims) == 1
    assert "It then formulates and implements a risk-treatment plan" in (
        result.claims[0].claim
    )


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

    assert len(result.claims) == 1
    assert "This ensures that identified risks" in result.claims[0].claim


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


def test_extractor_keeps_complete_bullet_with_multiple_clauses_intact():
    answer = (
        "- Adoption of an ISMS is presented as a strategic decision that must "
        "align with the organization's needs, objectives, security requirements, "
        "processes, size and structure, and it is expected to evolve over time.\n"
        "- The organization documents the decision and reviews it annually."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        answer.splitlines()[0][2:],
        answer.splitlines()[1][2:],
    ]
    assert all(
        answer[claim.claim_start : claim.claim_end] == claim.claim
        for claim in result.claims
    )
    assert not any(
        claim.claim == "it is expected to evolve over time."
        for claim in result.claims
    )


def test_extractor_extracts_all_factual_bullets_across_topic_headings():
    answer = (
        "# Determining the scope of the ISMS\n"
        "- The organization defines the boundaries and applicability of its "
        "ISMS and documents its scope. [Source 1]\n\n"
        "# Internal, external, and interested-party factors\n"
        "- The organization considers relevant external and internal issues. "
        "[Source 1]\n"
        "- The organization considers interested-party requirements and "
        "determines which requirements the ISMS addresses. [Source 1]\n"
        "- The organization considers interfaces and dependencies between its "
        "own activities and those performed by other organizations. [Source 1]\n"
        "- Interested-party requirements may include legal, regulatory, and "
        "contractual obligations. [Source 1]"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 5
    assert [claim.claim for claim in result.claims] == [
        "The organization defines the boundaries and applicability of its "
        "ISMS and documents its scope.",
        "The organization considers relevant external and internal issues.",
        "The organization considers interested-party requirements and "
        "determines which requirements the ISMS addresses.",
        "The organization considers interfaces and dependencies between its "
        "own activities and those performed by other organizations.",
        "Interested-party requirements may include legal, regulatory, and "
        "contractual obligations.",
    ]
    assert all(
        answer[claim.claim_start : claim.claim_end] == claim.claim
        for claim in result.claims
    )


def test_extractor_keeps_when_bullet_with_trailing_citation():
    answer = (
        "**Establishing information-security objectives**\n"
        "- The organization shall set information-security objectives at "
        "relevant functions and levels, and document them [Source 1].\n\n"
        "**Planning to achieve the objectives**\n"
        "- When planning how to achieve its information-security objectives, "
        "the organization shall specify what will be done, what resources "
        "will be required, who will be responsible, when it will be "
        "completed, and how the results will be evaluated [Source 1]."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 2
    assert result.claims[1].claim.startswith(
        "When planning how to achieve its information-security objectives"
    )
    assert all(
        answer[claim.claim_start : claim.claim_end] == claim.claim
        for claim in result.claims
    )


def test_extractor_recursively_extracts_nested_bullets_with_exact_spans():
    answer = (
        "The organization must consider interested parties:\n\n"
        "- Identify relevant interested parties.\n"
        "- Determine their requirements:\n\n"
        "  - Legal requirements.\n"
        "  - Regulatory requirements.\n"
        "  - Contractual requirements."
    )

    result = HeuristicClaimExtractor().extract(answer)
    claims = {claim.claim: claim for claim in result.claims}

    assert set(claims) == {
        "The organization must consider interested parties",
        "Identify relevant interested parties.",
        "Legal requirements.",
        "Regulatory requirements.",
        "Contractual requirements.",
    }
    for claim in result.claims:
        assert answer[claim.claim_start : claim.claim_end] == claim.claim
    assert claims["Legal requirements."].context == "Determine their requirements:"


def test_extractor_splits_inline_nested_bullets_without_splitting_their_text():
    bullet = "\u2022"
    answer = (
        "- The scope must be established by considering:     "
        f"{bullet} the external and internal issues relevant to the organization "
        "(clause 4.1) [Source 1];     "
        f"{bullet} the requirements of interested parties (clause 4.2) "
        "[Source 1];     "
        f"{bullet} the interfaces and dependencies between the organization's "
        "activities and those performed by other organizations [Source 1]."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "the external and internal issues relevant to the organization "
        "(clause 4.1)",
        "the requirements of interested parties (clause 4.2)",
        "the interfaces and dependencies between the organization's "
        "activities and those performed by other organizations",
    ]
    assert all(
        answer[claim.claim_start : claim.claim_end] == claim.claim
        for claim in result.claims
    )
    assert all(
        claim.context == "The scope must be established by considering:"
        for claim in result.claims
    )


def test_extractor_keeps_parent_context_for_all_dependent_children():
    answer = (
        "When defining the scope, the organization must consider:\n\n"
        "- The external and internal issues relevant to the organization.\n"
        "- The requirements of interested parties.\n"
        "- The interfaces and dependencies between its own activities and "
        "those performed by other organizations."
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 3
    assert all(
        claim.context
        == "When defining the scope, the organization must consider"
        for claim in result.claims
    )
    assert all(answer[claim.claim_start : claim.claim_end] == claim.claim for claim in result.claims)


def test_build_claim_extractor_always_returns_the_heuristic_extractor(
    settings: Settings,
):
    updated = settings.model_copy(update={"llm_provider": "groq"})

    extractor = build_claim_extractor(updated, ExplodingGenerator())

    assert isinstance(extractor, HeuristicClaimExtractor)
    assert extractor.method == "heuristic"
    result = extractor.extract("The policy requires annual reviews.")
    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews."
    ]


def test_build_claim_extractor_enforces_the_hard_claim_cap(
    settings: Settings,
):
    updated = settings.model_copy(
        update={"verification_max_extracted_claims": 100}
    )

    extractor = build_claim_extractor(updated)

    assert isinstance(extractor, HeuristicClaimExtractor)
    assert extractor._max_claims == 30


def test_extractor_groups_dependent_items_under_an_intro():
    answer = (
        "For each identified risk, the organization:\n"
        "- assesses the potential consequences of the risk.\n"
        "- evaluates the realistic likelihood of the risk occurring.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.claim == answer.strip()
    assert answer[claim.claim_start : claim.claim_end] == claim.claim
    assert claim.evaluable is True
    assert claim.hypothesis.startswith(
        "For each identified risk, the organization assesses"
    )


def test_extractor_does_not_group_independent_list_items():
    answer = (
        "The organization must:\n"
        "- The organization identifies the risk owners.\n"
        "- The organization documents the risk criteria.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "The organization identifies the risk owners.",
        "The organization documents the risk criteria.",
    ]


def test_extractor_does_not_group_a_standalone_heading():
    answer = (
        "Risk assessment\n"
        "- assesses the potential consequences of the risk.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert result.claims
    assert all(
        not claim.claim.startswith("Risk assessment") for claim in result.claims
    )


def test_extractor_skips_meta_commentary_about_missing_evidence():
    answer = (
        "The evidence does not provide the annual review frequency.\n"
        "The uploaded documents do not contain a retention period.\n"
        "It only outlines the required elements.\n"
        "The policy requires annual reviews.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews."
    ]
    assert len(result.skipped) == 3
    assert all(
        "comments on the available evidence" in fragment.reason
        for fragment in result.skipped
    )


def test_extractor_handles_blockquotes_as_narrative_text():
    answer = (
        "> The policy requires annual reviews.\n"
        "> - The policy requires daily audits.\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "The policy requires annual reviews.",
        "The policy requires daily audits.",
    ]
    for claim in result.claims:
        assert answer[claim.claim_start : claim.claim_end] == claim.claim


def test_extractor_prefers_a_more_complete_containing_claim():
    shorter_first = (
        "The policy requires annual reviews. "
        "The policy requires annual reviews for all staff."
    )
    longer_first = (
        "The policy requires annual reviews for all staff. "
        "The policy requires annual reviews."
    )

    for answer in (shorter_first, longer_first):
        result = HeuristicClaimExtractor().extract(answer)
        assert [claim.claim for claim in result.claims] == [
            "The policy requires annual reviews for all staff."
        ]


def test_extractor_caps_answers_at_thirty_claims():
    answer = " ".join(
        f"Policy {index} requires annual reviews." for index in range(40)
    )

    result = HeuristicClaimExtractor(max_claims=100).extract(answer)

    assert len(result.claims) == 30
    assert any("limit of 30 claims" in warning for warning in result.warnings)


def test_iso_risk_answer_extracts_only_complete_claims():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    assert [claim.claim for claim in result.claims] == [
        "The organization must establish and maintain risk criteria",
        "Risk acceptance criteria: The level of risk the organization is "
        "willing to tolerate.",
        "The organization must identify the risk owners.",
        "The organization must analyze the risks.",
        "Risk evaluation: The organization must compare risk analysis "
        "results with the risk criteria.",
        "If a risk exceeds the 5% tolerance, the organization must treat "
        "the risk.",
        "The organization must not accept a risk without documented "
        "approval.",
        "The organization must identify the risk owners, and the organization "
        "must document the risk criteria.",
    ]
    assert all(
        "must risk owners" not in claim.claim
        and "must risk-acceptance criteria" not in claim.claim
        for claim in result.claims
    )


def test_iso_risk_answer_skips_labels_and_introductory_text():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    skipped = {fragment.text: fragment.reason for fragment in result.skipped}
    assert "Risk assessment" in skipped
    assert "Risk criteria" in skipped
    assert "Risk-acceptance criteria" in skipped
    assert "Risk owners" in skipped
    assert "Risk analysis" in skipped
    assert "Risk assessment" not in {claim.claim for claim in result.claims}
    assert "Risk criteria" not in {claim.hypothesis for claim in result.claims}
    assert (
        "The organization must establish and maintain risk criteria"
        in {claim.claim for claim in result.claims}
    )
    assert "The organization must establish and maintain risk criteria that include:" not in skipped


def test_iso_risk_answer_marks_dependent_nominal_fragment_unevaluable():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    assert not any(
        claim.claim
        == "The level of risk the organization is willing to tolerate."
        for claim in result.claims
    )
    assert any(
        fragment.text
        == "The level of risk the organization is willing to tolerate."
        for fragment in result.skipped
    )


def test_iso_risk_answer_reconstructs_label_definition_bullet():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    claim = next(
        claim
        for claim in result.claims
        if claim.claim.startswith("Risk acceptance criteria:")
    )
    assert claim.hypothesis == (
        "Risk acceptance criteria define the level of risk the organization "
        "is willing to tolerate."
    )
    assert claim.context == "Risk acceptance criteria"
    assert claim.evaluable is True


def test_iso_risk_answer_preserves_condition_quantity_and_negation():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    hypotheses = {claim.hypothesis for claim in result.claims}
    assert (
        "If a risk exceeds the 5% tolerance, the organization must treat "
        "the risk."
    ) in hypotheses
    assert (
        "The organization must not accept a risk without documented approval."
    ) in hypotheses


def test_iso_risk_answer_splits_compound_assertions():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    assert (
        "The organization must identify the risk owners."
        in {claim.claim for claim in result.claims}
    )
    assert (
        "The organization must identify the risk owners, and the organization "
        "must document the risk criteria."
        in {claim.claim for claim in result.claims}
    )
    assert (
        "the organization must document the risk criteria."
        not in {claim.claim for claim in result.claims}
    )


def test_extractor_uses_markdown_table_cells_as_claims():
    answer = (
        "| Control | Description |\n"
        "| --- | --- |\n"
        "| Access control | Limits access to information. |\n"
        "| Monitoring | The organization monitors access attempts. |\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert [claim.claim for claim in result.claims] == [
        "Limits access to information.",
        "The organization monitors access attempts.",
    ]
    assert [claim.hypothesis for claim in result.claims] == [
        "Access control limits access to information.",
        "The organization monitors access attempts.",
    ]
    assert all(claim.evaluable for claim in result.claims)


def test_extractor_does_not_turn_table_fragments_into_claims():
    answer = (
        "| Item | Detail |\n"
        "| --- | --- |\n"
        "| Review | Annual. |\n"
    )

    result = HeuristicClaimExtractor().extract(answer)

    assert result.claims == ()
    assert any(fragment.text == "Annual." for fragment in result.skipped)


def test_iso_risk_answer_does_not_extract_section_headings_as_claims():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    claim_texts = {claim.claim for claim in result.claims}
    assert "Risk criteria" not in claim_texts
    assert "Risk owners" not in claim_texts
    assert "Risk analysis" not in claim_texts
    assert "The organization must establish and maintain risk criteria that include" not in claim_texts


def test_iso_risk_answer_has_no_duplicate_claims_and_reports_skips():
    result = HeuristicClaimExtractor().extract(ISO_RISK_ANSWER)

    assert len(result.claims) == 8
    assert len(result.skipped) == 6
    assert len({claim.hypothesis for claim in result.claims}) == 8
    assert not any(
        "duplicates an earlier claim" in fragment.reason
        for fragment in result.skipped
    )
