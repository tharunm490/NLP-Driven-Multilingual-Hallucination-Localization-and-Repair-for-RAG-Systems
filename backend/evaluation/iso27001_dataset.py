"""Small labelled ISO/IEC 27001:2022-derived evaluation fixture.

The text is a paraphrased excerpt for testing retrieval/abstention behaviour,
not the standard itself. It deliberately contains no monetary fine, staff-count,
or fee statement so that related-but-unanswerable questions can be measured.

Each query is labelled with the chunk IDs that actually contain the information
needed to answer its supported part. Topically related but non-answer-bearing
chunks are *not* labelled relevant; evidence sufficiency is a separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    clause: str
    text: str


@dataclass(frozen=True)
class LabelledQuery:
    query_id: str
    question: str
    language: str
    category: str
    split: str
    relevant_chunk_ids: tuple[str, ...]
    expected_status: str
    expected_missing_aspects: tuple[str, ...] = ()
    note: str = ""


ISO_CHUNKS: tuple[FixtureChunk, ...] = (
    FixtureChunk(
        chunk_id="iso-cia",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=3,
        chunk_index=0,
        clause="3.2",
        text=(
            "Information security is the preservation of confidentiality, integrity "
            "and availability of information. Confidentiality means that information "
            "is not made available or disclosed to unauthorised individuals, entities "
            "or processes. Integrity means that information is accurate and complete. "
            "Availability means that information is accessible and usable on demand "
            "by an authorised entity."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-risk",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=8,
        chunk_index=1,
        clause="6.1.2",
        text=(
            "Clause 6.1.2 Information security risk assessment requires the "
            "organisation to define and apply an information security risk assessment "
            "process. The organisation establishes and maintains information security "
            "risk criteria, ensures that repeated assessments produce consistent, "
            "valid and comparable results, identifies risks associated with the loss "
            "of confidentiality, integrity and availability of information, analyses "
            "and evaluates those risks, and determines which risks require treatment."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-audit",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=12,
        chunk_index=2,
        clause="9.2",
        text=(
            "Clause 9.2 Internal audit requires the organisation to conduct internal "
            "audits at planned intervals. The audit programme must consider the "
            "importance of the processes concerned, changes affecting the "
            "organisation, and the results of previous audits. The organisation "
            "defines the audit criteria and scope, selects competent auditors, ensures "
            "objectivity and impartiality, reports results to relevant management, and "
            "retains documented information as evidence of the audit programme and "
            "results."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-improve",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=14,
        chunk_index=3,
        clause="10.1",
        text=(
            "Clause 10.1 Continual improvement requires the organisation to "
            "continually improve the suitability, adequacy and effectiveness of the "
            "information security management system. The organisation considers the "
            "results of monitoring, measurement, analysis and evaluation, and the "
            "outputs of management review, to determine whether there are needs or "
            "opportunities for improvement."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-corrective",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=15,
        chunk_index=4,
        clause="10.2",
        text=(
            "Clause 10.2 Nonconformity and corrective action requires the "
            "organisation to react to nonconformity, evaluate the need to eliminate "
            "the root cause so that the nonconformity does not recur, implement "
            "corrective action, review the effectiveness of the corrective action, "
            "update risks and opportunities determined during planning if necessary, "
            "and retain documented information as evidence of the nature of the "
            "nonconformities and the results of corrective action."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-policy",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=6,
        chunk_index=5,
        clause="5.2",
        text=(
            "Clause 5.2 Information security policy requires top management to "
            "establish an information security policy that is appropriate to the "
            "purpose of the organisation, includes information security objectives, "
            "includes a commitment to satisfy applicable requirements, and includes a "
            "commitment to continual improvement of the information security "
            "management system."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-documented",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=10,
        chunk_index=6,
        clause="7.5",
        text=(
            "Clause 7.5 Documented information requires the information security "
            "management system to include documented information required by "
            "ISO/IEC 27001 and documented information determined by the organisation "
            "as necessary. Documented information must be controlled to ensure it is "
            "available and suitable where and when it is needed, and protected from "
            "loss of integrity, improper use, or unauthorised access."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-certification",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=18,
        chunk_index=7,
        clause="Certification",
        text=(
            "Certification to ISO/IEC 27001 is performed by an accredited "
            "certification body. The certification body plans and conducts audits, "
            "including an initial certification audit and surveillance audits at "
            "planned intervals, to confirm that the information security management "
            "system continues to conform to the requirements of the standard."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-annexa",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=22,
        chunk_index=8,
        clause="Annex A",
        text=(
            "Annex A of ISO/IEC 27001:2022 contains information security controls "
            "organised into four themes: organisational controls, people controls, "
            "physical controls and technological controls. The controls are selected "
            "through the risk assessment and risk treatment process and are applied "
            "as necessary to manage information security risks."
        ),
    ),
    FixtureChunk(
        chunk_id="iso-scope",
        document_id="doc-iso27001",
        filename="iso27001.pdf",
        page_number=5,
        chunk_index=9,
        clause="4.3",
        text=(
            "Clause 4.3 Determining the scope of the information security management "
            "system requires the organisation to determine the boundaries and "
            "applicability of the ISMS. The scope statement must be available as "
            "documented information and must consider external and internal issues, "
            "the requirements of interested parties, and interfaces and dependencies "
            "with other organisations."
        ),
    ),
)

CHUNKS_BY_ID: dict[str, FixtureChunk] = {chunk.chunk_id: chunk for chunk in ISO_CHUNKS}


# Validation queries are used to select the distance threshold. Test queries are
# held out and are only used for the final before/after report.
LABELLED_QUERIES: tuple[LabelledQuery, ...] = (
    LabelledQuery(
        "val-en-cia",
        "What three information security properties does an ISMS preserve?",
        "en",
        "answerable",
        "validation",
        ("iso-cia",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-hi-cia",
        "सूचना सुरक्षा के तीन मुख्य गुण कौन-से हैं?",
        "hi",
        "answerable",
        "validation",
        ("iso-cia",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-audit",
        "What does the internal audit clause require?",
        "en",
        "answerable",
        "validation",
        ("iso-audit",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-improve",
        "How does the standard require continual improvement?",
        "en",
        "answerable",
        "validation",
        ("iso-improve",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-doc",
        "What does documented information need to be protected from?",
        "en",
        "answerable",
        "validation",
        ("iso-documented",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-annex",
        "What are the four themes of Annex A controls?",
        "en",
        "answerable",
        "validation",
        ("iso-annexa",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-hi-audit",
        "आंतरिक ऑडिट कितने अंतराल पर किया जाना चाहिए?",
        "hi",
        "answerable",
        "validation",
        ("iso-audit",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-premise",
        "Does Clause 6.1.2 cover internal audits?",
        "en",
        "contradictory_premise",
        "validation",
        ("iso-risk",),
        "verification_pending",
        note="The correct response explains risk assessment, not internal audits.",
    ),
    LabelledQuery(
        "val-en-612-internal",
        "Does Clause 6.1.2 cover internal audits or risk assessment?",
        "en",
        "contradictory_premise",
        "validation",
        ("iso-risk",),
        "verification_pending",
    ),
    LabelledQuery(
        "val-en-partial",
        "What three properties does an ISMS preserve and what is the exact fine for violation?",
        "en",
        "partial",
        "validation",
        ("iso-cia",),
        "partially_grounded",
        ("exact fine for violation",),
    ),
    LabelledQuery(
        "val-hi-partial",
        "सूचना सुरक्षा के तीन गुण क्या हैं और जुर्माना कितना है?",
        "hi",
        "partial",
        "validation",
        ("iso-cia",),
        "partially_grounded",
        ("जुर्माना",),
    ),
    LabelledQuery(
        "val-en-fine",
        "What is the exact fine for violating ISO/IEC 27001?",
        "en",
        "related_unanswerable",
        "validation",
        (),
        "insufficient_evidence",
    ),
    LabelledQuery(
        "val-en-employees",
        "How many employees must an organisation have to use ISO 27001?",
        "en",
        "related_unanswerable",
        "validation",
        (),
        "insufficient_evidence",
    ),
    LabelledQuery(
        "val-en-cost",
        "What is the certification fee for ISO 27001?",
        "en",
        "related_unanswerable",
        "validation",
        (),
        "insufficient_evidence",
    ),
    LabelledQuery(
        "val-en-tharun",
        "What does this document say about Tharun M's personal behavior?",
        "en",
        "unrelated",
        "validation",
        (),
        "irrelevant_retrieval",
    ),
    LabelledQuery(
        "val-en-cake",
        "What is the recipe for chocolate cake?",
        "en",
        "unrelated",
        "validation",
        (),
        "irrelevant_retrieval",
    ),
    LabelledQuery(
        "val-hi-weather",
        "आज चेन्नई में मौसम कैसा है?",
        "hi",
        "unrelated",
        "validation",
        (),
        "irrelevant_retrieval",
    ),
    LabelledQuery(
        "test-en-612",
        "What does Clause 6.1.2 address?",
        "en",
        "answerable",
        "test",
        ("iso-risk",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-hi-612",
        "क्लॉज 6.1.2 किस बारे में है?",
        "hi",
        "answerable",
        "test",
        ("iso-risk",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-en-corrective",
        "What is required for nonconformity and corrective action?",
        "en",
        "answerable",
        "test",
        ("iso-corrective",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-en-scope",
        "What must the scope of the ISMS consider?",
        "en",
        "answerable",
        "test",
        ("iso-scope",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-en-cert",
        "How are surveillance audits conducted?",
        "en",
        "answerable",
        "test",
        ("iso-certification",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-en-audit-freq",
        "At what intervals must internal audits be conducted?",
        "en",
        "answerable",
        "test",
        ("iso-audit",),
        "verification_pending",
    ),
    LabelledQuery(
        "test-en-penalty",
        "What is the legal penalty for non-compliance with ISO 27001 in India?",
        "en",
        "related_unanswerable",
        "test",
        (),
        "insufficient_evidence",
    ),
    LabelledQuery(
        "test-hi-fine",
        "आईएसओ 27001 का उल्लंघन करने पर कितना जुर्माना लगता है?",
        "hi",
        "related_unanswerable",
        "test",
        (),
        "insufficient_evidence",
    ),
    LabelledQuery(
        "test-en-paris",
        "What is the capital of France?",
        "en",
        "unrelated",
        "test",
        (),
        "irrelevant_retrieval",
    ),
    LabelledQuery(
        "test-hi-tharun",
        "यह दस्तावेज़ थरुण एम के व्यक्तिगत व्यवहार के बारे में क्या कहता है?",
        "hi",
        "unrelated",
        "test",
        (),
        "irrelevant_retrieval",
    ),
    LabelledQuery(
        "test-en-premise",
        "Is Clause 9.2 about information security risk assessment?",
        "en",
        "contradictory_premise",
        "test",
        ("iso-audit",),
        "verification_pending",
        note="The correct response explains that 9.2 is internal audit.",
    ),
    LabelledQuery(
        "test-en-partial",
        "What does Clause 9.2 require and what is the exact certification fee?",
        "en",
        "partial",
        "test",
        ("iso-audit",),
        "partially_grounded",
        ("exact certification fee",),
    ),
    LabelledQuery(
        "test-hi-partial",
        "आंतरिक ऑडिट की आवश्यकता क्या है और प्रमाणन शुल्क कितना है?",
        "hi",
        "partial",
        "test",
        ("iso-audit",),
        "partially_grounded",
        ("प्रमाणन शुल्क",),
    ),
)


def queries_for_split(split: str) -> list[LabelledQuery]:
    return [query for query in LABELLED_QUERIES if query.split == split]
