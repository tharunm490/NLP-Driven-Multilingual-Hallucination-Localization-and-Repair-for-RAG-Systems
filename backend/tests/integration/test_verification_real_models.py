r"""Real NLI/NER verification tests (opt-in, no network after model download).

    $env:RUN_VERIFICATION_INTEGRATION="1"
    .\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_real_models.py -m integration -s
"""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.schemas.verification import VerificationEvidenceInput
from app.services.query_language_detection import QueryLanguageDetector
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import ClaimVerificationEngine
from app.services.verification.matching import LexicalEvidenceMatcher
from app.services.verification.ner import build_ner_tagger
from app.services.verification.nli import build_nli_model
from evaluation.iso27001_dataset import CHUNKS_BY_ID

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_VERIFICATION_INTEGRATION") != "1",
        reason="set RUN_VERIFICATION_INTEGRATION=1 to run real verification tests",
    ),
]


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(
        _env_file=None,
        verification_nli_device=os.environ.get("VERIFICATION_DEVICE", "cpu"),
        verification_ner_device=os.environ.get("VERIFICATION_DEVICE", "cpu"),
    )


def test_cross_lingual_nli_and_obvious_contradiction(settings: Settings):
    model = build_nli_model(settings)
    hindi_claim = "सूचना सुरक्षा गोपनीयता, अखंडता और उपलब्धता को बनाए रखती है।"
    supported = model.predict(
        [
            (
                "Information security is the preservation of confidentiality, "
                "integrity and availability of information.",
                hindi_claim,
            )
        ]
    )[0]
    contradicted = model.predict(
        [("The policy requires annual reviews.", "The policy requires daily reviews.")]
    )[0]
    neutral = model.predict(
        [("The policy requires annual reviews.", "The policy is available online.")]
    )[0]

    print(
        "cross-lingual:", supported.label, supported.confidence,
        "contradiction:", contradicted.label, contradicted.confidence,
        "neutral:", neutral.label, neutral.confidence,
    )
    assert supported.label == "entailment"
    assert contradicted.label == "contradiction"
    assert neutral.label == "neutral"


def test_real_model_label_mapping_matches_model_config(settings: Settings):
    model = build_nli_model(settings)
    model.load()
    config = getattr(getattr(model, "_model", None), "config", None)
    raw_labels = getattr(config, "id2label", {}) or {}
    assert raw_labels

    def canonical(raw: object) -> str:
        value = str(raw).casefold()
        if "entail" in value or "support" in value:
            return "entailment"
        if "contrad" in value:
            return "contradiction"
        return "neutral"

    label_map = getattr(model, "_id2label", {})
    assert {
        int(index): canonical(raw_label)
        for index, raw_label in raw_labels.items()
    } == label_map


def test_real_nli_page4_iso_claims(settings: Settings):
    model = build_nli_model(settings)
    premise = (
        "ISO (the International Organization for Standardization) and IEC "
        "(the International Electrotechnical Commission) form the specialized "
        "system for worldwide standardization."
    )
    predictions = model.predict(
        [
            (premise, "ISO is the International Organization for Standardization."),
            (
                premise,
                "ISO and IEC form the specialized system for worldwide "
                "standardization.",
            ),
            (premise, "ISO was founded in 1947."),
        ]
    )

    print(
        "page4:",
        [(prediction.label, prediction.confidence) for prediction in predictions],
    )
    assert [prediction.label for prediction in predictions] == [
        "entailment",
        "entailment",
        "neutral",
    ]


def _engine(settings: Settings) -> ClaimVerificationEngine:
    return ClaimVerificationEngine(
        settings=settings,
        detector=QueryLanguageDetector(enabled=True),
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=build_ner_tagger(settings),
        matcher=LexicalEvidenceMatcher(),
        nli_model=build_nli_model(settings),
    )


def test_real_engine_verifies_supported_claim(settings: Settings):
    engine = _engine(settings)
    answer = "An ISMS preserves confidentiality, integrity, and availability."
    result = engine.verify(
        answer,
        [
            VerificationEvidenceInput(
                text=(
                    "Information security is the preservation of confidentiality, "
                    "integrity and availability of information."
                ),
                evidence_id="e1",
                filename="iso27001.pdf",
                page_number=3,
            )
        ],
    )

    print("engine summary:", result.summary.model_dump())
    assert result.status == "completed"
    assert result.summary.supported == 1
    assert result.claims[0].claim.language.code == "en"


def test_iso_c2_c4_c6_c12_regression_with_all_retrieved_chunks(
    settings: Settings,
):
    engine = _engine(settings)
    answer = (
        "Information security preserves integrity. "
        "The scope statement must be available as documented information. "
        "Annex A controls are organised into four themes. "
        "The annual certification fee is 10,000 USD."
    )
    evidence_ids = (
        "iso-cia",
        "iso-audit",
        "iso-scope",
        "iso-certification",
        "iso-annexa",
        "iso-risk",
    )
    evidence = [
        VerificationEvidenceInput(
            text=CHUNKS_BY_ID[chunk_id].text,
            evidence_id=chunk_id,
            filename=CHUNKS_BY_ID[chunk_id].filename,
            page_number=CHUNKS_BY_ID[chunk_id].page_number,
        )
        for chunk_id in evidence_ids
    ]

    result = engine.verify(answer, evidence)

    statuses = {
        claim.claim.claim: claim.status
        for claim in result.claims
    }
    print("iso regression:", statuses)
    print("iso regression summary:", result.summary.model_dump())
    assert result.status == "completed"
    assert result.summary.claims_total == 4
    assert result.summary.pairs_expected == 24
    assert result.summary.pairs_evaluated == 24
    assert statuses["Information security preserves integrity."] == "supported"
    assert (
        statuses[
            "The scope statement must be available as documented information."
        ]
        == "supported"
    )
    assert statuses["Annex A controls are organised into four themes."] == "supported"
    assert statuses["The annual certification fee is 10,000 USD."] == "neutral"
    assert all(
        len(claim.matched_evidence) == len(evidence_ids)
        and claim.complete
        for claim in result.claims
    )


def test_real_engine_excludes_materially_weaker_secondary_support(
    settings: Settings,
):
    engine = _engine(settings)
    answer = (
        f"1. Monitor, measure, and evaluate performance {chr(8212)} The "
        "organization determines what to monitor, the methods to use, and "
        "when results must be evaluated."
    )
    result = engine.verify(
        answer,
        [
            VerificationEvidenceInput(
                text=(
                    "Information security objectives must be consistent with "
                    "the information security policy, measurable if "
                    "practicable, and take account of applicable information "
                    "security requirements and the results of risk assessment "
                    "and risk treatment. They must be monitored, communicated, "
                    "updated as appropriate, and available as documented "
                    "information. When planning to achieve the objectives, "
                    "the organization determines what will be done, the "
                    "required resources, responsibilities, completion dates, "
                    "and how results will be evaluated."
                ),
                evidence_id="weak",
            ),
            VerificationEvidenceInput(
                text=(
                    "The organization determines what needs to be monitored "
                    "and measured, the methods to use, when the activities "
                    "occur, who performs them, and when results are analysed "
                    "and evaluated."
                ),
                evidence_id="strong",
            ),
        ],
    )

    print("weak support summary:", result.summary.model_dump())
    claim = result.claims[0]
    counted = [
        item.evidence.evidence_id
        for item in claim.matched_evidence
        if item.counts_for_verdict
    ]
    assert result.status == "completed"
    assert claim.status == "supported"
    assert counted == ["strong"]
    assert claim.matched_evidence[0].relevance == "relevant"
    assert claim.matched_evidence[0].counts_for_verdict is False
    assert "materially weaker" in claim.matched_evidence[0].explanation


def test_real_engine_verifies_isms_property_list(settings: Settings):
    engine = _engine(settings)
    answer = "The ISMS preserves confidentiality, integrity, and availability."
    result = engine.verify(
        answer,
        [
            VerificationEvidenceInput(
                text=(
                    "The information security management system preserves the "
                    "confidentiality, integrity and availability of information "
                    "by applying a risk management process and gives confidence "
                    "to interested parties that risks are adequately managed."
                ),
                evidence_id="e1",
                filename="iso27001.pdf",
                page_number=5,
            )
        ],
    )

    print("isms summary:", result.summary.model_dump())
    print("isms claim:", result.claims[0].model_dump())
    assert result.status == "completed"
    assert result.summary.supported == 1
    assert result.claims[0].status == "supported"
