"""Controlled labelled claim-evidence cases for verification benchmarks.

The cases are derived from the paraphrased ISO/IEC 27001 fixture used elsewhere
in the repository. Each gold label is tied to an explicit supporting,
contradicting, or neutral sentence in the fixture. This is a small regression
set, not a substitute for a public NLI benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.iso27001_dataset import CHUNKS_BY_ID

GoldLabel = str


@dataclass(frozen=True)
class NliCase:
    case_id: str
    claim: str
    evidence: str
    focus: str
    gold_label: GoldLabel
    category: str
    split: str


@dataclass(frozen=True)
class EngineScenario:
    scenario_id: str
    claims: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    expected_statuses: tuple[str, ...]

    @property
    def answer(self) -> str:
        return " ".join(self.claims)


_CASE_SPECS: tuple[tuple[str, str, str, str, GoldLabel, str], ...] = (
    (
        "cia-confidentiality",
        "Information security preserves confidentiality.",
        "iso-cia",
        "Information security is the preservation of confidentiality, integrity "
        "and availability of information.",
        "entailment",
        "supported",
    ),
    (
        "audit-interval",
        "Internal audits are conducted at planned intervals.",
        "iso-audit",
        "Clause 9.2 Internal audit requires the organisation to conduct internal "
        "audits at planned intervals.",
        "entailment",
        "supported",
    ),
    (
        "scope-documented",
        "The scope statement must be available as documented information.",
        "iso-scope",
        "The scope statement must be available as documented information and "
        "must consider external and internal issues, the requirements of "
        "interested parties, and interfaces and dependencies with other "
        "organisations.",
        "entailment",
        "supported",
    ),
    (
        "documented-access",
        "Documented information must be protected from unauthorised access.",
        "iso-documented",
        "Documented information must be controlled to ensure it is available "
        "and suitable where and when it is needed, and protected from loss of "
        "integrity, improper use, or unauthorised access.",
        "entailment",
        "supported",
    ),
    (
        "certification-body",
        "Certification audits are performed by an accredited certification body.",
        "iso-certification",
        "Certification to ISO/IEC 27001 is performed by an accredited "
        "certification body.",
        "entailment",
        "supported",
    ),
    (
        "annex-four-themes",
        "Annex A controls are organised into four themes.",
        "iso-annexa",
        "Annex A of ISO/IEC 27001:2022 contains information security controls "
        "organised into four themes: organisational controls, people controls, "
        "physical controls and technological controls.",
        "entailment",
        "supported",
    ),
    (
        "audit-nonconformity-only",
        "Internal audits are conducted only after a nonconformity occurs.",
        "iso-audit",
        "Clause 9.2 Internal audit requires the organisation to conduct internal "
        "audits at planned intervals.",
        "contradiction",
        "contradicted",
    ),
    (
        "scope-confidential",
        "The scope statement must be kept confidential.",
        "iso-scope",
        "The scope statement must be available as documented information and "
        "must consider external and internal issues, the requirements of "
        "interested parties, and interfaces and dependencies with other "
        "organisations.",
        "contradiction",
        "contradicted",
    ),
    (
        "documented-unprotected",
        "Documented information does not need to be protected from unauthorised access.",
        "iso-documented",
        "Documented information must be controlled to ensure it is available "
        "and suitable where and when it is needed, and protected from loss of "
        "integrity, improper use, or unauthorised access.",
        "contradiction",
        "contradicted",
    ),
    (
        "certification-self",
        "Certification audits are performed by the organisation itself.",
        "iso-certification",
        "Certification to ISO/IEC 27001 is performed by an accredited "
        "certification body.",
        "contradiction",
        "contradicted",
    ),
    (
        "annex-three-themes",
        "Annex A controls are organised into three themes.",
        "iso-annexa",
        "Annex A of ISO/IEC 27001:2022 contains information security controls "
        "organised into four themes: organisational controls, people controls, "
        "physical controls and technological controls.",
        "contradiction",
        "contradicted",
    ),
    (
        "risk-inconsistent",
        "Repeated risk assessments must produce inconsistent and incomparable results.",
        "iso-risk",
        "The organisation establishes and maintains information security risk "
        "criteria, ensures that repeated assessments produce consistent, valid "
        "and comparable results, identifies risks associated with the loss of "
        "confidentiality, integrity and availability of information, analyses "
        "and evaluates those risks, and determines which risks require treatment.",
        "contradiction",
        "contradicted",
    ),
    (
        "cia-staff-count",
        "ISO/IEC 27001 requires a minimum staff count of 50 employees.",
        "iso-cia",
        "Information security is the preservation of confidentiality, integrity "
        "and availability of information.",
        "neutral",
        "neutral",
    ),
    (
        "certification-fee",
        "The annual certification fee is 10,000 USD.",
        "iso-certification",
        "Certification to ISO/IEC 27001 is performed by an accredited "
        "certification body.",
        "neutral",
        "neutral",
    ),
    (
        "risk-public-website",
        "The organisation must publish its risk assessment on a public website.",
        "iso-risk",
        "The organisation establishes and maintains information security risk "
        "criteria, ensures that repeated assessments produce consistent, valid "
        "and comparable results, identifies risks associated with the loss of "
        "confidentiality, integrity and availability of information, analyses "
        "and evaluates those risks, and determines which risks require treatment.",
        "neutral",
        "neutral",
    ),
    (
        "audit-government",
        "Internal audits must be performed by government inspectors.",
        "iso-audit",
        "Clause 9.2 Internal audit requires the organisation to conduct internal "
        "audits at planned intervals.",
        "neutral",
        "neutral",
    ),
    (
        "policy-quarterly",
        "The information security policy must be reviewed every quarter.",
        "iso-policy",
        "Clause 5.2 Information security policy requires top management to "
        "establish an information security policy that is appropriate to the "
        "purpose of the organisation, includes information security objectives, "
        "includes a commitment to satisfy applicable requirements, and includes "
        "a commitment to continual improvement of the information security "
        "management system.",
        "neutral",
        "neutral",
    ),
    (
        "annex-93-controls",
        "Annex A contains exactly 93 controls.",
        "iso-annexa",
        "Annex A of ISO/IEC 27001:2022 contains information security controls "
        "organised into four themes: organisational controls, people controls, "
        "physical controls and technological controls.",
        "neutral",
        "neutral",
    ),
)

_DISTRACTORS = {
    "iso-cia": ("iso-improve", "iso-policy"),
    "iso-risk": ("iso-audit", "iso-documented"),
    "iso-audit": ("iso-policy", "iso-improve"),
    "iso-improve": ("iso-risk", "iso-certification"),
    "iso-corrective": ("iso-documented", "iso-audit"),
    "iso-policy": ("iso-scope", "iso-certification"),
    "iso-documented": ("iso-improve", "iso-risk"),
    "iso-certification": ("iso-annexa", "iso-scope"),
    "iso-annexa": ("iso-certification", "iso-documented"),
    "iso-scope": ("iso-policy", "iso-improve"),
}


def _long_text(chunk_id: str) -> str:
    target = CHUNKS_BY_ID[chunk_id].text
    first, second = _DISTRACTORS[chunk_id]
    return " ".join(
        (CHUNKS_BY_ID[first].text, target, CHUNKS_BY_ID[second].text)
    )


def nli_cases() -> tuple[NliCase, ...]:
    cases: list[NliCase] = []
    for index, (case_id, claim, chunk_id, focus, gold, category) in enumerate(
        _CASE_SPECS, start=1
    ):
        short = CHUNKS_BY_ID[chunk_id].text
        long = _long_text(chunk_id)
        cases.append(
            NliCase(
                case_id=f"short-{case_id}",
                claim=claim,
                evidence=short,
                focus=focus,
                gold_label=gold,
                category=category,
                split="short",
            )
        )
        cases.append(
            NliCase(
                case_id=f"long-{case_id}",
                claim=claim,
                evidence=long,
                focus=focus,
                gold_label=gold,
                category=category,
                split="long",
            )
        )
    return tuple(cases)


_SHORT_CLAIMS = (
    "Information security preserves confidentiality.",
    "Internal audits are conducted at planned intervals.",
    "The scope statement must be available as documented information.",
)
_SHORT_EVIDENCE = ("iso-cia", "iso-audit", "iso-scope")
_SHORT_STATUSES = ("supported", "supported", "supported")

_LONG_CLAIMS = (
    "Information security preserves confidentiality.",
    "Information security preserves integrity.",
    "Internal audits are conducted at planned intervals.",
    "The scope statement must be available as documented information.",
    "Certification audits are performed by an accredited certification body.",
    "Annex A controls are organised into four themes.",
    "Repeated risk assessments must produce inconsistent results.",
    "Internal audits are conducted only after a nonconformity occurs.",
    "The scope statement must be kept confidential.",
    "Certification audits are performed by the organisation itself.",
    "ISO/IEC 27001 requires a minimum staff count of 50 employees.",
    "The annual certification fee is 10,000 USD.",
    "Annex A contains exactly 93 controls.",
)
_LONG_EVIDENCE = (
    "iso-cia",
    "iso-audit",
    "iso-scope",
    "iso-certification",
    "iso-annexa",
    "iso-risk",
)
_LONG_STATUSES = (
    "supported",
    "supported",
    "supported",
    "supported",
    "supported",
    "supported",
    "contradicted",
    "contradicted",
    "contradicted",
    "contradicted",
    "neutral",
    "neutral",
    "neutral",
)


def engine_scenarios() -> tuple[EngineScenario, ...]:
    return (
        EngineScenario(
            scenario_id="short-request",
            claims=_SHORT_CLAIMS,
            evidence_ids=_SHORT_EVIDENCE,
            expected_statuses=_SHORT_STATUSES,
        ),
        EngineScenario(
            scenario_id="long-request",
            claims=_LONG_CLAIMS,
            evidence_ids=_LONG_EVIDENCE,
            expected_statuses=_LONG_STATUSES,
        ),
    )


def evidence_texts(evidence_ids: tuple[str, ...]) -> list[str]:
    return [CHUNKS_BY_ID[chunk_id].text for chunk_id in evidence_ids]
