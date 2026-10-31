"""Prompt construction and strict response parsing for claim repair."""

from __future__ import annotations

import json
import re
from typing import Sequence

from app.core.errors import RepairResponseError
from app.schemas.repair import (
    RepairAttempt,
    RepairEvidence,
    RepairLLMOutput,
    RepairSupportingEvidence,
)
from app.services.generator import ChatMessage

REPAIR_SYSTEM_PROMPT = """\
You repair factual claims from a document-grounded answer independently.

Rules:
- The supplied document passages are the ONLY factual authority.
- Do not use external knowledge, assumptions, or pretrained knowledge.
- Do not invent information or document references.
- Do not try to justify the original claim when the evidence does not support it.
- Preserve the original meaning where the evidence supports it.
- Correct unsupported details.
- Narrow the statement when only part of the original claim is supported.
- Do not add unrelated factual assertions.
- Do not fabricate document names or page numbers.
- Keep the corrected claim grammatically complete.
- Preserve the original answer's language.
- Process every submitted claim independently. Never merge claims or create
  claims that were not submitted.
- If the supplied evidence cannot establish the claim or support a reliable
  correction, return INSUFFICIENT_EVIDENCE.
- If the claim cannot be usefully corrected and is unnecessary to the answer,
  return REMOVED.
- Return exactly one JSON object and nothing else. Do not use markdown fences.

Allowed status values:
- REPAIRED: the claim was rewritten into a statement supported by the evidence.
- QUALIFIED: only part of the claim was supported, so it was narrowed.
- REMOVED: the claim cannot be usefully corrected and is unnecessary.
- INSUFFICIENT_EVIDENCE: the evidence does not establish the claim or a reliable
  correction.

Return a JSON object with a `claims` array. The array must contain exactly one
entry for every submitted claim, each entry using this shape:
{
  "status": "REPAIRED",
  "original_claim": "Exact original claim",
  "repaired_claim": "Corrected factual claim",
  "reason": "Explanation of the correction",
  "supporting_evidence": [
    {
      "source": "iso27001.pdf",
      "page": 10,
      "evidence_quote": "Exact passage copied from the supplied evidence"
    }
  ]
}

For REMOVED or INSUFFICIENT_EVIDENCE, repaired_claim may be null and
supporting_evidence may be empty. For REPAIRED and QUALIFIED, repaired_claim is
required and at least one supporting_evidence entry must quote a supplied
passage exactly.
"""

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def build_repair_messages(
    *,
    question: str,
    claim: str,
    context: str | None,
    evidence: Sequence[RepairEvidence],
    original_nli_status: str | None,
    original_nli_confidence: float | None,
    original_guard_verdict: str,
    original_guard_support: float,
    previous_attempt: RepairAttempt | None,
) -> list[ChatMessage]:
    lines: list[str] = [
        "QUESTION:",
        question,
        "",
        "ORIGINAL CLAIM:",
        claim,
        "",
    ]
    if context:
        lines.extend(["SECTION OR CONTEXT:", context, ""])

    lines.extend(
        [
            "ORIGINAL VERIFICATION:",
            "- NLI: "
            + (
                f"{original_nli_status}"
                + (
                    f" (confidence {original_nli_confidence:.3f})"
                    if original_nli_confidence is not None
                    else ""
                )
                if original_nli_status
                else "unavailable"
            ),
            f"- Hallucination Guard: {original_guard_verdict.upper()} "
            f"(support {original_guard_support:.3f})",
            "",
        ]
    )

    if previous_attempt is not None:
        verification = previous_attempt.verification
        lines.extend(
            [
                "PREVIOUS ATTEMPT (it failed independent verification):",
                f"- Attempt number: {previous_attempt.attempt_number}",
                f"- Repair decision: {previous_attempt.decision or 'unknown'}",
                f"- Previous repaired claim: "
                f"{previous_attempt.repaired_claim or '(none)'}",
                f"- Previous explanation: {previous_attempt.reason or '(none)'}",
                "- Previous NLI: "
                + (verification.nli_status if verification and verification.nli_status else "unavailable"),
                "- Previous Hallucination Guard: "
                + (
                    verification.guard_verdict
                    if verification and verification.guard_verdict
                    else "unavailable"
                ),
                f"- Failure reason: {previous_attempt.failure_reason or 'verification failed'}",
                "",
            ]
        )

    lines.append(
        "SUPPLIED EVIDENCE (the only factual authority; quote exact text):"
    )
    for index, item in enumerate(evidence, start=1):
        location = item.filename
        if item.page_number is not None:
            location = f"{location}, page {item.page_number}"
        lines.extend(
            [
                f"[{index}] source={location} chunk_id={item.chunk_id} "
                f"guard_support={item.guard_support_probability:.3f}",
                item.text,
                "",
            ]
        )
    lines.append(
        "Return only the JSON object described in the system instructions."
    )
    return [
        ChatMessage(role="system", content=REPAIR_SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n".join(lines).strip()),
    ]


def build_repair_batch_messages(
    *,
    question: str,
    claims: Sequence[tuple[
        str,
        str | None,
        Sequence[RepairEvidence],
        str | None,
        float | None,
        str,
        float,
        RepairAttempt | None,
    ]],
) -> list[ChatMessage]:
    """Build one request containing independent claim repair tasks."""
    lines = ["QUESTION:", question, "", "SUBMITTED CLAIMS:"]
    for index, (
        claim,
        context,
        evidence,
        nli_status,
        nli_confidence,
        guard_verdict,
        guard_support,
        previous,
    ) in enumerate(claims, start=1):
        lines.extend(
            [
                "",
                f"CLAIM {index}:",
                "ORIGINAL CLAIM:",
                claim,
            ]
        )
        if context:
            lines.extend(["SECTION OR CONTEXT:", context])
        lines.extend(
            [
                "ORIGINAL VERIFICATION:",
                "- NLI: "
                + (nli_status or "unavailable")
                + (
                    f" (confidence {nli_confidence:.3f})"
                    if nli_confidence is not None
                    else ""
                ),
                f"- Hallucination Guard: {guard_verdict.upper()} "
                f"(support {guard_support:.3f})",
            ]
        )
        if previous is not None:
            verification = previous.verification
            lines.extend(
                [
                    "PREVIOUS ATTEMPT:",
                    f"- Attempt number: {previous.attempt_number}",
                    f"- Previous repaired claim: "
                    f"{previous.repaired_claim or '(none)'}",
                    f"- Previous NLI: "
                    f"{verification.nli_status if verification and verification.nli_status else 'unavailable'}",
                    f"- Previous Hallucination Guard: "
                    f"{verification.guard_verdict if verification and verification.guard_verdict else 'unavailable'}",
                    f"- Failure reason: "
                    f"{previous.failure_reason or 'verification failed'}",
                ]
            )
        lines.extend(["SUPPLIED EVIDENCE FOR THIS CLAIM:"])
        for evidence_item in evidence:
            location = evidence_item.filename
            if evidence_item.page_number is not None:
                location += f", page {evidence_item.page_number}"
            lines.extend(
                [
                    f"source={location} chunk_id={evidence_item.chunk_id}",
                    evidence_item.text,
                ]
            )
    lines.extend(["", "Return only the JSON object described in the system instructions."])
    return [
        ChatMessage(role="system", content=REPAIR_SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n".join(lines).strip()),
    ]


def parse_repair_response(text: str, original_claim: str) -> RepairLLMOutput:
    """Parse and validate the model's JSON; never accept a partial object."""
    payload = _extract_json(text)
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise RepairResponseError(
            "The repair model returned malformed JSON."
        ) from exc
    if not isinstance(data, dict):
        raise RepairResponseError(
            "The repair model did not return a JSON object."
        )
    try:
        output = RepairLLMOutput.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - normalised into a repair error
        raise RepairResponseError(
            "The repair model response did not match the required schema."
        ) from exc

    if _normalise(output.original_claim) != _normalise(original_claim):
        raise RepairResponseError(
            "The repair model changed the original claim instead of returning it."
        )
    repaired = (output.repaired_claim or "").strip()
    if output.status in {"REPAIRED", "QUALIFIED"}:
        if not repaired:
            raise RepairResponseError(
                "The repair model did not provide a corrected claim."
            )
        if not output.supporting_evidence:
            raise RepairResponseError(
                "The repair model did not cite any supplied evidence."
            )
    return output


def parse_repair_batch_response(
    text: str, original_claims: Sequence[str]
) -> dict[str, RepairLLMOutput]:
    """Parse a batch response and map each validated entry to its claim."""
    payload = _extract_json(text)
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise RepairResponseError(
            "The repair model returned malformed JSON."
        ) from exc
    entries = data.get("claims") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        if len(original_claims) == 1:
            return {
                _normalise(original_claims[0]): parse_repair_response(
                    text, original_claims[0]
                )
            }
        raise RepairResponseError(
            "The repair model did not return a claims array for the batch."
        )
    if len(entries) != len(original_claims):
        raise RepairResponseError(
            "The repair model returned a different number of claims than submitted."
        )
    outputs: dict[str, RepairLLMOutput] = {}
    for entry, original_claim in zip(entries, original_claims):
        output = parse_repair_response(json.dumps(entry), original_claim)
        key = _normalise(output.original_claim)
        if key in outputs:
            raise RepairResponseError("The repair model returned duplicate claims.")
        outputs[key] = output
    return outputs


def validate_supporting_evidence(
    output: RepairLLMOutput,
    evidence: Sequence[RepairEvidence],
) -> list[RepairSupportingEvidence]:
    """Keep only references whose quote really occurs in the supplied evidence."""
    if not evidence:
        return []
    normalised_passages = [
        (item, _normalise(item.text)) for item in evidence
    ]
    kept: list[RepairSupportingEvidence] = []
    for reference in output.supporting_evidence:
        quote = _normalise(reference.evidence_quote)
        if not quote:
            continue
        match = next(
            (item for item, passage in normalised_passages if quote in passage),
            None,
        )
        if match is None:
            continue
        kept.append(
            RepairSupportingEvidence(
                source=match.filename,
                page=match.page_number,
                evidence_quote=reference.evidence_quote,
            )
        )
    if output.status in {"REPAIRED", "QUALIFIED"} and not kept:
        raise RepairResponseError(
            "The repair model's evidence quotes do not appear in the supplied evidence."
        )
    return kept


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fenced = _JSON_BLOCK.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RepairResponseError("The repair model did not return JSON.")
    return cleaned[start : end + 1]


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())
