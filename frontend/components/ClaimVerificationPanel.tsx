"use client";

import { useState } from "react";

import {
  claimVerificationStatusLabel,
  claimVerificationStatusTone,
} from "../lib/format";
import type { RagVerificationInfo, VerifiedEvidence } from "../lib/types";

import StatusBadge from "./StatusBadge";

type ClaimVerificationPanelProps = {
  verification: RagVerificationInfo;
  defaultOpen?: boolean;
};

function formatScore(value: number | null | undefined): string {
  return value == null ? "n/a" : value.toFixed(4);
}

function formatClaimScores(
  probabilities: Record<string, number> | undefined,
  supportConfidence: number | null | undefined,
  contradictionConfidence: number | null | undefined,
): string | null {
  const scores = Object.entries(probabilities ?? {})
    .filter(([, value]) => Number.isFinite(value))
    .map(([label, value]) => `${label} ${value.toFixed(4)}`);
  if (scores.length > 0) {
    return scores.join(" / ");
  }
  const fallbacks: string[] = [];
  if (supportConfidence != null) {
    fallbacks.push(`support ${supportConfidence.toFixed(4)}`);
  }
  if (contradictionConfidence != null) {
    fallbacks.push(`contradiction ${contradictionConfidence.toFixed(4)}`);
  }
  return fallbacks.length > 0 ? fallbacks.join(" / ") : null;
}

function VerificationPairTable({
  evidence,
}: {
  evidence: VerifiedEvidence[];
}) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Claim</th>
            <th>Evidence</th>
            <th>Source</th>
            <th>Retrieval similarity</th>
            <th>Matcher similarity</th>
            <th>Hypothesis</th>
            <th>Evaluated premise</th>
            <th>Full evidence</th>
            <th>Relevance and quality</th>
            <th>NLI scores</th>
            <th>Token accounting</th>
            <th>Counted</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {evidence.map((item, index) => (
            <tr
              key={`${item.evidence.evidence_id}-${item.rank}-${index}`}
              className={item.evaluated ? undefined : "muted"}
              data-testid="verification-pair-row"
              data-evidence-id={item.evidence.evidence_id}
            >
              <td className="mono">{item.claim_id ?? "unknown"}</td>
              <td className="mono">{item.evidence.evidence_id}</td>
              <td>
                <div>{item.evidence.filename ?? "unknown"}</div>
                <div className="muted">
                  rank {item.rank}
                  {item.evidence.page_number != null
                    ? ` - page ${item.evidence.page_number}`
                    : ""}
                </div>
              </td>
              <td data-testid="retrieval-similarity">
                {formatScore(item.retrieval_similarity)}
                {item.retrieval_score != null && (
                  <div className="muted">
                    distance {formatScore(item.retrieval_score)}
                  </div>
                )}
              </td>
              <td data-testid="matcher-similarity">
                {formatScore(item.similarity)}
              </td>
              <td>
                <p className="result-text">
                  {item.hypothesis_text ?? "n/a"}
                </p>
                {item.nli_input_order && (
                  <p className="muted">
                    Input order: {item.nli_input_order}
                  </p>
                )}
              </td>
              <td>
                <details className="source-text">
                  <summary>
                    {item.truncated
                      ? "Focused window"
                      : item.premise_strategy === "full_evidence"
                        ? "Full evidence"
                        : "Not compared"}
                  </summary>
                  <p className="result-text">
                    {item.premise_text || item.evidence.text_preview}
                  </p>
                  {item.window_start != null && item.window_end != null && (
                    <p className="note">
                      Characters {item.window_start} to {item.window_end} of{" "}
                      {item.evidence_char_length}
                      {item.truncated
                        ? " - focused because the full chunk exceeded the NLI limit"
                        : ""}
                    </p>
                  )}
                </details>
              </td>
              <td>
                <details className="source-text">
                  <summary>Original retrieved chunk</summary>
                  <p className="result-text">
                    {item.evidence_text || item.evidence.text_preview}
                  </p>
                </details>
              </td>
              <td>
                <div>{item.relevance.replaceAll("_", " ")}</div>
                <div className="muted">
                  claim-term coverage {item.claim_term_coverage.toFixed(3)}
                </div>
                {item.evidence_quality && (
                  <div className="muted">
                    quality: {item.evidence_quality.replaceAll("_", " ")}
                  </div>
                )}
                {item.evidence_quality_reason && (
                  <div className="muted">{item.evidence_quality_reason}</div>
                )}
              </td>
              <td>
                <div>
                  {item.evaluated ? item.label ?? "unknown" : "not compared"}
                </div>
                <div className="muted">
                  confidence{" "}
                  {item.confidence != null ? item.confidence.toFixed(4) : "n/a"}
                </div>
                {item.probabilities && (
                  <div className="muted">
                    {Object.entries(item.probabilities)
                      .map(
                        ([label, score]) =>
                          `${label} ${score.toFixed(4)}`,
                      )
                      .join(" / ")}
                  </div>
                )}
              </td>
              <td>
                <div>
                  input {item.input_pair_token_count ?? "n/a"} / full{" "}
                  {item.full_pair_token_count ?? "n/a"} / max{" "}
                  {item.max_pair_tokens ?? "n/a"}
                </div>
                {item.truncated && (
                  <div className="muted">
                    Truncated to the focused window
                  </div>
                )}
              </td>
              <td>{item.counts_for_verdict ? "Yes" : "No"}</td>
              <td>{item.explanation ?? "n/a"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ClaimVerificationPanel({
  verification,
  defaultOpen = false,
}: ClaimVerificationPanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const claims = verification.details?.claims ?? [];
  const verificationWarnings = verification.details?.warnings ?? [];
  const claimsTotal = verification.claims_total ?? 0;
  const claimsComplete = verification.claims_complete ?? 0;
  const claimsIncomplete = verification.claims_incomplete ?? 0;
  const pairsExpected = verification.pairs_expected ?? 0;
  const pairsEvaluated = verification.pairs_evaluated ?? 0;
  const pairsMissing = verification.pairs_missing ?? 0;
  const pairsFailed = verification.pairs_failed ?? 0;
  const verificationFailed = verification.status === "error";

  return (
    <details
      className="disclosure"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary data-testid="verification-coverage-summary">
        Claim verification
        {claimsTotal > 0
          ? ` (${claimsComplete}/${claimsTotal} fully evaluated - ${pairsEvaluated}/${pairsExpected} evidence pairs - ${verification.supported ?? 0} supported - ${verification.contradicted ?? 0} contradicted - ${verification.neutral ?? 0} neutral${pairsMissing > 0 ? ` - ${pairsMissing} pairs missing` : ""}${pairsFailed > 0 ? ` - ${pairsFailed} pair failures` : ""})`
          : ""}
      </summary>
      <p className="muted">{verification.note}</p>
      {verification.details?.error && (
        <p className="note">
          Verification error: {verification.details.error}
        </p>
      )}
      {verificationWarnings.length > 0 && (
        <ul className="note">
          {verificationWarnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      {claims.length > 0 ? (
        <ol className="source-list">
          {claims.map((claim, claimIndex) => (
            <li
              key={`${claim.claim.claim_id}-${claimIndex}`}
              className="source"
              data-testid="verification-claim"
            >
              <div className="source-head">
                <span className="result-rank">{claim.claim.claim_id}</span>
                <StatusBadge
                  tone={claimVerificationStatusTone(claim.status ?? "neutral")}
                >
                  {claimVerificationStatusLabel(claim.status ?? "neutral")}
                </StatusBadge>
                <span className="muted">
                  {claim.pairs_evaluated}/{claim.pairs_expected} evidence
                  comparisons
                </span>
                {claim.label && (
                  <span className="muted">
                    best evaluated pair: {claim.label}
                    {claim.confidence != null
                      ? ` - ${claim.confidence.toFixed(3)}`
                      : ""}
                  </span>
                )}
              </div>
              <p className="result-text">{claim.claim.claim}</p>
              {claim.claim.context && (
                <p className="muted">
                  Lead-in context: {claim.claim.context}
                </p>
              )}
              {claim.claim.hypothesis &&
                claim.claim.hypothesis !== claim.claim.claim && (
                  <p className="muted">
                    NLI hypothesis: {claim.claim.hypothesis}
                  </p>
                )}
              {claim.error && (
                <p className="note">Verification error: {claim.error}</p>
              )}
              {claim.explanation && (
                <p className="muted">{claim.explanation}</p>
              )}
              {(() => {
                const scores = formatClaimScores(
                  claim.probabilities,
                  claim.support_confidence,
                  claim.contradiction_confidence,
                );
                return scores ? (
                  <p className="muted mono">NLI scores: {scores}</p>
                ) : null;
              })()}
              {claim.incomplete_reason && (
                <p className="note">{claim.incomplete_reason}</p>
              )}
              {claim.matched_evidence.length > 0 && (
                <details className="source-text">
                  <summary>
                    Evidence comparisons ({claim.pairs_evaluated}/
                    {claim.pairs_expected} evaluated)
                  </summary>
                  <VerificationPairTable evidence={claim.matched_evidence} />
                </details>
              )}
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">
          {verificationFailed
            ? "No claim results are available because verification failed."
            : "No factual claims were extracted for verification."}
        </p>
      )}
      {pairsMissing > 0 && (
        <p className="note">
          {pairsMissing} evidence comparison(s) were unavailable.
        </p>
      )}
      {claimsIncomplete > 0 && (
        <p className="note">
          {claimsIncomplete} claim(s) have partial evidence-pair coverage.
          Their verdicts use only pairs with usable NLI results.
        </p>
      )}
    </details>
  );
}
