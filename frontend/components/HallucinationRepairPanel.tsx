"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Info,
  Quote,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  RequestCancelledError,
  checkHallucinations,
  getRepairResult,
  getRepairSource,
  runRepair,
} from "@/lib/api";
import {
  claimVerificationStatusLabel,
  claimVerificationStatusTone,
  formatDateTime,
  formatNumber,
  hallucinationVerdictLabel,
  hallucinationVerdictTone,
  type Tone,
} from "@/lib/format";
import type {
  FinalAnswerClaim,
  HallucinationCheckResponse,
  RagQueryResponse,
  RepairAttempt,
  RepairClaimResult,
  RepairClaimStatus,
  RepairDecision,
  RepairOverallStatus,
  RepairOverview,
  RepairResult,
  RepairSourceResponse,
} from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { HallucinationHighlightedAnswer } from "./HallucinationCheckPanel";
import { useWorkspace } from "./WorkspaceProvider";
import styles from "./HallucinationRepairPanel.module.css";

const REPAIR_STATUS_META: Record<
  RepairClaimStatus,
  { label: string; tone: Tone }
> = {
  ORIGINAL: { label: "Supported — No repair required", tone: "success" },
  REPAIRING: { label: "Repairing", tone: "info" },
  REPAIRED_PENDING_VERIFICATION: {
    label: "Verifying candidate",
    tone: "info",
  },
  VERIFIED: { label: "Verified", tone: "success" },
  RETRY_REQUIRED: { label: "Retry required", tone: "warning" },
  UNRESOLVED: { label: "Unresolved", tone: "error" },
  REMOVED: { label: "Removed", tone: "neutral" },
  INSUFFICIENT_EVIDENCE: {
    label: "Insufficient evidence",
    tone: "warning",
  },
  REPAIR_ERROR: { label: "Processing failure", tone: "error" },
};

const DECISION_META: Record<RepairDecision, { label: string; tone: Tone }> = {
  REPAIRED: { label: "Repaired", tone: "success" },
  QUALIFIED: { label: "Qualified", tone: "info" },
  REMOVED: { label: "Removed", tone: "neutral" },
  INSUFFICIENT_EVIDENCE: {
    label: "Insufficient evidence",
    tone: "warning",
  },
};

const OVERALL_META: Record<
  RepairOverallStatus,
  { label: string; tone: Tone }
> = {
  FULLY_VERIFIED: { label: "Fully verified", tone: "success" },
  PARTIALLY_VERIFIED: { label: "Partially verified", tone: "warning" },
  REPAIR_INCOMPLETE: { label: "Repair incomplete", tone: "error" },
  NO_REPAIR_NEEDED: { label: "No repair needed", tone: "info" },
};

interface FinalSegment {
  text: string;
  claim?: FinalAnswerClaim;
}

export function buildFinalSegments(
  answer: string,
  claims: FinalAnswerClaim[],
): FinalSegment[] {
  const valid = claims
    .filter(
      (claim) =>
        Number.isInteger(claim.start) &&
        Number.isInteger(claim.end) &&
        claim.start >= 0 &&
        claim.end > claim.start &&
        claim.end <= answer.length,
    )
    .sort((left, right) => left.start - right.start);

  if (valid.length === 0) {
    return [{ text: answer }];
  }

  const segments: FinalSegment[] = [];
  let cursor = 0;
  for (const claim of valid) {
    if (claim.start < cursor) {
      continue;
    }
    if (claim.start > cursor) {
      segments.push({ text: answer.slice(cursor, claim.start) });
    }
    segments.push({ text: answer.slice(claim.start, claim.end), claim });
    cursor = claim.end;
  }
  if (cursor < answer.length) {
    segments.push({ text: answer.slice(cursor) });
  }
  return segments;
}

const FALLBACK_REPAIR_MAX_ATTEMPTS = 3;

function probability(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function isMissingEndpoint(caught: unknown): boolean {
  return (
    caught instanceof ApiError &&
    caught.status === 404 &&
    caught.message.trim().toLowerCase() === "not found"
  );
}

function buildStoredOverview(
  result: RagQueryResponse,
  check: HallucinationCheckResponse,
): RepairOverview {
  const summary = result.verification?.details?.summary;
  return {
    claims_total: check.summary.claims_total,
    supported: check.summary.supported,
    potentially_hallucinated: check.summary.potentially_hallucinated,
    neutral: summary?.neutral ?? result.verification?.neutral ?? 0,
    nli_disagreements: check.summary.nli_disagreements,
    claims_unevaluable:
      summary?.claims_unevaluable ?? result.verification?.claims_unevaluable ?? 0,
    invalid_fragments_skipped: summary?.invalid_fragments_skipped ?? 0,
    evidence_total: check.summary.evidence_total,
  };
}

async function prepareRepairSource(
  result: RagQueryResponse,
  signal: AbortSignal,
): Promise<RepairSourceResponse> {
  const resultId = result.result_id;
  if (!resultId) {
    throw new ApiError("The Ask result has no identifier. Run the question again.");
  }

  try {
    return await getRepairSource(resultId, { signal });
  } catch (caught) {
    if (!isMissingEndpoint(caught)) {
      throw caught;
    }
  }

  const check = await checkHallucinations({ result_id: resultId }, { signal });
  let repair: RepairResult | null = null;
  try {
    repair = await getRepairResult(resultId, { signal });
  } catch (caught) {
    if (!(caught instanceof ApiError && caught.status === 404)) {
      throw caught;
    }
  }

  return {
    result_id: resultId,
    original: result,
    hallucination: check,
    overview: buildStoredOverview(result, check),
    repair,
    max_attempts: FALLBACK_REPAIR_MAX_ATTEMPTS,
  };
}

function formatSourceError(caught: unknown): string {
  if (!(caught instanceof ApiError)) {
    return "Could not prepare the repair source.";
  }
  if (caught.status === 404) {
    return isMissingEndpoint(caught)
      ? "The repair preparation endpoint is unavailable. Restart the backend so the repair routes are loaded."
      : caught.message;
  }
  if (caught.status === 422) {
    return `The repair preparation request was rejected: ${caught.message}`;
  }
  if (caught.status >= 500) {
    return `The backend could not prepare the repair source: ${caught.message}`;
  }
  return caught.message;
}

function formatRunError(caught: unknown): string {
  if (!(caught instanceof ApiError)) {
    return "Unexpected error while repairing the answer.";
  }
  if (caught.status === 422) {
    return `The repair request was rejected: ${caught.message}`;
  }
  if (caught.status >= 500) {
    return `The backend could not complete the repair: ${caught.message}`;
  }
  return caught.message;
}

function finalClaimClass(claim: FinalAnswerClaim): string {
  const tone =
    claim.verdict === "verified"
      ? "supported"
      : "potentially_hallucinated";
  if (claim.verdict === "insufficient_evidence") {
    return styles.insufficientEvidenceMark;
  }
  return `hallucination-mark hallucination-mark-${
    tone
  }`;
}

function RepairOverviewCard({
  overview,
  check,
}: {
  overview: RepairOverview;
  check: HallucinationCheckResponse;
}) {
  const excluded =
    overview.claims_unevaluable + overview.invalid_fragments_skipped;
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>Repair overview</h2>
          <p className="section-description">
            Hallucination Guard candidates are repaired; every candidate must
            then pass both NLI and Hallucination Guard again.
          </p>
        </div>
        <StatusBadge tone="info">
          {overview.claims_total} claim(s) inspected
        </StatusBadge>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-head">
            <span className="stat-icon stat-icon-success">
              <CheckCircle2 size={16} aria-hidden="true" />
            </span>
            <span className="stat-label">Supported</span>
          </div>
          <p className="stat-value">{overview.supported}</p>
          <p className="stat-hint">Kept unchanged in the final answer</p>
        </div>
        <div className="stat-card">
          <div className="stat-head">
            <span className="stat-icon stat-icon-error">
              <AlertTriangle size={16} aria-hidden="true" />
            </span>
            <span className="stat-label">Repair candidates</span>
          </div>
          <p className="stat-value">{overview.potentially_hallucinated}</p>
          <p className="stat-hint">Not supported by the supplied evidence</p>
        </div>
        <div className="stat-card">
          <div className="stat-head">
            <span className="stat-icon stat-icon-warning">
              <ShieldCheck size={16} aria-hidden="true" />
            </span>
            <span className="stat-label">NLI disagreements</span>
          </div>
          <p className="stat-value">{overview.nli_disagreements}</p>
          <p className="stat-hint">Independent signals differ; no fusion</p>
        </div>
        <div className="stat-card">
          <div className="stat-head">
            <span className="stat-icon stat-icon-info">
              <Info size={16} aria-hidden="true" />
            </span>
            <span className="stat-label">Excluded fragments</span>
          </div>
          <p className="stat-value">{excluded}</p>
          <p className="stat-hint">
            {overview.neutral} neutral · {overview.evidence_total} evidence
            chunk(s)
          </p>
        </div>
      </div>

      <p className="note">
        The binary guard verdict and the independent NLI result remain
        separate. A correction is accepted only when both systems support it.
        The original Ask result and its stored verification are never modified.
      </p>
      <p className="muted">
        Guard model: {check.model_name} · {check.summary.pairs_evaluated}{" "}
        comparison(s) across {check.summary.evidence_total} evidence chunk(s)
      </p>
      {check.warnings.filter(
        (warning) =>
          !warning.endsWith(
            "heading or incomplete fragment(s) were excluded from Hallucination Guard verification.",
          ),
      ).length > 0 && (
        <div className="stack" style={{ marginTop: 12 }}>
          {check.warnings.filter(
            (warning) =>
              !warning.endsWith(
                "heading or incomplete fragment(s) were excluded from Hallucination Guard verification.",
              ),
          ).map((warning) => (
            <ProcessingStatus
              key={warning}
              status="warning"
              message={warning}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function RepairResultView({ repair }: { repair: RepairResult }) {
  const meta = OVERALL_META[repair.status];
  const segments = buildFinalSegments(repair.final_answer, repair.final_claims);

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>Repaired answer</h2>
          <p className="section-description">
            Only claims that passed both independent verifiers are presented as
            verified. Every original claim remains visible with its current
            verification status.
          </p>
        </div>
        <StatusBadge tone={meta.tone}>{meta.label}</StatusBadge>
      </div>

      <div className="answer">
        <p className={`answer-text ${styles.finalAnswerText}`}>
          {segments.length === 0 ? (
            <span className="muted">No text remains after repair.</span>
          ) : (
            segments.map((segment, index) =>
              segment.claim ? (
                <span
                  key={`${segment.claim.claim_id}-${index}`}
                  className={finalClaimClass(segment.claim)}
                >
                  {segment.text}
                </span>
              ) : (
                <span key={`gap-${index}`}>{segment.text}</span>
              ),
            )
          )}
        </p>
      </div>

      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <StatusBadge tone="success">Verified</StatusBadge>
          <span className="muted">passed NLI and Hallucination Guard</span>
        </span>
        <span className={styles.legendItem}>
          <StatusBadge tone="warning">Unresolved</StatusBadge>
          <span className="muted">did not pass both verifiers</span>
        </span>
        <span className={styles.legendItem}>
          <StatusBadge tone="warning">Insufficient evidence</StatusBadge>
          <span className="muted">
            could not be established from the evidence
          </span>
        </span>
      </div>

      <div className="stat-grid" style={{ marginTop: 16 }}>
        <div className="stat-card">
          <span className="stat-label">Total original claims</span>
          <p className="stat-value">{repair.claims.length}</p>
        </div>
        <div className="stat-card">
          <span className="stat-label">Originally supported</span>
          <p className="stat-value">
            {repair.claims.filter((claim) => !claim.requires_repair).length}
          </p>
        </div>
        <div className="stat-card">
          <span className="stat-label">Successfully repaired</span>
          <p className="stat-value">
            {repair.claims.filter((claim) => claim.status === "VERIFIED").length}
          </p>
        </div>
        <div className="stat-card">
          <span className="stat-label">Repair attempts</span>
          <p className="stat-value">
            {repair.claims.reduce(
              (total, claim) => total + claim.attempts.length,
              0,
            )}
          </p>
        </div>
      </div>
      <p className="note">
        Initially unsupported:{" "}
        {repair.claims.filter((claim) => claim.requires_repair).length} ·
        Unresolved:{" "}
        {
          repair.claims.filter(
            (claim) =>
              claim.status === "UNRESOLVED" || claim.status === "REMOVED",
          ).length
        }{" "}
        · Insufficient evidence:{" "}
        {
          repair.claims.filter(
            (claim) => claim.status === "INSUFFICIENT_EVIDENCE",
          ).length
        }
      </p>

      {(repair.iterations ?? []).length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Iteration history</h3>
          <div className="stack">
            {(repair.iterations ?? []).map((iteration) => (
              <div key={iteration.iteration_number} className="source">
                <div className="source-head">
                  <strong>Iteration {iteration.iteration_number}</strong>
                  <span className="muted">
                    {iteration.submitted_claim_ids.length} claim(s) submitted
                  </span>
                </div>
                <p className="muted">
                  Claims:{" "}
                  {iteration.submitted_claim_ids.length > 0
                    ? iteration.submitted_claim_ids.join(", ")
                    : "—"}
                </p>
                <p className="muted">
                  Verified: {iteration.verified_claim_ids.length} · Still
                  unsupported: {iteration.remaining_claim_ids.length}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {repair.status === "NO_REPAIR_NEEDED" && (
        <p className="note">
          Hallucination Guard found no unsupported claims, so the final answer
          keeps the original text.
        </p>
      )}
      {repair.status === "REPAIR_INCOMPLETE" && (
        <ProcessingStatus
          status="error"
          message="At least one claim encountered a non-recoverable repair error."
          detail="The affected claim is not presented as a verified fact."
        />
      )}

      {repair.omissions.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <h3>Claim status notes</h3>
          <ul className={styles.omissionList}>
            {repair.omissions.map((omission) => (
              <li key={omission.claim_id} className={styles.omission}>
                <strong>{omission.original_claim}</strong>
                <span className="muted">{omission.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {repair.warnings.length > 0 && (
        <div className="stack" style={{ marginTop: 12 }}>
          {repair.warnings.map((warning) => (
            <ProcessingStatus
              key={warning}
              status="warning"
              message={warning}
            />
          ))}
        </div>
      )}

      <div className={styles.resultMeta}>
        <span>Maximum repair attempts: 3</span>
        <span>
          {repair.model_provider ?? "unknown provider"} ·{" "}
          {repair.model_name ?? "unknown model"}
        </span>
        <span>{formatNumber(repair.timings.total_ms)} ms</span>
        <span>{formatDateTime(repair.created_at)}</span>
      </div>
    </section>
  );
}

function AttemptItem({
  attempt,
  claimId,
}: {
  attempt: RepairAttempt;
  claimId: string;
}) {
  const status = REPAIR_STATUS_META[attempt.status];
  const decision = attempt.decision ? DECISION_META[attempt.decision] : null;
  const verification = attempt.verification;

  return (
    <li className={styles.attempt}>
      <div className={styles.attemptHead}>
        <span className={styles.attemptTitle}>
          Attempt {attempt.attempt_number}
        </span>
        <StatusBadge tone={status.tone}>{status.label}</StatusBadge>
        {decision && (
          <StatusBadge tone={decision.tone}>{decision.label}</StatusBadge>
        )}
        <span className={styles.attemptMeta}>
          {formatNumber(attempt.duration_ms)} ms
        </span>
      </div>

      {attempt.repaired_claim && (
        <p>
          <span className={styles.attemptLabel}>Candidate: </span>
          {attempt.repaired_claim}
        </p>
      )}
      {attempt.reason && (
        <p className="muted">
          <span className={styles.attemptLabel}>Model reason: </span>
          {attempt.reason}
        </p>
      )}
      {attempt.failure_reason && (
        <p className="muted">
          <span className={styles.attemptLabel}>Verification: </span>
          {attempt.failure_reason}
        </p>
      )}
      {attempt.error && (
        <p className="muted">
          <span className={styles.attemptLabel}>Error: </span>
          {attempt.error}
        </p>
      )}

      {verification && (
        <dl className="details-grid details-grid-2">
          <div>
            <dt>NLI</dt>
            <dd>
              {verification.nli_status ? (
                <>
                  <StatusBadge
                    tone={claimVerificationStatusTone(verification.nli_status)}
                  >
                    {claimVerificationStatusLabel(verification.nli_status)}
                  </StatusBadge>
                  {verification.nli_confidence != null
                    ? ` · ${probability(verification.nli_confidence)}`
                    : ""}
                </>
              ) : (
                <span className="muted">Unavailable</span>
              )}
            </dd>
          </div>
          <div>
            <dt>Hallucination Guard</dt>
            <dd>
              {verification.guard_verdict ? (
                <>
                  <StatusBadge
                    tone={hallucinationVerdictTone(verification.guard_verdict)}
                  >
                    {hallucinationVerdictLabel(verification.guard_verdict)}
                  </StatusBadge>
                  {verification.guard_support_probability != null
                    ? ` · ${probability(
                        verification.guard_support_probability,
                      )}`
                    : ""}
                </>
              ) : (
                <span className="muted">Unavailable</span>
              )}
            </dd>
          </div>
          <div>
            <dt>Both supported</dt>
            <dd>{verification.both_supported ? "Yes" : "No"}</dd>
          </div>
        </dl>
      )}

      {attempt.supporting_evidence.length > 0 && (
        <ul className={styles.quoteList}>
          {attempt.supporting_evidence.map((reference, index) => (
            <li
              key={`${claimId}-${attempt.attempt_number}-quote-${index}`}
              className={styles.quote}
            >
              <cite>
                {reference.source}
                {reference.page != null ? `, page ${reference.page}` : ""}
              </cite>
              “{reference.evidence_quote}”
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function ClaimRow({
  claim,
  expanded,
  onToggle,
}: {
  claim: RepairClaimResult;
  expanded: boolean;
  onToggle: () => void;
}) {
  const status = REPAIR_STATUS_META[claim.status];
  const decision = claim.final_decision
    ? DECISION_META[claim.final_decision]
    : null;

  return (
    <>
      <tr>
        <td>
          <span className={styles.claimCell}>
            <span>{claim.original_claim}</span>
            {claim.original_context && (
              <span className={styles.subtle}>{claim.original_context}</span>
            )}
          </span>
        </td>
        <td>
          <StatusBadge tone={hallucinationVerdictTone(claim.original_verdict)}>
            {hallucinationVerdictLabel(claim.original_verdict)}
          </StatusBadge>
          <div className={styles.subtle}>
            {probability(claim.original_support_probability)}
          </div>
        </td>
        <td>
          <StatusBadge tone={status.tone}>{status.label}</StatusBadge>
          {decision && (
            <div className={styles.subtle}>{decision.label}</div>
          )}
        </td>
        <td>{claim.attempts.length}</td>
        <td className="table-actions-head">
          <button
            type="button"
            className="button button-secondary button-small"
            onClick={onToggle}
            aria-expanded={expanded}
          >
            {expanded ? (
              <ChevronUp size={14} aria-hidden="true" />
            ) : (
              <ChevronDown size={14} aria-hidden="true" />
            )}
            {expanded ? "Hide" : "Inspect"}
          </button>
        </td>
      </tr>
      {expanded && (
        <tr className="details-row">
          <td colSpan={5}>
            <p className="hallucination-expanded-claim">
              {claim.original_claim}
            </p>
            {claim.original_context && (
              <p className="muted">Section: {claim.original_context}</p>
            )}

            <dl className="details-grid details-grid-2">
              <div>
                <dt>Original NLI</dt>
                <dd>
                  {claim.original_nli_status ? (
                    <>
                      <StatusBadge
                        tone={claimVerificationStatusTone(
                          claim.original_nli_status,
                        )}
                      >
                        {claimVerificationStatusLabel(
                          claim.original_nli_status,
                        )}
                      </StatusBadge>
                      {claim.original_nli_confidence != null
                        ? ` · ${probability(claim.original_nli_confidence)}`
                        : ""}
                    </>
                  ) : (
                    <span className="muted">Unavailable</span>
                  )}
                </dd>
              </div>
              <div>
                <dt>Verification model label</dt>
                <dd className="mono">{claim.original_raw_model_label}</dd>
              </div>
              <div>
                <dt>Final decision</dt>
                <dd>{decision?.label ?? "—"}</dd>
              </div>
              <div>
                <dt>Included in final answer</dt>
                <dd>{claim.included_in_final ? "Yes" : "No"}</dd>
              </div>
            </dl>

            {claim.final_claim && (
              <p style={{ marginTop: 12 }}>
                <span className={styles.attemptLabel}>Final claim: </span>
                {claim.final_claim}
              </p>
            )}

            <div style={{ marginTop: 14 }}>
              <h4>Claim-specific evidence</h4>
              {claim.selected_evidence.length === 0 ? (
                <p className={styles.emptyBlock}>
                  No claim-specific evidence was available.
                </p>
              ) : (
                <ol className="source-list">
                  {claim.selected_evidence.map((evidence) => (
                    <li key={evidence.evidence_id} className="source">
                      <div className="source-head">
                        <Quote size={13} aria-hidden="true" />
                        <span
                          className="result-file"
                          title={evidence.filename}
                        >
                          {evidence.filename}
                        </span>
                        {evidence.page_number != null && (
                          <span className="source-page">
                            page {evidence.page_number}
                          </span>
                        )}
                        <StatusBadge tone="neutral">
                          support{" "}
                          {probability(evidence.guard_support_probability)}
                        </StatusBadge>
                        <span className="muted mono">
                          {evidence.chunk_id}
                        </span>
                      </div>
                      <p className="result-meta">
                        retrieval score{" "}
                        {evidence.retrieval_score.toFixed(4)} · similarity{" "}
                        {evidence.retrieval_similarity.toFixed(4)}
                      </p>
                      <details className="source-text">
                        <summary>View evidence text</summary>
                        <p className="result-text">{evidence.text}</p>
                      </details>
                    </li>
                  ))}
                </ol>
              )}
            </div>

            <div style={{ marginTop: 14 }}>
              <h4>Attempt history</h4>
              {claim.attempts.length === 0 ? (
                <p className={styles.emptyBlock}>
                  No repair was attempted because this claim is supported.
                </p>
              ) : (
                <ol className={styles.attemptList}>
                  {claim.attempts.map((attempt) => (
                    <AttemptItem
                      key={`${claim.claim_id}-${attempt.attempt_number}`}
                      attempt={attempt}
                      claimId={claim.claim_id}
                    />
                  ))}
                </ol>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function ClaimTrail({ repair }: { repair: RepairResult }) {
  const [expandedClaimId, setExpandedClaimId] = useState<string | null>(null);

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>Claim-by-claim repair trail</h2>
          <p className="section-description">
            Inspect the evidence supplied to the repair model and every
            independently verified attempt.
          </p>
        </div>
        <StatusBadge tone="neutral">
          {repair.claims.length} claim(s)
        </StatusBadge>
      </div>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Original claim</th>
              <th>Guard</th>
              <th>Repair result</th>
              <th>Attempts</th>
              <th className="table-actions-head">Details</th>
            </tr>
          </thead>
          <tbody>
            {repair.claims.map((claim) => (
              <ClaimRow
                key={claim.claim_id}
                claim={claim}
                expanded={expandedClaimId === claim.claim_id}
                onToggle={() =>
                  setExpandedClaimId(
                    expandedClaimId === claim.claim_id
                      ? null
                      : claim.claim_id,
                  )
                }
              />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function HallucinationRepairPanel() {
  const {
    latestRagResult,
    latestRagResultLoading,
    latestRagResultError,
    clearLatestRagResult,
  } = useWorkspace();
  const [source, setSource] = useState<Awaited<
    ReturnType<typeof getRepairSource>
  > | null>(null);
  const [repair, setRepair] = useState<RepairResult | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const sourceAbortRef = useRef<AbortController | null>(null);
  const runAbortRef = useRef<AbortController | null>(null);
  const runInFlightRef = useRef(false);

  const resultId = latestRagResult?.result_id ?? null;
  const hasAnswer = Boolean(latestRagResult?.generation.used_llm);
  const hasEvidence = (latestRagResult?.sources.length ?? 0) > 0;
  const canRepair = Boolean(resultId && hasAnswer && hasEvidence);

  useEffect(() => {
    sourceAbortRef.current?.abort();
    runAbortRef.current?.abort();
    sourceAbortRef.current = null;
    runAbortRef.current = null;
    runInFlightRef.current = false;
    setSource(null);
    setRepair(null);
    setSourceError(null);
    setRunError(null);
    setSourceLoading(false);
    setRunning(false);

    if (!resultId || !canRepair || !latestRagResult) {
      return;
    }

    const controller = new AbortController();
    sourceAbortRef.current = controller;
    setSourceLoading(true);

    void prepareRepairSource(latestRagResult, controller.signal)
      .then((response) => {
        if (sourceAbortRef.current !== controller) {
          return;
        }
        setSource(response);
        setRepair(response.repair);
      })
      .catch((caught: unknown) => {
        if (sourceAbortRef.current !== controller) {
          return;
        }
        if (caught instanceof RequestCancelledError) {
          return;
        }
        setSourceError(formatSourceError(caught));
      })
      .finally(() => {
        if (sourceAbortRef.current === controller) {
          sourceAbortRef.current = null;
          setSourceLoading(false);
        }
      });

    return () => {
      controller.abort();
      if (sourceAbortRef.current === controller) {
        sourceAbortRef.current = null;
      }
    };
  }, [canRepair, reloadToken, resultId]);

  useEffect(
    () => () => {
      sourceAbortRef.current?.abort();
      runAbortRef.current?.abort();
      sourceAbortRef.current = null;
      runAbortRef.current = null;
      runInFlightRef.current = false;
    },
    [],
  );

  async function handleRepair() {
    if (
      !resultId ||
      !source ||
      !canRepair ||
      repair ||
      running ||
      runInFlightRef.current
    ) {
      return;
    }
    runInFlightRef.current = true;
    const controller = new AbortController();
    runAbortRef.current = controller;
    setRunning(true);
    setRunError(null);
    try {
      const response = await runRepair(
        {
          result_id: resultId,
          max_attempts: 3,
        },
        { signal: controller.signal },
      );
      if (runAbortRef.current === controller) {
        setRepair(response);
      }
    } catch (caught) {
      if (runAbortRef.current !== controller) {
        return;
      }
      if (caught instanceof RequestCancelledError) {
        setRunError("The repair request was cancelled.");
      } else {
        setRunError(formatRunError(caught));
      }
    } finally {
      if (runAbortRef.current === controller) {
        runAbortRef.current = null;
        runInFlightRef.current = false;
        setRunning(false);
      }
    }
  }

  function handleClear() {
    sourceAbortRef.current?.abort();
    runAbortRef.current?.abort();
    sourceAbortRef.current = null;
    runAbortRef.current = null;
    runInFlightRef.current = false;
    clearLatestRagResult();
  }

  if (latestRagResultLoading) {
    return (
      <section className="card">
        <ProcessingStatus
          status="loading"
          message="Restoring the latest Ask result..."
          detail="Only the opaque result identifier is kept in this browser."
        />
      </section>
    );
  }

  if (!latestRagResult) {
    return (
      <section className="card">
        {latestRagResultError && (
          <ProcessingStatus status="error" message={latestRagResultError} />
        )}
        <EmptyState
          icon={Sparkles}
          title="No Ask result to repair"
          description="Ask a question in Ask Documents first. Hallucination Repair reuses that exact answer, its verification, and its retrieved evidence without rerunning retrieval or answer generation."
          action={
            <Link className="button button-primary" href="/ask">
              Go to Ask Documents
            </Link>
          }
        />
      </section>
    );
  }

  const alreadyRepaired = repair !== null;
  const repairButtonLabel = running
    ? "Repairing claims..."
    : alreadyRepaired
      ? "Repair completed"
      : sourceLoading
        ? "Preparing..."
        : "Repair unsupported claims";

  return (
    <div className="stack">
      <section className="card">
        <div className="card-header">
          <div>
            <h2>Repair source</h2>
            <p className="section-description">
              The repair layer reuses the latest Ask result and its retrieved
              evidence. It never reruns retrieval or the original generation.
            </p>
          </div>
          {repair ? (
            <StatusBadge tone={OVERALL_META[repair.status].tone}>
              {OVERALL_META[repair.status].label}
            </StatusBadge>
          ) : source ? (
            <StatusBadge
              tone={
                source.overview.potentially_hallucinated > 0
                  ? "error"
                  : "success"
              }
            >
              {source.overview.potentially_hallucinated} repair candidate(s)
            </StatusBadge>
          ) : (
            <StatusBadge tone="info">Preparing</StatusBadge>
          )}
        </div>

        <dl className="details-grid details-grid-2">
          <div>
            <dt>Question</dt>
            <dd>{latestRagResult.question}</dd>
          </div>
          <div>
            <dt>Answer model</dt>
            <dd>{latestRagResult.generation.model ?? "—"}</dd>
          </div>
          <div>
            <dt>Evidence chunks</dt>
            <dd>{latestRagResult.sources.length}</dd>
          </div>
          <div>
            <dt>Existing NLI</dt>
            <dd>
              {latestRagResult.verification?.status ?? "not_performed"} ·{" "}
              {latestRagResult.verification?.supported ?? 0} supported ·{" "}
              {latestRagResult.verification?.contradicted ?? 0} contradicted ·{" "}
              {latestRagResult.verification?.neutral ?? 0} neutral
            </dd>
          </div>
          <div>
            <dt>Ask result ID</dt>
            <dd className="mono">{resultId ?? "—"}</dd>
          </div>
        </dl>

        {!hasAnswer && (
          <ProcessingStatus
            status="warning"
            message="This result has no generated answer."
            detail="Ask Documents must return a generated answer before it can be repaired."
          />
        )}
        {!hasEvidence && (
          <ProcessingStatus
            status="warning"
            message="This result has no retrieved evidence."
            detail="There is no document evidence available to verify a correction."
          />
        )}

        <div className={styles.controls}>
          <div className={styles.attemptField}>
            <span className="muted">Maximum repair attempts: 3</span>
          </div>
          <div className={styles.controlsSpacer} />
          <button
            type="button"
            className="button button-primary"
            onClick={handleRepair}
            disabled={
              running ||
              sourceLoading ||
              !source ||
              !canRepair ||
              alreadyRepaired
            }
            aria-busy={running}
          >
            <Wrench size={15} aria-hidden="true" />
            {repairButtonLabel}
          </button>
          <button
            type="button"
            className="button"
            onClick={handleClear}
            disabled={running}
          >
            Clear shared result
          </button>
        </div>

        {sourceLoading && (
          <ProcessingStatus
            status="loading"
            message="Preparing repair source..."
            detail="Reusing the stored answer and evidence, then running the existing NLI and Hallucination Guard checks."
          />
        )}
        {sourceError && (
          <>
            <ProcessingStatus status="error" message={sourceError} />
            <div className="form-actions" style={{ marginTop: 10 }}>
              <button
                type="button"
                className="button button-secondary"
                onClick={() => setReloadToken((value) => value + 1)}
              >
                <RefreshCw size={14} aria-hidden="true" />
                Retry preparation
              </button>
            </div>
          </>
        )}
        {running && (
          <ProcessingStatus
            status="loading"
            message="Repairing unsupported claims..."
            detail="Each candidate is regenerated from the supplied evidence and independently re-verified. This can take several attempts."
          />
        )}
        {runError && <ProcessingStatus status="error" message={runError} />}
        {alreadyRepaired && (
          <p className="note">
            A repair result is already stored for this Ask result. It is reused
            when the page is refreshed; the attempt limit is locked.
          </p>
        )}
        {!alreadyRepaired &&
          source &&
          source.overview.potentially_hallucinated === 0 && (
            <p className="note">
              Hallucination Guard found no unsupported claims. Running repair
              confirms the answer without changing its text.
            </p>
          )}
      </section>

      {source && (
        <RepairOverviewCard
          overview={source.overview}
          check={source.hallucination}
        />
      )}

      {source && (
        <HallucinationHighlightedAnswer
          key={source.hallucination.check_id}
          answer={source.original.answer}
          check={source.hallucination}
        />
      )}

      {repair && <RepairResultView key={repair.repair_id} repair={repair} />}
      {repair && <ClaimTrail repair={repair} />}
    </div>
  );
}
