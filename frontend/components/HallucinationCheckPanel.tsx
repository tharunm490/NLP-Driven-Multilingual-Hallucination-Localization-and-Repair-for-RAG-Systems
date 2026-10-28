"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Eye,
  Quote,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ApiError, RequestCancelledError, checkHallucinations } from "@/lib/api";
import {
  claimVerificationStatusLabel,
  claimVerificationStatusTone,
  formatNumber,
  hallucinationVerdictLabel,
  hallucinationVerdictTone,
} from "@/lib/format";
import type {
  HallucinationCheckResponse,
  HallucinationClaimResult,
  HallucinationVerdict,
  RagQueryResponse,
} from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";

export interface HighlightSegment {
  text: string;
  claim?: HallucinationClaimResult;
}

export function buildHighlightSegments(
  answer: string,
  claims: HallucinationClaimResult[],
): HighlightSegment[] {
  const valid = claims
    .filter(
      (claim) =>
        Number.isInteger(claim.claim_start) &&
        Number.isInteger(claim.claim_end) &&
        claim.claim_start >= 0 &&
        claim.claim_end > claim.claim_start &&
        claim.claim_end <= answer.length,
    )
    .sort((left, right) => left.claim_start - right.claim_start);

  if (valid.length === 0) {
    return [{ text: answer }];
  }

  const segments: HighlightSegment[] = [];
  let cursor = 0;
  for (const claim of valid) {
    if (claim.claim_start < cursor) {
      continue;
    }
    if (claim.claim_start > cursor) {
      segments.push({ text: answer.slice(cursor, claim.claim_start) });
    }
    segments.push({
      text: answer.slice(claim.claim_start, claim.claim_end),
      claim,
    });
    cursor = claim.claim_end;
  }
  if (cursor < answer.length) {
    segments.push({ text: answer.slice(cursor) });
  }
  return segments;
}

function verdictClass(verdict: HallucinationVerdict): string {
  return `hallucination-mark hallucination-mark-${verdict}`;
}

function Probability({ value }: { value: number }) {
  return <span className="mono">{(value * 100).toFixed(1)}%</span>;
}

export function HallucinationResultView({
  answer,
  check,
}: {
  answer: string;
  check: HallucinationCheckResponse;
}) {
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(
    check.claims[0]?.claim_id ?? null,
  );
  const [expandedClaimId, setExpandedClaimId] = useState<string | null>(null);
  const segments = buildHighlightSegments(answer, check.claims);
  const selected =
    check.claims.find((claim) => claim.claim_id === selectedClaimId) ?? null;

  return (
    <>
      <section className="card">
        <div className="card-header">
          <div>
            <h2>Groundedness summary</h2>
            <p className="section-description">
              Dedicated local detector · {check.model_name}
            </p>
          </div>
          <StatusBadge tone="info">
            {check.summary.claims_total} claim(s) checked
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
            <p className="stat-value">{check.summary.supported}</p>
            <p className="stat-hint">Grounded by supplied evidence</p>
          </div>
          <div className="stat-card">
            <div className="stat-head">
              <span className="stat-icon stat-icon-error">
                <AlertTriangle size={16} aria-hidden="true" />
              </span>
              <span className="stat-label">Potentially hallucinated</span>
            </div>
            <p className="stat-value">
              {check.summary.potentially_hallucinated}
            </p>
            <p className="stat-hint">Low support from supplied evidence</p>
          </div>
          <div className="stat-card">
            <div className="stat-head">
              <span className="stat-icon stat-icon-warning">
                <Eye size={16} aria-hidden="true" />
              </span>
              <span className="stat-label">Needs review</span>
            </div>
            <p className="stat-value">{check.summary.needs_review}</p>
            <p className="stat-hint">Near the model decision boundary</p>
          </div>
          <div className="stat-card">
            <div className="stat-head">
              <span className="stat-icon stat-icon-info">
                <ShieldCheck size={16} aria-hidden="true" />
              </span>
              <span className="stat-label">NLI disagreements</span>
            </div>
            <p className="stat-value">{check.summary.nli_disagreements}</p>
            <p className="stat-hint">Independent signals differ; no fusion</p>
          </div>
        </div>

        <p className="note">
          The dedicated model is binary. “Potentially hallucinated” means the
          claim was not supported by the supplied evidence; it does not prove a
          contradiction. Support threshold {check.support_threshold.toFixed(2)},
          review band ±{check.review_margin.toFixed(2)}.
        </p>
        <p className="muted">
          {check.summary.pairs_evaluated} comparison(s) across{" "}
          {check.summary.evidence_total} evidence chunk(s) ·{" "}
          {formatNumber(check.timings.total_ms)} ms
        </p>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <h2>Answer with claim highlights</h2>
            <p className="section-description">
              Hover, focus, or select a highlighted claim to inspect its result.
            </p>
          </div>
        </div>
        <div className="answer">
          <p className="answer-text answer-highlight-text">
            {segments.map((segment, index) =>
              segment.claim ? (
                <button
                  key={`${segment.claim.claim_id}-${index}`}
                  type="button"
                  className={`${verdictClass(segment.claim.verdict)}${
                    segment.claim.claim_id === selectedClaimId
                      ? " hallucination-mark-selected"
                      : ""
                  }`}
                  onMouseEnter={() => setSelectedClaimId(segment.claim!.claim_id)}
                  onFocus={() => setSelectedClaimId(segment.claim!.claim_id)}
                  onClick={() => setSelectedClaimId(segment.claim!.claim_id)}
                >
                  {segment.text}
                </button>
              ) : (
                <span key={`gap-${index}`}>{segment.text}</span>
              ),
            )}
          </p>
        </div>

        {selected && (
          <div className="hallucination-inspector">
            <div className="hallucination-inspector-head">
              <StatusBadge tone={hallucinationVerdictTone(selected.verdict)}>
                {hallucinationVerdictLabel(selected.verdict)}
              </StatusBadge>
              <span className="muted">
                support <Probability value={selected.support_probability} /> · raw{" "}
                <span className="mono">{selected.raw_model_label}</span>
              </span>
            </div>
            <p>{selected.claim}</p>
            <p className="muted">{selected.explanation}</p>
            {selected.evidence[0] && (
              <p className="muted">
                Best evidence: {selected.evidence[0].filename}
                {selected.evidence[0].page_number != null
                  ? `, page ${selected.evidence[0].page_number}`
                  : ""}
              </p>
            )}
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <h2>Claim-level results</h2>
            <p className="section-description">
              NLI is shown independently. Disagreement is not resolved
              automatically.
            </p>
          </div>
        </div>

        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Claim</th>
                <th>Verdict</th>
                <th>Support</th>
                <th>NLI</th>
                <th>Evidence</th>
                <th className="table-actions-head">Details</th>
              </tr>
            </thead>
            <tbody>
              {check.claims.map((claim) => {
                const expanded = expandedClaimId === claim.claim_id;
                return (
                  <ClaimRows
                    key={claim.claim_id}
                    claim={claim}
                    expanded={expanded}
                    onSelect={() => setSelectedClaimId(claim.claim_id)}
                    onToggle={() =>
                      setExpandedClaimId(expanded ? null : claim.claim_id)
                    }
                  />
                );
              })}
            </tbody>
          </table>
        </div>

        {check.claims.length === 0 && (
          <p className="muted">No factual claims were extracted from the answer.</p>
        )}
        {check.warnings.length > 0 && (
          <div className="stack" style={{ marginTop: 12 }}>
            {check.warnings.map((warning) => (
              <ProcessingStatus
                key={warning}
                status="warning"
                message={warning}
              />
            ))}
          </div>
        )}
        <p className="note">
          Raw model outputs are shown without converting uncertainty into a
          factual contradiction. Dedicated model input order: evidence first,
          claim second.
        </p>
      </section>
    </>
  );
}

function ClaimRows({
  claim,
  expanded,
  onSelect,
  onToggle,
}: {
  claim: HallucinationClaimResult;
  expanded: boolean;
  onSelect: () => void;
  onToggle: () => void;
}) {
  return (
    <>
      <tr>
        <td>
          <button type="button" className="claim-text-button" onClick={onSelect}>
            {claim.claim}
          </button>
        </td>
        <td>
          <StatusBadge tone={hallucinationVerdictTone(claim.verdict)}>
            {hallucinationVerdictLabel(claim.verdict)}
          </StatusBadge>
        </td>
        <td>
          <Probability value={claim.support_probability} />
          <div className="muted mono">{claim.raw_model_label}</div>
        </td>
        <td>
          {claim.nli_status ? (
            <>
              <StatusBadge tone={claimVerificationStatusTone(claim.nli_status)}>
                {claimVerificationStatusLabel(claim.nli_status)}
              </StatusBadge>
              {claim.nli_disagrees && (
                <div className="hallucination-disagreement">Disagrees</div>
              )}
            </>
          ) : (
            <span className="muted">Unavailable</span>
          )}
        </td>
        <td>
          {claim.evidence[0] ? (
            <span>
              {claim.evidence[0].filename}
              {claim.evidence[0].page_number != null
                ? ` · p.${claim.evidence[0].page_number}`
                : ""}
            </span>
          ) : (
            <span className="muted">None</span>
          )}
        </td>
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
          <td colSpan={6}>
            <p className="hallucination-expanded-claim">{claim.claim}</p>
            {claim.nli_status && claim.nli_disagrees && (
              <ProcessingStatus
                status="warning"
                message="The dedicated detector and independent NLI result disagree."
                detail="Both signals are retained; neither overrides the other."
              />
            )}
            {claim.evidence.length === 0 ? (
              <p className="muted">No evidence comparisons were recorded.</p>
            ) : (
              <ol className="source-list">
                {claim.evidence.map((evidence) => (
                  <li key={evidence.evidence_id} className="source">
                    <div className="source-head">
                      <Quote size={13} aria-hidden="true" />
                      <span className="result-file" title={evidence.filename}>
                        {evidence.filename}
                      </span>
                      {evidence.page_number != null && (
                        <span className="source-page">
                          page {evidence.page_number}
                        </span>
                      )}
                      <StatusBadge tone="neutral">
                        support {(evidence.support_probability * 100).toFixed(1)}%
                      </StatusBadge>
                      <span className="muted mono">
                        {evidence.raw_model_label}
                      </span>
                    </div>
                    <p className="result-meta">
                      retrieval score {evidence.retrieval_score.toFixed(4)} ·
                      similarity {evidence.retrieval_similarity.toFixed(4)}
                    </p>
                  </li>
                ))}
              </ol>
            )}
            <dl className="details-grid details-grid-2">
              <div>
                <dt>NLI status</dt>
                <dd>
                  {claim.nli_status
                    ? claimVerificationStatusLabel(claim.nli_status)
                    : "Unavailable"}
                  {claim.nli_confidence != null
                    ? ` (${(claim.nli_confidence * 100).toFixed(1)}%)`
                    : ""}
                </dd>
              </div>
              <div>
                <dt>Dedicated raw label</dt>
                <dd className="mono">{claim.raw_model_label}</dd>
              </div>
              <div>
                <dt>Dedicated support</dt>
                <dd>
                  <Probability value={claim.support_probability} />
                </dd>
              </div>
              <div>
                <dt>Answer offsets</dt>
                <dd className="mono">
                  {claim.claim_start}–{claim.claim_end}
                </dd>
              </div>
            </dl>
          </td>
        </tr>
      )}
    </>
  );
}

export default function HallucinationCheckPanel() {
  const {
    latestRagResult,
    latestRagResultLoading,
    latestRagResultError,
    clearLatestRagResult,
  } = useWorkspace();
  const [check, setCheck] = useState<HallucinationCheckResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const checkInFlightRef = useRef(false);

  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    checkInFlightRef.current = false;
    setLoading(false);
    setCheck(null);
    setError(null);
    setCancelled(false);
  }, [latestRagResult?.result_id]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      abortRef.current = null;
      checkInFlightRef.current = false;
    },
    [],
  );

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
          <ProcessingStatus
            status="error"
            message={latestRagResultError}
          />
        )}
        <EmptyState
          icon={Sparkles}
          title="No Ask result to check"
          description="Ask a question in Ask Documents first. Hallucination Checks reuses that exact answer and its retrieved evidence without rerunning retrieval or generation."
          action={
            <Link className="button button-primary" href="/ask">
              Go to Ask Documents
            </Link>
          }
        />
      </section>
    );
  }

  const resultId = latestRagResult.result_id;
  const hasAnswer = latestRagResult.generation.used_llm;
  const hasEvidence = latestRagResult.sources.length > 0;
  const canCheck = Boolean(resultId && hasAnswer && hasEvidence);

  async function handleCheck() {
    if (!resultId || !canCheck || checkInFlightRef.current) {
      return;
    }
    checkInFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    setCancelled(false);
    try {
      const response = await checkHallucinations(
        { result_id: resultId },
        { signal: controller.signal },
      );
      if (abortRef.current === controller) {
        setCheck(response);
      }
    } catch (caught) {
      if (abortRef.current !== controller) {
        return;
      }
      if (caught instanceof RequestCancelledError) {
        setCancelled(true);
      } else if (caught instanceof ApiError) {
        setError(caught.message);
      } else {
        setError("Unexpected error while checking hallucinations.");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        checkInFlightRef.current = false;
        setLoading(false);
      }
    }
  }

  function handleClear() {
    abortRef.current?.abort();
    abortRef.current = null;
    checkInFlightRef.current = false;
    clearLatestRagResult();
  }

  return (
    <div className="stack">
      <section className="card">
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

        <div className="answer">
          <p className="answer-text">{latestRagResult.answer}</p>
        </div>

        {!hasAnswer && (
          <ProcessingStatus
            status="warning"
            message="This result has no generated answer."
            detail="Ask Documents must return a generated answer before it can be checked."
          />
        )}
        {!hasEvidence && (
          <ProcessingStatus
            status="warning"
            message="This result has no retrieved evidence."
            detail="There is nothing to compare the generated claims against."
          />
        )}

        <div className="form-actions">
          <button
            type="button"
            className="button button-primary"
            onClick={handleCheck}
            disabled={loading || !canCheck}
            aria-busy={loading}
          >
            <ShieldCheck size={15} aria-hidden="true" />
            {loading ? "Checking..." : "Check for Hallucinations"}
          </button>
          {loading && (
            <button
              type="button"
              className="button"
              onClick={() => abortRef.current?.abort()}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            className="button"
            onClick={handleClear}
          >
            Clear shared result
          </button>
        </div>

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Running the local hallucination model..."
            detail="Every extracted claim is compared with the supplied evidence chunks. NLI results are left unchanged."
          />
        )}
        {cancelled && !loading && (
          <ProcessingStatus
            status="warning"
            message="Hallucination check cancelled."
            detail="No local verdict was recorded for this run."
          />
        )}
        {error && <ProcessingStatus status="error" message={error} />}
      </section>

      {check && (
        <HallucinationResultView
          answer={latestRagResult.answer}
          check={check}
        />
      )}
    </div>
  );
}
