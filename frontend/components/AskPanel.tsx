"use client";

import { AlertTriangle, Send, ShieldCheck, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  RequestCancelledError,
  askQuestion,
  verifyRagResult,
} from "@/lib/api";
import {
  answerStatusLabel,
  answerStatusShowsAbstention,
  answerStatusTone,
  formatNumber,
} from "@/lib/format";

import ClaimVerificationPanel from "./ClaimVerificationPanel";
import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import RetrievedEvidenceList from "./RetrievedEvidenceList";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";
import { DocumentFilter } from "./fields";

export default function AskPanel() {
  const {
    documents,
    latestRagResult,
    latestRagResultLoading,
    latestRagResultError,
    setLatestRagResult,
    clearLatestRagResult,
  } = useWorkspace();
  const [question, setQuestion] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const [verifyingResultId, setVerifyingResultId] = useState<string | null>(
    null,
  );
  const [verificationError, setVerificationError] = useState<string | null>(
    null,
  );
  const [verificationCancelled, setVerificationCancelled] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const verificationAbortRef = useRef<AbortController | null>(null);
  const verificationInFlightRef = useRef(false);
  const currentResultIdRef = useRef<string | null>(null);

  const response = latestRagResult;
  currentResultIdRef.current = response?.result_id ?? null;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get("document");
    if (requested) {
      setDocumentId(requested);
    }
    return () => {
      abortRef.current?.abort();
      verificationAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    verificationAbortRef.current?.abort();
    verificationAbortRef.current = null;
    verificationInFlightRef.current = false;
    setVerifyingResultId(null);
    setVerificationError(null);
    setVerificationCancelled(false);
  }, [response?.result_id]);

  function handleCancel() {
    abortRef.current?.abort();
  }

  function handleCancelVerification() {
    verificationAbortRef.current?.abort();
    verificationAbortRef.current = null;
    verificationInFlightRef.current = false;
    setVerifyingResultId(null);
    setVerificationCancelled(true);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || loading) {
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    verificationAbortRef.current?.abort();
    verificationAbortRef.current = null;
    verificationInFlightRef.current = false;
    setVerifyingResultId(null);
    setVerificationError(null);
    setVerificationCancelled(false);
    clearLatestRagResult();
    setLoading(true);
    setError(null);
    setCancelled(false);
    try {
      const result = await askQuestion(
        {
          question: question.trim(),
          top_k: 5,
          document_ids: documentId ? [documentId] : undefined,
        },
        { signal: controller.signal },
      );
      setLatestRagResult(result);
    } catch (caught) {
      if (caught instanceof RequestCancelledError) {
        setCancelled(true);
      } else if (caught instanceof ApiError) {
        if (caught.status === 503 && /provider|api key|model/i.test(caught.message)) {
          setError(
            `${caught.message} Retrieval still works in the Retrieval Playground.`,
          );
        } else {
          setError(caught.message);
        }
      } else {
        setError("Unexpected error while generating an answer.");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setLoading(false);
    }
  }

  const answerStatus = response?.answer_status ?? response?.answer_mode;
  const showsAbstention = answerStatusShowsAbstention(answerStatus);
  const showsPartialSupport = answerStatus === "partially_grounded";
  const showsContradiction = answerStatus === "contradicted";
  const verification = response?.verification;
  const verificationFailed = verification?.status === "error";
  const verificationPartial = verification?.status === "partial";
  const claimsChecked = verification?.claims_checked ?? 0;
  const claimsTotal = verification?.claims_total ?? 0;
  const claimsComplete = verification?.claims_complete ?? 0;
  const claimsIncomplete = verification?.claims_incomplete ?? 0;
  const verificationErrors = verification?.verification_error ?? 0;
  const pairsExpected = verification?.pairs_expected ?? 0;
  const pairsEvaluated = verification?.pairs_evaluated ?? 0;
  const pairsMissing = verification?.pairs_missing ?? 0;
  const resultId = response?.result_id ?? null;
  const isVerifying = Boolean(resultId && verifyingResultId === resultId);
  const canVerify = Boolean(
    resultId &&
      (verification?.status === "pending" || verification?.status === "error"),
  );
  const showsVerificationPanel =
    verification?.status === "verified" || verification?.status === "partial";

  async function handleVerify() {
    if (!resultId || !canVerify || verificationInFlightRef.current) {
      return;
    }
    verificationInFlightRef.current = true;
    const controller = new AbortController();
    verificationAbortRef.current = controller;
    setVerifyingResultId(resultId);
    setVerificationError(null);
    setVerificationCancelled(false);
    try {
      const updated = await verifyRagResult(resultId, {
        signal: controller.signal,
      });
      if (
        !controller.signal.aborted &&
        currentResultIdRef.current === resultId
      ) {
        setLatestRagResult(updated);
      }
    } catch (caught) {
      if (controller.signal.aborted || currentResultIdRef.current !== resultId) {
        return;
      }
      if (caught instanceof RequestCancelledError) {
        setVerificationCancelled(true);
      } else if (caught instanceof ApiError) {
        setVerificationError(caught.message);
      } else {
        setVerificationError("Unexpected error while verifying answer claims.");
      }
    } finally {
      if (verificationAbortRef.current === controller) {
        verificationAbortRef.current = null;
        verificationInFlightRef.current = false;
        setVerifyingResultId(null);
      }
    }
  }

  return (
    <div className="stack">
      <section className="card">
        <div className="card-header">
          <div>
            <h2>Ask a question</h2>
            <p className="section-description">
              Answers are generated only from retrieved evidence from your documents.
            </p>
          </div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="ask-question">Question</label>
            <textarea
              id="ask-question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="e.g. Where is the Eiffel Tower located?"
              rows={3}
              maxLength={2000}
              required
            />
          </div>

          <div className="form-row">
            <DocumentFilter
              id="ask-document"
              documents={documents}
              value={documentId}
              onChange={setDocumentId}
            />
          </div>

          <div className="form-actions">
            <button type="submit" className="button button-primary" disabled={loading || !question.trim()}>
              <Send size={15} aria-hidden="true" />
              {loading ? "Generating..." : "Ask documents"}
            </button>
            {loading && (
              <button type="button" className="button" onClick={handleCancel}>
                <X size={15} aria-hidden="true" />
                Cancel
              </button>
            )}
          </div>
        </form>

        {latestRagResultLoading && (
          <ProcessingStatus
            status="loading"
            message="Restoring the latest completed answer..."
            detail="The result is being fetched by its opaque identifier, without rerunning retrieval, generation, or NLI."
          />
        )}

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Generating a grounded answer..."
            detail="Retrieving evidence and calling the language model. NLI verification is not run automatically; use Verify Answer Claims after the answer appears."
          />
        )}

        {latestRagResultError && !response && !loading && (
          <ProcessingStatus
            status="error"
            message={latestRagResultError}
          />
        )}

        {cancelled && !loading && (
          <ProcessingStatus
            status="warning"
            message="Request cancelled."
            detail="Cancel stops waiting in this browser. The backend may finish the current generation step, but no new result is shown here."
          />
        )}

        {error && <ProcessingStatus status="error" message={error} />}
      </section>

      {response && (
        <section className="card">
          <div className="card-header">
            <div>
              <h2>Answer</h2>
              <p className="section-description">
                Question: {response.question}
              </p>
              <p className="section-description">
                {response.sources.length} retrieved chunk(s) · {formatNumber(response.timings.total_ms)} ms
                total
              </p>
            </div>
            <StatusBadge tone={answerStatusTone(answerStatus)}>
              {answerStatusLabel(answerStatus)}
            </StatusBadge>
          </div>

          <div
            className={`answer${
              showsAbstention ||
              showsPartialSupport ||
              showsContradiction ||
              verificationPartial ||
              verificationFailed
                ? " answer-warning"
                : ""
            }`}
          >
            {showsAbstention && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                {answerStatus === "irrelevant_retrieval"
                  ? "No retrieved passage met the relevance criteria, so no answer was generated."
                  : "The uploaded documents did not contain enough relevant evidence to answer this question."}
              </p>
            )}
            {showsPartialSupport && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                Only part of the requested information is supported by the retrieved evidence.
              </p>
            )}
            {showsContradiction && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                The evidence contradicts at least one factual claim in this answer.
              </p>
            )}
            {verificationPartial && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                {pairsEvaluated} of {pairsExpected} evidence comparison(s) were
                evaluated; {claimsComplete} of {claimsTotal} claim(s) have
                complete evidence coverage.
                {pairsMissing > 0
                  ? ` ${pairsMissing} comparison(s) are missing.`
                  : ""}
                {claimsIncomplete > 0
                  ? ` ${claimsIncomplete} claim(s) have partial evidence-pair coverage.`
                  : ""}
                {verificationErrors > 0
                  ? ` ${verificationErrors} could not be checked due to an error.`
                  : ""}
                {" "}This answer was not fully evaluated against all evidence.
              </p>
            )}
            {verificationFailed && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                Claim-level verification could not be completed, so this answer was
                not checked against the evidence.
              </p>
            )}
            <p className="answer-text">{response.answer}</p>
            <p className="answer-meta">
              <Sparkles size={13} aria-hidden="true" />
              Detected language: {response.detected_language ?? "unknown"}
              {!response.language_reliable && " (low confidence)"} · model{" "}
              {response.generation.model ?? "—"}
              {response.generation.used_llm ? "" : " (no LLM call)"}
            </p>
          </div>

          <details className="disclosure">
            <summary>Retrieved evidence ({response.sources.length})</summary>
            <RetrievedEvidenceList
              sources={response.sources}
              emptyMessage="No chunks passed the evidence policy for this question."
            />
          </details>

          {verification &&
            response.generation.used_llm &&
            response.sources.length > 0 && (
              <div className="stack" style={{ marginTop: 14 }}>
                {(canVerify || isVerifying) && (
                  <div className="form-actions">
                    <button
                      type="button"
                      className="button button-primary"
                      onClick={handleVerify}
                      disabled={isVerifying}
                      aria-busy={isVerifying}
                    >
                      <ShieldCheck size={15} aria-hidden="true" />
                      {isVerifying
                        ? "Verifying claims..."
                        : verification.status === "error"
                          ? "Retry verification"
                          : "Verify Answer Claims"}
                    </button>
                    {isVerifying && (
                      <button
                        type="button"
                        className="button"
                        onClick={handleCancelVerification}
                      >
                        <X size={15} aria-hidden="true" />
                        Cancel
                      </button>
                    )}
                  </div>
                )}
                {verification.status === "pending" && !isVerifying && (
                  <p className="muted">
                    The answer and evidence are ready. Claim-level NLI runs only
                    when you request it.
                  </p>
                )}
                {verification.status === "not_performed" && (
                  <ProcessingStatus
                    status="warning"
                    message="Claim-level verification is unavailable for this answer."
                    detail={verification.note}
                  />
                )}
                {isVerifying && (
                  <p className="muted">
                    Comparing extracted claims with the retrieved evidence. NLI
                    work stops starting new comparisons after 2 minutes.
                  </p>
                )}
                {verificationCancelled && !isVerifying && (
                  <ProcessingStatus
                    status="warning"
                    message="Verification cancelled."
                    detail="No new NLI result was applied to this answer."
                  />
                )}
                {verificationError && !isVerifying && (
                  <ProcessingStatus
                    status="error"
                    message={verificationError}
                  />
                )}
              </div>
            )}

          {showsVerificationPanel && verification && (
            <ClaimVerificationPanel verification={verification} />
          )}

          <details className="disclosure">
            <summary>Technical details</summary>
            <dl className="details-grid details-grid-2">
              <div>
                <dt>Sufficient evidence</dt>
                <dd>{response.retrieval.sufficient_evidence ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Chunks returned</dt>
                <dd>{response.retrieval.returned_chunks}</dd>
              </div>
              <div>
                <dt>Min evidence chunks</dt>
                <dd>{response.retrieval.min_evidence_chunks}</dd>
              </div>
              <div>
                <dt>Max distance</dt>
                <dd>{response.retrieval.max_distance ?? "—"}</dd>
              </div>
              <div>
                <dt>Distance filter applied</dt>
                <dd>{response.retrieval.distance_filter_applied ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Metric</dt>
                <dd>{response.retrieval.metric}</dd>
              </div>
              <div>
                <dt>Verification</dt>
                <dd>{response.verification?.status ?? "not_performed"}</dd>
              </div>
              <div>
                <dt>Verification coverage</dt>
                <dd>
                  {pairsEvaluated}/{pairsExpected} evidence pairs ·{" "}
                  {claimsComplete}/{claimsTotal} claims with complete evidence coverage ·{" "}
                  {claimsIncomplete} incomplete
                  {pairsMissing > 0 ? ` · ${pairsMissing} pairs missing` : ""}
                </dd>
              </div>
              <div>
                <dt>Verified claims</dt>
                <dd>
                  {claimsChecked}/{claimsTotal} checked ·{" "}
                  {response.verification?.supported ?? 0} supported ·{" "}
                  {response.verification?.contradicted ?? 0} contradicted ·{" "}
                  {response.verification?.neutral ?? 0} neutral
                  {verificationErrors > 0 ? ` · ${verificationErrors} errors` : ""}
                </dd>
              </div>
              <div>
                <dt>Verification time</dt>
                <dd>{formatNumber(response.timings.verification_ms ?? 0)} ms</dd>
              </div>
              <div>
                <dt>Claim extraction / evidence selection</dt>
                <dd>
                  {formatNumber(
                    response.verification?.details?.timings
                      ?.claim_extraction_ms ?? 0,
                  )}{" "}
                  ms /{" "}
                  {formatNumber(
                    response.verification?.details?.timings
                      ?.evidence_selection_ms ?? 0,
                  )}{" "}
                  ms
                </dd>
              </div>
              <div>
                <dt>NLI / aggregation</dt>
                <dd>
                  {formatNumber(
                    response.verification?.details?.timings?.nli_ms ?? 0,
                  )}{" "}
                  ms /{" "}
                  {formatNumber(
                    response.verification?.details?.timings?.aggregation_ms ?? 0,
                  )}{" "}
                  ms
                </dd>
              </div>
              <div>
                <dt>Provider</dt>
                <dd>{response.generation.provider ?? "—"}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{response.generation.model ?? "—"}</dd>
              </div>
              <div>
                <dt>Prompt version</dt>
                <dd className="mono">{response.generation.prompt_version}</dd>
              </div>
              <div>
                <dt>Tokens (prompt / completion)</dt>
                <dd>
                  {response.generation.prompt_tokens ?? "—"} /{" "}
                  {response.generation.completion_tokens ?? "—"}
                </dd>
              </div>
              <div>
                <dt>Retrieval time</dt>
                <dd>{formatNumber(response.timings.retrieval_ms)} ms</dd>
              </div>
              <div>
                <dt>Generation time</dt>
                <dd>{formatNumber(response.timings.generation_ms)} ms</dd>
              </div>
            </dl>
            <p className="note">{response.retrieval.score_interpretation}</p>
          </details>
        </section>
      )}

      {!response &&
        !loading &&
        !latestRagResultLoading &&
        !error &&
        !latestRagResultError && (
        <section className="card">
          <EmptyState
            icon={Sparkles}
            title="No answer yet"
            description="Ask a question above. The answer will cite the chunks retrieved from your documents."
          />
        </section>
      )}
    </div>
  );
}
