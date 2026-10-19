"use client";

import { AlertTriangle, Quote, Send, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { ApiError, askQuestion } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { RagQueryResponse } from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";
import { DocumentFilter } from "./fields";

export default function AskPanel() {
  const { documents } = useWorkspace();
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [documentId, setDocumentId] = useState("");
  const [response, setResponse] = useState<RagQueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get("document");
    if (requested) {
      setDocumentId(requested);
    }
  }, []);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || loading) {
      return;
    }
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await askQuestion({
        question: question.trim(),
        top_k: topK,
        document_ids: documentId ? [documentId] : undefined,
      });
      setResponse(result);
    } catch (caught) {
      if (caught instanceof ApiError) {
        if (caught.status === 503) {
          setError(
            `${caught.message} Configure an LLM provider to generate answers. Retrieval still works in the Retrieval Playground.`,
          );
        } else {
          setError(caught.message);
        }
      } else {
        setError("Unexpected error while generating an answer.");
      }
    } finally {
      setLoading(false);
    }
  }

  const grounded = response?.answer_mode === "grounded";

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
            <div className="field">
              <label htmlFor="ask-top-k">Evidence chunks (top_k)</label>
              <input
                id="ask-top-k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </div>
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
              {loading ? "Generating…" : "Ask documents"}
            </button>
          </div>
        </form>

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Generating a grounded answer…"
            detail="Retrieving evidence and calling the language model. This can take a few seconds."
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
                {response.sources.length} source(s) used · {formatNumber(response.timings.total_ms)} ms
                total
              </p>
            </div>
            {grounded ? (
              <StatusBadge tone="success">Grounded answer</StatusBadge>
            ) : (
              <StatusBadge tone="warning">Insufficient evidence</StatusBadge>
            )}
          </div>

          <div className={`answer${grounded ? "" : " answer-warning"}`}>
            {!grounded && (
              <p className="answer-warning-line">
                <AlertTriangle size={14} aria-hidden="true" />
                The knowledge base did not contain enough relevant evidence, so no grounded answer
                was produced.
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
            {response.sources.length === 0 ? (
              <p className="muted">No chunks passed the evidence policy for this question.</p>
            ) : (
              <ol className="source-list">
                {response.sources.map((source, index) => (
                  <li key={source.chunk_id} className="source">
                    <div className="source-head">
                      <span className="result-rank">#{index + 1}</span>
                      <Quote size={13} aria-hidden="true" />
                      <span className="result-file" title={source.filename}>
                        {source.filename}
                      </span>
                      {source.page_number != null && <span className="source-page">page {source.page_number}</span>}
                      <StatusBadge tone="neutral">
                        retrieval_score {source.retrieval_score.toFixed(4)}
                      </StatusBadge>
                    </div>
                    <details className="source-text">
                      <summary>Show passage</summary>
                      <p className="result-text">{source.text}</p>
                    </details>
                  </li>
                ))}
              </ol>
            )}
          </details>

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

      {!response && !loading && !error && (
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
