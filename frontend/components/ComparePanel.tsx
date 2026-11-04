"use client";

import { GitCompare, Send } from "lucide-react";
import { useState } from "react";

import { ApiError, compareQuestion } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { CompareQueryResponse } from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";
import { DocumentFilter } from "./fields";

export default function ComparePanel() {
  const { documents } = useWorkspace();
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [documentId, setDocumentId] = useState("");
  const [response, setResponse] = useState<CompareQueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await compareQuestion({
        question: question.trim(),
        top_k: topK,
        document_ids: documentId ? [documentId] : undefined,
      });
      setResponse(result);
    } catch (caught) {
      if (caught instanceof ApiError) {
        setError(caught.status === 503
          ? `${caught.message} Configure an LLM provider to generate answers.`
          : caught.message);
      } else {
        setError("Unexpected error while comparing answers.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack">
      <section className="card">
        <div className="card-header">
          <div>
            <h2>RAG vs LLM-only comparison</h2>
            <p className="section-description">
              Ask the same question to both the RAG pipeline (grounded in your documents) and a
              bare LLM (general knowledge only) to evaluate the improvement.
            </p>
          </div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="compare-question">Question</label>
            <textarea
              id="compare-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. Where is the Eiffel Tower located?"
              rows={3}
              maxLength={2000}
              required
            />
          </div>

          <div className="form-row">
            <div className="field">
              <label htmlFor="compare-top-k">Evidence chunks (top_k)</label>
              <input
                id="compare-top-k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <DocumentFilter
              id="compare-document"
              documents={documents}
              value={documentId}
              onChange={setDocumentId}
            />
          </div>

          <div className="form-actions">
            <button
              type="submit"
              className="button button-primary"
              disabled={loading || !question.trim()}
            >
              <Send size={15} aria-hidden="true" />
              {loading ? "Comparing…" : "Compare answers"}
            </button>
          </div>
        </form>

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Running both pipelines…"
            detail="Calling RAG pipeline and bare LLM in sequence. This may take a few seconds."
          />
        )}
        {error && <ProcessingStatus status="error" message={error} />}
      </section>

      {response && (
        <div className="compare-grid">
          {/* RAG answer */}
          <section className="card">
            <div className="card-header">
              <div>
                <h2>RAG answer</h2>
                <p className="section-description">
                  Grounded in retrieved document evidence ·{" "}
                  {formatNumber(response.rag.timings.total_ms)} ms
                </p>
              </div>
              {response.rag.answer_mode === "grounded" ? (
                <StatusBadge tone="success">Grounded</StatusBadge>
              ) : (
                <StatusBadge tone="warning">Insufficient evidence</StatusBadge>
              )}
            </div>
            <p className="answer-text">{response.rag.answer}</p>

            <details className="disclosure">
              <summary>Retrieved evidence ({response.rag.sources.length})</summary>
              {response.rag.sources.length === 0 ? (
                <p className="muted">No chunks passed the evidence policy.</p>
              ) : (
                <ol className="source-list">
                  {response.rag.sources.map((source, i) => (
                    <li key={source.chunk_id} className="source">
                      <div className="source-head">
                        <span className="result-rank">#{i + 1}</span>
                        <span className="result-file" title={source.filename}>
                          {source.filename}
                        </span>
                        {source.page_number != null && (
                          <span className="source-page">page {source.page_number}</span>
                        )}
                        <StatusBadge tone="neutral">
                          score {source.retrieval_score.toFixed(4)}
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
          </section>

          {/* LLM-only answer */}
          <section className="card">
            <div className="card-header">
              <div>
                <h2>LLM-only answer</h2>
                <p className="section-description">
                  General knowledge only, no document retrieval ·{" "}
                  {formatNumber(response.llm_only_generation_ms)} ms
                </p>
              </div>
              <StatusBadge tone="neutral">Baseline</StatusBadge>
            </div>
            <p className="answer-text">{response.llm_only_answer}</p>

            <details className="disclosure">
              <summary>Generation details</summary>
              <dl className="details-grid details-grid-2">
                <div>
                  <dt>Provider</dt>
                  <dd>{response.llm_only_generation.provider ?? "—"}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{response.llm_only_generation.model ?? "—"}</dd>
                </div>
                <div>
                  <dt>Tokens (prompt / completion)</dt>
                  <dd>
                    {response.llm_only_generation.prompt_tokens ?? "—"} /{" "}
                    {response.llm_only_generation.completion_tokens ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>Generation time</dt>
                  <dd>{formatNumber(response.llm_only_generation_ms)} ms</dd>
                </div>
              </dl>
            </details>
          </section>
        </div>
      )}

      {!response && !loading && !error && (
        <section className="card">
          <EmptyState
            icon={GitCompare}
            title="No comparison yet"
            description="Submit a question above to see the RAG pipeline answer alongside the bare LLM baseline."
          />
        </section>
      )}
    </div>
  );
}
