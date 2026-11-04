"use client";

import { FlaskConical, Send } from "lucide-react";
import { useState } from "react";

import { ApiError, evaluateAnswer } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { EvalResponse } from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";
import { DocumentFilter } from "./fields";

function MetricRow({
  label,
  rag,
  llm,
  hint,
}: {
  label: string;
  rag: number;
  llm: number;
  hint: string;
}) {
  const ragWins = rag >= llm;
  const delta = (rag - llm).toFixed(4);
  return (
    <div className="eval-metric">
      <div className="eval-metric-header">
        <span className="eval-metric-label">
          {label}
          <span className="eval-metric-hint"> — {hint}</span>
        </span>
        <span className={`eval-metric-delta ${ragWins ? "eval-delta-positive" : "eval-delta-negative"}`}>
          RAG {ragWins ? "▲" : "▼"} {Math.abs(Number(delta)).toFixed(4)}
        </span>
      </div>
      <div className="eval-bars">
        <div className="eval-bar-row">
          <span className="eval-bar-tag">RAG</span>
          <div className="eval-bar-track">
            <div
              className={`eval-bar-fill ${ragWins ? "eval-bar-rag-better" : "eval-bar-rag-worse"}`}
              style={{ width: `${Math.min(rag * 100, 100)}%` }}
            />
          </div>
          <span className="eval-bar-value">{rag.toFixed(4)}</span>
        </div>
        <div className="eval-bar-row">
          <span className="eval-bar-tag">LLM</span>
          <div className="eval-bar-track">
            <div className="eval-bar-fill eval-bar-llm" style={{ width: `${Math.min(llm * 100, 100)}%` }} />
          </div>
          <span className="eval-bar-value">{llm.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

export default function EvalPanel() {
  const { documents } = useWorkspace();
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [documentId, setDocumentId] = useState("");
  const [response, setResponse] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await evaluateAnswer({
        question: question.trim(),
        top_k: topK,
        document_ids: documentId ? [documentId] : undefined,
      });
      setResponse(result);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Unexpected error during evaluation.",
      );
    } finally {
      setLoading(false);
    }
  }

  const ragWinsOverall =
    response &&
    response.rag_metrics.rouge_l + response.rag_metrics.semantic_similarity >
      response.llm_only_metrics.rouge_l + response.llm_only_metrics.semantic_similarity;

  return (
    <div className="stack">
      <section className="card">
        <div className="card-header">
          <div>
            <h2>Pipeline evaluation</h2>
            <p className="section-description">
              Ask a question about your uploaded PDF. Both the RAG pipeline and a bare LLM answer
              it. Their answers are then scored against the retrieved document evidence — showing
              how much more grounded the RAG answer is compared to the LLM baseline.
            </p>
          </div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="eval-question">Question</label>
            <textarea
              id="eval-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. Where is the Eiffel Tower located?"
              rows={2}
              maxLength={2000}
              required
            />
          </div>
          <div className="form-row">
            <div className="field">
              <label htmlFor="eval-top-k">Evidence chunks (top_k)</label>
              <input
                id="eval-top-k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <DocumentFilter
              id="eval-document"
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
              {loading ? "Evaluating…" : "Run evaluation"}
            </button>
          </div>
        </form>

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Running evaluation…"
            detail="Calling both pipelines and scoring against retrieved evidence. This may take a few seconds."
          />
        )}
        {error && <ProcessingStatus status="error" message={error} />}
      </section>

      {response && (
        <>
          {/* Metrics card */}
          <section className="card">
            <div className="card-header">
              <div>
                <h2>Evaluation metrics</h2>
                <p className="section-description">
                  Both answers scored against the retrieved PDF evidence ({response.rag_sources_count} chunk
                  {response.rag_sources_count !== 1 ? "s" : ""}). Higher = more aligned with the document.
                </p>
              </div>
              {ragWinsOverall ? (
                <StatusBadge tone="success">RAG pipeline wins</StatusBadge>
              ) : (
                <StatusBadge tone="warning">LLM baseline wins</StatusBadge>
              )}
            </div>

            <MetricRow
              label="ROUGE-L F1"
              rag={response.rag_metrics.rouge_l}
              llm={response.llm_only_metrics.rouge_l}
              hint="lexical overlap with retrieved evidence"
            />
            <MetricRow
              label="Semantic similarity"
              rag={response.rag_metrics.semantic_similarity}
              llm={response.llm_only_metrics.semantic_similarity}
              hint="embedding cosine similarity to retrieved evidence"
            />

            <dl className="details-grid details-grid-2" style={{ marginTop: "1.25rem" }}>
              <div><dt>RAG answer mode</dt><dd>{response.rag_answer_mode}</dd></div>
              <div><dt>Sources retrieved</dt><dd>{response.rag_sources_count}</dd></div>
              <div><dt>RAG retrieval</dt><dd>{formatNumber(response.rag_retrieval_ms)} ms</dd></div>
              <div><dt>RAG generation</dt><dd>{formatNumber(response.rag_generation_ms)} ms</dd></div>
              <div><dt>LLM-only generation</dt><dd>{formatNumber(response.llm_only_generation_ms)} ms</dd></div>
            </dl>
          </section>

          {/* Answers side by side */}
          <div className="compare-grid">
            <section className="card">
              <div className="card-header">
                <h2>RAG answer</h2>
                <StatusBadge tone={response.rag_answer_mode === "grounded" ? "success" : "warning"}>
                  {response.rag_answer_mode === "grounded" ? "Grounded" : "Insufficient evidence"}
                </StatusBadge>
              </div>
              <p className="answer-text">{response.rag_answer}</p>
            </section>

            <section className="card">
              <div className="card-header">
                <h2>LLM-only answer</h2>
                <StatusBadge tone="neutral">Baseline — no document access</StatusBadge>
              </div>
              <p className="answer-text">{response.llm_only_answer}</p>
            </section>
          </div>

          {/* Evidence used as reference */}
          <section className="card">
            <details className="disclosure">
              <summary>Retrieved evidence used as reference</summary>
              <p className="result-text">{response.evidence_used_as_reference}</p>
            </details>
          </section>
        </>
      )}

      {!response && !loading && !error && (
        <section className="card">
          <EmptyState
            icon={FlaskConical}
            title="No evaluation yet"
            description="Ask a question about your uploaded PDF to compare RAG vs LLM-only accuracy."
          />
        </section>
      )}
    </div>
  );
}
