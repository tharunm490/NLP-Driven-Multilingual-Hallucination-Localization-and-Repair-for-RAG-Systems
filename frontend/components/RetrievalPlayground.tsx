"use client";

import { Info, Search as SearchIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { ApiError, search } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { SearchResponse } from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";
import { DocumentFilter } from "./fields";

export default function RetrievalPlayground() {
  const { documents } = useWorkspace();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [documentId, setDocumentId] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
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
    if (!query.trim() || loading) {
      return;
    }
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await search({
        query: query.trim(),
        top_k: topK,
        document_ids: documentId ? [documentId] : undefined,
      });
      setResponse(result);
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.message : "Unexpected error while searching.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack">
      <section className="card">
        <div className="card-header">
          <div>
            <h2>Retrieval query</h2>
            <p className="section-description">
              Inspect which chunks the vector store returns for a question before any answer is
              generated.
            </p>
          </div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="retrieval-query">Query</label>
            <textarea
              id="retrieval-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. Where is the Eiffel Tower located?"
              rows={3}
              maxLength={2000}
              required
            />
          </div>

          <div className="form-row">
            <div className="field">
              <label htmlFor="retrieval-top-k">Results (top_k)</label>
              <input
                id="retrieval-top-k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </div>
            <DocumentFilter
              id="retrieval-document"
              documents={documents}
              value={documentId}
              onChange={setDocumentId}
            />
          </div>

          <div className="form-actions">
            <button type="submit" className="button button-primary" disabled={loading || !query.trim()}>
              <SearchIcon size={15} aria-hidden="true" />
              {loading ? "Searching…" : "Search knowledge base"}
            </button>
          </div>
        </form>

        {loading && (
          <ProcessingStatus
            status="loading"
            message="Retrieving matching chunks…"
            detail="Queries are embedded with the same model used for your documents."
          />
        )}

        {error && <ProcessingStatus status="error" message={error} />}
      </section>

      {response && (
        <section className="card">
          <div className="card-header">
            <div>
              <h2>Results</h2>
              <p className="section-description">
                {formatNumber(response.total)} chunk(s) · model {response.embedding_model} · metric{" "}
                {response.metric}
              </p>
            </div>
            <StatusBadge tone="info">{response.metric}</StatusBadge>
          </div>

          <p className="note">
            <Info size={14} aria-hidden="true" />
            Scores are retrieval distances, not confidence values. A lower distance means the chunk
            is closer to your query; the similarity value is simply 1 − distance.
          </p>

          {response.results.length === 0 ? (
            <EmptyState
              icon={SearchIcon}
              title="No matching chunks"
              description="Try a different question, increase top_k, or upload more documents."
            />
          ) : (
            <ol className="result-cards">
              {response.results.map((chunk, index) => (
                <li key={chunk.chunk_id} className="result-card">
                  <div className="result-card-head">
                    <span className="result-rank">#{index + 1}</span>
                    <span className="result-file" title={chunk.filename}>
                      {chunk.filename}
                    </span>
                    <span className="result-meta">
                      {chunk.page_number != null && <span>page {chunk.page_number}</span>}
                      <span>chunk {chunk.chunk_index}</span>
                      {chunk.language && <span>language: {chunk.language}</span>}
                    </span>
                    <span className="result-scores">
                      <StatusBadge tone="info">distance {chunk.distance.toFixed(4)}</StatusBadge>
                      <StatusBadge tone="neutral">
                        similarity {chunk.similarity.toFixed(4)}
                      </StatusBadge>
                    </span>
                  </div>
                  <p className="result-text">{chunk.text}</p>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      {!response && !loading && !error && (
        <section className="card">
          <EmptyState
            icon={SearchIcon}
            title="No query yet"
            description="Enter a question above to see the evidence your knowledge base retrieves."
          />
        </section>
      )}
    </div>
  );
}
