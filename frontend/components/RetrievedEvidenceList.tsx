import { Quote } from "lucide-react";

import type { RagSource } from "@/lib/types";

import StatusBadge from "./StatusBadge";

export default function RetrievedEvidenceList({
  sources,
  emptyMessage = "No chunks were retrieved for this result.",
}: {
  sources: RagSource[];
  emptyMessage?: string;
}) {
  if (sources.length === 0) {
    return <p className="muted">{emptyMessage}</p>;
  }

  return (
    <ol className="source-list" data-testid="retrieved-evidence-list">
      {sources.map((source, index) => (
        <li key={source.chunk_id} className="source">
          <div className="source-head">
            <span className="result-rank">#{index + 1}</span>
            <Quote size={13} aria-hidden="true" />
            <span className="result-file" title={source.filename}>
              {source.filename}
            </span>
            {source.page_number != null && (
              <span className="source-page">page {source.page_number}</span>
            )}
            <StatusBadge tone="neutral">
              retrieval_score {source.retrieval_score.toFixed(4)}
            </StatusBadge>
          </div>
          <p className="result-meta">
            similarity {source.similarity.toFixed(4)} · chunk{" "}
            {source.chunk_index}
            {source.passed_relevance_filter != null
              ? ` · relevance ${source.passed_relevance_filter ? "passed" : "rejected"}`
              : ""}
          </p>
          <details className="source-text">
            <summary>Show full passage</summary>
            <p className="result-text">{source.text}</p>
          </details>
        </li>
      ))}
    </ol>
  );
}
