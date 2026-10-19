"use client";

import { ChevronDown, ChevronRight, FileText, RefreshCw, Search, Trash2 } from "lucide-react";
import Link from "next/link";
import { Fragment, useState } from "react";

import { ApiError, deleteDocument } from "@/lib/api";
import {
  documentStatusLabel,
  documentStatusTone,
  fileExtensionLabel,
  formatBytes,
  formatDateTime,
} from "@/lib/format";
import type { DocumentInfo } from "@/lib/types";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";

interface Props {
  documents: DocumentInfo[];
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export default function DocumentsTable({ documents, loading, error, onRefresh }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function handleDelete(documentId: string) {
    setDeletingId(documentId);
    setActionError(null);
    try {
      await deleteDocument(documentId);
      setConfirmingId(null);
      if (expandedId === documentId) {
        setExpandedId(null);
      }
      onRefresh();
    } catch (caught) {
      setActionError(
        caught instanceof ApiError ? caught.message : "Could not delete the document.",
      );
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>Knowledge base</h2>
          <p className="section-description">
            Documents available as evidence for retrieval and question answering.
          </p>
        </div>
        <button
          type="button"
          className="button button-secondary"
          onClick={onRefresh}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "icon-spin" : undefined} aria-hidden="true" />
          Reload
        </button>
      </div>

      {error && <ProcessingStatus status="error" message={error} />}
      {actionError && <ProcessingStatus status="error" message={actionError} />}

      {loading && documents.length === 0 ? (
        <div className="table-skeleton" aria-hidden="true">
          {[0, 1, 2].map((row) => (
            <span key={row} className="skeleton skeleton-row" />
          ))}
        </div>
      ) : documents.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Upload a PDF or TXT file to build your knowledge base, then test retrieval and ask questions."
        />
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Document</th>
                <th scope="col">Status</th>
                <th scope="col">Chunks</th>
                <th scope="col">Language</th>
                <th scope="col">Size</th>
                <th scope="col">Uploaded</th>
                <th scope="col" className="table-actions-head">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {documents.map((document) => {
                const expanded = expandedId === document.document_id;
                const confirming = confirmingId === document.document_id;
                const deleting = deletingId === document.document_id;
                return (
                  <Fragment key={document.document_id}>
                    <tr>
                      <td>
                        <div className="doc-cell">
                          <span className="doc-icon" aria-hidden="true">
                            {fileExtensionLabel(document.filename)}
                          </span>
                          <span className="doc-name" title={document.filename}>
                            {document.filename}
                          </span>
                        </div>
                      </td>
                      <td>
                        <StatusBadge tone={documentStatusTone(document.status)}>
                          {documentStatusLabel(document.status)}
                        </StatusBadge>
                      </td>
                      <td>{document.chunks_stored}</td>
                      <td>{document.language ?? "—"}</td>
                      <td>{formatBytes(document.size_bytes)}</td>
                      <td>{formatDateTime(document.created_at)}</td>
                      <td>
                        <div className="row-actions">
                          <button
                            type="button"
                            className="icon-button"
                            onClick={() => setExpandedId(expanded ? null : document.document_id)}
                            aria-expanded={expanded}
                            aria-label={expanded ? "Hide details" : "Show details"}
                            title="Details"
                          >
                            {expanded ? (
                              <ChevronDown size={16} aria-hidden="true" />
                            ) : (
                              <ChevronRight size={16} aria-hidden="true" />
                            )}
                          </button>
                          <Link
                            href={`/retrieval?document=${encodeURIComponent(document.document_id)}`}
                            className="icon-button"
                            title="Test retrieval on this document"
                            aria-label="Test retrieval on this document"
                          >
                            <Search size={16} aria-hidden="true" />
                          </Link>
                          {confirming ? (
                            <>
                              <button
                                type="button"
                                className="button button-danger button-small"
                                onClick={() => handleDelete(document.document_id)}
                                disabled={deleting}
                              >
                                {deleting ? "Deleting…" : "Confirm"}
                              </button>
                              <button
                                type="button"
                                className="button button-ghost button-small"
                                onClick={() => setConfirmingId(null)}
                                disabled={deleting}
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              type="button"
                              className="icon-button icon-button-danger"
                              onClick={() => setConfirmingId(document.document_id)}
                              title="Delete document"
                              aria-label="Delete document"
                            >
                              <Trash2 size={16} aria-hidden="true" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="details-row">
                        <td colSpan={7}>
                          <dl className="details-grid">
                            <div>
                              <dt>Document ID</dt>
                              <dd className="mono">{document.document_id}</dd>
                            </div>
                            <div>
                              <dt>Content type</dt>
                              <dd>{document.content_type ?? "—"}</dd>
                            </div>
                            <div>
                              <dt>Pages</dt>
                              <dd>{document.page_count ?? "—"}</dd>
                            </div>
                            <div>
                              <dt>Characters</dt>
                              <dd>{document.character_count ?? "—"}</dd>
                            </div>
                            <div>
                              <dt>Tokens</dt>
                              <dd>{document.token_count ?? "—"}</dd>
                            </div>
                            <div>
                              <dt>Chunks created</dt>
                              <dd>{document.chunk_count}</dd>
                            </div>
                            <div>
                              <dt>Embedding model</dt>
                              <dd>{document.embedding_model ?? "—"}</dd>
                            </div>
                            <div>
                              <dt>Updated</dt>
                              <dd>{formatDateTime(document.updated_at)}</dd>
                            </div>
                          </dl>
                          {document.error && (
                            <p className="error-text">Processing error: {document.error}</p>
                          )}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
