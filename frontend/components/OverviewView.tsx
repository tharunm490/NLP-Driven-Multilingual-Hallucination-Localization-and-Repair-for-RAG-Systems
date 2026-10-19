"use client";

import {
  Activity,
  ArrowRight,
  Boxes,
  FileText,
  Layers,
  MessageSquare,
  Search,
  Server,
  Upload,
} from "lucide-react";
import Link from "next/link";

import { getApiBaseUrl } from "@/lib/api";
import { componentTone, documentStatusTone, formatDateTime, formatNumber } from "@/lib/format";

import EmptyState from "./EmptyState";
import ProcessingStatus from "./ProcessingStatus";
import StatCard from "./StatCard";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";

const QUICK_START = [
  {
    href: "/documents",
    icon: Upload,
    title: "Upload documents",
    description: "Add PDF or TXT files to the knowledge base.",
  },
  {
    href: "/retrieval",
    icon: Search,
    title: "Test retrieval",
    description: "Inspect which chunks match a question.",
  },
  {
    href: "/ask",
    icon: MessageSquare,
    title: "Ask documents",
    description: "Generate answers grounded in your evidence.",
  },
];

export default function OverviewView() {
  const {
    health,
    healthLoading,
    healthError,
    documents,
    documentsLoading,
    documentsError,
  } = useWorkspace();

  const loading = healthLoading || documentsLoading;
  const completed = documents.filter((document) => document.status === "completed").length;
  const failed = documents.filter((document) => document.status === "failed").length;
  const totalChunks = documents.reduce((sum, document) => sum + document.chunks_stored, 0);

  const recent = [...documents]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5);

  const backendTone = healthError ? "error" : health?.status === "degraded" ? "warning" : "success";

  return (
    <div className="stack">
      {(healthError || documentsError) && (
        <section className="card">
          {healthError && (
            <ProcessingStatus
              status="error"
              message="The backend is unavailable."
              detail={`${healthError} Start the FastAPI server, then use Refresh to try again.`}
            />
          )}
          {documentsError && <ProcessingStatus status="error" message={documentsError} />}
        </section>
      )}

      <section className="stat-grid">
        <StatCard
          icon={FileText}
          label="Documents"
          value={formatNumber(documents.length)}
          hint={
            documents.length > 0
              ? `${completed} completed${failed > 0 ? `, ${failed} failed` : ""}`
              : "No documents ingested yet"
          }
          tone="info"
          loading={loading}
        />
        <StatCard
          icon={Layers}
          label="Stored chunks"
          value={formatNumber(totalChunks)}
          hint="Chunks indexed in the vector store"
          tone="neutral"
          loading={loading}
        />
        <StatCard
          icon={Boxes}
          label="Embedding model"
          value={health?.embedding_model ?? "—"}
          hint={
            health?.embedding_loaded
              ? `Loaded${health.embedding_device ? ` · ${health.embedding_device}` : ""}`
              : "Loaded on first request"
          }
          tone="info"
          loading={loading}
        />
        <StatCard
          icon={Server}
          label="Backend"
          value={healthError ? "Unavailable" : health?.status ?? "Checking…"}
          hint={health ? `v${health.version} · ${health.environment}` : "Awaiting connection"}
          tone={backendTone}
          loading={loading && !health}
        />
      </section>

      <div className="split">
        <section className="card">
          <div className="card-header">
            <div>
              <h2>Recent documents</h2>
              <p className="section-description">The five most recently uploaded documents.</p>
            </div>
            <Link href="/documents" className="button button-ghost button-small">
              View all
              <ArrowRight size={14} aria-hidden="true" />
            </Link>
          </div>

          {recent.length === 0 ? (
            <EmptyState
              icon={FileText}
              title="No documents yet"
              description="Upload a PDF or TXT file to populate the knowledge base."
              action={
                <Link href="/documents" className="button button-primary button-small">
                  <Upload size={14} aria-hidden="true" />
                  Upload documents
                </Link>
              }
            />
          ) : (
            <ul className="recent-list">
              {recent.map((document) => (
                <li key={document.document_id} className="recent-item">
                  <div className="recent-main">
                    <span className="recent-name" title={document.filename}>
                      {document.filename}
                    </span>
                    <span className="recent-meta">
                      {document.chunks_stored} chunk(s) · {formatDateTime(document.created_at)}
                    </span>
                  </div>
                  <StatusBadge tone={documentStatusTone(document.status)}>
                    {document.status}
                  </StatusBadge>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2>System health</h2>
              <p className="section-description">Component status reported by the backend.</p>
            </div>
            <Activity size={16} aria-hidden="true" />
          </div>

          {healthLoading && !health ? (
            <div className="component-list" aria-hidden="true">
              {[0, 1, 2].map((row) => (
                <span key={row} className="skeleton skeleton-row" />
              ))}
            </div>
          ) : health ? (
            <ul className="component-list">
              {Object.entries(health.components).map(([name, component]) => (
                <li key={name} className="component-item">
                  <div className="component-main">
                    <span className="component-name">{name}</span>
                    {component.detail && <span className="component-detail">{component.detail}</span>}
                  </div>
                  <StatusBadge tone={componentTone(component.status)}>{component.status}</StatusBadge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">Component status is unavailable.</p>
          )}
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <div>
            <h2>Quick start</h2>
            <p className="section-description">The core workflow of the workspace.</p>
          </div>
        </div>
        <div className="quick-grid">
          {QUICK_START.map((step) => {
            const Icon = step.icon;
            return (
              <Link key={step.href} href={step.href} className="quick-card">
                <span className="quick-icon" aria-hidden="true">
                  <Icon size={17} />
                </span>
                <span className="quick-title">{step.title}</span>
                <span className="quick-description">{step.description}</span>
                <ArrowRight size={15} className="quick-arrow" aria-hidden="true" />
              </Link>
            );
          })}
        </div>
      </section>

      <section className="card">
        <details className="disclosure">
          <summary>Technical details</summary>
          <dl className="details-grid details-grid-2">
            <div>
              <dt>API base URL</dt>
              <dd className="mono">{getApiBaseUrl()}</dd>
            </div>
            <div>
              <dt>Upload size limit</dt>
              <dd>{health?.max_upload_size_mb ?? 20} MB per file</dd>
            </div>
            <div>
              <dt>Files per upload</dt>
              <dd>{health?.max_files_per_request ?? 5}</dd>
            </div>
            <div>
              <dt>Embedding device</dt>
              <dd>{health?.embedding_device ?? "—"}</dd>
            </div>
          </dl>
        </details>
      </section>
    </div>
  );
}
