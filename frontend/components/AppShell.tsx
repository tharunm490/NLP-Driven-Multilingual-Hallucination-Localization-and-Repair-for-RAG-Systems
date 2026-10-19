"use client";

import { Menu, RefreshCw, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { getApiBaseUrl } from "@/lib/api";
import { getPageMeta, NAV_ITEMS } from "@/lib/navigation";

import ThemeToggle from "./ThemeToggle";
import { useWorkspace } from "./WorkspaceProvider";

function backendStatusLabel(
  healthError: string | null,
  status: "ok" | "degraded" | undefined,
): { label: string; tone: "success" | "warning" | "error" } {
  if (healthError) {
    return { label: "Backend unavailable", tone: "error" };
  }
  if (status === "degraded") {
    return { label: "Backend degraded", tone: "warning" };
  }
  if (status === "ok") {
    return { label: "Backend connected", tone: "success" };
  }
  return { label: "Checking backend…", tone: "warning" };
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const meta = getPageMeta(pathname);
  const { health, healthError, refreshAll, refreshing } = useWorkspace();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [pathname]);

  const backend = backendStatusLabel(healthError, health?.status);

  return (
    <div className="app-shell">
      <aside className={`sidebar${navOpen ? " sidebar-open" : ""}`} aria-label="Main navigation">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            MR
          </span>
          <span className="brand-text">
            <span className="brand-name">Multilingual RAG Workspace</span>
            <span className="brand-sub">Document intelligence</span>
          </span>
          <button
            type="button"
            className="icon-button sidebar-close"
            onClick={() => setNavOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <nav className="nav" aria-label="Primary">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            if (item.soon) {
              return (
                <span
                  key={item.href}
                  className="nav-item nav-item-disabled"
                  aria-disabled="true"
                  title="Planned for a later phase"
                >
                  <Icon size={17} aria-hidden="true" />
                  <span className="nav-label">{item.label}</span>
                  <span className="nav-badge">Soon</span>
                </span>
              );
            }
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-item${active ? " nav-item-active" : ""}`}
                aria-current={active ? "page" : undefined}
              >
                <Icon size={17} aria-hidden="true" />
                <span className="nav-label">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <details className="tech-details">
            <summary>Technical details</summary>
            <dl className="tech-list">
              <div>
                <dt>Backend URL</dt>
                <dd className="mono">{getApiBaseUrl()}</dd>
              </div>
              <div>
                <dt>Version</dt>
                <dd>{health?.version ?? "—"}</dd>
              </div>
              <div>
                <dt>Environment</dt>
                <dd>{health?.environment ?? "—"}</dd>
              </div>
            </dl>
          </details>
        </div>
      </aside>

      {navOpen && (
        <button
          type="button"
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setNavOpen(false)}
        />
      )}

      <div className="app-main">
        <header className="app-header">
          <button
            type="button"
            className="icon-button menu-button"
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={18} aria-hidden="true" />
          </button>

          <div className="header-titles">
            <h1>{meta.title}</h1>
            <p className="header-description">{meta.description}</p>
          </div>

          <div className="header-actions">
            <span className={`status-pill status-pill-${backend.tone}`}>
              <span className="status-dot" aria-hidden="true" />
              {backend.label}
            </span>
            <ThemeToggle />
            <button
              type="button"
              className="button button-secondary"
              onClick={refreshAll}
              disabled={refreshing}
            >
              <RefreshCw size={15} className={refreshing ? "icon-spin" : undefined} aria-hidden="true" />
              {refreshing ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </header>

        <main className="app-content">{children}</main>
      </div>
    </div>
  );
}
