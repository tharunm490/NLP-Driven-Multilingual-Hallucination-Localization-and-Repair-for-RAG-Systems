# Frontend — Multilingual RAG Workspace

Next.js (App Router, TypeScript, React) SaaS-style workspace for uploading
documents, viewing the knowledge base, testing retrieval, and **asking questions
answered from the retrieved evidence** against the FastAPI backend.

Screens: **Overview** (`/`), **Documents** (`/documents`),
**Retrieval Playground** (`/retrieval`), and **Ask Documents** (`/ask`).

## Setup

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>.

`.env.example`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Only public, non-secret values belong in `NEXT_PUBLIC_*` variables.

## Scripts

| Script | Purpose |
| --- | --- |
| `npm run dev` | Start the dev server |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Vitest unit tests for the API utility |

## Layout

- `app/layout.tsx` — metadata, `WorkspaceProvider`, and the `AppShell`.
- `app/page.tsx`, `app/documents/page.tsx`, `app/retrieval/page.tsx`,
  `app/ask/page.tsx` — route pages (server components that export metadata).
- `app/globals.css` — the light design system (single stylesheet, no CSS framework).
- `components/AppShell.tsx` — persistent sidebar + top header (status, refresh).
- `components/WorkspaceProvider.tsx` — shared health + documents state (React context).
- `components/` — screen views (`OverviewView`, `DocumentsView`), feature panels
  (`UploadPanel`, `DocumentsTable`, `RetrievalPlayground`, `AskPanel`), and shared
  primitives (`StatCard`, `StatusBadge`, `EmptyState`, `ProcessingStatus`, `fields`).
- `lib/api.ts` — centralised `fetch` wrapper (timeouts, typed errors).
- `lib/types.ts` — shared response types mirroring the backend schemas.
- `lib/format.ts` — formatting helpers (bytes, dates, numbers, status labels).
- `lib/navigation.ts` — sidebar navigation + page metadata.
- `tests/api.test.ts` — error-handling tests with a mocked `fetch`.

## Behaviour

- Persistent sidebar navigation and a header showing the current page title, a
  short description, backend status, a light/dark theme toggle, and a Refresh action.
- Light and dark themes driven by CSS variables (`[data-theme]` on `<html>`).
  The theme defaults to the OS `prefers-color-scheme` and the choice is persisted
  in `localStorage`; an inline script applies it before first paint to avoid a flash.
- **Overview**: document/chunk counts, embedding model + device, backend status,
  component health, recent documents, and collapsible technical details.
- **Documents**: drag-and-drop multiple PDF/TXT upload with client-side type/size
  limits, per-file results, a knowledge base table with expandable metadata,
  delete with inline confirmation, and a shortcut to test retrieval per document.
- **Retrieval Playground**: runs retrieval queries and displays chunk text, source
  metadata, and cosine `distance`/`similarity` (explicitly labelled as
  non-probabilities).
- **Ask Documents** (`POST /api/rag/query`): shows the generated answer, the
  detected query language, and collapsible **Retrieved evidence** and
  **Technical details** sections. Duplicate submissions are blocked while a
  request is in flight. Insufficient evidence is shown explicitly.
- All backend failures are surfaced as readable messages (network errors,
  timeouts, validation details, server errors, LLM not configured (503), LLM
  provider error (502)).

Hallucination detection, localization, and repair are **not** implemented yet;
the corresponding sidebar entry is disabled and marked "Soon".

No LLM calls are made from the browser and no API keys or credentials are used or
exposed by the frontend; all generation happens in the backend.
