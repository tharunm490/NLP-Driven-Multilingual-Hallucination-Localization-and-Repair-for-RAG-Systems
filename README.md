# NLP-Driven Multilingual Hallucination Localization and Repair for RAG Systems

Phases 1–3 of a multilingual RAG system: project setup, document ingestion +
vector storage, and a working retrieval-augmented question-answering pipeline.

Current capability: upload **PDF/TXT** documents, extract and clean text, chunk it
with overlap, generate **multilingual embeddings**, store chunks in a persistent
**ChromaDB** collection, retrieve relevant evidence, and **answer questions with a
generative LLM grounded in the retrieved evidence** — all through a Next.js UI.

> Phase 4+ (claim extraction, NLI verification, hallucination classification,
> span localization, LLM repair, re-verification) is **not** implemented yet.
> The code is structured so those services can be added without rewriting the app.
> The Phase 3 RAG pipeline reduces unsupported answers but does **not** guarantee
> they never occur — verification is a later phase.

The project root in this repository is the working folder (`NLP/`), which plays
the role of `multilingual-hallucination-project/` in the specification.

---

## 1. Architecture

```
Next.js UI (app/, components/, lib/api.ts)
        │  fetch (NEXT_PUBLIC_API_BASE_URL)
        ▼
FastAPI  /api/health  /api/documents/*  /api/retrieval/search  /api/rag/query
        │
        ├── api/          route handlers + dependency wiring
        ├── services/     document_processor, text_cleaner, text_chunker,
        │                 embedding_service, ingestion_service, retrieval_service,
        │                 document_registry, language_detection, token_counter,
        │                 query_language_detection, query_processor,
        │                 prompt_builder, generator, rag_service
        └── vectorstore/  chroma_store (explicit embeddings, cosine space)
```

Ingestion pipeline:

```
PDF / TXT upload → validation → extraction → cleaning → chunking (overlap)
      → multilingual embeddings → ChromaDB chunks + metadata → registry row
```

RAG pipeline (`POST /api/rag/query`):

```
question → language detection → conservative preprocessing → query embedding
   (same model as ingestion) → ChromaDB top-K → evidence policy
      → system prompt + question + retrieved evidence → LLM → answer + sources
```

The **SQLite document registry** (`data/document_registry.sqlite3`) tracks
documents that are `processing`, `completed`, or `failed` (ChromaDB only stores
chunks), and stores aggregate counts used by the UI.

---

## 2. Directory structure

```
.
├── frontend/                     # Next.js (App Router, TypeScript)
│   ├── app/                      # layout.tsx, page.tsx, globals.css
│   ├── components/               # DocumentUpload, DocumentList,
│   │                             # ProcessingStatus, SearchPanel, AskPanel
│   ├── lib/                      # api.ts (centralised fetch), types.ts
│   ├── tests/                    # Vitest unit tests for api.ts
│   ├── .env.example
│   └── README.md
├── backend/                      # FastAPI + ingestion + RAG pipeline
│   ├── app/
│   │   ├── main.py
│   │   ├── api/                  # health.py, documents.py, retrieval.py,
│   │   │                         # rag.py, deps.py
│   │   ├── core/                 # config.py, logging.py, errors.py
│   │   ├── schemas/              # document.py, retrieval.py, rag.py, health.py
│   │   ├── services/             # processor, cleaner, chunker, embeddings,
│   │   │                         # ingestion, retrieval, registry, language,
│   │   │                         # token_counter, query_language_detection,
│   │   │                         # query_processor, prompt_builder, generator,
│   │   │                         # rag_service
│   │   └── vectorstore/          # chroma_store.py
│   ├── tests/                    # pytest unit/API tests + integration/
│   ├── samples/                  # original multilingual + English sample docs
│   ├── scripts/                  # download model, generate PDF, smoke test
│   ├── data/                     # runtime: chroma + registry (gitignored)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── .env.example
│   └── README.md
├── .env.example                  # documents all variables
├── .gitignore
└── README.md
```

---

## 3. Prerequisites

- Python **3.10–3.12** (verified on 3.12.10).
- Node.js **20+** (verified on 24) and npm.
- ~5 GB free disk for the default embedding model (`BAAI/bge-m3`, ~2.2 GB) and
  its dependencies. A GPU is optional; CPU works.

---

## 4. Setup — backend

### Windows PowerShell

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Optionally pre-download the embedding model (recommended, so the first upload
is not delayed):

```powershell
python scripts\download_embedding_model.py                 # uses EMBEDDING_MODEL_NAME
python scripts\download_embedding_model.py BAAI/bge-m3
```

Start the API (keep this terminal open):

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: <http://127.0.0.1:8000/api/health>
- Interactive docs: <http://127.0.0.1:8000/docs>

---

## 5. Setup — frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>. The page shows backend status, upload results,
the document list, a retrieval tester, and an **Ask a question** panel.

`frontend/.env.example`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

No secrets are stored in `NEXT_PUBLIC_*` variables.

### Start both (two terminals)

```powershell
# Terminal 1
cd backend; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload

# Terminal 2
cd frontend; npm run dev
```

---

## 6. Configuration

All backend settings live in `backend/.env` (see `backend/.env.example` and the
root `.env.example`). Highlights:

| Variable | Default | Notes |
| --- | --- | --- |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Comma-separated CORS origins |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Persistent ChromaDB path |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Change when switching models |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Primary multilingual model |
| `EMBEDDING_DEVICE` | `auto` | `auto` picks CUDA → MPS → CPU |
| `EMBEDDING_BATCH_SIZE` | `4` | Conservative for limited memory |
| `EMBEDDING_MAX_SEQ_LENGTH` | `1024` | Limits memory use |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `auto` | E5 gets `query: `/`passage: ` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Initial experimental values |
| `TOP_K` | `5` | Default retrieval depth |
| `MAX_UPLOAD_SIZE_MB` | `20` | Per-file limit |
| `ALLOW_DUPLICATE_DOCUMENTS` | `false` | Reuse identical content by checksum |
| `LANGUAGE_DETECTION_ENABLED` | `true` | Document-level label + query detection switch |
| `QUERY_LANGUAGE_MIN_CONFIDENCE` | `0.5` | Below this the detected language is not reported |
| `QUERY_LANGUAGE_FALLBACK` | empty | Optional ISO code used when detection is unreliable |
| `RAG_MAX_QUESTION_LENGTH` | `2000` | Rejects overlong questions (HTTP 400) |
| `RAG_MIN_EVIDENCE_CHUNKS` | `1` | Minimum chunks required for grounded answering |
| `RAG_MAX_DISTANCE` | `0.75` | Cosine-distance gate for "usable" evidence (tune per model) |
| `RAG_DISTANCE_FILTER_ENABLED` | `true` | Set `false` to accept any retrieved chunk |
| `RAG_GENERATION_WITHOUT_EVIDENCE` | `false` | If `true`, still call the LLM when evidence is weak |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `openrouter`, `together`, `ollama`, `openai-compatible`, `mock`, or empty |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Provider model id |
| `LLM_API_KEY` | empty | Secret; set in `backend/.env`, never committed |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `LLM_TEMPERATURE` | `0` | Low temperature for factual answering |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request provider timeout |
| `LLM_MAX_OUTPUT_TOKENS` | `800` | Answer length cap |

Relative paths are resolved from `backend/`, so commands work from any shell
directory. `.env` files and secrets are gitignored.

### Configuring the LLM provider (Groq example)

Answer generation uses an **OpenAI-compatible chat-completions API**, so Groq,
OpenAI, OpenRouter, Together, and local Ollama-compatible servers all work by
changing `LLM_BASE_URL`.

1. Create a Groq API key at <https://console.groq.com/keys>.
2. Open `backend/.env` and paste it:

   ```dotenv
   LLM_PROVIDER=groq
   LLM_MODEL=openai/gpt-oss-120b
   LLM_API_KEY=gsk_your_key_here
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_TEMPERATURE=0
   LLM_TIMEOUT_SECONDS=60
   LLM_MAX_OUTPUT_TOKENS=800
   ```

3. Restart the backend. The key stays on the server: it is never sent to the
   frontend, never logged, and never returned in responses.

Notes:

- The app **starts and serves uploads/retrieval even without a key**. Calling
  `/api/rag/query` then returns a clear HTTP 503 configuration error.
- Set `LLM_PROVIDER=mock` for a network-free deterministic answer (tests/demo
  only — it echoes the prompt, it does not generate real text).
- `LLM_PROVIDER=ollama` does not require a key.

### Embedding models

Only one model is loaded at a time, chosen by `EMBEDDING_MODEL_NAME`, behind the
shared `EmbeddingService` interface:

- Primary: `BAAI/bge-m3` (1024-dim).
- Benchmark: `intfloat/multilingual-e5-base` (768-dim, uses `query:`/`passage:` prefixes).

The collection records its embedding model and dimension. If you switch models
you must use a new `CHROMA_COLLECTION_NAME` (or clear the Chroma data); the store
refuses mismatched vectors instead of returning meaningless results.

---

## 7. Verifying ingestion, retrieval, and RAG answering

1. Generate the sample PDF (once):

   ```powershell
   cd backend
   python scripts\generate_sample_pdf.py
   ```

2. Start the backend (Section 4) and the frontend (Section 5).

3. Upload `backend/samples/sample_multilingual.txt` and
   `backend/samples/sample_english.pdf` in the UI (or via `/docs`).

4. Retrieval tester queries:

   - `Where is the Eiffel Tower?`
   - `भारत की राजधानी कौन सी है?`
   - `தமிழ்நாட்டின் தலைநகரம் என்ன?`

   Results show chunk text, filename, page number, chunk id, `distance`
   (cosine distance, **lower is more similar**), and `similarity` (`1 - distance`).
   These are similarity scores, **not** probabilities or confidence values.

5. **Ask a question** (RAG) — with `LLM_API_KEY` configured:

   - `Where is the Eiffel Tower located?` → grounded answer + collapsible sources.
   - An unrelated question (e.g. `What is the recipe for chocolate cake?`) → the
     system must not invent an answer; it either reports insufficient evidence or
     the LLM explains the evidence does not cover the question.

### Manual test procedure (uploaded document)

1. Start the backend with a valid `LLM_API_KEY`, then the frontend.
2. Upload `backend/samples/sample_english.pdf` and wait for `completed`.
3. In **Ask a question**, type `Where is the Eiffel Tower located?`, keep `top_k=5`,
   submit. Expect: detected language `en`, a grounded answer, and a **Sources**
   section listing the PDF chunk with its page number and retrieval score.
4. Expand/collapse **Sources** and confirm the cited chunk text actually contains
   the answer.
5. Ask an unrelated question. Expect an explicit insufficient-evidence/limitation
   response, not a fabricated answer.
6. Stop the backend and ask again → expect a clear "Could not reach the backend"
   error (the UI must not crash).

Automated end-to-end check against a running backend (start with
`LLM_PROVIDER=mock` to exercise RAG without an API key):

```powershell
cd backend
$env:LLM_PROVIDER="mock"                     # only to set on the server process
.\.venv\Scripts\python.exe scripts\smoke_test_api.py
```

Set `SMOKE_REQUIRE_LLM=1` to make an unconfigured LLM a failure.

Example output (real BGE-M3 run, backend started with `LLM_PROVIDER=mock`):

```
Uploading sample_multilingual.txt ... status=completed chunks=1 chars=1670
Uploading sample_english.pdf ... status=completed chunks=3 chars=641
Query: Where is the Eiffel Tower?
  distance=0.2409 similarity=0.7591 file=sample_english.pdf page=1 chunk=0
Query: भारत की राजधानी कौन सी है?
  distance=0.3691 similarity=0.6309 file=sample_multilingual.txt
Deleting PDF document ... chunks_deleted=3
RAG question: Where is the Eiffel Tower located?
  lang=en reliable=True mode=grounded chunks=1 sufficient=True llm=True total_ms=384.3
  answer: [mock:openai/gpt-oss-120b] USER QUESTION: ...
    score=0.2912 file=sample_multilingual.txt page=- chunk=0
SMOKE_TEST_OK
```

---

## 8. Tests

Backend (uses a deterministic fake encoder — no large download):

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Frontend:

```powershell
cd frontend
npm run typecheck
npm test
npm run build
```

Real embedding integration test (opt-in; downloads/loads the model):

```powershell
cd backend
$env:RUN_EMBEDDING_INTEGRATION="1"
$env:EMBEDDING_MODEL_NAME="BAAI/bge-m3"       # or intfloat/multilingual-e5-base
.\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s
```

The integration suite now also covers the RAG pipeline
(`tests/integration/test_rag_real_embeddings.py`) using the real embedding model
and a deterministic **mock** generator (no API credits spent).

Latest local results: **backend 138 passed, 2 skipped** (integration opt-in);
**integration 2 passed** each for `BAAI/bge-m3` (dim 1024) and
`intfloat/multilingual-e5-base` (dim 768); **frontend 11 passed**, typecheck and
production build clean. Phase 3 tests include language detection (EN/HI), query
preprocessing, prompt construction, generation-service error mapping (mocked
HTTP), the full RAG flow with a mocked LLM, and the `/api/rag/query` API.

---

## 9. API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, version, component health (does not load the model) |
| `POST` | `/api/documents/upload` | Upload one or more PDF/TXT files (multipart field `files`) |
| `GET` | `/api/documents` | List documents with status and counts |
| `GET` | `/api/documents/{document_id}` | Document detail |
| `DELETE` | `/api/documents/{document_id}` | Delete chunks + registry row |
| `POST` | `/api/retrieval/search` | `{ query, top_k?, document_ids? }` → ranked chunks |
| `POST` | `/api/rag/query` | `{ question, top_k?, document_ids? }` → grounded answer + sources |

Uploads return HTTP 200 with a per-file result (`completed`, `failed`, or
`rejected`) so a batch survives individual failures. A duplicate file (same
SHA-256) is reused instead of re-ingested unless `ALLOW_DUPLICATE_DOCUMENTS=true`.

### `POST /api/rag/query`

Request:

```json
{ "question": "When was the company founded?", "top_k": 5 }
```

Response (abridged; IDs/scores are real values from your data):

```json
{
  "question": "When was the company founded?",
  "normalized_question": "When was the company founded?",
  "detected_language": "en",
  "language_confidence": 0.93,
  "language_reliable": true,
  "answer": "According to the retrieved document, the company was founded in 2015 [Source 1].",
  "answer_mode": "grounded",
  "sources": [
    {
      "document_id": "…", "filename": "company_profile.pdf",
      "chunk_id": "…", "chunk_index": 7, "page_number": 2, "language": "en",
      "text": "The company was founded in 2015.",
      "retrieval_score": 0.21, "similarity": 0.79
    }
  ],
  "retrieval": {
    "top_k": 5, "returned_chunks": 1, "sufficient_evidence": true,
    "metric": "cosine_distance",
    "score_interpretation": "lower distance = more similar; not a probability",
    "min_evidence_chunks": 1, "max_distance": 0.75, "distance_filter_applied": true
  },
  "generation": {
    "used_llm": true, "provider": "groq", "model": "openai/gpt-oss-120b",
    "prompt_version": "rag-grounded-answer-1.0.0",
    "finish_reason": "stop", "prompt_tokens": 512,
    "completion_tokens": 40, "total_tokens": 552
  },
  "timings": { "retrieval_ms": 235.7, "generation_ms": 812.4, "total_ms": 1052.3 }
}
```

Status codes: `400` invalid/overlong question, `422` schema validation error,
`503` LLM not configured / vector store unavailable, `502` LLM provider error.
`answer_mode` is `grounded` when a chunk passed the retrieval policy, otherwise
`insufficient_evidence` (and, by default, the LLM is **not** called).
`retrieval_score` is a cosine **distance** (lower = more similar); it is **not** a
probability, confidence, or factual-verification result.

---

## 10. Future phases

Phase 3 delivers retrieval-grounded answering. The following are intentionally
out of scope now but planned for Phase 4+: multilingual Transformer NLI
(`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, `joeddav/xlm-roberta-large-xnli`),
multilingual NER, claim extraction, claim–evidence matching, hallucination
classification, span localization, LLM repair, and re-verification. Inference will
stay isolated from route handlers, thresholds will be configurable, and models will
be evaluated (Recall@K, MRR, Precision/Recall/F1/Macro-F1, span F1, IoU) per
language before selection.

---

## 11. Troubleshooting

- **`Activate.ps1` cannot be loaded** → `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- **`pip`/file-lock errors on OneDrive folders** → close any running Python
  process and re-run `pip install`; avoid syncing `.venv` while installing.
- **First upload seems to hang** → it is downloading the model. Run
  `python scripts\download_embedding_model.py` in advance. BGE-M3 is ~2.2 GB.
- **Out of memory** → lower `EMBEDDING_BATCH_SIZE`, set `EMBEDDING_DEVICE=cpu`,
  and/or reduce `EMBEDDING_MAX_SEQ_LENGTH`.
- **CORS error in the browser** → add the exact frontend origin to
  `FRONTEND_ORIGIN` (comma-separated) and restart the backend.
- **Port already in use** → change `API_PORT` (and `NEXT_PUBLIC_API_BASE_URL`),
  or start uvicorn with `--port 8001`.
- **"model X was created with embedding model Y"** → you changed
  `EMBEDDING_MODEL_NAME`; use a new `CHROMA_COLLECTION_NAME` or delete
  `backend/data/chroma`.
- **PDF rejected with an OCR message** → the PDF has no text layer; OCR is not
  implemented. Upload a text-based PDF or TXT.
- **`npm install` peer conflicts** → use a current Node LTS and `npm install`
  (no `--force`); the pinned versions in `package.json` are compatible.
- **`/api/rag/query` returns 503 "No LLM provider is configured"** → set
  `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` in `backend/.env` and restart.
  For a quick offline demo set `LLM_PROVIDER=mock`.
- **`/api/rag/query` returns 503 "Authentication ... failed"** → the API key is
  invalid/expired; generate a new one and restart the backend.
- **`/api/rag/query` returns 502** → the provider is rate-limiting or down
  (check the backend log line for the HTTP status); retry later or raise
  `LLM_TIMEOUT_SECONDS`.
- **The answer is always "insufficient evidence"** → retrieval found no chunk
  within `RAG_MAX_DISTANCE`. Inspect `/api/retrieval/search` distances for your
  model and raise `RAG_MAX_DISTANCE`, lower `RAG_MIN_EVIDENCE_CHUNKS`, or set
  `RAG_DISTANCE_FILTER_ENABLED=false`.

---

## 12. Known limitations

- OCR is not implemented; image-only PDFs are reported, never silently "succeeded".
- Language detection runs on the whole document and may label mixed-language
  files with the dominant language (e.g. `en`).
- Query language detection is metadata only: it is not proof that the embedding
  model or the LLM will perform equally well in that language, and the query is
  never translated.
- Retrieval scores are cosine distances/similarities, not truth or confidence.
- Upload processing is synchronous; the first request includes model load time.
- Only `.pdf` and `.txt` are supported; no authentication is implemented.
- The evidence policy uses a single `RAG_MAX_DISTANCE` per embedding model. In an
  observed run an unrelated question still landed at distance ≈ 0.63 (< 0.75) and
  was treated as "sufficient"; the LLM was then instructed to state the limitation.
  Tune the threshold (or disable the filter) per model/corpus.
- The LLM answer is grounded by prompt design only. A system prompt reduces but
  cannot eliminate unsupported answers; the dedicated verification/localization
  engine is a later phase.

---

## 13. Acceptance checklist

Phase 1: all pass — frontend starts/builds; backend starts; `GET /api/health`
returns ok; frontend calls it via `lib/api.ts`; CORS returns
`Access-Control-Allow-Origin: http://localhost:3000`; no secrets in frontend.

Phase 2: all pass — PDF/TXT upload works (real BGE-M3 run: 1 TXT chunk, 3 PDF
chunks); cleaning preserves multilingual facts; overlapping chunks; embeddings
generated; metadata persists in ChromaDB across restarts; multilingual retrieval
returns relevant chunks; UI shows results and evidence; invalid/corrupt/oversize
files handled; automated tests cover processing, chunking, storage, retrieval.

Phase 3: all pass —
user can ask a question in the UI (**Ask a question** panel);
query language detected where possible (EN/HI/TA, with confidence + reliability);
query embedding uses the same model/config as ingestion;
ChromaDB returns top-K chunks;
the LLM receives a separate system prompt, the user question, and clearly
delimited untrusted evidence;
the API returns the generated answer plus **actual** source metadata (document id,
filename, chunk id, page, scores);
the UI shows the answer, detected language, and a collapsible **Sources** section;
missing/insufficient evidence is handled explicitly (`answer_mode` =
`insufficient_evidence`, no fabricated answer);
LLM provider errors are mapped to clean `503`/`502` responses;
automated tests pass, including mocked-LLM tests;
ingestion/persistence still work; READMEs explain LLM configuration and the manual
test procedure.
