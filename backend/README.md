# Backend — Multilingual RAG API

FastAPI service for Phases 1–3: document upload, extraction, cleaning, chunking,
multilingual embeddings, ChromaDB storage, retrieval, and **retrieval-augmented
question answering** with a generative LLM.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1   # or: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

## Setup (Linux / macOS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Run

```powershell
python scripts\download_embedding_model.py         # optional pre-download
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: `http://127.0.0.1:8000/api/health`
- Docs: `http://127.0.0.1:8000/docs`

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Unit/API tests inject a deterministic fake encoder, so they do not download a
large model. Real-model tests are opt-in:

```powershell
$env:RUN_EMBEDDING_INTEGRATION="1"
$env:EMBEDDING_MODEL_NAME="BAAI/bge-m3"    # or intfloat/multilingual-e5-base
.\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s
```

End-to-end smoke test against a running server (includes a RAG question; start the
backend with `LLM_PROVIDER=mock` to exercise RAG without a real key):

```powershell
python scripts\smoke_test_api.py
```

## RAG pipeline (Phase 3)

`POST /api/rag/query` with `{ "question": "...", "top_k": 5 }` runs:

```
language detection → conservative preprocessing → query embedding
   (same model as ingestion) → ChromaDB top-K → evidence policy
      → system prompt + question + delimited untrusted evidence → LLM
         → answer + actual source metadata
```

- Generation lives in `app/services/generator.py` (OpenAI-compatible HTTP client;
  works with Groq/OpenAI/OpenRouter/Together/Ollama, plus a network-free `mock`).
- Prompt construction and the versioned system prompt live in
  `app/services/prompt_builder.py`.
- The route (`app/api/rag.py`) is thin; orchestration is in
  `app/services/rag_service.py`.

### Configure the LLM (Groq example)

Put these in `backend/.env` (git-ignored) and restart:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=60
LLM_MAX_OUTPUT_TOKENS=800
```

The service still starts without a key; `/api/rag/query` then returns HTTP 503
with a clear configuration error. The key is never logged or returned. Use
`LLM_PROVIDER=mock` for offline tests/demos.

## Layout

- `app/api/` — FastAPI routers and dependency providers (`deps.py`).
- `app/core/` — settings, logging, error types + exception handlers.
- `app/schemas/` — Pydantic request/response models (incl. `rag.py`).
- `app/services/` — processor, cleaner, chunker, embeddings, ingestion,
  retrieval, registry, language detection, token counting, **query language
  detection, query preprocessing, prompt builder, generator, rag service**.
- `app/vectorstore/chroma_store.py` — persistent ChromaDB (explicit embeddings).
- `tests/` — pytest suite; `tests/integration/` — real-model tests.
- `scripts/` — model download, sample PDF generation, API smoke test.
- `samples/` — original sample documents.
- `data/` — runtime ChromaDB + SQLite registry (gitignored).

## Notes

- Only one embedding model is loaded at a time (`EMBEDDING_MODEL_NAME`).
- The collection stores its embedding model/dimension; switching models requires
  a new `CHROMA_COLLECTION_NAME` or clearing `data/chroma`.
- Documents are tracked in a SQLite registry so failed/processing uploads are
  visible even though they have no chunks.
- Duplicate uploads (identical SHA-256) are reused unless
  `ALLOW_DUPLICATE_DOCUMENTS=true`.
- Retrieval `distance`/`retrieval_score` values are cosine distances, never
  probabilities or factual-verification results.
