# Phase Documentation — Phases 1, 2 and 3

**Project:** NLP-Driven Multilingual Hallucination Localization and Repair for RAG Systems
**Application version:** `0.1.0`
**Scope of this document:** the backend and the data/NLP pipeline implemented in **Phase 1, Phase 2 and Phase 3**.
This document deliberately does **not** describe the user interface, screens, components, styling or theming.

> Phase 4+ (claim extraction, NLI verification, hallucination classification, span localization,
> LLM repair, re-verification) is **not** implemented. The code is structured so those services can be
> added without rewriting the existing pipeline.

---

## 1. Phase summary

| Phase | Goal | Delivered |
| --- | --- | --- |
| **Phase 1** | Foundation / project setup | FastAPI application skeleton, typed settings, structured logging + request logging, centralised error types and HTTP exception mapping, CORS, health endpoint, SQLite registry bootstrap, deterministic test harness. |
| **Phase 2** | Document ingestion + vector storage | PDF/TXT validation, text extraction, conservative cleaning, sentence-aware overlapping chunking, multilingual embeddings, persistent ChromaDB storage with metadata, document registry, retrieval-only search endpoint, delete flow, and full test coverage. |
| **Phase 3** | Retrieval-augmented question answering | Query language detection, conservative query preprocessing, query embedding with the ingestion model, ChromaDB top-K retrieval, an evidence policy, a versioned evidence-grounded system prompt, an OpenAI-compatible generative LLM client, and the `/api/rag/query` endpoint returning an answer plus real source metadata and timings. |

---

## 2. Tech stack and versions

Verified locally (Python 3.12.10, Windows). Pins are ranges in `backend/requirements.txt`; the installed
versions at the time of writing are listed for reproducibility.

| Component | Requirement (range) | Installed | Role |
| --- | --- | --- | --- |
| Python | 3.10–3.12 | 3.12.10 | Runtime |
| fastapi | `>=0.115,<1.0` | 0.147.0 | HTTP API |
| uvicorn[standard] | `>=0.30,<1.0` | 0.54.0 | ASGI server |
| pydantic | `>=2.8,<3.0` | 2.14.1 | Schemas / validation |
| pydantic-settings | `>=2.4,<3.0` | 2.16.0 | Env-based settings |
| python-multipart | `>=0.0.9,<1.0` | — | Multipart uploads |
| pymupdf | `>=1.24,<2.0` | 1.28.2 | PDF text extraction |
| chromadb | `>=0.5.5,<2.0` | 1.5.9 | Persistent vector store |
| sentence-transformers | `>=3.0,<6.0` | 5.7.0 | Embedding model runtime |
| transformers | `>=4.44,<6.0` | 5.20.1 | Tokenizers |
| torch | (transitive) | 2.14.1 | Model execution (CUDA/MPS/CPU) |
| charset-normalizer | `>=3.3,<4.0` | 3.5.2 | Legacy TXT encoding detection |
| langdetect | `>=1.0.9,<2.0` | 1.0.9 | Document language + query fallback |
| httpx | `>=0.27,<1.0` | 0.28.1 | LLM HTTP client |
| lingua-language-detector | `>=2.0,<3.0` | 2.2.0 | Query language detection (primary) |

Development tooling lives in `backend/requirements-dev.txt` (pytest, etc.).

---

## 3. Models used

This section consolidates **every model** used by the pipeline, where it is used, and how it is configured.

### 3.1 Embedding models

One embedding model is loaded per process, selected by `EMBEDDING_MODEL_NAME`, behind the shared
`EmbeddingService` (`backend/app/services/embedding_service.py`). The model is loaded **lazily** on first
embedding call (uploads/queries), never by the health endpoint.

| Purpose | Model | Vector dimension | Text prefixes | Notes |
| --- | --- | --- | --- | --- |
| **Primary** (default) | `BAAI/bge-m3` | **1024** | none | Multilingual (100+ languages), no instruction prefix required. Default in config and `.env.example`. ~2.2 GB on first download. |
| **Benchmark / alternative** | `intfloat/multilingual-e5-base` | **768** | `query: ` / `passage: ` | Requires E5 prefixes. Used to validate pipeline portability across models. |

Configuration that applies to whichever model is active:

| Setting | Value | Effect |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Model id passed to `SentenceTransformer` |
| `EMBEDDING_DEVICE` | `auto` | `auto` resolves CUDA → MPS → CPU; falls back to CPU if the requested device fails to load |
| `EMBEDDING_BATCH_SIZE` | `4` | Conservative batch size for limited memory |
| `EMBEDDING_MAX_SEQ_LENGTH` | `1024` | Caps context length to limit memory use |
| `EMBEDDING_NORMALIZE` | `true` | Vectors are L2-normalised (required for cosine space) |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `auto` | `auto` = `"query: "` / `"passage: "` **only** for E5-style models (name contains "e5"); empty for BGE-M3 |
| `EMBEDDING_TRUST_REMOTE_CODE` | `false` | Remote code execution is disabled |

Implementation guarantees:
- `embed_documents()` applies the **passage** prefix; `embed_query()` applies the **query** prefix.
- Output validation: dimension match, non-empty, all values finite, not all-zero — otherwise `EmbeddingError`.
- Vectors are converted to plain Python `float` lists before being handed to ChromaDB.

### 3.2 Tokenizer / token counting

| Purpose | Implementation | Details |
| --- | --- | --- |
| Chunk sizing and token metadata | `HuggingFaceTokenCounter` | Loads `AutoTokenizer` for `EMBEDDING_MODEL_NAME` (or `TOKENIZER_MODEL_NAME` when set) and counts with `add_special_tokens=False`. |
| Fallback | `HeuristicTokenCounter` | Deterministic estimate used when the tokenizer cannot be loaded (e.g. offline tests): `ceil(ascii_chars / 4) + ceil(non_ascii_chars / 1.5)`, minimum 1. |

So `CHUNK_SIZE` / `CHUNK_OVERLAP` are measured in **model tokens** when a tokenizer is available.

### 3.3 Document language detection (ingestion metadata)

| Aspect | Detail |
| --- | --- |
| Library / model | `langdetect` 1.0.9 (`detect_langs`), `DetectorFactory.seed = 0` for determinism |
| Runs on | The **full document text** (not per chunk — short chunks are too unreliable), first 6000 characters |
| Minimum input | 120 characters, otherwise no label (`null`) |
| Acceptance rule | Accept only if `top.prob >= 0.85` **and** `margin = top.prob - second.prob >= 0.15` |
| Toggle | `LANGUAGE_DETECTION_ENABLED` (default `true`) |
| Stored as | `language` on the **registry row only** (returned by the documents endpoints). It is **not** written into ChromaDB chunk metadata, so `sources[].language` in retrieval/RAG responses is currently always `null`. |

This is a **metadata label only**; it never affects chunking, embedding or retrieval.

### 3.4 Query language detection (Phase 3)

Implemented in `backend/app/services/query_language_detection.py` behind a replaceable
`LanguageBackend` protocol.

| Aspect | Detail |
| --- | --- |
| Primary backend | **Lingua** (`lingua-language-detector` 2.2.0), built with **31 curated languages** (11 Indian languages incl. Hindi/Tamil/Telugu/Bengali/Marathi/Urdu/Gujarati/Kannada/Malayalam/Punjabi, plus English, French, German, Spanish, Portuguese, Italian, Dutch, Russian, Ukrainian, Polish, Turkish, Arabic, Persian, Hebrew, Chinese, Japanese, Korean, Vietnamese, Thai, Indonesian, Swahili) |
| Confidence source | `compute_language_confidence_values(text)` → sorted `(iso_code_639_1, confidence)` candidates |
| Fallback backend | **langdetect** (`detect_langs` → probability) used automatically when Lingua is unavailable or returns nothing |
| Sample size | First 1000 characters (`max_chars`) |
| Minimum input | `QUERY_LANGUAGE_MIN_CHARS` (default 3); also requires at least one letter |
| Reliability gate | `confidence >= QUERY_LANGUAGE_MIN_CONFIDENCE` (0.5) **and** `margin >= QUERY_LANGUAGE_MIN_MARGIN` (0.15) **and** not mixed-script / not multi-language |
| Mixed detection | Unicode name-based script buckets; two or more scripts with ≥3 letters each ⇒ flagged `mixed`; a runner-up language at ≥0.5 confidence also flags mixed |
| Optional override | `QUERY_LANGUAGE_FALLBACK` (ISO code) used only when detection is unreliable; empty ⇒ report "unknown" |
| Guarantees | Detection never raises (failures are logged and degrade to `null`); the query is **never translated or rewritten** |

The detected languages/confidence/reliability are returned in the RAG response as **metadata only** — they are
not proof that the embedding model or LLM performs equally well in that language.

### 3.5 Generative LLM (Phase 3)

Implemented in `backend/app/services/generator.py`. The only real provider client speaks the
**OpenAI-compatible chat-completions** API (`POST {LLM_BASE_URL}/chat/completions`), which covers
Groq, OpenAI, OpenRouter, Together and local Ollama-compatible servers by changing the base URL.

Provider defaults (`PROVIDER_DEFAULTS`):

| `LLM_PROVIDER` | Default base URL | Default model |
| --- | --- | --- |
| `groq` | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `openrouter` | `https://openrouter.ai/api/v1` | (must be set) |
| `together` | `https://api.together.xyz/v1` | (must be set) |
| `ollama` | `http://localhost:11434/v1` | `llama3.1` (no API key required) |
| `openai-compatible` / `custom` | (must be set) | (must be set) |
| `mock` | — | deterministic, network-free (`MockGenerator`) |
| empty / `none` | — | `UnconfiguredGenerator` (503 on use) |

> **Important model note:** Groq **retired `llama-3.3-70b-versatile`** (the endpoint now returns HTTP 404).
> The default Groq model is therefore **`openai/gpt-oss-120b`** (verified working end-to-end). Working
> alternatives on Groq include `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`.

Request parameters actually sent:

| Parameter | Setting | Default |
| --- | --- | --- |
| `model` | `LLM_MODEL` | provider default |
| `messages` | `[{"role": "system", ...}, {"role": "user", ...}]` | built by `PromptBuilder` |
| `temperature` | `LLM_TEMPERATURE` | `0` |
| `max_tokens` | `LLM_MAX_OUTPUT_TOKENS` | `800` |
| `stream` | hard-coded | `false` |

Credential handling: `LLM_API_KEY` is a `SecretStr`, only read from the backend environment/`.env`, never
logged, never returned in any response, and never sent to a client. Response parsing extracts the answer text
and `usage` token counts.

Auxiliary generators:
- **`MockGenerator`** (`provider="mock"`) — deterministic, makes no network call, records every message list
  so tests can assert exactly which evidence was passed. For tests/development only.
- **`UnconfiguredGenerator`** — keeps the app running when no provider is configured; raises
  `GenerationConfigurationError` (503) only when generation is actually requested.

### 3.6 Vector store / index

| Aspect | Detail |
| --- | --- |
| Engine | ChromaDB **PersistentClient** 1.5.9, path `CHROMA_PERSIST_DIRECTORY` (default `backend/data/chroma`) |
| Collection | `CHROMA_COLLECTION_NAME` (default `rag_documents`) |
| Distance space | `hnsw:space = cosine` |
| Embeddings | **Always supplied explicitly**; ChromaDB's default embedding function is disabled (`embedding_function=None`) |
| Writes | Batched (`CHROMA_BATCH_SIZE`, default 256) |
| Telemetry | Disabled (`anonymized_telemetry=False`) |

### 3.7 Model summary table

| # | Stage | Model / library | Version | Output |
| --- | --- | --- | --- | --- |
| 1 | Text extraction (PDF) | PyMuPDF | 1.28.2 | Ordered page text |
| 2 | TXT decoding | charset-normalizer (+ BOM/UTF-8/CP1252) | 3.5.2 | Unicode text |
| 3 | Token counting | `AutoTokenizer` of the embedding model | transformers 5.20.1 | Token counts |
| 4 | Document language | langdetect | 1.0.9 | ISO code / null |
| 5 | Embeddings | `BAAI/bge-m3` (1024) or `intfloat/multilingual-e5-base` (768) | sentence-transformers 5.7.0 | Normalised vectors |
| 6 | Vector search | ChromaDB (cosine HNSW) | 1.5.9 | Top-K chunks + distances |
| 7 | Query language | Lingua (31 languages) → langdetect fallback | 2.2.0 / 1.0.9 | ISO code + confidence + reliability |
| 8 | Answer generation | OpenAI-compatible LLM, default Groq `openai/gpt-oss-120b` | httpx 0.28.1 | Grounded answer text |

---

## 4. Phase 1 — Foundation

- **Application factory** `backend/app/main.py`: `create_app()` builds FastAPI, installs CORS
  (`FRONTEND_ORIGIN`, comma-separated), registers exception handlers, adds HTTP request logging, mounts all
  routers under `/api`, and exposes `GET /` (name/version/docs/health) and `/docs`.
- **Lifespan**: creates `data/` and `data/chroma/` if missing and logs startup configuration
  (env, embedding model, collection, LLM provider).
- **Typed settings** `backend/app/core/config.py`: a single cached `Settings` (`pydantic-settings`) reading
  environment variables and `backend/.env`, with relative paths resolved from `backend/` so behaviour is
  independent of the shell's working directory. Includes cross-field validation
  (`CHUNK_OVERLAP < CHUNK_SIZE`) and derived helpers (`is_e5_model`, effective prefixes, `llm_configured`, …).
- **Centralised errors** `backend/app/core/errors.py` with a single `AppError` base and FastAPI handlers
  mapping them to HTTP status codes:

  | Error | HTTP |
  | --- | --- |
  | `DocumentNotFoundError` | 404 |
  | `DocumentValidationError` | 400 |
  | `DocumentProcessingError` (incl. `ImageOnlyPdfError`) | 422 |
  | `EmbeddingError` | 503 |
  | `VectorStoreError` | 503 |
  | `QueryValidationError` | 400 |
  | `GenerationConfigurationError` | 503 |
  | `GenerationError` | 502 |
  | any unhandled exception | 500 (generic message; details to logs) |

- **Logging** `backend/app/core/logging.py`: single configured logger tree; every request logs method, path,
  status and duration.
- **Health** `backend/app/api/health.py`: `GET /api/health` reports `status` (`ok`/`degraded`), `version`,
  `environment`, `embedding_model`, `embedding_device`, `embedding_loaded`, `max_upload_size_mb`,
  `max_files_per_request`, and a `components` map (`vector_store`, `document_registry`, `embedding_model`)
  with per-component `status` + `detail`. It **does not** load the embedding model (lazy loading).
- **Dependency wiring** `backend/app/api/deps.py`: cached providers for settings, embedding service, Chroma
  store, registry, processor, chunker, ingestion, retrieval, generator, prompt builder, language detectors and
  the RAG service, plus status probes used by health. Tests override these providers.

---

## 5. Phase 2 — Ingestion, chunking, embeddings and vector storage

### 5.1 Ingestion pipeline

`backend/app/services/ingestion_service.py` orchestrates:
**validate → SHA-256 dedupe check → registry row (`processing`) → extract → clean → chunk → embed → store →
registry update (`completed`)**.

Per-file error handling keeps a multi-file upload alive: expected failures are recorded as
`status="failed"` (or `"rejected"` before ingestion) with a message, orphaned chunks are cleaned up, and
the HTTP request still returns 200 with one structured result per file.

Extraction (`document_processor.py`):
- Supported types: `.pdf`, `.txt` (extension **and** content-type validated).
- Filenames are sanitised (no path components, no control characters, length-capped) and used as display
  metadata only; uploaded bytes are never written to disk or executed. Document IDs are UUID4.
- **PDF**: PyMuPDF (`import pymupdf as fitz`, with `fitz` fallback); page-by-page `get_text("text")`; encrypted
  or password-protected PDFs are rejected; PDFs with no extractable text (image-only/scans) raise
  `ImageOnlyPdfError` (OCR is **not** implemented).
- **TXT**: decoding order = UTF BOMs (UTF-8/16/32) → strict UTF-8 → charset-normalizer → CP1252 fallback;
  obviously binary payloads are rejected; implausible decodings (control-character ratio > 2 %) are discarded.

Cleaning (`text_cleaner.py`): deliberately conservative — NFC normalisation, line-ending normalisation,
zero-width/control-character removal, exotic-space collapsing, whitespace/blank-line tidy-up. It preserves
punctuation, numbers, dates, names, casing and sentence boundaries; no stemming, stop-word removal,
translation or destructive normalisation.

### 5.2 Chunking algorithm

`backend/app/services/text_chunker.py` — sentence-aware, recursive, with token overlap; chunks never cross
page boundaries.

1. Split each page into paragraphs (`\n\s*\n`), then into sentences using a multilingual regex
   (Latin `.!?`, Devanagari `।॥`, CJK `。！？`, Arabic/Urdu `؟۔`).
2. Accumulate sentences into a chunk until adding the next would exceed `CHUNK_SIZE` tokens (default **700**).
3. On flush, carry a tail of complete sentences totalling up to `CHUNK_OVERLAP` tokens (default **100**) into
   the next chunk when it still fits.
4. A single oversized sentence is split on word boundaries with the same overlap logic.
5. Duplicate chunk text within a document is dropped (SHA-256 of the normalised text).
6. Metadata per chunk: `chunk_id`, `document_id`, `filename`, `chunk_index`, `text`, `token_count`,
   `page_number`.

Chunk IDs are stable and deterministic:
`"{document_id}:{chunk_index:06d}:{sha256(chunk_text)[:12]}"`.

### 5.3 Embedding + storage

- `EmbeddingService.embed_documents()` embeds all chunk texts in `EMBEDDING_BATCH_SIZE` batches and validates
  every vector.
- `ChromaStore.add_chunks()` writes ids, documents, **explicit** embeddings and metadata to ChromaDB in
  `CHROMA_BATCH_SIZE` batches. Stored metadata per chunk:
  `document_id`, `filename`, `chunk_id`, `chunk_index`, `token_count`, `embedding_model`, and `page_number`
  (when known).
- The collection stores `embedding_model` in its metadata, and every chunk records `embedding_model` too.
  On startup, add and query operations verify the stored model and vector dimension and **refuse** mismatched
  data with a clear `VectorStoreError` (prevents mixing embeddings from different models, which would produce
  meaningless similarity). Switching models requires a new `CHROMA_COLLECTION_NAME` or clearing Chroma data.

### 5.4 Document registry (SQLite)

`backend/app/services/document_registry.py` — ChromaDB stores only chunks; the registry stores documents in
any state plus aggregate metadata. Single table `documents` (WAL mode, 30 s busy timeout):

`document_id` (PK), `filename`, `checksum`, `status`, `size_bytes`, `content_type`, `page_count`,
`character_count`, `token_count`, `chunk_count`, `chunks_stored`, `language`, `embedding_model`, `error`,
`created_at`, `updated_at`; index on `(checksum, status)`.

Duplicate handling: when `ALLOW_DUPLICATE_DOCUMENTS=false` (default), a completed document with the same
SHA-256 checksum is **reused** instead of re-ingested (result marked `duplicate=true`).

Deletion removes both the Chroma chunks and the registry row.

### 5.5 Retrieval (search-only endpoint)

`backend/app/services/retrieval_service.py` + `POST /api/retrieval/search` — embeds the query with the
**same** model/config used for ingestion, queries ChromaDB for top-K (clamped to 1–50), optionally filtered by
`document_ids`, and returns chunk text and metadata.

**Score semantics (documented explicitly everywhere):** `metric = "cosine_distance"`,
`distance` = ChromaDB cosine distance (**lower = more similar**), `similarity = 1 - distance`. These are
similarity scores only — **not** probabilities, confidence values, or factual-verification results.

---

## 6. Phase 3 — Retrieval-augmented question answering

### 6.1 Pipeline stages (`POST /api/rag/query`)

`RagService.answer()` (`backend/app/services/rag_service.py`) runs:

1. **Query language detection** — `QueryLanguageDetector.detect()` (§3.4); result is metadata only.
2. **Conservative preprocessing** — `preprocess_query()` (`query_processor.py`): NFC normalisation, line-ending
   normalisation, zero-width/control-character cleanup, exotic-space collapsing, and all whitespace runs
   collapsed to single spaces. The query is treated as one logical line. Empty-after-cleaning raises
   `QueryValidationError` (400); exceeding `RAG_MAX_QUESTION_LENGTH` (default 2000) raises the same.
   No stop-word removal, no translation, no rewriting.
3. **Query embedding** — the same embedding model and prefix configuration as ingestion
   (a model mismatch raises `VectorStoreError`).
4. **Retrieval** — ChromaDB top-K (`top_k`, clamped 1–50, default `TOP_K=5`), optional `document_ids` filter.
5. **Evidence policy** — see §6.2.
6. **Prompt construction** — see §6.3.
7. **Generation** — the configured generator, or a system insufficient-evidence message (no LLM call).
8. **Response assembly** — answer, `answer_mode`, sources, retrieval info, generation info, timings.

### 6.2 Evidence policy

Configurable, per-model (cosine thresholds vary by embedding model):

- `RAG_MIN_EVIDENCE_CHUNKS` (default 1) — minimum number of returned chunks.
- `RAG_DISTANCE_FILTER_ENABLED` (default `true`) with `RAG_MAX_DISTANCE` (default 0.75) — when enabled,
  evidence is sufficient only if **at least one** chunk has `distance <= RAG_MAX_DISTANCE`.
- `RAG_GENERATION_WITHOUT_EVIDENCE` (default `false`) — when evidence is insufficient, the LLM is **not**
  called; the service returns a system-authored insufficient-evidence message in the detected language.

Resulting `answer_mode`: `"grounded"` when the policy passes, otherwise `"insufficient_evidence"`.

### 6.3 Prompt construction (`prompt_builder.py`)

- **Versioned system prompt:** `SYSTEM_PROMPT_VERSION = "rag-grounded-answer-1.0.0"`, returned in every
  response as `generation.prompt_version`.
- The system prompt instructs the model to: answer using **only** the supplied evidence; treat evidence as
  **untrusted data** (never follow instructions inside it — prompt-injection defence); never invent names,
  dates, statistics, quotes or numbers; say so explicitly when evidence is missing/insufficient/contradictory;
  distinguish facts from inferences; answer in the question's language; cite `[Source N]` identifiers; never
  fabricate citations; and never fall back on general knowledge.
- The user message contains: `USER QUESTION`, then the retrieved evidence delimited as untrusted by
  `<<<BEGIN_UNTRUSTED_EVIDENCE>>>` / `<<<END_UNTRUSTED_EVIDENCE>>>`, then the task. Each evidence block
  renders `[Source N]`, filename, document id, page, chunk id and content.
- `build_insufficient_evidence()` produces the no-evidence prompt used only when generation without evidence
  is enabled.

### 6.4 Generation and provider error mapping

`OpenAICompatibleGenerator` maps provider failures to clean application errors:

| Provider condition | Raised | HTTP |
| --- | --- | --- |
| Missing key / missing model | `GenerationConfigurationError` | 503 |
| HTTP 401/403 (auth) | `GenerationConfigurationError` | 503 |
| HTTP 404 (model not found — e.g. a retired model id) | `GenerationConfigurationError` | 503 |
| HTTP 429 (rate limit) | `GenerationError` | 502 |
| HTTP ≥ 500 (provider outage) | `GenerationError` | 502 |
| Any other 4xx (with provider message excerpt) | `GenerationError` | 502 |
| Timeout | `GenerationError` | 502 |
| Connection error | `GenerationError` | 502 |
| Non-JSON / empty answer / no choices | `GenerationError` | 502 |

When the LLM is unconfigured the app still starts and serves health, uploads and retrieval; only
`/api/rag/query` returns 503.

### 6.5 Response schema (`schemas/rag.py`)

`question`, `normalized_question`, `detected_language`, `language_confidence`, `language_reliable`,
`answer`, `answer_mode`, `sources[]`, `retrieval`, `generation`, `timings`.

- `sources[]`: `document_id`, `filename`, `chunk_id`, `chunk_index`, `page_number`, `language`, `text`,
  `retrieval_score` (= cosine distance), `similarity`. (`language` is currently always `null` because
  chunk-level language is not persisted — see §3.3 and §11.)
- `retrieval`: `top_k`, `returned_chunks`, `sufficient_evidence`, `metric`, `score_interpretation`,
  `min_evidence_chunks`, `max_distance`, `distance_filter_applied`.
- `generation`: `used_llm`, `provider`, `model`, `prompt_version`, `finish_reason`, `prompt_tokens`,
  `completion_tokens`, `total_tokens`.
- `timings`: `retrieval_ms`, `generation_ms`, `total_ms`.

**All source metadata is real** (taken from the retrieved chunks); nothing is fabricated.

### 6.6 Insufficient-evidence messages

System-authored, language-aware (no claims about the documents):

- `en` — "I could not find sufficiently relevant evidence in the uploaded documents to answer this question."
- `hi` — Hindi equivalent.
- `ta` — Tamil equivalent.
- Unknown/unsupported language codes fall back to the English message; non-English messages append a note that
  no retrieved chunk passed the configured retrieval policy.

---

## 7. Configuration reference

Read from environment variables and `backend/.env` (see `backend/.env.example`, root `.env.example`).
Relative paths resolve from `backend/`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` / `APP_VERSION` | `development` / `0.1.0` | Reported by health |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Server binding |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Comma-separated CORS origins |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | ChromaDB persistence path |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Collection name (change when switching models) |
| `CHROMA_BATCH_SIZE` | `256` | Chroma write batch size |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding model (§3.1) |
| `EMBEDDING_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `EMBEDDING_BATCH_SIZE` | `4` | Inference batch size |
| `EMBEDDING_MAX_SEQ_LENGTH` | `1024` | Max tokens per text |
| `EMBEDDING_NORMALIZE` | `true` | L2-normalise vectors |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `auto` | `auto` → E5 prefixes only |
| `EMBEDDING_TRUST_REMOTE_CODE` | `false` | Disable remote code |
| `TOKENIZER_MODEL_NAME` | (empty) | Override tokenizer model; empty = embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Token-based chunking (overlap must be < size) |
| `TOP_K` | `5` | Default retrieval depth |
| `MAX_UPLOAD_SIZE_MB` | `20` | Per-file upload limit |
| `MAX_FILES_PER_REQUEST` | `5` | Files per upload request |
| `ALLOW_DUPLICATE_DOCUMENTS` | `false` | Reuse identical content by checksum |
| `LANGUAGE_DETECTION_ENABLED` | `true` | Document language label + query detection |
| `DATA_DIRECTORY` | `./data` | Runtime data root |
| `DOCUMENT_REGISTRY_PATH` | `./data/document_registry.sqlite3` | SQLite registry file |
| `QUERY_LANGUAGE_MIN_CHARS` | `3` | Minimum query length for detection |
| `QUERY_LANGUAGE_MIN_CONFIDENCE` | `0.5` | Minimum top-language confidence |
| `QUERY_LANGUAGE_MIN_MARGIN` | `0.15` | Minimum lead over the runner-up |
| `QUERY_LANGUAGE_FALLBACK` | (empty) | Optional ISO code when detection is unreliable |
| `RAG_MAX_QUESTION_LENGTH` | `2000` | Rejects overlong questions (400) |
| `RAG_MIN_EVIDENCE_CHUNKS` | `1` | Minimum chunks for grounded answering |
| `RAG_DISTANCE_FILTER_ENABLED` | `true` | Enable the distance gate |
| `RAG_MAX_DISTANCE` | `0.75` | Cosine-distance gate (tune per model) |
| `RAG_GENERATION_WITHOUT_EVIDENCE` | `false` | If `true`, still call the LLM with weak evidence |
| `LLM_PROVIDER` | `groq` | Provider id (§3.5) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Provider model id |
| `LLM_API_KEY` | (empty) | Secret; set in `backend/.env`, never committed |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `LLM_TEMPERATURE` | `0` | Low temperature for factual answering |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request provider timeout |
| `LLM_MAX_OUTPUT_TOKENS` | `800` | Answer length cap |

---

## 8. Storage layout and data model

```
backend/data/                        (gitignored runtime data)
├── chroma/                          ChromaDB PersistentClient files
│   └── collection "rag_documents"   hnsw:space=cosine; explicit embeddings
└── document_registry.sqlite3        documents table (WAL)
```

ChromaDB chunk metadata: `document_id`, `filename`, `chunk_id`, `chunk_index`, `token_count`,
`embedding_model`, `page_number` (optional).

Registry columns and the derived aggregate counts are listed in §5.4.

---

## 9. API reference

All routes are mounted under the `/api` prefix. `GET /` returns app name/version/docs/health; `/docs` is the
interactive OpenAPI UI.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, version, environment, embedding model/device/loaded, upload limits, component health (does not load the model) |
| `POST` | `/api/documents/upload` | Multipart upload of 1..N PDF/TXT files (field `files`); returns one result per file |
| `GET` | `/api/documents` | List documents with status and counts |
| `GET` | `/api/documents/{document_id}` | Document detail |
| `DELETE` | `/api/documents/{document_id}` | Delete chunks + registry row |
| `POST` | `/api/retrieval/search` | `{ query, top_k?, document_ids? }` → ranked chunks (no answer generation) |
| `POST` | `/api/rag/query` | `{ question, top_k?, document_ids? }` → grounded answer + sources + timings |

Uploads always return HTTP 200 with a per-file result status of `completed`, `failed`, or `rejected`, so a
batch survives individual failures. `POST /api/rag/query` returns: 400 invalid/overlong question, 422 schema
validation error, 503 LLM not configured / vector store unavailable, 502 LLM provider error.

---

## 10. Testing and verification

### 10.1 Automated backend tests

Command:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Latest result: **138 passed, 3 skipped** (~16 s). The 3 skips are the opt-in real-embedding integration tests.

Test design: the unit/API suite **never downloads a model**. `tests/conftest.py` injects a deterministic fake
encoder (16-dimensional) through `EmbeddingService`'s model factory, runs ChromaDB and SQLite against
per-test temporary directories, and provides a `TestClient` with dependency overrides. It also provides
fake/unavailable query-language backends for deterministic detection tests.

Coverage by area (15 unit/API modules in `tests/`, plus the 2 opt-in integration modules in §10.2):

| Area | Module(s) |
| --- | --- |
| Health + documents API | `test_api_documents.py` |
| Retrieval API | `test_api_retrieval.py` |
| RAG API | `test_api_rag.py` |
| Extraction / validation / decoding | `test_document_processor.py` |
| Cleaning | `test_text_cleaner.py` |
| Chunking | `test_text_chunker.py` |
| Embeddings (incl. E5 prefixes, validation) | `test_embedding_service.py` |
| Ingestion orchestration + dedupe | `test_ingestion_service.py` |
| Registry | `test_document_registry.py` |
| Chroma store + model/dimension guard | `test_chroma_store.py` |
| Document language detection | (covered via ingestion) |
| Query language detection | `test_query_language_detection.py` |
| Query preprocessing | `test_query_processor.py` |
| Prompt construction | `test_prompt_builder.py` |
| Generator error mapping (mocked HTTP) | `test_generator.py` |
| RAG evidence policy + full flow (mocked LLM) | `test_rag_service.py` |

### 10.2 Real-model integration tests (opt-in)

```powershell
cd backend
$env:RUN_EMBEDDING_INTEGRATION="1"
$env:EMBEDDING_MODEL_NAME="BAAI/bge-m3"            # or intfloat/multilingual-e5-base
.\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s
```

- `test_real_embeddings.py` — ingestion/retrieval with the real model
  (verified: `BAAI/bge-m3` → dim **1024**; `intfloat/multilingual-e5-base` → dim **768**).
- `test_rag_real_embeddings.py` — the full RAG pipeline with the real embedding model and the deterministic
  **mock** generator (no API credits spent).
- Verified: **2 tests passed** per model.

### 10.3 End-to-end smoke test

```powershell
cd backend
$env:LLM_PROVIDER="mock"                            # only to exercise RAG without an API key
.\.venv\Scripts\python.exe scripts\smoke_test_api.py
```

`scripts/smoke_test_api.py` uploads `samples/sample_multilingual.txt` and `samples/sample_english.pdf`, runs
multilingual retrieval queries, deletes a document, and runs a RAG question end-to-end. Set
`SMOKE_REQUIRE_LLM=1` to make an unconfigured LLM a failure.

A real Groq run (default model `openai/gpt-oss-120b`) produced a grounded answer
("The Eiffel Tower is located in Paris, France [Source 1]") and correctly refused to fabricate an answer for
an unrelated question. Example output: `SMOKE_TEST_OK`.

### 10.4 Phase acceptance

- **Phase 1:** app starts; `GET /api/health` returns ok; CORS allows the configured frontend origin; no secrets
  exposed.
- **Phase 2:** PDF/TXT upload works; cleaning preserves multilingual facts; overlapping chunks are produced;
  embeddings are generated; metadata persists in ChromaDB across restarts; multilingual retrieval returns
  relevant chunks; invalid/corrupt/oversize/encrypted/image-only files are handled with clear errors.
- **Phase 3:** questions are answered from retrieved evidence; query language is detected where possible
  (EN/HI/TA with confidence + reliability); the query uses the ingestion embedding model; the LLM receives a
  separate system prompt plus delimited untrusted evidence; the response contains the answer and **real**
  source metadata; insufficient evidence is handled explicitly without fabrication; provider errors map to
  clean 503/502 responses; automated tests (including mocked-LLM) pass.

---

## 11. Known limitations

- **OCR is not implemented**; image-only PDFs are reported as errors, never silently "succeeded".
- **Document language detection** runs on the whole document and may label mixed-language files with the
  dominant language (e.g. `en`); labels shorter than 120 characters stay `null`.
- **Query language detection is metadata only** — it is not proof the embedding model or the LLM performs
  equally well in that language, and the query is never translated.
- **Retrieval scores are cosine distances/similarities**, not truth or confidence values.
- **Chunk-level language is not persisted** (the `DocumentChunk` model has no `language` field and the vector
  store writes only document/chunk metadata), so `sources[].language` in retrieval/RAG responses is currently
  always `null`; the document-level language label is still stored in the registry and returned by the
  documents endpoints.
- **Evidence policy uses a single `RAG_MAX_DISTANCE` per embedding model.** In an observed run an unrelated
  question still landed at distance ≈ 0.626 (< 0.75) and was treated as "sufficient"; the LLM was then
  instructed to state the limitation. Tune the threshold (or disable the filter) per model/corpus.
- **Upload processing is synchronous**; the first request includes embedding-model load time.
- **Grounding is prompt-based only.** A system prompt reduces but cannot eliminate unsupported answers; a
  dedicated verification/localization engine is a later phase.
- Only `.pdf` and `.txt` are supported; there is no authentication/authorisation layer.

---

## 12. Out of scope (Phase 4+)

Planned but not implemented: multilingual Transformer NLI
(`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, `joeddav/xlm-roberta-large-xnli`), multilingual NER, claim
extraction, claim–evidence matching, hallucination classification, span localization, LLM repair, and
re-verification. Intended constraints: inference stays isolated from route handlers, thresholds remain
configurable, and models are evaluated per language (Recall@K, MRR, Precision/Recall/F1/Macro-F1, span F1, IoU)
before selection.

---

## 13. Backend repository map

```
backend/
├── app/
│   ├── main.py                      FastAPI factory, CORS, middleware, routers
│   ├── api/
│   │   ├── deps.py                  cached dependency providers + status probes
│   │   ├── health.py                GET /api/health
│   │   ├── documents.py             upload / list / detail / delete
│   │   ├── retrieval.py             POST /api/retrieval/search
│   │   └── rag.py                   POST /api/rag/query
│   ├── core/
│   │   ├── config.py                Settings (env + .env)
│   │   ├── errors.py                error types + HTTP handlers
│   │   └── logging.py               logging configuration
│   ├── schemas/                     health, document, retrieval, rag
│   ├── services/
│   │   ├── document_processor.py    validation, PDF/TXT extraction, decoding
│   │   ├── text_cleaner.py          conservative cleaning
│   │   ├── text_chunker.py          sentence-aware overlapping chunking
│   │   ├── token_counter.py         model tokenizer + heuristic fallback
│   │   ├── embedding_service.py     lazy multilingual embeddings
│   │   ├── language_detection.py    document language label (langdetect)
│   │   ├── ingestion_service.py     ingestion orchestration + dedupe
│   │   ├── retrieval_service.py     query embedding + ChromaDB search
│   │   ├── document_registry.py     SQLite registry
│   │   ├── query_language_detection.py  Lingua + langdetect fallback
│   │   ├── query_processor.py       conservative query preprocessing
│   │   ├── prompt_builder.py        versioned grounded system prompt
│   │   ├── generator.py             OpenAI-compatible LLM + mock/unconfigured
│   │   └── rag_service.py           RAG orchestration + evidence policy
│   └── vectorstore/
│       └── chroma_store.py          persistent cosine vector store
├── tests/                           pytest unit/API tests + integration/
├── samples/                         sample_multilingual.txt, sample_english.pdf
├── scripts/                         download_embedding_model, generate_sample_pdf, smoke_test_api
├── data/                            runtime chroma + registry (gitignored)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## 14. How to run the backend

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env

# optional: pre-download the embedding model (~2.2 GB for BAAI/bge-m3)
python scripts\download_embedding_model.py

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: <http://127.0.0.1:8000/api/health>
- OpenAPI docs: <http://127.0.0.1:8000/docs>

For answer generation, set `LLM_PROVIDER`, `LLM_MODEL` and `LLM_API_KEY` in `backend/.env`
(Groq default: `openai/gpt-oss-120b`). Use `LLM_PROVIDER=mock` for a network-free deterministic answer in
tests/demos. Without a key the app still serves health, uploads and retrieval; only `/api/rag/query`
returns 503.


## Phase 4: NLP Verification Engine and Model Training

**Objective:** Develop an NLP-based verification engine that evaluates individual factual claims against retrieved evidence and learns to distinguish supported claims from contradictions and unsupported claims.

### Implementation tasks

- Detect the language of generated answers and apply text normalization and sentence segmentation.
- Tokenize the answer into words or subword units.
- Implement factual claim extraction as an independent, testable component rather than treating tokenization as claim extraction.
- Identify named entities using a multilingual Named Entity Recognition (NER) model.
- Generate semantic representations of claims and retrieved evidence using multilingual Transformer embeddings.
- Match each claim with its most relevant evidence from the existing RAG retrieval pipeline.
- Implement a baseline using TF-IDF features and Logistic Regression for claim–evidence classification where suitable.
- Implement a pretrained multilingual Transformer-based Natural Language Inference (NLI) model.
- Prepare labelled training examples with entailment, contradiction, and neutral classes, ensuring the labels are supported by the original annotations.
- Fine-tune a suitable multilingual Transformer on compatible labelled data.
- Compare the fine-tuned model against the traditional ML baseline and pretrained Transformer.
- Save the trained model, tokenizer, label mapping, and evaluation configuration for integration into the backend.
- Return a structured verification result for every claim, including its predicted label, confidence, and supporting evidence.

**Datasets to investigate:** RAGTruth, PsiloQA, and a suitable Natural Language Inference dataset. Inspect their schemas and annotation compatibility before selecting training data.

**Expected deliverable:** A working NLP verification engine with an evaluated, fine-tuned claim–evidence verification model.

## Phase 5: Hallucination Detection and Span Localization

**Objective:** Identify unsupported or contradicted claims and locate the exact problematic information within generated answers.

### Implementation tasks

- Map NLI predictions to three application-level outcomes: supported, contradicted, and insufficiently supported.
- Apply configurable decision thresholds and confidence checks to reduce unreliable classifications.
- Identify hallucinated claims using the relationship between the generated claim and retrieved evidence.
- Implement span-level hallucination localization using suitable annotated training data and an appropriate token-level or span-prediction approach.
- Identify problematic words or spans, including incorrect dates, numbers, named entities, and factual statements.
- Associate each problematic span with its verification label and relevant evidence.
- Evaluate claim-level classification separately from span-level localization.
- Return structured results containing the original answer, claim, localized span, verification status, confidence, and evidence.

**Expected deliverable:** An evidence-grounded hallucination detection and localization module that explains what is wrong and why.

## Phase 6: Targeted Repair and Re-verification

**Objective:** Correct problematic information using retrieved evidence while preserving content that has already been verified.

### Implementation tasks

- Pass the original answer, identified problematic span, verified evidence, and repair instructions to the generative LLM.
- Repair only the affected information wherever possible.
- Preserve supported claims and the original answer's intended meaning.
- Re-run the repaired answer through the NLP verification engine.
- Accept the repaired answer only when it satisfies the configured verification criteria.
- Flag unresolved or insufficiently supported claims rather than presenting them as verified.
- Limit repair attempts and detect repeated failures to prevent endless repair loops.
- Record the original answer, detected spans, repaired answer, and verification results before and after repair.
- Evaluate correction accuracy, faithfulness, over-correction, and newly introduced hallucinations.

**Expected deliverable:** An integrated detection, localization, targeted repair, and re-verification pipeline.

## Phase 7: Next.js Interface Integration

**Objective:** Integrate the NLP research pipeline into the existing web application without disrupting the working document ingestion and RAG functionality.

### Implementation tasks

- Preserve the existing document upload, processing status, embedding generation, ChromaDB storage, and retrieval features.
- Add a multilingual chat interface for document-based questions.
- Display generated answers with supported, contradicted, and insufficiently supported spans highlighted.
- Provide expandable evidence sections for individual claims.
- Display the verification label and confidence for each claim.
- Show corrected answers and their re-verification status.
- Integrate the interface with the FastAPI verification, localization, and repair endpoints.
- Add clear loading, error, and verification-failure states.
- Ensure that model inference and repair operations run through the backend rather than exposing API credentials in the frontend.

**Expected deliverable:** A complete web application demonstrating multilingual RAG, trained NLP verification, hallucination localization, and evidence-grounded repair.

## Phase 8: Dataset Preparation, Training, Evaluation, and Final Testing

**Objective:** Establish measurable evidence of the effectiveness of the proposed NLP verification and repair approach.

### Dataset preparation

- Investigate RAGTruth for RAG-specific hallucination examples.
- Investigate PsiloQA for multilingual, span-level hallucination detection.
- Investigate Mu-SHROOM for annotated hallucination spans and relevant multilingual evaluation.
- Investigate BHRAM-IL for relevant Indian-language hallucination evaluation.
- Inspect dataset schemas, language coverage, licences, annotation types, and train/validation/test splits.
- Convert compatible annotations into the formats required by the selected verification and localization models.
- Avoid inventing entailment or contradiction labels when the original dataset does not establish them.
- Prevent data leakage by separating training, validation, and test examples appropriately.

### Model training and comparison

- Train a traditional ML baseline using suitable labelled features.
- Evaluate the pretrained multilingual Transformer without fine-tuning.
- Fine-tune a compatible Transformer on labelled training data.
- Compare the three approaches using the same evaluation split and metrics.
- Evaluate the impact of the trained verification model on the complete RAG pipeline.
- Record training configuration, model versions, random seeds where applicable, and computational requirements.

### Evaluation metrics

- **Retrieval:** Recall@K and Mean Reciprocal Rank (MRR).
- **Claim verification:** Precision, Recall, F1-score, and Macro-F1.
- **Span localization:** Span F1 and Intersection over Union (IoU).
- **Confidence:** AUROC where applicable and Expected Calibration Error (ECE).
- **Repair:** Correction accuracy, faithfulness, over-correction rate, and newly introduced hallucination rate.
- **Multilingual performance:** Per-language metrics and macro-averaged results across evaluated languages.

### Final testing

- Test questions in English and Hindi, followed by other supported languages where the models and datasets permit.
- Test cases where evidence supports a claim, contradicts it, or does not contain enough information.
- Test incorrect dates, numbers, named entities, and multi-sentence factual claims.
- Test whether localized spans correspond to the actual annotated errors.
- Test whether targeted repair preserves verified information.
- Compare the original RAG output with the verified and repaired output.
- Document quantitative results, limitations, error analysis, and the final demonstration.

**Expected deliverable:** A reproducible experimental evaluation showing the performance of the trained NLP models, comparisons with baselines, multilingual results, and the effectiveness of the complete detection-and-repair pipeline.

## Research Contribution

The proposed system extends a working multilingual RAG application with a trainable, evidence-grounded NLP verification component. Its contributions will be evaluated through:

1. Explicit claim extraction and claim-level evidence verification.
2. Fine-tuning and comparison of multilingual verification models.
3. Hallucination localization at the word or span level.
4. Evidence-grounded targeted repair followed by independent re-verification.
5. Multilingual evaluation using compatible public datasets and controlled test cases.

The research contribution will be supported by the implemented methods and measured experimental results, rather than by the web application alone.