# NLP-Driven Multilingual Hallucination Localization and Repair for RAG Systems

Phases 1–4A of a multilingual RAG system: project setup, document ingestion +
vector storage, retrieval-augmented question answering, and **pretrained
claim-level verification** against retrieved evidence.

Current capability: upload **PDF/TXT** documents, extract and clean text, chunk it
with overlap, generate **multilingual embeddings**, store chunks in a persistent
**ChromaDB** collection, retrieve relevant evidence, **answer questions with a
generative LLM grounded in the retrieved evidence**, and **extract claims and
verify them against that evidence** — all through a Next.js UI.

> Phase 4A provides pretrained inference only: language detection, claim
> extraction, multilingual NER, claim/evidence matching, and multilingual NLI.
> Fine-tuning, the TF-IDF baseline comparison, span localization, repair, and
> re-verification are **not** implemented yet.
>
> The RAG pipeline and verification reduce unsupported answers but do **not**
> guarantee correctness. NLI confidence is a model score, not calibrated factual
> truth. Retrieval similarity is not a verification result.

The project root in this repository is the working folder (`NLP/`), which plays
the role of `multilingual-hallucination-project/` in the specification.

---

## 1. Architecture

```
Next.js UI (app/, components/, lib/api.ts)
        │  fetch (NEXT_PUBLIC_API_BASE_URL)
        ▼
FastAPI  /api/health  /api/documents/*  /api/retrieval/search
         /api/rag/query  /api/verification/check
         /api/verification/results/{result_id}
        │
        ├── api/          route handlers + dependency wiring
        ├── services/     document_processor, text_cleaner, text_chunker,
        │                 embedding_service, ingestion_service, retrieval_service,
        │                 document_registry, language_detection, token_counter,
        │                 query_language_detection, query_processor,
        │                 prompt_builder, generator, rag_service,
        │                 verification/ (text, claims, ner, matching, nli, onnx, engine)
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
   (same model as ingestion) → ChromaDB top-K → evidence assessment
      → system prompt + question + retrieved evidence → LLM → answer + sources
         → language detection → factual claim extraction → multilingual NER
            → claim/evidence matching → multilingual NLI
               → per-claim status + answer status
```

Verification is optional and isolated: if models fail, the generated answer is
preserved and the response reports a verification error.

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
│   │   │                         # rag.py, verification.py, deps.py
│   │   ├── core/                 # config.py, logging.py, errors.py
│   │   ├── schemas/              # document.py, retrieval.py, rag.py,
│   │   │                         # verification.py, health.py
│   │   ├── services/             # processor, cleaner, chunker, embeddings,
│   │   │                         # ingestion, retrieval, registry, language,
│   │   │                         # token_counter, query_language_detection,
│   │   │                         # query_processor, prompt_builder, generator,
│   │   │                         # rag_service, verification/
│   │   └── vectorstore/          # chroma_store.py
│   ├── tests/                    # pytest unit/API tests + integration/
│   ├── samples/                  # original multilingual + English sample docs
│   ├── evaluation/               # verification benchmark + ONNX export
│   ├── scripts/                  # download/export models, generate PDF, smoke test
│   ├── data/                     # runtime: chroma + registry + ONNX (gitignored)
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
- ~7 GB free disk for the default embedding model (`BAAI/bge-m3`, ~2.2 GB), the
  default verification models, and their dependencies. A GPU is optional; CPU
  works but first model load and inference are slower.

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

With the default local NLI backend, the first verification request also
downloads/loads `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` and
`Davlan/xlm-roberta-base-wikiann-ner` unless they are already cached. Expect the
first verification request to include model-load latency. Set
`VERIFICATION_ENABLED=false` to skip verification entirely. To keep the NLI
model off this machine, configure a hosted endpoint as described in
[Verification models](#verification-models-phase-4a).

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
| `VERIFICATION_ENABLED` | `true` | Enable on-demand claim-level verification for stored Ask results |
| `VERIFICATION_MAX_EXTRACTED_CLAIMS` | `64` | Extracted claims retained for reporting; must be ≥ verified claims |
| `VERIFICATION_MAX_CLAIMS` | `20` | Maximum claims verified per answer; skipped claims are reported with `verification_error` and a `not_checked` reason |
| `VERIFICATION_MAX_EVIDENCE_PER_CLAIM` | `50` | Distinct evidence chunks considered per claim; default matches the largest retrievable `top_k` |
| `VERIFICATION_MAX_NLI_PAIRS` | `250` | Hard ceiling for exhaustive claim/evidence NLI pairs; skipped pairs are reported missing |
| `VERIFICATION_MIN_EVIDENCE_RELEVANCE` | `0.5` | Minimum claim-term or semantic relevance before an NLI verdict counts |
| `VERIFICATION_PRIMARY_WINDOW_TOKENS` | `256` | Focused window tried first; ambiguous claims may use the larger limit |
| `VERIFICATION_DEADLINE_SECONDS` | `120` | Stop starting new NLI work; unstarted claims retain `verification_error` with the deadline reason |
| `VERIFICATION_MAX_CONCURRENT_REQUESTS` | `1` | Local verification jobs allowed at once |
| `VERIFICATION_QUEUE_WAIT_SECONDS` | `5` | Maximum wait for a verification slot |
| `VERIFICATION_MAX_QUEUED_REQUESTS` | `2` | Waiting requests allowed before HTTP 503 overload |
| `VERIFICATION_NLI_MODEL_NAME` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Multilingual NLI; slower on CPU but passed cross-lingual and contradiction integration checks |
| `VERIFICATION_NLI_BACKEND` | `local` | `local`, `onnx`, or `hf_endpoint` |
| `VERIFICATION_NLI_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` (PyTorch backend only) |
| `VERIFICATION_NLI_BATCH_SIZE` | `8` | Local in-process inference batch size |
| `VERIFICATION_NLI_MAX_BATCH_TOKENS` | `8192` | Padded token cap per local NLI batch |
| `VERIFICATION_NLI_CACHE_SIZE` | `256` | Exact claim/evidence prediction cache entries; `0` disables it |
| `VERIFICATION_NLI_ONNX_MODEL_PATH` | empty | Required for `onnx`; path to exported `model.onnx` |
| `VERIFICATION_NLI_ONNX_PROVIDER` | `CPUExecutionProvider` | ONNX Runtime execution provider |
| `VERIFICATION_NLI_ONNX_INTRA_THREADS` / `INTER_THREADS` | `0` / `1` | ONNX Runtime thread caps; `0` lets it choose |
| `VERIFICATION_NLI_HOSTED_BATCH_SIZE` | `1` | Pairs per hosted request; raise only after confirming endpoint support |
| `VERIFICATION_NLI_MAX_CONCURRENCY` | `2` | Maximum concurrent hosted requests |
| `VERIFICATION_NLI_MAX_TOKENS` | `256` | NLI pair-token limit |
| `VERIFICATION_NLI_ENDPOINT_URL` | empty | Required for `hf_endpoint`; full Inference Endpoint URL |
| `VERIFICATION_NLI_API_KEY` | empty | Required for `hf_endpoint`; secret, backend-only |
| `VERIFICATION_NLI_TIMEOUT_SECONDS` | `30` | Hosted request timeout |
| `VERIFICATION_NLI_MAX_RETRIES` | `2` | Retries for timeout, `429`, and transient `5xx` |
| `VERIFICATION_NLI_RETRY_BACKOFF_SECONDS` | `0.5` | Base exponential backoff; honours `Retry-After` |
| `VERIFICATION_NLI_ENTAILMENT_THRESHOLD` | `0.5` | Minimum entailment score for `supported` |
| `VERIFICATION_NLI_CONTRADICTION_THRESHOLD` | `0.6` | Minimum contradiction score for `contradicted` |
| `VERIFICATION_NER_ENABLED` | `true` | Extract named entities; failures are non-fatal |
| `VERIFICATION_NER_MODEL_NAME` | `Davlan/xlm-roberta-base-wikiann-ner` | Multilingual NER |
| `VERIFICATION_NER_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `VERIFICATION_NER_MAX_TOKENS` | `512` | NER chunk token limit |

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

### Verification models (Phase 4A)

- **NLI:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` classifies evidence/claim
  pairs as `entailment`, `neutral`, or `contradiction`. Actual label indices are
  read from the model config. It is slower on CPU than a compact MiniLM but
  passed the real cross-lingual and obvious-contradiction integration checks.
  The compact MiniLM option was rejected after it labelled
  "annual reviews" vs "daily reviews" as entailment.
- **NER:** `Davlan/xlm-roberta-base-wikiann-ner` extracts dates, persons,
  organizations, and locations. NER failure is non-fatal.
- **Matching:** claim/evidence candidates use the configured multilingual
  embedding model, with a lexical fallback if semantic matching fails.
- Models load lazily on the first verification request. First use may download
  weights and take substantially longer than later requests.
- **Local inference:** every checked claim is compared with every retrieved
  evidence chunk, subject to the configured budget. Identical pairs are
  evaluated once per run, recent predictions are reused through a bounded LRU
  cache, and pairs are ordered by token length within the batch/token caps. The
  default verifies up to 20 claims against up to 50 chunks each, capped at 250
  NLI pairs and 120 seconds. Pairs beyond the budget remain visible as missing
  rather than disappearing.
- **Measured result:** the current exhaustive benchmark on the 6-physical-core
  test machine used 13 claims and 6 evidence chunks (78 pairs). The batched
  configuration completed in **16.9 s cold** and **1.0 s warm**, with 13/13
  claims complete, 78/78 pairs evaluated, 92.3% accuracy on the small labelled
  fixture, and 0 false-support outcomes. The unbatched configuration took
  **21.9 s cold**. The old 8-claim/2-chunk/12-pair cap evaluated only 12/78
  pairs and left every claim incomplete, so it is retained only as a comparison
  row. These are single-machine, single-run measurements, not guarantees.
- Claims beyond the verified cap are preserved with `verification_error` and a
  `not_checked` aggregation reason rather than disappearing or being shown as
  verified. A claim is reported as having complete evidence coverage only when
  all of its expected pairs were evaluated; partial coverage uses the usable
  pairs and is reported separately.
- NLI confidence is **not** a calibrated probability that a claim is true.
  Evidence windows over the token limit raise/record warnings rather than being
  silently truncated.

#### ONNX Runtime backend

The NLI model can run through ONNX Runtime instead of PyTorch. Export it locally
once (the model file is large and is gitignored):

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\export_nli_onnx.py
```

This writes `backend/data/onnx/mdeberta-v3-base-mnli-xnli/model.onnx`, a
`model_int8.onnx` only when `--quantize-int8` is passed, and a manifest with the
model label mapping. Then configure:

```dotenv
VERIFICATION_NLI_BACKEND=onnx
VERIFICATION_NLI_ONNX_MODEL_PATH=./data/onnx/mdeberta-v3-base-mnli-xnli/model.onnx
VERIFICATION_NLI_ONNX_PROVIDER=CPUExecutionProvider
VERIFICATION_NLI_ONNX_INTRA_THREADS=0
VERIFICATION_NLI_ONNX_INTER_THREADS=1
```

The ONNX FP32 export previously matched the PyTorch labels and ran faster per
pair in the NLI-only benchmark; INT8 was faster still but fell to **72.2%**
accuracy on the labelled fixture, so it is not recommended. The latest report
contains the exhaustive engine comparison; re-run the full benchmark before
changing either backend or quantization on another machine.

#### Hosted NLI endpoint

The default verifier, tests, and benchmark use local inference only. This
optional endpoint backend is disabled unless explicitly configured.

To avoid local NLI inference, deploy `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
as a Hugging Face **Inference Endpoint** using the **Text Classification**
task, then configure:

```dotenv
VERIFICATION_NLI_BACKEND=hf_endpoint
VERIFICATION_NLI_ENDPOINT_URL=https://<endpoint-id>.<region>.<cloud>.endpoints.huggingface.cloud
VERIFICATION_NLI_API_KEY=hf_...
VERIFICATION_NLI_HOSTED_BATCH_SIZE=1
VERIFICATION_NLI_MAX_CONCURRENCY=2
VERIFICATION_NLI_TIMEOUT_SECONDS=30
VERIFICATION_NLI_MAX_RETRIES=2
```

The endpoint URL must use HTTPS and may be the endpoint's direct URL shown in
the Hugging Face console. The endpoint must accept this text-pair payload and
expose three labels (`entailment`, `neutral`, `contradiction`):

```json
{
  "inputs": {"text": "evidence passage", "text_pair": "claim"},
  "parameters": {"function_to_apply": "softmax", "top_k": 3}
}
```

The routed Hugging Face Inference Providers text-classification API is **not**
used: its documented input schema accepts one string, with no `text_pair`
field, so it cannot score evidence against a claim. The serverless zero-shot
API is also unsuitable because it treats candidate labels as hypotheses. A
dedicated Text Classification endpoint supports the actual premise/hypothesis
pair. Inference Providers remains useful for other tasks, but not this
three-way NLI step.

Operational notes:

- Hosted mode loads the NLI tokenizer locally for window sizing but never
  downloads or instantiates the local NLI model.
- Only the generated claims and selected evidence windows are sent to the
  endpoint; retrieval documents and the full answer are not. The endpoint
  provider's logging, retention, availability, rate limits, and pricing are
  governed by your Hugging Face account/plan and endpoint configuration.
- Failures are surfaced as verification errors; there is no automatic fallback
  to the local model. Set `VERIFICATION_NLI_BACKEND=local` to switch back.
- Identical evidence/claim pairs are sent once per verification request.
  Hosted requests default to one pair each; raise
  `VERIFICATION_NLI_HOSTED_BATCH_SIZE` only after confirming the endpoint
  accepts a list of text-pair objects. `VERIFICATION_NLI_MAX_CONCURRENCY`
  bounds parallel requests.
- The response includes per-stage verification timings in
  `verification.details.timings`: `claim_extraction_ms`,
  `evidence_selection_ms`, `nli_ms`, and `aggregation_ms`.

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
   - The answer and retrieved evidence appear without running NLI. Select
     **Verify Answer Claims** to check that exact stored answer. The first
     verification request downloads/loads the verification models.

### Manual test procedure (uploaded document)

1. Start the backend with a valid `LLM_API_KEY`, then the frontend.
2. Upload `backend/samples/sample_english.pdf` and wait for `completed`.
3. In **Ask a question**, type `Where is the Eiffel Tower located?` and submit.
   Ask always uses exactly five evidence chunks; there is no per-request top-K
   control. Expect: detected language `en`, an answer, and a **Sources** section
   listing the PDF chunk with its page number and retrieval score.
4. Expand/collapse **Sources** and confirm the cited chunk text actually contains
   the answer. The response starts with `verification.status="pending"`.
5. Select **Verify Answer Claims**, then expand **Claim verification** and confirm
   the claim(s), status, NLI score, and evidence metadata are shown. A model
   failure must not remove the answer.
6. Ask an unrelated question. Expect an explicit insufficient-evidence/limitation
   response, not a fabricated answer.
7. Stop the backend and ask again → expect a clear "Could not reach the backend"
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

The integration suite also covers the RAG pipeline
(`tests/integration/test_rag_real_embeddings.py`) using the real embedding model
and a deterministic **mock** generator (no API credits spent).

Real verification integration test (opt-in; downloads/loads NLI + NER):

```powershell
cd backend
$env:RUN_VERIFICATION_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_real_models.py -m integration -s
```

Hosted endpoint smoke test (opt-in, makes paid API calls):

```powershell
cd backend
$env:RUN_HOSTED_NLI_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_hosted_endpoint.py -m integration -s
```

Latest local results: **backend 271 passed, 11 skipped** (real-model and hosted-
endpoint tests are opt-in); **frontend 27 passed** with typecheck clean. Phase 4A
tests cover text offsets/Unicode tokenization, claim extraction and deduplication,
label/body list reconstruction, nested/pronoun context, NLI label
mapping/windowing, ONNX label mapping/token-type inputs, hosted
payload/retry/response validation, exhaustive 2x4 and 15x5 claim/evidence
coverage, pair-budget and deadline behavior, original retrieval-score
pass-through, ISO C2/C4/C6/C12 real-model regression, claim prioritization and
verification-error reporting, bounded request-queue behavior,
exhaustive/context/conflicting evidence, NER aggregation, engine status
mapping, RAG verification isolation, frontend request cancellation, per-pair UI
rendering, and the standalone endpoint. The real-model verification tests
remain opt-in; a live hosted endpoint is not part of the default suite.

Verification performance benchmark (loads the real local NLI model; pass
`--include-int8` only when the quantized export exists):

```powershell
cd backend
.\.venv\Scripts\python.exe -m evaluation.verification_benchmark --engine-only
```

Use `--engine-only` for the exhaustive verifier comparison. Results are written
to `backend/evaluation/results/`. Current results and their limitations are
described in [Verification models](#verification-models-phase-4a).

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
| `POST` | `/api/rag/query` | `{ question, top_k?, document_ids? }` → answer + sources; verification pending |
| `GET` | `/api/rag/results/{result_id}` | Restore a completed Ask result |
| `POST` | `/api/verification/results/{result_id}` | Verify the stored answer against its stored evidence |
| `POST` | `/api/verification/check` | `{ answer, evidence[] }` → claim-level verification without retrieval |
| `POST` | `/api/hallucination/check` | `{ result_id }` → dedicated hallucination detector for a stored result |

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
  "answer_mode": "verification_pending",
  "answer_status": "verification_pending",
  "evidence_state": "sufficient_evidence",
  "sources": [
    {
      "document_id": "…", "filename": "company_profile.pdf",
      "chunk_id": "…", "chunk_index": 7, "page_number": 2, "language": null,
      "text": "The company was founded in 2015.",
      "retrieval_score": 0.21, "similarity": 0.79,
      "passed_relevance_filter": true
    }
  ],
  "retrieval": {
    "top_k": 5, "candidate_chunks": 1, "filtered_chunks": 1,
    "returned_chunks": 1, "sufficient_evidence": true,
    "metric": "cosine_distance",
    "score_interpretation": "lower distance = more similar; not a probability",
    "min_evidence_chunks": 1, "max_distance": 0.75,
    "distance_filter_applied": true
  },
  "assessment": {
    "state": "sufficient_evidence", "method": "llm", "used_llm": true
  },
  "verification": {
    "claim_level_verification": false,
    "status": "pending",
    "method": "none",
    "note": "Claim-level verification has not been run.",
    "details": null
  },
  "generation": {
    "used_llm": true, "provider": "groq", "model": "openai/gpt-oss-120b",
    "prompt_version": "rag-grounded-answer-1.0.0",
    "finish_reason": "stop", "prompt_tokens": 512,
    "completion_tokens": 40, "total_tokens": 552
  },
  "diagnostics": {
    "candidates_retrieved": 1, "candidates_after_filter": 1,
    "assessment_method": "llm", "verification_error": null
  },
  "timings": {
    "retrieval_ms": 235.7, "assessment_ms": 640.2,
    "generation_ms": 812.4, "verification_ms": 0.0, "total_ms": 1688.3
  }
}
```

Status codes: `400` invalid/overlong question, `422` schema validation error,
`503` LLM not configured / vector store unavailable, `502` LLM provider error.
`answer_mode` is kept for Phase 3 client compatibility and carries the same value
as `answer_status`: `grounded`, `partially_grounded`, `contradicted`,
`insufficient_evidence`, `irrelevant_retrieval`, or `verification_pending`.
When evidence is insufficient, the LLM is **not** called by default.
`retrieval_score` is a cosine **distance** (lower = more similar); it is **not** a
probability, confidence, or factual-verification result.

Ask responses are stored under `result_id` and start unverified. Run NLI on that
exact answer without rerunning retrieval or generation:

```http
POST /api/verification/results/{result_id}
```

The response is the updated `RagQueryResponse` with `verification.status` set to
`verified`, `partial`, `error`, or `not_performed`. Already-verified results are
returned unchanged, and late/duplicate verification requests cannot alter a newer
Ask result in the UI.

### `POST /api/verification/check`

Runs verification without retrieval or generation:

```json
{
  "answer": "ISO published the standard in 2024.",
  "evidence": [
    {
      "text": "ISO published the standard in 2024.",
      "filename": "report.pdf",
      "page_number": 3
    }
  ]
}
```

The response is a `VerificationResult` with `status` (`completed`/`error`),
answer language, extraction method, per-claim NLI label/confidence/windows,
matched evidence references, summary counts, warnings, and timing. Set
`VERIFICATION_ENABLED=false` to return a disabled engine result. First use loads
the NLI/NER models lazily.

---

## 10. Future phases

Phase 4A delivers **pretrained** claim-level verification. The following are
intentionally out of scope and planned for later phases:

- **Phase 4B/training:** TF-IDF + Logistic Regression baseline, fine-tuning,
  comparison against the pretrained NLI model, and dataset/label preparation.
- **Phase 5:** hallucination classification and word/span localization.
- **Phase 6:** targeted LLM repair and re-verification.
- **Phase 7:** richer chat/span-highlighting UI integration.
- **Phase 8:** reproducible training/evaluation across RAGTruth, PsiloQA,
  Mu-SHROOM, BHRAM-IL, with per-language metrics and calibration.

New inference stays isolated from route handlers and thresholds remain
configurable.

---

## 11. Troubleshooting

- **`Activate.ps1` cannot be loaded** → `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- **`pip`/file-lock errors on OneDrive folders** → close any running Python
  process and re-run `pip install`; avoid syncing `.venv` while installing.
- **First upload seems to hang** → it is downloading the model. Run
  `python scripts\download_embedding_model.py` in advance. BGE-M3 is ~2.2 GB.
- **First RAG answer is much slower** → verification is downloading/loading the
  NLI and NER models. Later requests reuse the cached models. Set
  `VERIFICATION_ENABLED=false` for Phase 3-only latency.
- **Browser says the request timed out while backend logs continue** → the
  browser stopped waiting, but FastAPI may still finish the current step. Wait
  for the final `RAG decision` log and do not resubmit immediately. The UI waits
  up to 10 minutes for RAG, but verification stops starting new NLI work after
  2 minutes and leaves unstarted claims in `verification_error` with the deadline
  reason. Use **Cancel** to stop waiting without leaving the page stuck.
- **Hosted NLI returns HTTP 401/403** → check `VERIFICATION_NLI_API_KEY` and
  endpoint permissions. Keys stay in `backend/.env`; restart after changing them.
- **Hosted NLI returns HTTP 400 or malformed labels** → the endpoint is not
  serving a text-classification model with exactly entailment, neutral, and
  contradiction labels. Deploy the model with the Text Classification task.
- **Hosted NLI times out or returns 429** → increase
  `VERIFICATION_NLI_TIMEOUT_SECONDS`, keep the bounded retry settings, or reduce
  `VERIFICATION_NLI_BATCH_SIZE`. The app does not silently switch to local NLI.
- **Out of memory** → lower `EMBEDDING_BATCH_SIZE`,
  `VERIFICATION_NLI_BATCH_SIZE`, and/or the sequence limits; set the embedding,
  NLI, and NER devices to `cpu`.
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
- Verification is pretrained inference only. It does not include fine-tuning, a
  TF-IDF/Logistic Regression baseline, span localization, or repair.
- NLI confidence is a model score, not a calibrated probability that a claim is
  true. `supported` means the evidence passed the configured entailment
  threshold, not that the claim is objectively correct.
- Claim-level outcomes are exactly `supported`, `contradicted`, `neutral`,
  `unevaluable`, or `verification_error`. `neutral` means evaluated evidence did
  not meet either threshold; it is not a factual refutation. If evaluated pairs
  contain both thresholded support and contradiction, contradiction takes
  precedence for safety, `conflict=true` is retained, and the answer-level status
  is `contradicted`.
- Claim extraction is local and deterministic. It keeps headings as context while
  excluding them from claim text, extracts markdown/numbered/bullet/nested list
  items, separates short list labels from complete proposition bodies (for
  example `Risk treatment – The organization...`), reconstructs imperative and
  noun-led items from a lead-in, heading, or parent item, and resolves immediate
  pronoun references when the referent is unambiguous. Original claim text,
  reconstructed NLI hypothesis, and reconstruction context are all returned.
  Metadata, questions, and introductory text are excluded. It can still miss or
  merge claims in long or ambiguous sentences.
- At most `VERIFICATION_MAX_CLAIMS` claims are verified. Additional extracted
  claims retain `verification_error` with a `not_checked` reason, not verified
  status. Every checked claim is compared with up to
  `VERIFICATION_MAX_EVIDENCE_PER_CLAIM` retrieved chunks, subject to
  `VERIFICATION_MAX_NLI_PAIRS`; pairs beyond the budget are reported missing.
  The focused window adds neighbouring sentences when they fit the model limit,
  and the selected passages are re-ranked around the claim sentence. Warnings are
  returned when limits apply. Queue limits can return HTTP 503 rather than
  allowing an unbounded number of waiting requests.
- NER failure degrades entity extraction without failing verification. NLI or
  other verification failures preserve the generated answer and return
  `verification.status="error"`.
- Verification quality is model- and language-dependent. English/Hindi have been
  exercised; broader language claims require evaluation rather than assumption.
- The LLM answer is still grounded by prompt design as well as verification.
  Verification reduces but cannot eliminate unsupported answers; localization and
  repair are later phases.

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

Phase 4A: all pass — answer language and sentence/claim offsets are preserved;
factual claims are extracted separately from tokenization; NER and
claim/evidence matching run behind replaceable interfaces; NLI uses actual model
labels and evidence-as-premise; per-claim statuses, confidence, evidence, warnings,
and summary are returned; Ask returns immediately with verification pending;
`POST /api/verification/results/{result_id}` verifies the exact stored answer and
evidence; RAG status then integrates `grounded` / `partially_grounded` /
`contradicted` / `insufficient_evidence`; verification errors preserve the
answer; standalone `/api/verification/check` works; real NLI, NER, and
cross-lingual smoke tests pass; deterministic backend, frontend, and typecheck
suites pass; bounded queue/deadline behavior is covered; the frontend can cancel
an in-flight verification; and the measured benchmark report is checked in.
Fine-tuning/baseline comparison are explicitly deferred.
