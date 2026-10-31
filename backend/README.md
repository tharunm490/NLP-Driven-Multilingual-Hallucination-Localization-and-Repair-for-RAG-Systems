# Backend — Multilingual RAG API

FastAPI service for Phases 1–4A: document upload, extraction, cleaning, chunking,
multilingual embeddings, ChromaDB storage, retrieval, **retrieval-augmented
question answering**, and **claim-level verification** against retrieved evidence.

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

Unit/API tests inject deterministic fake models, so they do not download large
weights. Latest local result: **329 passed, 12 skipped**. Real-model tests are
opt-in:

```powershell
$env:RUN_EMBEDDING_INTEGRATION="1"
$env:EMBEDDING_MODEL_NAME="BAAI/bge-m3"    # or intfloat/multilingual-e5-base
.\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s
```

```powershell
$env:RUN_VERIFICATION_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_real_models.py -m integration -s
```

```powershell
$env:RUN_HOSTED_NLI_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_verification_hosted_endpoint.py -m integration -s
```

The real-model verification test downloads/loads the NLI and NER models on
first use. Set `VERIFICATION_NLI_DEVICE` / `VERIFICATION_NER_DEVICE` to choose a
device. The hosted smoke test requires a configured `hf_endpoint` and makes
paid API calls.

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
- Ask generation never runs NLI. The response is stored under `result_id` with
  `verification.status="pending"` (or `not_performed` for abstentions).
- `POST /api/verification/results/{result_id}` runs verification on demand over
  the exact stored answer/evidence; `rag_verification_service.py` owns this flow.
  Duplicate calls for `verified`/`partial` results are idempotent.

## Claim-level verification (Phase 4A)

`app/services/verification/` implements a pretrained-only verification pipeline:

```
answer language detection → normalization/sentence segmentation
   → factual claim extraction → NER → claim/evidence matching
      → multilingual NLI (evidence as premise, claim as hypothesis)
         → per-claim supported/contradicted/conflicting/insufficient + summary
```

- NLI: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` by default. It is slower on
  CPU, but passed the real cross-lingual and obvious-contradiction integration
  checks. A compact MiniLM option was rejected because it misclassified an
  obvious contradiction as entailment. Keep the default unless you re-evaluate
  a replacement on your own data. The model can also run on a dedicated Hugging
  Face Inference Endpoint; see [Hosted NLI](#hosted-nli).
- NER: `Davlan/xlm-roberta-base-wikiann-ner`.
- Claim extraction is local, deterministic, and language-aware; it never calls
  an LLM or downloads a claim-extraction model. English text uses sentence,
  heading, list, and clause rules. Predominantly Devanagari text uses
  conservative Hindi predicate rules and accepts a claim only when the original
  text already contains a complete predicate.
- Headings, metadata, questions, introductory lead-ins, and meta-commentary
  about missing evidence are excluded. Dependent list items under an
  introductory lead-in are grouped into one claim that preserves the original
  source span; complete bullets remain independent. Coordinated Hindi
  `अथवा/तथा/और/;` clauses are split only when both sides are independently
  complete, and Devanagari tokens use the shared `word_spans` tokenizer so
  combining marks are preserved.
- Every accepted claim has exact original-answer offsets and
  `location_status="verified"`. Extraction is hard-capped at 30 claims,
  deduplicated by normalized containment (preferring the more complete claim),
  and ordered by source position. `VERIFICATION_MAX_EXTRACTED_CLAIMS` may lower
  the cap but cannot raise it above 30.
- Claim matching reuses the configured multilingual embedding model.
- Models are lazy-loaded; label maps are read from model config, not hardcoded.
- Local inference deduplicates identical claim/evidence pairs, reuses recent
  predictions through a bounded LRU cache, orders pairs by token length, and
  caps padded tokens per batch. Progress and cache/batch metrics are logged.
- Defaults verify up to 20 claims against up to 50 retrieved chunks each, capped
  at 250 NLI pairs and 120 seconds. Every checked claim is compared with every
  retrieved chunk within that budget. Extracted claims beyond the cap retain
  `verification_error` with a `not_checked` reason, and pairs beyond the budget
  are reported missing. On
  the 6-physical-core benchmark machine the exhaustive long request (13 claims,
  6 chunks, 78 pairs) took 16.9 s cold / 1.0 s warm with complete claim and pair
  coverage; the unbatched variant took 21.9 s. Accuracy on the small labelled
  fixture was 92.3% with 0 false-support outcomes. The old 8-claim/12-pair cap
  evaluated only 12/78 pairs and left every claim incomplete. These are
  single-run measurements.
- NLI confidence is a model score, not calibrated factual truth.
- Claim-level outcomes are exactly `supported`, `contradicted`, or `neutral`.
  Operational failures are represented by `diagnostic_status="verification_error"`
  while the displayed status remains neutral; there is no separate
  “not evaluated” verdict. If a claim has both thresholded support and
  contradiction, contradiction takes precedence for safety, `conflict=true`
  is retained, and answer-level status remains `contradicted`.
- Evidence windows that exceed the model limit are surfaced with warnings rather
  than silently truncated.
- Verification failures do not discard the generated answer; the response keeps
  the answer and reports `verification.status="error"`.

Stored `POST /api/verification/results/{result_id}` responses include
`verification` counts/details and `timings.verification_ms`;
`verification.details.timings` breaks verification down into claim extraction,
evidence selection, NLI, and aggregation milliseconds on successful runs. After
verification, answer status can become `grounded`, `partially_grounded`,
`contradicted`, or `insufficient_evidence`; before it runs, Ask responses use
`verification_pending`. Local verification is serialized by a bounded request
limiter (one active job, 5 s wait, at most two waiters). The UI can cancel an
in-flight verification; the generated answer is never replaced by a partial
result.

Verify a stored Ask result on demand:

```http
POST /api/verification/results/{result_id}
```

This reruns neither retrieval nor generation and returns the updated
`RagQueryResponse`. Standalone endpoint:

```http
POST /api/verification/check
Content-Type: application/json

{
  "answer": "The standard was published in 2024.",
  "evidence": [
    {
      "text": "ISO published the standard in 2024.",
      "filename": "report.pdf",
      "page_number": 3
    }
  ]
}
```

Configure verification in `backend/.env` with the `VERIFICATION_*` variables in
`.env.example`. Set `VERIFICATION_ENABLED=false` to keep Phase 3 behavior.

## Hallucination Checks

The Hallucination Checks page reuses the latest completed Ask result instead of
rerunning retrieval/generation:

1. `POST /api/rag/query` stores the completed response in bounded process
   memory and returns an opaque `result_id` (`RAG_RESULT_CACHE_SIZE`,
   `RAG_RESULT_TTL_SECONDS`).
2. `GET /api/rag/results/{result_id}` restores the exact question, answer,
   evidence, and NLI result after navigation/refresh while the backend lives.
3. `POST /api/hallucination/check` accepts `{"result_id": "..."}` and runs the
   dedicated local model over every extracted claim and every supplied evidence
   chunk. The browser keeps only the result ID in session storage.

The detector defaults to `Horizon-Labs/hallucination-guard-base`, Apache-2.0,
308M parameters, 8k context (trained at 2k). Its documented input order is
**source first, claim second**, with labels `UNSUPPORTED` (0) and `SUPPORTED`
(1). The app uses the `SUPPORTED` probability, never a hosted inference API.
Verify a local download with:

```powershell
$env:RUN_HALLUCINATION_INTEGRATION="1"
$env:HALLUCINATION_REAL_MODEL="Horizon-Labs/hallucination-guard-base"
$env:HALLUCINATION_REAL_LOCAL_ONLY="true"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_hallucination_real_model.py -m integration -s
```

Set `HALLUCINATION_MODEL_LOCAL_FILES_ONLY=true` for strictly offline inference
after pre-downloading the checkpoint. Configure the model, thresholds, claim
limits, review band, timeout, and memory cache with the `HALLUCINATION_*`
variables in `.env.example`. Set `HALLUCINATION_ENABLED=false` to disable the
endpoint without affecting Ask Documents. NLI results are displayed separately;
the two signals are never fused.

### ONNX Runtime NLI

Export the local model once; the output is gitignored and large:

```powershell
.\.venv\Scripts\python.exe scripts\export_nli_onnx.py
```

Then set:

```dotenv
VERIFICATION_NLI_BACKEND=onnx
VERIFICATION_NLI_ONNX_MODEL_PATH=./data/onnx/mdeberta-v3-base-mnli-xnli/model.onnx
VERIFICATION_NLI_ONNX_PROVIDER=CPUExecutionProvider
```

The export writes `manifest.json` with the model label mapping. FP32 ONNX
previously matched PyTorch labels and ran faster per pair in the NLI-only
benchmark. `--quantize-int8` also produces `model_int8.onnx`, but INT8 scored
only 72.2% on the labelled fixture and is not recommended. The latest report
contains the exhaustive engine comparison; run the full benchmark before
changing backends on other hardware:

```powershell
.\.venv\Scripts\python.exe -m evaluation.verification_benchmark --engine-only
```

Reports are written to `evaluation/results/`.

### Hosted NLI

The default verifier, tests, and benchmark use local inference only. This
optional backend is disabled unless explicitly configured.

Deploy `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` as a Hugging Face Inference
Endpoint with the **Text Classification** task, then set:

```dotenv
VERIFICATION_NLI_BACKEND=hf_endpoint
VERIFICATION_NLI_ENDPOINT_URL=https://<endpoint-id>.<region>.<cloud>.endpoints.huggingface.cloud
VERIFICATION_NLI_API_KEY=hf_...
VERIFICATION_NLI_HOSTED_BATCH_SIZE=1
VERIFICATION_NLI_MAX_CONCURRENCY=2
VERIFICATION_NLI_TIMEOUT_SECONDS=30
VERIFICATION_NLI_MAX_RETRIES=2
```

The endpoint URL must use HTTPS. It receives
`{"inputs": {"text": <evidence>, "text_pair": <claim>}}` with
`function_to_apply=softmax` and `top_k=3`, and must return exactly
`entailment`, `neutral`, and `contradiction`. Labels are mapped from the
response, not assumed to be ordered. The routed Inference Providers
text-classification API is not supported because its official input schema is a
single string with no `text_pair`, and serverless zero-shot classification
cannot preserve the premise/hypothesis direction or return all three NLI
scores.

Hosted mode runs the NLI tokenizer locally for window sizing but never loads
the NLI model. Only claims and selected evidence windows are sent; endpoint
provider logging, retention, pricing, quotas, and rate limits are controlled by
your Hugging Face account. Failures return verification errors and never
auto-fallback to local inference. Set `VERIFICATION_NLI_BACKEND=local` to revert.
Identical pairs are deduplicated. Hosted requests default to one pair each and
use at most `VERIFICATION_NLI_MAX_CONCURRENCY` parallel requests. Raise
`VERIFICATION_NLI_HOSTED_BATCH_SIZE` only after confirming the endpoint accepts
batched text-pair objects.

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
  retrieval, registry, language detection, token counting, query language
  detection, query preprocessing, prompt builder, generator, rag service, and
  **claim-level verification**.
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
- Verification is pretrained inference only. Fine-tuning, TF-IDF baseline
  comparison, and span localization belong to later phases. Hallucination
  Repair uses a fixed three-iteration claim queue and re-verifies every
  correction with both existing verification systems.
