"""End-to-end Phase 2/3 smoke test against a running backend.

Uploads the sample TXT and PDF, lists documents, runs multilingual searches,
asks a RAG question, and deletes one document. Start the backend first, then run:

    python scripts/smoke_test_api.py

Set SMOKE_REQUIRE_LLM=1 to fail when the RAG endpoint reports an unconfigured
LLM (for example when the backend was started with LLM_PROVIDER=mock).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

# Windows consoles default to cp1252 and cannot print Hindi/Tamil text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
UPLOAD_TIMEOUT = 900.0
REQUIRE_LLM = os.environ.get("SMOKE_REQUIRE_LLM") == "1"

QUERIES = [
    "Where is the Eiffel Tower?",
    "भारत की राजधानी कौन सी है?",
    "தமிழ்நாட்டின் தலைநகரம் என்ன?",
]

RAG_QUESTION = "Where is the Eiffel Tower located?"
RAG_UNRELATED_QUESTION = "What is the recipe for chocolate cake?"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def upload(client: httpx.Client, path: Path, content_type: str) -> dict:
    print(f"\nUploading {path.name} ...")
    with path.open("rb") as handle:
        response = client.post(
            "/api/documents/upload",
            files={"files": (path.name, handle, content_type)},
        )
    response.raise_for_status()
    result = response.json()["results"][0]
    print(
        f"  status={result['status']} document_id={result['document_id']} "
        f"chunks={result['chunks_stored']} chars={result['character_count']} "
        f"language={result['language']} error={result['error']}"
    )
    if result["status"] != "completed":
        fail(f"upload of {path.name} did not complete")
    return result


def ask(client: httpx.Client, question: str) -> dict | None:
    print(f"\nRAG question: {question}")
    response = client.post("/api/rag/query", json={"question": question, "top_k": 3})
    if response.status_code == 503:
        detail = response.json().get("detail", "")
        if REQUIRE_LLM:
            fail(f"RAG generation is not configured: {detail}")
        print(f"  RAG skipped (LLM not configured): {detail}")
        return None
    if response.status_code != 200:
        response.raise_for_status()
    payload = response.json()
    print(
        f"  lang={payload['detected_language']} reliable={payload['language_reliable']} "
        f"mode={payload['answer_mode']} chunks={payload['retrieval']['returned_chunks']} "
        f"sufficient={payload['retrieval']['sufficient_evidence']} "
        f"llm={payload['generation']['used_llm']} total_ms={payload['timings']['total_ms']}"
    )
    print(f"  answer: {payload['answer'][:200]}")
    for source in payload["sources"][:2]:
        page = source["page_number"] if source["page_number"] is not None else "-"
        print(
            f"    score={source['retrieval_score']:.4f} file={source['filename']} "
            f"page={page} chunk={source['chunk_index']}"
        )
    if payload["answer_mode"] == "grounded" and not payload["sources"]:
        fail("a grounded answer was returned without sources")
    return payload


def main() -> int:
    txt_path = SAMPLES_DIR / "sample_multilingual.txt"
    pdf_path = SAMPLES_DIR / "sample_english.pdf"
    for path in (txt_path, pdf_path):
        if not path.exists():
            fail(
                f"missing sample {path}. Run scripts/generate_sample_pdf.py for the PDF."
            )

    with httpx.Client(base_url=BASE_URL, timeout=UPLOAD_TIMEOUT) as client:
        health = client.get("/api/health").json()
        print(f"Health: status={health['status']} version={health['version']}")
        if health["status"] != "ok":
            fail("backend health is not ok")

        txt_result = upload(client, txt_path, "text/plain")
        pdf_result = upload(client, pdf_path, "application/pdf")

        listing = client.get("/api/documents").json()
        print(f"\nDocuments stored: {listing['total']}")
        for document in listing["documents"]:
            print(
                f"  {document['filename']}: status={document['status']} "
                f"chunks={document['chunks_stored']} pages={document['page_count']}"
            )
        if listing["total"] < 2:
            fail("expected at least two documents")

        for query in QUERIES:
            response = client.post(
                "/api/retrieval/search", json={"query": query, "top_k": 3}
            )
            response.raise_for_status()
            payload = response.json()
            print(f"\nQuery: {query}")
            if not payload["results"]:
                fail(f"no results for query: {query}")
            for item in payload["results"]:
                page = item["page_number"] if item["page_number"] is not None else "-"
                print(
                    f"  distance={item['distance']:.4f} similarity={item['similarity']:.4f} "
                    f"file={item['filename']} page={page} chunk={item['chunk_index']}"
                )
                print(f"    {item['text'][:120]}...")

        print(f"\nDeleting PDF document {pdf_result['document_id']} ...")
        deleted = client.delete(f"/api/documents/{pdf_result['document_id']}")
        deleted.raise_for_status()
        print(f"  chunks_deleted={deleted.json()['chunks_deleted']}")
        remaining = client.get("/api/documents").json()["total"]
        print(f"  documents remaining={remaining}")

        grounded = ask(client, RAG_QUESTION)
        if grounded is not None and grounded["answer_mode"] == "grounded":
            if not grounded["sources"]:
                fail("expected retrieved sources for the grounded question")
        ask(client, RAG_UNRELATED_QUESTION)

    print("\nSMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
