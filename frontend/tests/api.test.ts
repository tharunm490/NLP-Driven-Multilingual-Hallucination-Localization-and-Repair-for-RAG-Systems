import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  RequestCancelledError,
  askQuestion,
  checkHallucinations,
  checkHealth,
  getApiBaseUrl,
  getRagResult,
  listDocuments,
  verifyRagResult,
} from "../lib/api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.NEXT_PUBLIC_API_BASE_URL;
});

describe("getApiBaseUrl", () => {
  it("falls back to localhost when unset", () => {
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
    expect(getApiBaseUrl()).toBe("http://localhost:8000");
  });

  it("strips trailing slashes from the configured URL", () => {
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:9000/";
    expect(getApiBaseUrl()).toBe("http://127.0.0.1:9000");
  });
});

describe("api error handling", () => {
  it("reports network failures clearly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await expect(checkHealth()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
    });
    await expect(checkHealth()).rejects.toThrow(/Could not reach the backend/);
  });

  it("reports timeouts", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new DOMException("aborted", "AbortError")),
    );
    await expect(checkHealth()).rejects.toThrow(/timed out/);
  });

  it("extracts a string detail from backend errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Unsupported file type" }, 400)),
    );
    await expect(listDocuments()).rejects.toMatchObject({
      message: "Unsupported file type",
      status: 400,
    });
  });

  it("joins validation detail arrays", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: [{ msg: "field required" }, { msg: "invalid value" }] }, 422),
      ),
    );
    await expect(listDocuments()).rejects.toThrow("field required; invalid value");
  });

  it("returns parsed data on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "ok",
          version: "0.1.0",
          environment: "test",
          embedding_model: "BAAI/bge-m3",
          components: {},
        }),
      ),
    );
    const health = await checkHealth();
    expect(health.status).toBe("ok");
    expect(health.embedding_model).toBe("BAAI/bge-m3");
  });

  it("wraps ApiError instances with the status code", () => {
    const error = new ApiError("boom", 503, { detail: "boom" });
    expect(error.status).toBe(503);
    expect(error).toBeInstanceOf(Error);
  });
});

describe("askQuestion", () => {
  const answer = {
    question: "When was the company founded?",
    normalized_question: "When was the company founded?",
    detected_language: "en",
    language_confidence: 0.97,
    language_reliable: true,
    answer: "The company was founded in 2015 [Source 1].",
    answer_mode: "grounded",
    sources: [
      {
        document_id: "doc_1",
        filename: "acme.txt",
        chunk_id: "chunk_0",
        chunk_index: 0,
        page_number: null,
        language: "en",
        text: "Acme was founded in 2015.",
        retrieval_score: 0.21,
        similarity: 0.79,
      },
    ],
    retrieval: {
      top_k: 5,
      returned_chunks: 1,
      sufficient_evidence: true,
      metric: "cosine_distance",
      score_interpretation: "lower distance = more similar; not a probability",
      min_evidence_chunks: 1,
      max_distance: 0.75,
      distance_filter_applied: true,
    },
    generation: {
      used_llm: true,
      provider: "mock",
      model: "mock-model",
      prompt_version: "rag-grounded-answer-1.0.0",
      finish_reason: null,
      prompt_tokens: null,
      completion_tokens: null,
      total_tokens: null,
    },
    timings: { retrieval_ms: 12.5, generation_ms: 0, total_ms: 12.5 },
  };

  it("posts to /api/rag/query and parses the response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(answer));
    vi.stubGlobal("fetch", fetchMock);

    const result = await askQuestion({ question: "When was the company founded?" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/rag/query");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      question: "When was the company founded?",
    });
    expect(result.answer_mode).toBe("grounded");
    expect(result.sources[0].filename).toBe("acme.txt");
  });

  it("distinguishes caller cancellation from a timeout", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((_url: string, init: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }),
    );

    const pending = askQuestion({ question: "long request" }, {
      signal: controller.signal,
    });
    controller.abort();

    await expect(pending).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(pending).rejects.toThrow(/cancelled/);
  });

  it("surfaces a 503 configuration error from the backend", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "No LLM provider is configured." }, 503),
      ),
    );
    await expect(askQuestion({ question: "hi" })).rejects.toMatchObject({
      status: 503,
      message: "No LLM provider is configured.",
    });
  });

  it("handles an insufficient-evidence response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          ...answer,
          answer_mode: "insufficient_evidence",
          sources: [],
          retrieval: { ...answer.retrieval, returned_chunks: 0, sufficient_evidence: false },
          generation: { ...answer.generation, used_llm: false, provider: null, model: null },
        }),
      ),
    );
    const result = await askQuestion({ question: "unrelated" });
    expect(result.answer_mode).toBe("insufficient_evidence");
    expect(result.sources).toEqual([]);
    expect(result.retrieval.sufficient_evidence).toBe(false);
  });
});


describe("shared result and hallucination endpoints", () => {
  it("retrieves a shared Ask result by identifier", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ result_id: "abc", answer: "shared answer" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await getRagResult("abc");

    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "/api/rag/results/abc",
    );
    expect(result.answer).toBe("shared answer");
  });

  it("verifies a stored Ask result by identifier", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ result_id: "abc", verification: { status: "verified" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await verifyRagResult("abc");

    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "/api/verification/results/abc",
    );
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(result.verification?.status).toBe("verified");
  });

  it("posts the selected result id to the hallucination endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ check_id: "check-1", result_id: "abc" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await checkHallucinations({ result_id: "abc" });

    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "/api/hallucination/check",
    );
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      result_id: "abc",
    });
    expect(result.check_id).toBe("check-1");
  });
});
