import type {
  DeleteDocumentResponse,
  DocumentInfo,
  DocumentListResponse,
  HallucinationCheckRequest,
  HallucinationCheckResponse,
  HealthResponse,
  RagQueryRequest,
  RagQueryResponse,
  SearchRequest,
  SearchResponse,
  UploadResponse,
} from "./types";

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 10 * 60_000;
const RAG_TIMEOUT_MS = 10 * 60_000;

export class ApiError extends Error {
  readonly status: number;
  readonly details: unknown;

  constructor(message: string, status = 0, details: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export class RequestCancelledError extends ApiError {
  constructor(message = "The request was cancelled.") {
    super(message, 0);
    this.name = "RequestCancelledError";
  }
}

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const base = configured && configured.length > 0 ? configured : DEFAULT_BASE_URL;
  return base.replace(/\/+$/, "");
}

function extractErrorMessage(payload: unknown): string {
  if (!payload) {
    return "The backend returned an empty error response.";
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          typeof item === "object" && item !== null && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : String(item),
        )
        .join("; ");
    }
  }
  if (typeof payload === "object" && payload !== null && "message" in payload) {
    return String((payload as { message: unknown }).message);
  }
  return "The request failed.";
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", abortFromCaller, { once: true });
    }
  }
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let response: Response;
    try {
      response = await fetch(`${getApiBaseUrl()}${path}`, {
        ...init,
        signal: controller.signal,
        cache: "no-store",
      });
    } catch (error) {
      if (signal?.aborted) {
        throw new RequestCancelledError();
      }
      if (error instanceof Error && error.name === "AbortError") {
        throw new ApiError("The request timed out. Is the backend still processing?", 0);
      }
      throw new ApiError(
        `Could not reach the backend at ${getApiBaseUrl()}. Is the FastAPI server running?`,
        0,
      );
    }

    const contentType = response.headers.get("content-type") ?? "";
    let payload: unknown;
    try {
      payload = contentType.includes("application/json")
        ? await response.json().catch(() => null)
        : await response.text().catch(() => "");
    } catch (error) {
      if (signal?.aborted) {
        throw new RequestCancelledError();
      }
      throw new ApiError("The backend response could not be read.", response.status);
    }

    if (signal?.aborted) {
      throw new RequestCancelledError();
    }
    if (!response.ok) {
      throw new ApiError(extractErrorMessage(payload), response.status, payload);
    }
    return payload as T;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", {}, 8_000);
}

export function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/api/documents");
}

export function getDocument(documentId: string): Promise<DocumentInfo> {
  return request<DocumentInfo>(`/api/documents/${encodeURIComponent(documentId)}`);
}

export function deleteDocument(documentId: string): Promise<DeleteDocumentResponse> {
  return request<DeleteDocumentResponse>(
    `/api/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}

export function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  return request<UploadResponse>(
    "/api/documents/upload",
    { method: "POST", body: formData },
    UPLOAD_TIMEOUT_MS,
  );
}

export function search(payload: SearchRequest): Promise<SearchResponse> {
  return request<SearchResponse>("/api/retrieval/search", jsonInit("POST", payload));
}

export function askQuestion(
  payload: RagQueryRequest,
  options: { signal?: AbortSignal } = {},
): Promise<RagQueryResponse> {
  return request<RagQueryResponse>(
    "/api/rag/query",
    jsonInit("POST", payload),
    RAG_TIMEOUT_MS,
    options.signal,
  );
}

export function getRagResult(
  resultId: string,
  options: { signal?: AbortSignal } = {},
): Promise<RagQueryResponse> {
  return request<RagQueryResponse>(
    `/api/rag/results/${encodeURIComponent(resultId)}`,
    {},
    DEFAULT_TIMEOUT_MS,
    options.signal,
  );
}

export function verifyRagResult(
  resultId: string,
  options: { signal?: AbortSignal } = {},
): Promise<RagQueryResponse> {
  return request<RagQueryResponse>(
    `/api/verification/results/${encodeURIComponent(resultId)}`,
    { method: "POST" },
    RAG_TIMEOUT_MS,
    options.signal,
  );
}

export function checkHallucinations(
  payload: HallucinationCheckRequest,
  options: { signal?: AbortSignal } = {},
): Promise<HallucinationCheckResponse> {
  return request<HallucinationCheckResponse>(
    "/api/hallucination/check",
    jsonInit("POST", payload),
    RAG_TIMEOUT_MS,
    options.signal,
  );
}
