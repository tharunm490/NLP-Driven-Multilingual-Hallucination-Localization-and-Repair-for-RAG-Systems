import type {
  DeleteDocumentResponse,
  DocumentInfo,
  DocumentListResponse,
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
const RAG_TIMEOUT_MS = 120_000;

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
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError("The request timed out. Is the backend still processing?", 0);
    }
    throw new ApiError(
      `Could not reach the backend at ${getApiBaseUrl()}. Is the FastAPI server running?`,
      0,
    );
  } finally {
    clearTimeout(timer);
  }

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : await response.text().catch(() => "");

  if (!response.ok) {
    throw new ApiError(extractErrorMessage(payload), response.status, payload);
  }
  return payload as T;
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

export function askQuestion(payload: RagQueryRequest): Promise<RagQueryResponse> {
  return request<RagQueryResponse>(
    "/api/rag/query",
    jsonInit("POST", payload),
    RAG_TIMEOUT_MS,
  );
}
