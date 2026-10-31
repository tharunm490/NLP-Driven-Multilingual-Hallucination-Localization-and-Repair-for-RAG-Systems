import type {
  AnswerStatus,
  ClaimVerificationStatus,
  DocumentStatus,
  HallucinationVerdict,
  UploadItemStatus,
} from "./types";

export type Tone = "success" | "warning" | "error" | "neutral" | "info";

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes)) {
    return "—";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let size = bytes / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function documentStatusLabel(status: DocumentStatus): string {
  switch (status) {
    case "completed":
      return "Completed";
    case "processing":
      return "Processing";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

export function uploadStatusLabel(status: UploadItemStatus): string {
  switch (status) {
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "rejected":
      return "Rejected";
    default:
      return status;
  }
}

export function documentStatusTone(status: DocumentStatus): Tone {
  switch (status) {
    case "completed":
      return "success";
    case "processing":
      return "warning";
    case "failed":
      return "error";
    default:
      return "neutral";
  }
}

export function uploadStatusTone(status: UploadItemStatus): Tone {
  switch (status) {
    case "completed":
      return "success";
    case "failed":
      return "error";
    case "rejected":
      return "warning";
    default:
      return "neutral";
  }
}

export function answerStatusLabel(status: AnswerStatus | null | undefined): string {
  switch (status) {
    case "grounded":
      return "Grounded answer";
    case "contradicted":
      return "Contradicted by evidence";
    case "verification_pending":
      return "Verification pending";
    case "partially_grounded":
      return "Partially supported";
    case "insufficient_evidence":
      return "Insufficient evidence";
    case "irrelevant_retrieval":
      return "No relevant evidence found";
    default:
      return "Status unavailable";
  }
}

export function answerStatusTone(status: AnswerStatus | null | undefined): Tone {
  switch (status) {
    case "grounded":
      return "success";
    case "contradicted":
      return "error";
    case "verification_pending":
      return "info";
    case "partially_grounded":
      return "warning";
    case "insufficient_evidence":
    case "irrelevant_retrieval":
      return "warning";
    default:
      return "neutral";
  }
}

export function answerStatusShowsAbstention(status: AnswerStatus | null | undefined): boolean {
  return status === "insufficient_evidence" || status === "irrelevant_retrieval";
}

export function claimVerificationStatusLabel(
  status: ClaimVerificationStatus | null | undefined,
): string {
  switch (status) {
    case "supported":
      return "Supported";
    case "contradicted":
      return "Contradicted";
    case "neutral":
    default:
      return "Neutral";
  }
}

export function claimVerificationStatusTone(
  status: ClaimVerificationStatus | null | undefined,
): Tone {
  switch (status) {
    case "supported":
      return "success";
    case "contradicted":
      return "error";
    case "neutral":
    default:
      return "neutral";
  }
}

export function hallucinationVerdictLabel(
  verdict: HallucinationVerdict | null | undefined,
): string {
  switch (verdict) {
    case "supported":
      return "Supported";
    case "potentially_hallucinated":
      return "Potentially hallucinated";
    default:
      return "Not checked";
  }
}

export function hallucinationVerdictTone(
  verdict: HallucinationVerdict | null | undefined,
): Tone {
  switch (verdict) {
    case "supported":
      return "success";
    case "potentially_hallucinated":
      return "error";
    default:
      return "neutral";
  }
}

export function componentTone(status: "ok" | "error"): Tone {
  return status === "ok" ? "success" : "error";
}

export function fileExtensionLabel(filename: string): "PDF" | "TXT" | "FILE" {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf")) {
    return "PDF";
  }
  if (lower.endsWith(".txt")) {
    return "TXT";
  }
  return "FILE";
}
