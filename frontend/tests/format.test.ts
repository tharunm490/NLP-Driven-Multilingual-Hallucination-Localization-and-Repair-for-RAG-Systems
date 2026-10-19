import { describe, expect, it } from "vitest";

import {
  documentStatusTone,
  fileExtensionLabel,
  formatBytes,
  formatDateTime,
  formatNumber,
  uploadStatusTone,
} from "../lib/format";

describe("format helpers", () => {
  it("formats byte sizes with sensible units", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(15 * 1024 * 1024)).toBe("15 MB");
    expect(formatBytes(null)).toBe("—");
  });

  it("formats numbers with thousands separators", () => {
    expect(formatNumber(12345)).toBe("12,345");
  });

  it("falls back to the raw string for invalid dates", () => {
    expect(formatDateTime("not-a-date")).toBe("not-a-date");
    expect(formatDateTime(null)).toBe("—");
  });

  it("maps document and upload statuses to tones", () => {
    expect(documentStatusTone("completed")).toBe("success");
    expect(documentStatusTone("processing")).toBe("warning");
    expect(documentStatusTone("failed")).toBe("error");
    expect(uploadStatusTone("rejected")).toBe("warning");
    expect(uploadStatusTone("completed")).toBe("success");
  });

  it("labels file extensions", () => {
    expect(fileExtensionLabel("notes.PDF")).toBe("PDF");
    expect(fileExtensionLabel("readme.txt")).toBe("TXT");
    expect(fileExtensionLabel("data.csv")).toBe("FILE");
  });
});
