"use client";

import { FileUp, Lock, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";

import { ApiError, uploadDocuments } from "@/lib/api";
import { formatBytes, uploadStatusLabel, uploadStatusTone } from "@/lib/format";
import type { UploadItemResult } from "@/lib/types";

import ProcessingStatus from "./ProcessingStatus";
import StatusBadge from "./StatusBadge";
import { useWorkspace } from "./WorkspaceProvider";

const ALLOWED_EXTENSIONS = [".pdf", ".txt"];

interface Props {
  onUploaded: () => void;
}

export default function UploadPanel({ onUploaded }: Props) {
  const { health } = useWorkspace();
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<UploadItemResult[]>([]);

  const maxSizeMb = health?.max_upload_size_mb ?? 20;
  const maxFiles = health?.max_files_per_request ?? 5;
  const maxBytes = maxSizeMb * 1024 * 1024;

  function accept(incoming: File[]) {
    setResults([]);
    const problems: string[] = [];
    const accepted: File[] = [];

    for (const file of incoming) {
      const name = file.name.toLowerCase();
      if (!ALLOWED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
        problems.push(`${file.name} is not a PDF or TXT file`);
        continue;
      }
      if (file.size > maxBytes) {
        problems.push(`${file.name} exceeds ${maxSizeMb} MB`);
        continue;
      }
      if (accepted.some((existing) => existing.name === file.name && existing.size === file.size)) {
        continue;
      }
      accepted.push(file);
    }

    let selected = accepted;
    if (accepted.length > maxFiles) {
      selected = accepted.slice(0, maxFiles);
      problems.push(`Only ${maxFiles} files can be uploaded at once`);
    }

    setFiles(selected);
    setError(problems.length > 0 ? problems.join(". ") + "." : null);
  }

  function handleInputChange(event: React.ChangeEvent<HTMLInputElement>) {
    accept(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (uploading) {
      return;
    }
    accept(Array.from(event.dataTransfer.files ?? []));
  }

  function removeFile(target: File) {
    setFiles((current) => current.filter((file) => file !== target));
  }

  async function handleUpload() {
    if (files.length === 0 || uploading) {
      return;
    }
    setUploading(true);
    setError(null);
    setResults([]);
    try {
      const response = await uploadDocuments(files);
      setResults(response.results);
      const accepted = response.results.filter((result) => result.status === "completed");
      setFiles(accepted.length === files.length ? [] : files);
      onUploaded();
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.message : "Unexpected error while uploading.",
      );
    } finally {
      setUploading(false);
    }
  }

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);

  return (
    <section className="card" id="upload">
      <div className="card-header">
        <div>
          <h2>Upload documents</h2>
          <p className="section-description">
            PDF and TXT files are validated, cleaned, chunked, embedded, and stored in the
            knowledge base.
          </p>
        </div>
      </div>

      <div
        className={`upload-zone${dragging ? " upload-zone-active" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading) {
            setDragging(true);
          }
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <span className="upload-icon" aria-hidden="true">
          <UploadCloud size={22} />
        </span>
        <p className="upload-title">Upload your documents</p>
        <p className="upload-subtitle">Add PDF or TXT files to your knowledge base.</p>
        <div className="upload-actions">
          <input
            id="document-upload-input"
            ref={inputRef}
            type="file"
            accept=".pdf,.txt"
            multiple
            className="visually-hidden"
            onChange={handleInputChange}
            disabled={uploading}
          />
          <button
            type="button"
            className="button button-secondary"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
          >
            <FileUp size={15} aria-hidden="true" />
            Choose files
          </button>
          <button
            type="button"
            className="button button-primary"
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
          >
            {uploading ? "Uploading…" : `Upload${files.length > 0 ? ` (${files.length})` : ""}`}
          </button>
        </div>
        <p className="upload-hint">
          <Lock size={12} aria-hidden="true" />
          PDF or TXT · up to {maxSizeMb} MB each · up to {maxFiles} files per upload
        </p>
      </div>

      {files.length > 0 && (
        <ul className="file-chips">
          {files.map((file) => (
            <li key={`${file.name}-${file.size}`} className="file-chip">
              <span className="file-chip-name">{file.name}</span>
              <span className="file-chip-size">{formatBytes(file.size)}</span>
              <button
                type="button"
                className="icon-button"
                onClick={() => removeFile(file)}
                disabled={uploading}
                aria-label={`Remove ${file.name}`}
              >
                <X size={14} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {files.length > 0 && (
        <p className="upload-total">
          {files.length} file{files.length === 1 ? "" : "s"} selected · {formatBytes(totalBytes)}
        </p>
      )}

      {uploading && (
        <ProcessingStatus
          status="loading"
          message="Processing documents…"
          detail="Extracting text, generating embeddings, and writing to ChromaDB. The first upload may also load the embedding model."
        />
      )}

      {error && <ProcessingStatus status="error" message={error} />}

      {results.length > 0 && (
        <div className="upload-results">
          <h3>Upload results</h3>
          <ul className="result-list">
            {results.map((result, index) => (
              <li key={`${result.original_filename}-${index}`} className="upload-result">
                <div className="upload-result-head">
                  <span className="upload-result-name">{result.original_filename}</span>
                  <StatusBadge tone={uploadStatusTone(result.status)}>
                    {uploadStatusLabel(result.status)}
                  </StatusBadge>
                </div>
                <div className="upload-result-meta">
                  {result.status === "completed" && (
                    <>
                      <span>{result.chunks_stored} chunk(s) stored</span>
                      <span>{result.character_count ?? 0} characters</span>
                      {result.language && <span>language: {result.language}</span>}
                    </>
                  )}
                  {result.duplicate && <span>Identical content already ingested.</span>}
                  {result.message && <span>{result.message}</span>}
                </div>
                {result.error && <p className="error-text">{result.error}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
