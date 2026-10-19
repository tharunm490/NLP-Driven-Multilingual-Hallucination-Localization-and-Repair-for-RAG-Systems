"use client";

import { Upload } from "lucide-react";

import DocumentsTable from "./DocumentsTable";
import UploadPanel from "./UploadPanel";
import { useWorkspace } from "./WorkspaceProvider";

export default function DocumentsView() {
  const { documents, documentsLoading, documentsError, refreshDocuments } = useWorkspace();

  function openUploadPicker() {
    document.getElementById("upload")?.scrollIntoView({ behavior: "smooth", block: "start" });
    document.getElementById("document-upload-input")?.click();
  }

  return (
    <div className="stack">
      <div className="toolbar">
        <p className="toolbar-note">
          {documentsLoading
            ? "Loading knowledge base…"
            : `${documents.length} document(s) in the knowledge base`}
        </p>
        <button type="button" className="button button-primary" onClick={openUploadPicker}>
          <Upload size={15} aria-hidden="true" />
          Upload documents
        </button>
      </div>

      <UploadPanel onUploaded={refreshDocuments} />

      <DocumentsTable
        documents={documents}
        loading={documentsLoading}
        error={documentsError}
        onRefresh={refreshDocuments}
      />
    </div>
  );
}
