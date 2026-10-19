import type { DocumentInfo } from "@/lib/types";

interface DocumentFilterProps {
  id: string;
  documents: DocumentInfo[];
  value: string;
  onChange: (value: string) => void;
}

export function DocumentFilter({ id, documents, value, onChange }: DocumentFilterProps) {
  const completed = documents.filter((document) => document.status === "completed");
  return (
    <div className="field">
      <label htmlFor={id}>Document scope</label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={completed.length === 0}
      >
        <option value="">All documents</option>
        {completed.map((document) => (
          <option key={document.document_id} value={document.document_id}>
            {document.filename}
          </option>
        ))}
      </select>
    </div>
  );
}
