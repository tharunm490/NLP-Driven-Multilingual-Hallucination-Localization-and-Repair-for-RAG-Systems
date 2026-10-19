"use client";

import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";

export type StatusKind = "idle" | "loading" | "success" | "error" | "warning";

interface Props {
  status: StatusKind;
  message?: string;
  detail?: string;
}

const ICONS = {
  loading: Loader2,
  success: CheckCircle2,
  error: AlertTriangle,
  warning: AlertTriangle,
  idle: Info,
} as const;

export default function ProcessingStatus({ status, message, detail }: Props) {
  if (!message && !detail) {
    return null;
  }

  const Icon = ICONS[status];
  const spinning = status === "loading";

  return (
    <div className={`status status-${status}`} role="status" aria-live="polite">
      <Icon size={16} className={spinning ? "icon-spin" : undefined} aria-hidden="true" />
      <div className="status-body">
        {message && <p className="status-message">{message}</p>}
        {detail && <p className="status-detail">{detail}</p>}
      </div>
    </div>
  );
}
