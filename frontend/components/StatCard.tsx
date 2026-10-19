import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { Tone } from "@/lib/format";

interface Props {
  icon: LucideIcon;
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
  loading?: boolean;
}

export default function StatCard({ icon: Icon, label, value, hint, tone = "neutral", loading }: Props) {
  return (
    <article className="stat-card">
      <header className="stat-head">
        <span className={`stat-icon stat-icon-${tone}`}>
          <Icon size={16} aria-hidden="true" />
        </span>
        <span className="stat-label">{label}</span>
      </header>
      {loading ? (
        <span className="skeleton skeleton-value" aria-hidden="true" />
      ) : (
        <p className="stat-value">{value}</p>
      )}
      {hint ? (
        loading ? (
          <span className="skeleton skeleton-hint" aria-hidden="true" />
        ) : (
          <p className="stat-hint">{hint}</p>
        )
      ) : null}
    </article>
  );
}
