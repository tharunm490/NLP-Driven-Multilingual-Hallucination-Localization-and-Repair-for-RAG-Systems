import type { Tone } from "@/lib/format";
import type { ReactNode } from "react";

interface Props {
  tone: Tone;
  children: ReactNode;
}

export default function StatusBadge({ tone, children }: Props) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
