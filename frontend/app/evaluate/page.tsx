import type { Metadata } from "next";

import EvalPanel from "@/components/EvalPanel";

export const metadata: Metadata = {
  title: "Pipeline Evaluation",
};

export default function EvaluatePage() {
  return <EvalPanel />;
}
