import type { Metadata } from "next";

import HallucinationCheckPanel from "@/components/HallucinationCheckPanel";

export const metadata: Metadata = {
  title: "Hallucination Checks",
};

export default function VerificationPage() {
  return <HallucinationCheckPanel />;
}
