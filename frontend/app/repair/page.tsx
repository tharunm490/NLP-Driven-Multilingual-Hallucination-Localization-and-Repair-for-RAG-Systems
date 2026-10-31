import type { Metadata } from "next";

import HallucinationRepairPanel from "@/components/HallucinationRepairPanel";

export const metadata: Metadata = {
  title: "Hallucination Repair",
};

export default function RepairPage() {
  return <HallucinationRepairPanel />;
}
