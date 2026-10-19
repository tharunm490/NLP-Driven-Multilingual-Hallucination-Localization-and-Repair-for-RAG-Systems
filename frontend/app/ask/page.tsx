import type { Metadata } from "next";

import AskPanel from "@/components/AskPanel";

export const metadata: Metadata = {
  title: "Ask Documents",
};

export default function AskPage() {
  return <AskPanel />;
}
