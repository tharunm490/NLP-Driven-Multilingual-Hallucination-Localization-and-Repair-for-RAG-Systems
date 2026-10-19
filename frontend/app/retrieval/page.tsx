import type { Metadata } from "next";

import RetrievalPlayground from "@/components/RetrievalPlayground";

export const metadata: Metadata = {
  title: "Retrieval Playground",
};

export default function RetrievalPage() {
  return <RetrievalPlayground />;
}
