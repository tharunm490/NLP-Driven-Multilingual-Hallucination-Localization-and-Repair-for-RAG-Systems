import {
  FileText,
  LayoutDashboard,
  MessageSquare,
  Search,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  href: string;
  label: string;
  description: string;
  icon: LucideIcon;
  soon?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Overview",
    description: "Workspace status, document and chunk counts, and system health.",
    icon: LayoutDashboard,
  },
  {
    href: "/documents",
    label: "Documents",
    description:
      "Upload, process, and manage the documents used as evidence by your RAG system.",
    icon: FileText,
  },
  {
    href: "/retrieval",
    label: "Retrieval Playground",
    description: "Test how effectively your knowledge base retrieves evidence for a question.",
    icon: Search,
  },
  {
    href: "/ask",
    label: "Ask Documents",
    description:
      "Ask questions about your uploaded documents and inspect the evidence used to answer them.",
    icon: MessageSquare,
  },
  {
    href: "/verification",
    label: "Hallucination Checks",
    description: "Claim verification and hallucination localization.",
    icon: ShieldCheck,
    soon: true,
  },
];

const DEFAULT_META = {
  title: "Multilingual RAG Workspace",
  description: "Document intelligence workspace for multilingual retrieval-augmented generation.",
};

export interface PageMeta {
  title: string;
  description: string;
}

export function getPageMeta(pathname: string): PageMeta {
  const normalised = pathname !== "/" ? pathname.replace(/\/+$/, "") : "/";
  const match = NAV_ITEMS.find((item) => item.href === normalised);
  if (match) {
    return { title: match.label, description: match.description };
  }
  return DEFAULT_META;
}
