import type { Metadata } from "next";

import AppShell from "@/components/AppShell";
import WorkspaceProvider from "@/components/WorkspaceProvider";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Multilingual RAG Workspace",
    template: "%s · Multilingual RAG Workspace",
  },
  description:
    "Upload multilingual documents, embed them, test retrieval, and ask questions answered strictly from retrieved evidence.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("theme");var d=t==="dark"||(t!=="light"&&window.matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.dataset.theme=d?"dark":"light";}catch(e){}})();`,
          }}
        />
        <WorkspaceProvider>
          <AppShell>{children}</AppShell>
        </WorkspaceProvider>
      </body>
    </html>
  );
}
