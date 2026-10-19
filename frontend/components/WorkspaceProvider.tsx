"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { ApiError, checkHealth, listDocuments } from "@/lib/api";
import type { DocumentInfo, HealthResponse } from "@/lib/types";

interface WorkspaceState {
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  refreshHealth: () => Promise<void>;
  documents: DocumentInfo[];
  documentsLoading: boolean;
  documentsError: string | null;
  refreshDocuments: () => Promise<void>;
  refreshAll: () => void;
  refreshing: boolean;
}

const WorkspaceContext = createContext<WorkspaceState | null>(null);

export function useWorkspace(): WorkspaceState {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used inside a WorkspaceProvider");
  }
  return context;
}

export default function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      setHealth(await checkHealth());
    } catch (caught) {
      setHealth(null);
      setHealthError(
        caught instanceof ApiError ? caught.message : "Could not contact the backend.",
      );
    } finally {
      setHealthLoading(false);
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const response = await listDocuments();
      setDocuments(response.documents);
    } catch (caught) {
      setDocumentsError(
        caught instanceof ApiError ? caught.message : "Could not load documents.",
      );
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
    void refreshDocuments();
  }, [refreshHealth, refreshDocuments]);

  const refreshAll = useCallback(() => {
    void refreshHealth();
    void refreshDocuments();
  }, [refreshHealth, refreshDocuments]);

  const value = useMemo<WorkspaceState>(
    () => ({
      health,
      healthLoading,
      healthError,
      refreshHealth,
      documents,
      documentsLoading,
      documentsError,
      refreshDocuments,
      refreshAll,
      refreshing: healthLoading || documentsLoading,
    }),
    [
      health,
      healthLoading,
      healthError,
      refreshHealth,
      documents,
      documentsLoading,
      documentsError,
      refreshDocuments,
      refreshAll,
    ],
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
