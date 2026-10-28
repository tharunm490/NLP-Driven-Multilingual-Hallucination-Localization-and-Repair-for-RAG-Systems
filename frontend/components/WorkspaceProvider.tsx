"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  ApiError,
  checkHealth,
  getRagResult,
  listDocuments,
} from "@/lib/api";
import type {
  DocumentInfo,
  HealthResponse,
  RagQueryResponse,
} from "@/lib/types";

const LATEST_RESULT_KEY = "rag.latestResultId";

interface WorkspaceState {
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  refreshHealth: () => Promise<void>;
  documents: DocumentInfo[];
  documentsLoading: boolean;
  documentsError: string | null;
  refreshDocuments: () => Promise<void>;
  latestRagResult: RagQueryResponse | null;
  latestRagResultLoading: boolean;
  latestRagResultError: string | null;
  setLatestRagResult: (result: RagQueryResponse) => void;
  clearLatestRagResult: () => void;
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

  const [latestRagResult, setLatestRagResultState] =
    useState<RagQueryResponse | null>(null);
  const [latestRagResultLoading, setLatestRagResultLoading] = useState(false);
  const [latestRagResultError, setLatestRagResultError] = useState<string | null>(
    null,
  );
  const latestRagResultVersion = useRef(0);
  const restoreAbortRef = useRef<AbortController | null>(null);

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

  const setLatestRagResult = useCallback((result: RagQueryResponse) => {
    latestRagResultVersion.current += 1;
    restoreAbortRef.current?.abort();
    restoreAbortRef.current = null;
    setLatestRagResultState(result);
    setLatestRagResultLoading(false);
    setLatestRagResultError(null);
    try {
      if (result.result_id) {
        window.sessionStorage.setItem(LATEST_RESULT_KEY, result.result_id);
      } else {
        window.sessionStorage.removeItem(LATEST_RESULT_KEY);
      }
    } catch {
      // Storage can be unavailable in private or restricted browser modes.
    }
  }, []);

  const clearLatestRagResult = useCallback(() => {
    latestRagResultVersion.current += 1;
    restoreAbortRef.current?.abort();
    restoreAbortRef.current = null;
    setLatestRagResultState(null);
    setLatestRagResultError(null);
    setLatestRagResultLoading(false);
    try {
      window.sessionStorage.removeItem(LATEST_RESULT_KEY);
    } catch {
      // Storage can be unavailable in private or restricted browser modes.
    }
  }, []);

  useEffect(() => {
    let resultId: string | null = null;
    try {
      resultId = window.sessionStorage.getItem(LATEST_RESULT_KEY);
    } catch {
      resultId = null;
    }
    if (!resultId) {
      return;
    }

    const controller = new AbortController();
    const version = latestRagResultVersion.current;
    restoreAbortRef.current = controller;
    setLatestRagResultLoading(true);
    setLatestRagResultError(null);

    function isCurrent() {
      return (
        !controller.signal.aborted &&
        latestRagResultVersion.current === version
      );
    }

    void getRagResult(resultId, { signal: controller.signal })
      .then((result) => {
        if (isCurrent()) {
          setLatestRagResultState(result);
        }
      })
      .catch((caught: unknown) => {
        if (!isCurrent()) {
          return;
        }
        setLatestRagResultState(null);
        try {
          if (window.sessionStorage.getItem(LATEST_RESULT_KEY) === resultId) {
            window.sessionStorage.removeItem(LATEST_RESULT_KEY);
          }
        } catch {
          // Storage can be unavailable in private or restricted browser modes.
        }
        setLatestRagResultError(
          caught instanceof ApiError
            ? caught.message
            : "The latest Ask result could not be restored.",
        );
      })
      .finally(() => {
        if (isCurrent()) {
          restoreAbortRef.current = null;
          setLatestRagResultLoading(false);
        }
      });
    return () => {
      controller.abort();
      if (restoreAbortRef.current === controller) {
        restoreAbortRef.current = null;
      }
    };
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
      latestRagResult,
      latestRagResultLoading,
      latestRagResultError,
      setLatestRagResult,
      clearLatestRagResult,
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
      latestRagResult,
      latestRagResultLoading,
      latestRagResultError,
      setLatestRagResult,
      clearLatestRagResult,
      refreshAll,
    ],
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
