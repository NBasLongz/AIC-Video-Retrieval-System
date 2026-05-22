import { useCallback, useEffect, useState } from "react";
import { submitFrame } from "../api/submitApi";
import { timeMsFromFrame } from "../utils/frameUtils";
import type { RetrievalResult, SubmitHistoryItem } from "../types/retrieval.types";

const STORAGE_KEY = "aic-submit-history";

function loadHistory(): SubmitHistoryItem[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as SubmitHistoryItem[];
  } catch {
    return [];
  }
}

export function useSubmitFrame(query: string) {
  const [history, setHistory] = useState<SubmitHistoryItem[]>(() => loadHistory());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  const submit = useCallback(
    async (item: RetrievalResult, frameOverride?: number) => {
      const frame = frameOverride ?? item.frame;
      const fps = Number(item.raw.fps || 25);
      const timeMs = timeMsFromFrame(frame, fps);
      const id = `${item.videoId}-${frame}`;

      try {
        await submitFrame(item.videoId, timeMs);
        setHistory((prev) => [
          {
            id,
            videoId: item.videoId,
            frame,
            timestamp: timeMs / 1000,
            query,
            score: item.score,
            status: "success",
            createdAt: new Date().toISOString(),
          },
          ...prev.filter((entry) => entry.id !== id),
        ]);
      } catch (err) {
        setHistory((prev) => [
          {
            id,
            videoId: item.videoId,
            frame,
            timestamp: timeMs / 1000,
            query,
            score: item.score,
            status: "failed",
            createdAt: new Date().toISOString(),
            message: err instanceof Error ? err.message : "Submit failed",
          },
          ...prev.filter((entry) => entry.id !== id),
        ]);
        throw err;
      }
    },
    [query],
  );

  return {
    history,
    submit,
    submittedIds: new Set(history.filter((item) => item.status === "success").map((item) => item.id)),
  };
}

