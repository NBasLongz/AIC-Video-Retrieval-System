import { useMemo } from "react";
import type { RetrievalMode, RetrievalResult } from "../types/retrieval.types";

export function useVisibleResults(
  results: RetrievalResult[],
  activeVideo: string,
  mode: RetrievalMode,
  minScore: string,
  keyframeLimit: number | "all",
) {
  return useMemo(() => {
    const filtered = results.filter((item) => {
      const passVideo = activeVideo === "all" || item.videoId === activeVideo;
      const passScore = item.score >= Number(minScore);
      const source = item.source.toLowerCase();
      const passMode = mode === "hybrid" || mode === "audio" || source === mode || source === "hybrid";
      return passVideo && passScore && passMode;
    });
    if (keyframeLimit === "all") return filtered;
    return filtered.slice(0, keyframeLimit);
  }, [activeVideo, keyframeLimit, minScore, mode, results]);
}
