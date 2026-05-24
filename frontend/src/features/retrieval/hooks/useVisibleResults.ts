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
    const baseFiltered = results.filter((item) => {
      const passVideo = activeVideo === "all" || item.videoId === activeVideo;
      const passScore = item.score >= Number(minScore);
      return passVideo && passScore;
    });

    const modeFiltered = baseFiltered.filter((item) => {
      const source = item.source.toLowerCase();
      const passMode = mode === "hybrid" || mode === "audio" || source === mode || source === "hybrid";
      return passMode;
    });

    const visible = modeFiltered.length > 0 || mode === "hybrid" || mode === "audio"
      ? modeFiltered
      : baseFiltered;

    if (keyframeLimit === "all") return visible;
    return visible.slice(0, keyframeLimit);
  }, [activeVideo, keyframeLimit, minScore, mode, results]);
}
