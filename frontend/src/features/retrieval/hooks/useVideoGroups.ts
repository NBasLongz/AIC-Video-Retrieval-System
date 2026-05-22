import { useMemo } from "react";
import type { RetrievalResult } from "../types/retrieval.types";

export function useVideoGroups(results: RetrievalResult[]) {
  return useMemo(() => {
    const map = new Map<string, RetrievalResult[]>();
    results.forEach((item) => {
      if (!map.has(item.videoId)) map.set(item.videoId, []);
      map.get(item.videoId)!.push(item);
    });
    return Array.from(map.entries()).map(([videoId, frames]) => ({
      videoId,
      frames: frames.sort((a, b) => a.timestamp - b.timestamp),
    }));
  }, [results]);
}

