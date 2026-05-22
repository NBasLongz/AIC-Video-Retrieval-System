import { useMemo } from "react";
import { timestampToFrame } from "../utils/frameUtils";
import type { NearbyFrame, RetrievalResult } from "../types/retrieval.types";

export function useNearbyFrames(openedFrame: RetrievalResult | null, offsets: number[] = [-5, -3, 0, 3, 5]): NearbyFrame[] {
  return useMemo(() => {
    if (!openedFrame) return [];
    const fps = Number(openedFrame.raw.fps || 25);
    return offsets.map((offset) => {
      const timestamp = Math.max(0, openedFrame.timestamp + offset);
      return {
        id: `${openedFrame.videoId}-${timestamp}`,
        videoId: openedFrame.videoId,
        timestamp,
        frame: timestampToFrame(timestamp, fps),
        label: offset === 0 ? "Current" : `${offset > 0 ? "+" : ""}${offset}s`,
      };
    });
  }, [offsets, openedFrame]);
}
