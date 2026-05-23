import { useMemo } from "react";
import { timestampToFrame } from "../utils/frameUtils";
import type { NearbyFrame, RetrievalResult } from "../types/retrieval.types";

export function useNearbyFrames(openedFrame: RetrievalResult | null, offsets: number[] = [-5, -3, 0, 3, 5]): NearbyFrame[] {
  return useMemo(() => {
    if (!openedFrame) return [];
    if (openedFrame.raw.neighbors?.length) {
      return openedFrame.raw.neighbors.map((neighbor) => ({
        id: `${neighbor.video_id}-${neighbor.time_ms}`,
        videoId: neighbor.video_id,
        timestamp: neighbor.timestamp,
        frame: neighbor.frame_number,
        label: neighbor.label === "current" ? "Current" : neighbor.label,
      }));
    }
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
