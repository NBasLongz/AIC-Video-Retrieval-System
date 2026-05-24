import { useMemo } from "react";
import { keyframeUrl, timestampToFrame } from "../utils/frameUtils";
import type { NearbyFrame, RetrievalResult } from "../types/retrieval.types";

export function useNearbyFrames(openedFrame: RetrievalResult | null, offsets: number[] = [-4, -3, -2, -1, 0, 1, 2, 3, 4]): NearbyFrame[] {
  return useMemo(() => {
    if (!openedFrame) return [];
    const fps = Number(openedFrame.raw.fps || 25);
    return offsets.map((offset) => {
      const timestamp = Math.max(0, openedFrame.timestamp + offset);
      const keyframeIndex = Math.max(0, openedFrame.keyframeIndex + offset);
      const frame = timestampToFrame(timestamp, fps);
      return {
        id: `${openedFrame.videoId}-${keyframeIndex}`,
        videoId: openedFrame.videoId,
        keyframeIndex,
        timestamp,
        frame,
        label: offset === 0 ? "Current" : `${offset > 0 ? "+" : ""}${offset}`,
        thumbnailUrl: keyframeUrl(openedFrame.videoId, keyframeIndex),
      };
    });
  }, [offsets, openedFrame]);
}
