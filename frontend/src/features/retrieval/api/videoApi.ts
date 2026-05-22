import { keyframeUrl, videoUrl } from "../utils/frameUtils";

export const getVideoUrl = videoUrl;
export const getThumbnailUrl = keyframeUrl;
export function getPreviewUrl(videoId: string, timestamp: number) {
  return `${videoUrl(videoId)}#t=${Math.max(0, timestamp).toFixed(2)}`;
}

