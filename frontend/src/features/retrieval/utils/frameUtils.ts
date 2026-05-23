export function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remaining = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

export function timestampToFrame(timestamp: number, fps = 25): number {
  return Math.max(0, Math.floor(timestamp * fps));
}

export function frameToTimestamp(frame: number, fps = 25): number {
  return Math.max(0, frame / fps);
}

export function timeMsFromFrame(frame: number, fps = 25): number {
  return Math.round(frameToTimestamp(frame, fps) * 1000);
}

export function keyframeUrl(videoId: string, keyframeIndex: number): string {
  return `/keyframes/${encodeURIComponent(videoId)}/keyframe_${keyframeIndex}.webp`;
}

export function videoUrl(videoId: string): string {
  return `/videos/${encodeURIComponent(videoId)}`;
}
