import type { RetrievalResult } from "../types/retrieval.types";
import { timeMsFromFrame, timestampToFrame } from "./frameUtils";

type SubmitTime = {
  frame: number;
  timestamp: number;
  timeMs: number;
  basis: "frame_override" | "frame_number" | "transcript_timestamp" | "item_frame" | "timestamp";
};

function finiteNumber(value: unknown): number | undefined {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function normalizedSources(item: RetrievalResult): string[] {
  return [
    ...item.sources,
    item.raw.source_type,
    item.raw.doc_type,
  ]
    .filter(Boolean)
    .map((source) => String(source).toLowerCase());
}

function shouldSubmitByTranscriptTimestamp(item: RetrievalResult): boolean {
  const sources = normalizedSources(item);
  const hasTranscript = sources.some((source) => source === "transcript" || source === "asr");
  const hasVisualFrameSource = sources.some((source) => ["visual", "clip", "ocr", "caption"].includes(source));
  return hasTranscript && !hasVisualFrameSource;
}

export function computeSubmitTime(item: RetrievalResult, frameOverride?: number): SubmitTime {
  const fps = finiteNumber(item.raw.fps) || 25;

  if (frameOverride !== undefined) {
    const frame = Math.max(0, Math.floor(frameOverride));
    const timeMs = timeMsFromFrame(frame, fps);
    return {
      frame,
      timestamp: timeMs / 1000,
      timeMs,
      basis: "frame_override",
    };
  }

  const timestamp = finiteNumber(item.raw.start_seconds ?? item.raw.start ?? item.timestamp);
  if (shouldSubmitByTranscriptTimestamp(item) && timestamp !== undefined) {
    const timeMs = Math.round(Math.max(0, timestamp) * 1000);
    return {
      frame: timestampToFrame(timeMs / 1000, fps),
      timestamp: timeMs / 1000,
      timeMs,
      basis: "transcript_timestamp",
    };
  }

  const rawFrameNumber = finiteNumber(item.raw.frame_number);
  if (rawFrameNumber !== undefined) {
    const frame = Math.max(0, Math.floor(rawFrameNumber));
    const timeMs = timeMsFromFrame(frame, fps);
    return {
      frame,
      timestamp: timeMs / 1000,
      timeMs,
      basis: "frame_number",
    };
  }

  const itemFrame = finiteNumber(item.frame);
  if (itemFrame !== undefined) {
    const frame = Math.max(0, Math.floor(itemFrame));
    const timeMs = timeMsFromFrame(frame, fps);
    return {
      frame,
      timestamp: timeMs / 1000,
      timeMs,
      basis: "item_frame",
    };
  }

  const fallbackTimestamp = Math.max(0, timestamp || 0);
  const timeMs = Math.round(fallbackTimestamp * 1000);
  return {
    frame: timestampToFrame(fallbackTimestamp, fps),
    timestamp: timeMs / 1000,
    timeMs,
    basis: "timestamp",
  };
}
