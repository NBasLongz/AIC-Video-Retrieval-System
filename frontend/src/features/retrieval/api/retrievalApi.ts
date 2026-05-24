import { apiFetch } from "@/lib/apiClient";
import { formatTime, keyframeUrl, timestampToFrame, videoUrl } from "../utils/frameUtils";
import { clampScore, normalizeSource } from "../utils/scoreUtils";
import type { BackendSearchResult, RetrievalResult, SearchPayload } from "../types/retrieval.types";

export async function searchFrames(payload: SearchPayload, ocrHints: string[] = []): Promise<RetrievalResult[]> {
  const data = await apiFetch<BackendSearchResult[]>("/search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data.map((item, index) => mapBackendResult(item, index, ocrHints));
}

function mapBackendResult(item: BackendSearchResult, index: number, ocrHints: string[]): RetrievalResult {
  const videoId = String(item.video_id || item.videoId || "");
  const keyframeIndex = Number(item.keyframe_index ?? 0);
  const timestamp = Number(item.start_seconds ?? item.start ?? 0);
  const fps = Number(item.fps || 25);
  const frame = Number(item.frame_number ?? timestampToFrame(timestamp, fps));
  const sources = Array.isArray(item.sources) && item.sources.length
    ? item.sources
    : [item.source_type || item.doc_type].filter(Boolean) as string[];
  const source = normalizeSource(sources[0] || item.source_type || item.doc_type) as RetrievalResult["source"];

  const sourceScores = item.source_scores || {};
  const visual = clampScore(item.visual_score ?? item.clip_score ?? sourceScores.clip_score ?? sourceScores.visual_score);
  const ocr = clampScore(item.ocr_score ?? sourceScores.ocr_score);
  const transcript = clampScore(item.transcript_score ?? sourceScores.transcript_score);
  const textDense = clampScore(item.text_dense_score ?? sourceScores.text_dense_score);
  const fusion = clampScore(item.fusion_score ?? Math.max(visual, ocr, transcript, textDense));
  const rerank = item.rerank_score === undefined ? undefined : clampScore(item.rerank_score);
  const rank = item.rank_score === undefined ? undefined : clampScore(item.rank_score);
  const display = item.display_score === undefined ? undefined : clampScore(item.display_score);
  const score = display ?? (source === "Visual" ? visual : rerank ?? rank ?? fusion ?? visual);

  const ocrText = item.ocr_text || "";
  const ocrMatches = ocrHints
    .map((hint) => hint.trim())
    .filter((hint) => hint.length >= 2)
    .filter((hint) => ocrText.toLowerCase().includes(hint.toLowerCase()))
    .slice(0, 6);

  return {
    id: `${videoId}-${keyframeIndex}-${index}`,
    videoId,
    keyframeIndex,
    frame,
    timestamp,
    timeLabel: formatTime(timestamp),
    score,
    source,
    thumbnailUrl: keyframeUrl(videoId, keyframeIndex),
    videoUrl: videoUrl(videoId),
    evidence: {
      ocr: item.ocr_text,
      transcript: item.transcript_text,
      caption: item.caption_text,
      text: item.text,
    },
    scores: {
      visual,
      ocr,
      transcript,
      textDense,
      fusion,
      rerank,
      rank,
    },
    ocrMatches,
    sources,
    raw: item,
  };
}
