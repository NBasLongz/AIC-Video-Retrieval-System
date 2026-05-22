export type RetrievalMode = "hybrid" | "visual" | "transcript" | "ocr" | "audio";

export type RetrievalSource = "Visual" | "OCR" | "Transcript" | "Hybrid" | "Caption" | "Text";

export type RetrievalScores = {
  visual: number;
  ocr: number;
  transcript: number;
  fusion: number;
  rerank?: number;
};

export type NearbyFrame = {
  id: string;
  videoId: string;
  frame: number;
  timestamp: number;
  label: string;
  thumbnailUrl?: string;
};

export type RetrievalResult = {
  id: string;
  videoId: string;
  keyframeIndex: number;
  frame: number;
  timestamp: number;
  timeLabel: string;
  score: number;
  source: RetrievalSource;
  thumbnailUrl: string;
  videoUrl: string;
  evidence: {
    caption?: string;
    ocr?: string;
    transcript?: string;
    text?: string;
  };
  scores: RetrievalScores;
  ocrMatches: string[];
  sources: string[];
  raw: BackendSearchResult;
};

export type BackendSearchResult = {
  video_id?: string;
  videoId?: string;
  keyframe_index?: number;
  frame_number?: number;
  start_seconds?: number;
  start?: number;
  fps?: number;
  clip_score?: number;
  visual_score?: number;
  ocr_score?: number;
  transcript_score?: number;
  caption_score?: number;
  text_score?: number;
  fusion_score?: number;
  rerank_score?: number;
  source_type?: string;
  doc_type?: string;
  sources?: string[];
  source_scores?: Record<string, number>;
  source_ranks?: Record<string, number>;
  ocr_text?: string;
  transcript_text?: string;
  caption_text?: string;
  text?: string;
};

export type SearchPayload = {
  description?: string;
  ocr?: string;
  transcript?: string;
  audio?: string;
  caption?: string;
  negative?: string;
  fusion?: "rrf" | "intersection";
  rerank_top_k?: number;
  neighbor_seconds?: number[];
  explain?: boolean;
};

export type SubmitHistoryItem = {
  id: string;
  videoId: string;
  frame: number;
  timestamp: number;
  query: string;
  score: number;
  status: "success" | "failed";
  createdAt: string;
  message?: string;
};

