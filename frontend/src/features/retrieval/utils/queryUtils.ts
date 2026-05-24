import type { RetrievalMode, SearchPayload } from "../types/retrieval.types";

type BuildPayloadInput = {
  mode: RetrievalMode;
  query: string;
  ocrHint: string;
  transcriptHint: string;
  signals: { visual: boolean; ocr: boolean; transcript: boolean };
  rerank: boolean;
};

export function buildSearchPayload(input: BuildPayloadInput): SearchPayload {
  const query = input.query.trim();
  const ocr = input.ocrHint.trim();
  const transcript = input.transcriptHint.trim();
  const payload: SearchPayload = {
    fusion: "rrf",
    rerank_top_k: 0,
    neighbor_seconds: [-5, -3, 0, 3, 5],
    explain: true,
  };

  if (input.mode === "visual") {
    if (query) payload.description = query;
    if (ocr) payload.ocr = ocr;
    if (transcript) payload.transcript = transcript;
    return payload;
  }

  if (input.mode === "ocr") {
    payload.ocr = ocr || query;
    if (transcript) payload.transcript = transcript;
    return payload;
  }

  if (input.mode === "transcript") {
    payload.transcript = transcript || query;
    if (ocr) payload.ocr = ocr;
    return payload;
  }

  if (input.mode === "audio") {
    payload.audio = transcript || query;
    if (ocr) payload.ocr = ocr;
    return payload;
  }

  if (query && input.signals.visual) payload.description = query;
  if ((ocr || query) && input.signals.ocr) payload.ocr = ocr || query;
  if ((transcript || query) && input.signals.transcript) payload.transcript = transcript || query;
  return payload;
}

export function extractOcrMatches(text: string | undefined, hints: string[]): string[] {
  if (!text) return [];
  const normalized = text.toLowerCase();
  return hints
    .map((hint) => hint.trim())
    .filter((hint) => hint.length >= 2)
    .filter((hint) => normalized.includes(hint.toLowerCase()))
    .slice(0, 5);
}
