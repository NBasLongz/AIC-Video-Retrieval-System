export function clampScore(score: unknown): number {
  const value = Number(score);
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export function normalizeSource(source?: string): string {
  const normalized = source?.toLowerCase();
  if (!normalized) return "Hybrid";
  if (normalized === "visual" || normalized === "clip") return "Visual";
  if (normalized === "ocr") return "OCR";
  if (normalized === "transcript" || normalized === "asr") return "Transcript";
  if (normalized === "caption") return "Caption";
  if (normalized === "text") return "Text";
  if (normalized === "hybrid") return "Hybrid";
  const original = source ?? "Hybrid";
  return original.charAt(0).toUpperCase() + original.slice(1);
}
