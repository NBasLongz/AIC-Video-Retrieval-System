import { Captions, FileText, Image as ImageIcon, Mic, Search, type LucideIcon } from "lucide-react";
import type { RetrievalMode } from "../types/retrieval.types";

export const retrievalModes: Array<{ id: RetrievalMode; label: string; icon: LucideIcon }> = [
  { id: "hybrid", label: "Hybrid", icon: Search },
  { id: "visual", label: "Visual", icon: ImageIcon },
  { id: "transcript", label: "Transcript", icon: Captions },
  { id: "ocr", label: "OCR", icon: FileText },
  { id: "audio", label: "Audio", icon: Mic },
];

export const neighborSeconds = [-5, -3, 0, 3, 5];

export const defaultSignals = {
  visual: true,
  ocr: true,
  transcript: true,
};
