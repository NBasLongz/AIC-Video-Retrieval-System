import { CheckCircle2, Pin, Play, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { RetrievalMode, RetrievalResult } from "../types/retrieval.types";
import { OcrChips } from "./OcrChips";
import { ScoreBreakdown } from "./ScoreBreakdown";

function evidenceForMode(item: RetrievalResult, mode: RetrievalMode) {
  if (mode === "ocr") return item.evidence.ocr || item.evidence.text;
  if (mode === "transcript" || mode === "audio") return item.evidence.transcript || item.evidence.text;
  if (mode === "visual") return item.evidence.caption || item.evidence.text || "Visual scene matches the query.";
  if (item.source === "OCR") return item.evidence.ocr || item.evidence.text;
  if (item.source === "Transcript") return item.evidence.transcript || item.evidence.text;
  return item.evidence.caption || item.evidence.ocr || item.evidence.transcript || item.evidence.text || "Visual scene matches the query.";
}

export function ResultCard({
  item,
  onOpen,
  onSubmit,
  onPin,
  pinned,
  submitted,
  mode,
}: {
  item: RetrievalResult;
  onOpen: (item: RetrievalResult) => void;
  onSubmit: (item: RetrievalResult) => void;
  onPin: (item: RetrievalResult) => void;
  pinned: boolean;
  submitted: boolean;
  mode: RetrievalMode;
}) {
  const evidence = evidenceForMode(item, mode);
  const showOcrChips = mode === "ocr" || (mode === "hybrid" && item.source === "OCR");

  return (
    <div className="group overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 transition hover:-translate-y-0.5 hover:shadow-lg">
      <button onClick={() => onOpen(item)} className="block w-full text-left">
        <div className="relative aspect-video bg-gradient-to-br from-sky-100 via-cyan-100 to-amber-100">
          <img src={item.thumbnailUrl} alt={`${item.videoId} frame ${item.keyframeIndex}`} className="h-full w-full object-cover" loading="lazy" />
          <div className="absolute inset-0 flex items-center justify-center bg-black/0 text-sky-700 opacity-0 transition group-hover:bg-black/15 group-hover:opacity-100">
            <div className="rounded-full bg-white/90 p-2.5 shadow-sm transition group-hover:scale-105">
              <Play size={22} fill="currentColor" />
            </div>
          </div>
          <span className="absolute left-2.5 top-2.5 rounded-full bg-slate-950/75 px-2 py-1 text-[11px] font-black text-white">
            {item.timeLabel}
          </span>
          <span className="absolute right-2.5 top-2.5 rounded-full bg-white px-2 py-1 text-[11px] font-black text-sky-700 shadow-sm">
            {item.score.toFixed(3)}
          </span>
          <span className="absolute bottom-2.5 left-2.5 rounded-full bg-white/90 px-2 py-1 text-[11px] font-black text-slate-700 shadow-sm">
            Frame {item.frame}
          </span>
        </div>
      </button>

      <div className="space-y-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="text-sm font-black text-slate-950">{item.videoId}</p>
            <p className="line-clamp-1 text-xs text-slate-500">{evidence}</p>
          </div>
          <span className="shrink-0 rounded-full bg-sky-50 px-2 py-1 text-[11px] font-black text-sky-700">{item.source}</span>
        </div>

        {showOcrChips && <OcrChips matches={item.ocrMatches} text={item.evidence.ocr} />}
        <ScoreBreakdown mode={mode} scores={item.scores} />

        <div className="grid grid-cols-3 gap-2">
          <Button onClick={() => onOpen(item)} variant="outline" className="h-9 rounded-xl border-slate-200 bg-white text-xs font-black">
            <Play size={14} className="mr-1.5" /> View
          </Button>
          <Button
            onClick={() => onPin(item)}
            variant="outline"
            className={`h-9 rounded-xl text-xs font-black ${pinned ? "border-amber-200 bg-amber-100 text-amber-700" : ""}`}
          >
            <Pin size={14} className="mr-1.5" /> {pinned ? "Pinned" : "Pin"}
          </Button>
          <Button
            onClick={() => onSubmit(item)}
            className={`h-9 rounded-xl text-xs font-black ${submitted ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-100" : "bg-emerald-500 text-white hover:bg-emerald-600"}`}
          >
            {submitted ? <CheckCircle2 size={14} className="mr-1.5" /> : <Send size={14} className="mr-1.5" />}
            {submitted ? "Done" : "Submit"}
          </Button>
        </div>
      </div>
    </div>
  );
}
