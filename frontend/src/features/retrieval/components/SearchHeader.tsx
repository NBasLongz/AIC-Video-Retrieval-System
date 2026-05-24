import { Loader2, RotateCcw, Search, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { RetrievalMode } from "../types/retrieval.types";
import { EvaluationConfigBar } from "./EvaluationConfigBar";
import { ModeTabs } from "./ModeTabs";
import { QuickAssistBar } from "./QuickAssistBar";
import { SignalToggle } from "./SignalToggle";

type Signals = {
  visual: boolean;
  ocr: boolean;
  transcript: boolean;
};

export function SearchHeader({
  query,
  setQuery,
  mode,
  setMode,
  minScore,
  setMinScore,
  signals,
  setSignals,
  ocrHint,
  setOcrHint,
  transcriptHint,
  setTranscriptHint,
  rerank,
  setRerank,
  keyframeLimit,
  setKeyframeLimit,
  totalResults,
  visibleCount,
  submittedCount,
  isSearching,
  error,
  onSearch,
  onReset,
}: {
  query: string;
  setQuery: (value: string) => void;
  mode: RetrievalMode;
  setMode: (value: RetrievalMode) => void;
  minScore: string;
  setMinScore: (value: string) => void;
  signals: Signals;
  setSignals: (value: Signals | ((current: Signals) => Signals)) => void;
  ocrHint: string;
  setOcrHint: (value: string) => void;
  transcriptHint: string;
  setTranscriptHint: (value: string) => void;
  rerank: boolean;
  setRerank: (value: boolean | ((current: boolean) => boolean)) => void;
  keyframeLimit: number | "all";
  setKeyframeLimit: (value: number | "all") => void;
  totalResults: number;
  visibleCount: number;
  submittedCount: number;
  isSearching: boolean;
  error: string | null;
  onSearch: () => void;
  onReset: () => void;
}) {
  const sliderMax = Math.max(40, totalResults);
  const sliderValue = keyframeLimit === "all" ? sliderMax : Math.min(keyframeLimit, sliderMax);
  const limitLabel = keyframeLimit === "all" || totalResults <= Number(keyframeLimit) ? "All" : String(keyframeLimit);
  const modeLabel = mode === "visual" ? "Visual only" : mode === "ocr" ? "OCR only" : mode === "transcript" ? "Transcript only" : mode === "audio" ? "Audio only" : "Hybrid";

  return (
    <header className="shrink-0 rounded-2xl bg-white/95 p-3 shadow-sm ring-1 ring-slate-200/70 backdrop-blur-xl">
      <div className="mb-2 grid gap-2 xl:grid-cols-[minmax(300px,1fr)_auto] xl:items-center">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-950 lg:text-2xl">AIC Video Retrieval</h1>
          <p className="text-xs text-slate-500">Search the selected signal, inspect nearby frames, then submit the exact frame.</p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <EvaluationConfigBar />
          <ModeTabs mode={mode} setMode={setMode} />
        </div>
      </div>

      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_112px]">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={19} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onSearch()}
            placeholder={mode === "visual" || mode === "hybrid" ? "Search visual scene in English..." : mode === "ocr" ? "Search OCR text..." : "Search transcript text..."}
            className="h-10 w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 text-sm font-medium outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
          />
        </div>
        <Button
          onClick={onSearch}
          disabled={isSearching || (!query.trim() && !ocrHint.trim() && !transcriptHint.trim())}
          className="h-10 rounded-xl bg-sky-600 px-6 text-sm font-black text-white hover:bg-sky-700"
        >
          {isSearching ? <Loader2 size={17} className="mr-2 animate-spin" /> : <Search size={17} className="mr-2" />}
          Search
        </Button>
      </div>

      {mode === "hybrid" && (
        <QuickAssistBar
          ocrHint={ocrHint}
          setOcrHint={setOcrHint}
          transcriptHint={transcriptHint}
          setTranscriptHint={setTranscriptHint}
          rerank={rerank}
          setRerank={setRerank}
        />
      )}

      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
        <span className="flex items-center gap-1.5 font-black text-slate-700">
          <SlidersHorizontal size={15} /> Min score
        </span>
        <input
          type="range"
          min="0.00"
          max="1.00"
          step="0.01"
          value={minScore}
          onChange={(event) => setMinScore(event.target.value)}
          className="w-32 accent-sky-600"
        />
        <span className="rounded-full bg-sky-50 px-2.5 py-0.5 text-xs font-black text-sky-700">{minScore}</span>
        <span className="hidden text-slate-300 sm:inline">/</span>
        {mode === "hybrid" ? (
          <>
            <SignalToggle checked={signals.visual} onChange={() => setSignals((current) => ({ ...current, visual: !current.visual }))} label="Visual" />
            <SignalToggle checked={signals.ocr} onChange={() => setSignals((current) => ({ ...current, ocr: !current.ocr }))} label="OCR" />
            <SignalToggle checked={signals.transcript} onChange={() => setSignals((current) => ({ ...current, transcript: !current.transcript }))} label="Transcript" />
          </>
        ) : (
          <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-black text-emerald-700">{modeLabel}</span>
        )}
        <span className="hidden text-slate-300 sm:inline">/</span>
        <span className="font-semibold text-slate-500">{visibleCount}/{totalResults} frames shown</span>
        <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 font-black text-emerald-700">Submitted {submittedCount}</span>

        <div className="ml-auto flex flex-wrap items-center gap-2 rounded-xl bg-slate-50 px-2.5 py-1.5 ring-1 ring-slate-200">
          <span className="text-xs font-black text-slate-600">Keyframes</span>
          <input
            type="range"
            min="20"
            max={sliderMax}
            step="10"
            value={sliderValue}
            onChange={(event) => {
              const value = Number(event.target.value);
              setKeyframeLimit(value >= totalResults ? "all" : value);
            }}
            className="w-36 accent-sky-600"
          />
          <span className="min-w-8 rounded-full bg-white px-2 py-0.5 text-center text-xs font-black text-sky-700 ring-1 ring-slate-200">{limitLabel}</span>
          <button onClick={onReset} className="flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-xs font-black text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100">
            <RotateCcw size={13} /> Reset
          </button>
        </div>
      </div>

      {error && <p className="mt-2 rounded-xl bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700">{error}</p>}
    </header>
  );
}
