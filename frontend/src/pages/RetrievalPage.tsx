import { useCallback, useMemo, useState } from "react";
import { Keyboard, Pin, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { defaultSignals, neighborSeconds } from "@/features/retrieval/constants/retrieval.constants";
import { useNearbyFrames } from "@/features/retrieval/hooks/useNearbyFrames";
import { useRetrievalSearch } from "@/features/retrieval/hooks/useRetrievalSearch";
import { useSubmitFrame } from "@/features/retrieval/hooks/useSubmitFrame";
import { useVideoGroups } from "@/features/retrieval/hooks/useVideoGroups";
import { useVisibleResults } from "@/features/retrieval/hooks/useVisibleResults";
import type { RetrievalMode, RetrievalResult } from "@/features/retrieval/types/retrieval.types";
import { buildSearchPayload } from "@/features/retrieval/utils/queryUtils";
import { ResultGrid } from "@/features/retrieval/components/ResultGrid";
import { SearchHeader } from "@/features/retrieval/components/SearchHeader";
import { VideoList } from "@/features/retrieval/components/VideoList";
import { VideoModal } from "@/features/retrieval/components/VideoModal";

function splitHints(value: string) {
  return value
    .split(/[,;/|]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function RetrievalPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("visual");
  const [minScore, setMinScore] = useState("0.00");
  const [activeVideo, setActiveVideo] = useState("all");
  const [openedFrame, setOpenedFrame] = useState<RetrievalResult | null>(null);
  const [isVideoListCollapsed, setIsVideoListCollapsed] = useState(false);
  const [signals, setSignals] = useState(defaultSignals);
  const [ocrHint, setOcrHint] = useState("");
  const [transcriptHint, setTranscriptHint] = useState("");
  const [rerank, setRerank] = useState(false);
  const [keyframeLimit, setKeyframeLimit] = useState<number | "all">(80);
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(new Set());

  const { results, isSearching, error, search } = useRetrievalSearch();
  const { history, submit, submittedIds } = useSubmitFrame(query);

  const visibleResults = useVisibleResults(results, activeVideo, mode, minScore, keyframeLimit);
  const videoGroups = useVideoGroups(results.filter((item) => item.score >= Number(minScore)));
  const nearbyFrames = useNearbyFrames(openedFrame, neighborSeconds);

  const pinnedResults = useMemo(
    () => results.filter((item) => pinnedIds.has(item.id)).slice(0, 4),
    [pinnedIds, results],
  );

  const doSearch = useCallback(() => {
    const payload = buildSearchPayload({
      mode,
      query,
      ocrHint,
      transcriptHint,
      signals,
      rerank,
    });
    search(payload, splitHints(ocrHint));
  }, [mode, ocrHint, query, rerank, search, signals, transcriptHint]);

  const togglePin = useCallback((item: RetrievalResult) => {
    setPinnedIds((current) => {
      const next = new Set(current);
      if (next.has(item.id)) next.delete(item.id);
      else {
        if (next.size >= 4) next.delete(Array.from(next)[0]);
        next.add(item.id);
      }
      return next;
    });
  }, []);

  const submitCurrent = useCallback(
    async (item: RetrievalResult, frameOverride?: number) => {
      try {
        await submit(item, frameOverride);
      } catch (err) {
        console.error(err);
      }
    },
    [submit],
  );

  const resetFilters = () => {
    setMode("visual");
    setActiveVideo("all");
    setMinScore("0.00");
    setSignals(defaultSignals);
    setOcrHint("");
    setTranscriptHint("");
    setRerank(false);
    setKeyframeLimit(80);
  };

  useKeyboardShortcuts({
    onSearch: doSearch,
    onClose: () => setOpenedFrame(null),
    onSubmit: () => openedFrame && submitCurrent(openedFrame),
    onPin: () => openedFrame && togglePin(openedFrame),
  });

  return (
    <div className="h-screen w-full overflow-hidden bg-gradient-to-br from-sky-50 via-cyan-50 to-amber-50 px-3 py-4 text-slate-950 lg:px-4">
      <div className="flex h-full w-full max-w-none flex-col gap-4">
        <SearchHeader
          query={query}
          setQuery={setQuery}
          mode={mode}
          setMode={setMode}
          minScore={minScore}
          setMinScore={setMinScore}
          signals={signals}
          setSignals={setSignals}
          ocrHint={ocrHint}
          setOcrHint={setOcrHint}
          transcriptHint={transcriptHint}
          setTranscriptHint={setTranscriptHint}
          rerank={rerank}
          setRerank={setRerank}
          keyframeLimit={keyframeLimit}
          setKeyframeLimit={setKeyframeLimit}
          totalResults={results.length}
          visibleCount={visibleResults.length}
          submittedCount={submittedIds.size}
          isSearching={isSearching}
          error={error}
          onSearch={doSearch}
          onReset={resetFilters}
        />

        {pinnedResults.length > 0 && (
          <section className="flex shrink-0 items-center gap-2 overflow-x-auto rounded-2xl bg-white/95 p-2 shadow-sm ring-1 ring-slate-200">
            <div className="flex items-center gap-1.5 px-2 text-xs font-black text-amber-700">
              <Pin size={14} /> Shortlist
            </div>
            {pinnedResults.map((item) => (
              <button
                key={item.id}
                onClick={() => setOpenedFrame(item)}
                className="flex items-center gap-2 rounded-xl bg-amber-50 p-1.5 pr-3 text-left ring-1 ring-amber-100 hover:bg-amber-100"
              >
                <img src={item.thumbnailUrl} alt="" className="h-10 w-16 rounded-lg object-cover" />
                <span className="text-xs font-black text-slate-800">{item.videoId}</span>
                <span className="text-xs font-semibold text-slate-500">{item.timeLabel}</span>
              </button>
            ))}
            <button onClick={() => setPinnedIds(new Set())} className="ml-auto rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700" title="Clear shortlist">
              <X size={16} />
            </button>
          </section>
        )}

        <main className={`grid min-h-0 flex-1 gap-4 overflow-hidden ${isVideoListCollapsed ? "lg:grid-cols-[52px_minmax(0,1fr)]" : "lg:grid-cols-[220px_minmax(0,1fr)]"}`}>
          <VideoList
            groups={videoGroups}
            activeVideo={activeVideo}
            setActiveVideo={setActiveVideo}
            onOpenFrame={setOpenedFrame}
            collapsed={isVideoListCollapsed}
            setCollapsed={setIsVideoListCollapsed}
          />

          <section className="min-w-0 space-y-3 overflow-y-auto pr-1 pb-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-black text-slate-950">Ranked Frames</h2>
                <p className="text-sm text-slate-500">Click a frame to inspect nearby timestamps, pin it, or submit directly.</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="rounded-full bg-white px-3 py-1.5 text-xs font-black text-slate-500 shadow-sm ring-1 ring-slate-200">
                  History {history.length}
                </div>
                <div className="flex items-center gap-1.5 rounded-full bg-white px-3 py-1.5 text-xs font-black text-slate-500 shadow-sm ring-1 ring-slate-200">
                  <Keyboard size={14} /> Enter / P / S / Esc
                </div>
              </div>
            </div>

            <ResultGrid
              results={visibleResults}
              pinnedIds={pinnedIds}
              submittedIds={submittedIds}
              onOpen={setOpenedFrame}
              onSubmit={(item) => submitCurrent(item)}
              onPin={togglePin}
              mode={mode}
            />
          </section>
        </main>
      </div>

      <VideoModal
        item={openedFrame}
        neighbors={nearbyFrames}
        onClose={() => setOpenedFrame(null)}
        onSubmit={submitCurrent}
        onPin={togglePin}
        pinned={openedFrame ? pinnedIds.has(openedFrame.id) : false}
        submitted={openedFrame ? submittedIds.has(`${openedFrame.videoId}-${openedFrame.frame}`) : false}
        mode={mode}
      />
    </div>
  );
}
