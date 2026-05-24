import type { RetrievalMode, RetrievalResult } from "../types/retrieval.types";
import { ResultCard } from "./ResultCard";

export function ResultGrid({
  results,
  pinnedIds,
  submittedIds,
  onOpen,
  onSubmit,
  onPin,
  mode,
}: {
  results: RetrievalResult[];
  pinnedIds: Set<string>;
  submittedIds: Set<string>;
  onOpen: (item: RetrievalResult) => void;
  onSubmit: (item: RetrievalResult) => void;
  onPin: (item: RetrievalResult) => void;
  mode: RetrievalMode;
}) {
  if (!results.length) {
    return (
      <div className="rounded-3xl bg-white p-10 text-center text-sm font-semibold text-slate-500 ring-1 ring-slate-200">
        No frames found. Try lowering min score, adding OCR/transcript hints, or switching to Hybrid.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5">
      {results.map((item) => (
        <ResultCard
          key={item.id}
          item={item}
          onOpen={onOpen}
          onSubmit={onSubmit}
          onPin={onPin}
          pinned={pinnedIds.has(item.id)}
          submitted={submittedIds.has(`${item.videoId}-${item.frame}`)}
          mode={mode}
        />
      ))}
    </div>
  );
}
