import type { RetrievalMode, RetrievalScores } from "../types/retrieval.types";

function displayValue(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "0.00";
  return value > 1 ? value.toFixed(2) : value.toFixed(3);
}

function barWidth(value: number | undefined) {
  const safe = Number(value || 0);
  if (safe <= 1) return Math.max(4, safe * 100);
  return Math.min(100, safe * 12);
}

export function ScoreBreakdown({ mode, scores }: { mode: RetrievalMode; scores: RetrievalScores }) {
  const rows = mode === "hybrid"
    ? ([
        ["V", scores.visual],
        ["O", scores.ocr],
        ["T", scores.transcript],
      ] as const)
    : mode === "ocr"
      ? ([["O", scores.ocr]] as const)
      : mode === "transcript" || mode === "audio"
        ? ([["T", scores.transcript]] as const)
        : ([["V", scores.visual]] as const);

  return (
    <div className={`grid gap-1 ${rows.length === 1 ? "grid-cols-1" : "grid-cols-3"}`}>
      {rows.map(([label, value]) => (
        <div key={label} className="rounded-lg bg-slate-50 px-2 py-1">
          <div className="mb-1 flex items-center justify-between text-[10px] font-black text-slate-500">
            <span>{label}</span>
            <span>{displayValue(value)}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-200">
            <div className="h-full rounded-full bg-sky-500" style={{ width: `${barWidth(value)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
