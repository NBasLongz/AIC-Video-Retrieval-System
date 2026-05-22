import type { NearbyFrame } from "../types/retrieval.types";

export function NearbyFrames({
  frames,
  currentFrame,
  onSelect,
}: {
  frames: NearbyFrame[];
  currentFrame: number;
  onSelect: (frame: NearbyFrame) => void;
}) {
  return (
    <div className="grid grid-cols-5 gap-1.5">
      {frames.map((frame) => {
        const active = Math.abs(frame.frame - currentFrame) <= 1;
        return (
          <button
            key={frame.id}
            onClick={() => onSelect(frame)}
            className={`rounded-xl px-2 py-1.5 text-[11px] font-black ring-1 transition ${
              active
                ? "bg-sky-600 text-white ring-sky-600"
                : "bg-white text-slate-600 ring-slate-200 hover:bg-sky-50 hover:text-sky-700"
            }`}
          >
            <span className="block">{frame.label}</span>
            <span className={`block text-[9px] ${active ? "text-sky-100" : "text-slate-400"}`}>{frame.frame}f</span>
          </button>
        );
      })}
    </div>
  );
}
